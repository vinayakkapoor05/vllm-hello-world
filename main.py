import argparse
import base64
from pathlib import Path
import time
import json
import logging
import os

from waggle.plugin import Plugin
from pydantic import BaseModel, ValidationError
from openai import OpenAI


class ImageSummary(BaseModel):
    short_description: str
    long_description: str
    objects: list[str]


def run(plugin: Plugin, host: str, model: str, prompt: str, images: list[Path]):
    logging.info("Connecting to vLLM")
    client = OpenAI(base_url=f"http://{host}/v1", api_key="EMPTY")
    for image in images:
        encoded = base64.b64encode(image.read_bytes()).decode()
        data_url = f"data:image/jpeg;base64,{encoded}"
        system_prompt = (
            "You are a vision model. Examine the image and respond ONLY with JSON "
            "matching this exact schema:\n"
            "{\n"
            '  "short_description": "string",\n'
            '  "long_description": "string",\n'
            '  "objects": ["string", ...]\n'
            "}\n"
            "Do not add any text outside the JSON."
        )
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        duration = round(time.monotonic() - start, 3)
        raw_output = response.choices[0].message.content
        try:
            parsed = ImageSummary.model_validate_json(raw_output)
        except ValidationError:
            parsed = ImageSummary(
                short_description="See long_description",
                long_description=raw_output,
                objects=[],
            )
        output = {
            "input": str(image),
            "model": model,
            "prompt": prompt,
            "output": json.loads(parsed.model_dump_json()),
            "duration": duration,
        }
        output_json = json.dumps(output, separators=(",", ":"), sort_keys=True)
        logging.info("Publishing inference results")
        plugin.publish("structured_inference_log", output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--host",
        default=os.getenv("VLLM_HOST", "vllm.default.svc.cluster.local:8000"),
        help="vLLM host",
    )
    parser.add_argument("-m", "--model", default="/model", help="model served by vLLM")
    parser.add_argument(
        "-p",
        "--prompt",
        default="Describe the contents of this image.",
    )
    parser.add_argument("images", nargs="*", type=Path, help="Input images")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with Plugin() as plugin:
        run(
            plugin=plugin,
            host=args.host,
            model=args.model,
            prompt=args.prompt,
            images=args.images,
        )
