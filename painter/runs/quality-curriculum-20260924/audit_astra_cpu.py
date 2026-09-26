#!/usr/bin/env python3
"""Linux CPU token-shape preflight for the public Astra SFT export.

This uses the pinned Qwen processor but is not the Prime loss-mask audit. The
training node must still run the exact Prime audit before any optimizer step.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor


MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DATA_REPO = "CK0607/komorebi-painter-teachers"
DATA_FILE = "curricula/astra-high-100/sft-combined60.jsonl"
DATA_SHA = "0348b556a9f02e956c6f8d582f9b731be4b01ec1f45754c2f13cfcbde25657ca"


def decode_image(url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        raise ValueError("unexpected image URL format")
    image = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):], validate=True)))
    image.load()
    return image.convert("RGB")


def main() -> None:
    if sys.platform != "linux":
        raise RuntimeError("CPU preflight belongs on Linux")
    path = Path(hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset", token=False))
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("public SFT data digest mismatch")
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION,
                                              trust_remote_code=False)
    observations = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        conversation, images = [], []
        for message in row["messages"]:
            content = []
            for part in message["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    images.append(decode_image(part["image_url"]["url"]))
                    content.append({"type": "image"})
                else:
                    raise ValueError(f"unknown message part: {part['type']}")
            conversation.append({"role": message["role"], "content": content})
        rendered = processor.apply_chat_template(conversation, tokenize=False,
                                                 add_generation_prompt=False)
        encoded = processor(text=[rendered], images=images or None, padding=False)
        ids = encoded["input_ids"]
        token_count = len(ids[0] if ids and isinstance(ids[0], list) else ids)
        observations.append({"id": row["id"], "tokens": token_count,
                             "images": len(images), "kind": row["example_kind"]})
    if len(observations) != 60:
        raise ValueError("expected 60 reviewed rows")
    report = {
        "schema": "painter.astra-qwen-cpu-shape-preflight.v1",
        "status": "passed" if max(row["tokens"] for row in observations) <= 16384 else "overlength",
        "limit": "Processor token shape only; not the exact Prime assistant loss-mask audit",
        "model": MODEL, "model_revision": MODEL_REVISION,
        "processor_class": type(processor).__name__,
        "transformers_version": __import__("transformers").__version__,
        "data_sha256": DATA_SHA,
        "rows": len(observations), "max_tokens": max(row["tokens"] for row in observations),
        "over_16384": [row for row in observations if row["tokens"] > 16384],
        "largest": sorted(observations, key=lambda row: row["tokens"], reverse=True)[:10],
    }
    output = Path(os.environ.get("ASTRA_CPU_AUDIT_OUTPUT", "astra-cpu-audit.json"))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
