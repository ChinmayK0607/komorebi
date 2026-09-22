#!/usr/bin/env python3
"""CPU-only prompt-route parity diagnostic for the native painting run.

This compares Prime's hand-written ``qwen3.5`` renderer with the pinned
Hugging Face tokenizer/processor on the same canonical conversations.  It
loads tokenizer and image-processor metadata only: no model weights, CUDA,
HTTP endpoint, generation, or rendering are used.

Run on the Linux node with the Prime checkout and a locally cached model, for
example::

    python diagnostics/compare_qwen35_prompt_routes.py \
      --prime-root /root/painter/prime-rl \
      --model /tmp/photo-curriculum-sft-20260922-models/<revision> \
      --image /root/painter/.../reference.jpg \
      --output diagnostics/qwen35-route-parity.json

The image case uses the Qwen-VL reference path from the renderers' own parity
tests: ``processor.apply_chat_template(..., tokenize=False)`` followed by
``processor(images=..., text=...)``.  A missing local model/processor is
reported as ``limited`` rather than silently treated as parity.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from typing import Any


# Make accidental GPU use impossible before importing transformers/Pillow.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prime-root",
        default=os.environ.get("PRIME_ROOT", "/root/painter/prime-rl"),
        help="pinned Prime checkout containing deps/renderers",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_DIR"),
        required=os.environ.get("MODEL_DIR") is None,
        help="local model/tokenizer snapshot; never downloaded by this script",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="existing local reference PNG/JPEG for image cases",
    )
    parser.add_argument("--output", default=None, help="optional JSON result path")
    return parser


def _load_messages(image: Any | None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return representative fresh, text-only, and multi-turn histories."""
    # Importing the contract keeps this diagnostic tied to the active native
    # prompt text rather than copying a second prompt implementation.
    from painter.contract import NEXT_VERSION, SYSTEM

    text_user = [
        {"type": "text", "text": "REFERENCE"},
        {"type": "text", "text": NEXT_VERSION},
    ]
    cases: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "text_only_fresh",
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text_user},
            ],
        )
    ]
    if image is not None:
        image_part = {"type": "image", "image": image}
        fresh = [
            {"type": "text", "text": "REFERENCE"},
            image_part,
            {"type": "text", "text": NEXT_VERSION},
        ]
        cases.append(
            (
                "image_fresh",
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": fresh},
                ],
            )
        )
        # Keep the same reference/current-canvas image bytes. This exercises
        # the history path without creating or transforming any image.
        current = [
            {"type": "text", "text": "REFERENCE"},
            image_part,
            {"type": "text", "text": "CURRENT CANVAS"},
            image_part,
            {"type": "text", "text": "Current program:\nfunction draw(){ noLoop(); }"},
            {"type": "text", "text": NEXT_VERSION},
        ]
        cases.append(
            (
                "image_multiturn_history",
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": fresh},
                    {
                        "role": "assistant",
                        "content": "Plan.\n```javascript\nfunction draw(){ noLoop(); }\n```",
                    },
                    {"role": "user", "content": current},
                ],
            )
        )
    return cases


def _parts(messages: list[dict[str, Any]]) -> list[Any]:
    """Collect image objects in message order for HF processing."""
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in {"image", "image_url"}:
                continue
            raw = item.get("image")
            if raw is not None:
                images.append(raw)
    return images


def _coerce_ids(value: Any) -> list[int]:
    """Extract one input-id row from HF list, array, or BatchEncoding output."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Hugging Face output has no input_ids field")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"input_ids has unsupported type {type(value).__name__}")
    # Transformers may return a one-row BatchEncoding even when the caller
    # asks for one conversation. Keep this strict so a surprising batch is
    # reported as a diagnostic limitation instead of silently comparing row 0.
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"expected one input_ids row, found {len(value)}")
        value = list(value[0])
    return [int(token) for token in value]


def _hf_ids(processor: Any, tokenizer: Any, messages: list[dict[str, Any]], *, add_gp: bool) -> tuple[list[int], dict[str, Any]]:
    """Build the HF reference token IDs without loading a model."""
    image_paths = _parts(messages)
    if not image_paths:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_gp,
            enable_thinking=False,
        )
        return _coerce_ids(value), {"route": "tokenizer.apply_chat_template", "images": 0}

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_gp,
        enable_thinking=False,
    )
    out = processor(
        images=image_paths,
        text=text,
        return_tensors="np",
    )
    ids = _coerce_ids(out["input_ids"])
    grid = out.get("image_grid_thw")
    grid_shape = list(getattr(grid, "shape", ())) if grid is not None else []
    return ids, {
        "route": "processor.apply_chat_template+processor",
        "images": len(image_paths),
        "image_grid_thw_shape": grid_shape,
    }


def _prime_ids(renderer: Any, messages: list[dict[str, Any]], *, add_gp: bool) -> tuple[list[int], dict[str, Any]]:
    rendered = renderer.render(messages, add_generation_prompt=add_gp)
    mm = rendered.multi_modal_data
    metadata: dict[str, Any] = {"images": 0}
    if mm is not None:
        metadata["images"] = len(mm.mm_hashes.get("image", []))
        metadata["placeholder_ranges"] = [
            {"offset": p.offset, "length": p.length}
            for p in mm.mm_placeholders.get("image", [])
        ]
        metadata["image_item_keys"] = [
            sorted(item.keys()) for item in mm.mm_items.get("image", [])
        ]
    return [int(x) for x in rendered.token_ids], metadata


def _first_mismatch(left: list[int], right: list[int]) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {"index": index, "prime": left[index], "hf": right[index]}
    if len(left) != len(right):
        return {"index": limit, "prime": left[limit:limit + 8], "hf": right[limit:limit + 8]}
    return None


def _load_runtime(prime_root: Path, model: str) -> tuple[Any, Any, Any]:
    renderers_root = prime_root / "deps" / "renderers"
    if not renderers_root.is_dir():
        raise FileNotFoundError(f"Prime renderers checkout is missing: {renderers_root}")
    sys.path.insert(0, str(renderers_root))
    try:
        from transformers import AutoProcessor, AutoTokenizer
        from renderers import Qwen35Renderer, Qwen35RendererConfig
    except Exception as exc:  # pragma: no cover - node dependency diagnostic
        raise RuntimeError(f"pinned tokenizer/renderer imports unavailable: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    renderer = Qwen35Renderer(
        tokenizer,
        Qwen35RendererConfig(enable_thinking=False),
        processor=processor,
    )
    return tokenizer, processor, renderer


def main() -> int:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    prime_root = Path(args.prime_root).expanduser().resolve()
    model = str(Path(args.model).expanduser().resolve()) if Path(args.model).exists() else args.model
    image = Path(args.image).expanduser().resolve() if args.image else None
    result: dict[str, Any] = {
        "schema": "qwen35-prompt-route-parity.v1",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "prime_root": str(prime_root),
        "model": model,
        "image": str(image) if image else None,
        "cases": [],
    }
    if image is not None and not image.is_file():
        result["status"] = "limited"
        result["limitation"] = f"image path is missing: {image}"
        image = None
    if image is None and "limitation" not in result:
        result["limitation"] = "no --image supplied; multimodal cases were skipped"
    image_object: Any | None = None
    if image is not None:
        try:
            from PIL import Image

            # Loading an existing image is needed for the HF processor parity
            # path; no pixels are resized, composited, or written.
            with Image.open(image) as opened:
                image_object = opened.convert("RGB")
        except Exception as exc:
            result["status"] = "limited"
            result["limitation"] = f"image could not be opened: {type(exc).__name__}: {exc}"
            image_object = None
    try:
        tokenizer, processor, renderer = _load_runtime(prime_root, model)
    except Exception as exc:
        result["status"] = "limited"
        result["limitation"] = f"local tokenizer/processor unavailable: {type(exc).__name__}: {exc}"
        _write_result(result, args.output)
        return 2

    case_errors = False
    for name, messages in _load_messages(image_object):
        record: dict[str, Any] = {"name": name, "add_generation_prompt": True, "message_count": len(messages)}
        try:
            prime, prime_meta = _prime_ids(renderer, messages, add_gp=True)
            hf, hf_meta = _hf_ids(processor, tokenizer, messages, add_gp=True)
            record.update(
                {
                    "prime_length": len(prime),
                    "hf_length": len(hf),
                    "equal": prime == hf,
                    "first_mismatch": _first_mismatch(prime, hf),
                    "prime": prime_meta,
                    "hf": hf_meta,
                }
            )
        except Exception as exc:
            case_errors = True
            record.update({"equal": None, "error": f"{type(exc).__name__}: {exc}"})
        result["cases"].append(record)
    if "limitation" in result or case_errors:
        result["status"] = "limited"
    else:
        result["status"] = "ok" if all(case.get("equal") is True for case in result["cases"]) else "mismatch"
    _write_result(result, args.output)
    return 0 if result["status"] == "ok" else 1


def _write_result(result: dict[str, Any], output: str | None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if output:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
