"""Audit the exact Prime tokenizer/processor path before SFT.

The audit deliberately expands each sample with a very large temporary
window.  Any target longer than the configured sequence budget fails before
training; no truncation or silent loss-mask repair is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


DATA_FIELDS = (
    "id", "source_group", "split", "family", "source_kind", "example_kind",
    "complete_target", "reference_sha256", "target_sha256", "current_canvas_sha256",
    "current_program_sha256", "messages",
)


def training_projection(row: dict) -> dict:
    """Drop heterogeneous provenance fields before Arrow/Prime schema inference."""
    return {key: row[key] for key in DATA_FIELDS if key in row}


def dataset_path(data_dir: Path, split: str) -> Path:
    """Resolve the same input for processing and the reported digest."""
    candidates = data_dir / f"candidates-{split}.jsonl"
    if candidates.is_file():
        return candidates
    return data_dir / f"{split}.jsonl"


def audit(config_path: Path, output: Path, brush_root: Path) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("processor audit runs on the Linux training node")
    sys.path[:0] = [
        str(brush_root / "integrations/speedpainting/mixed_sft"),
        str(brush_root / "integrations/prime/watercolour_sft"),
    ]
    from datasets import Dataset
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.trainer.model import setup_processor, setup_tokenizer
    from renderers.base import create_renderer
    strict_source = brush_root / "integrations/speedpainting/mixed_sft/train.py"
    strict_spec = importlib.util.spec_from_file_location("photo_curriculum_mixed_sft_audit", strict_source)
    if strict_spec is None or strict_spec.loader is None:
        raise ImportError(f"missing pinned mixed SFT source: {strict_source}")
    strict_module = importlib.util.module_from_spec(strict_spec)
    strict_spec.loader.exec_module(strict_module)
    StrictSFTDataset = strict_module.StrictSFTDataset

    config = SFTConfig.model_validate(tomllib.loads(config_path.read_text()))
    tokenizer = setup_tokenizer(config.tokenizer)
    renderer = create_renderer(tokenizer, config.renderer)
    renderer._processor = setup_processor(config.model)
    result = []
    source_paths = {}
    for split in ("train", "validation"):
        source = dataset_path(Path(config.data.name), split)
        source_paths[split] = source
        raw_rows = list(rows(source))
        if not raw_rows:
            raise ValueError(f"{split} is empty: {source}")
        clean_rows = [training_projection(row) for row in raw_rows]
        wrapper = StrictSFTDataset(Dataset.from_list([clean_rows[0]]), renderer,
                                   seq_len=2**62, multimodal=True,
                                   loss_mask_config=config.data.loss_mask)
        for raw, row in zip(raw_rows, clean_rows):
            wrapper.dataset = Dataset.from_list([row])
            sample = wrapper._process(row)
            if sample is None:
                raise ValueError(f"{raw.get('id')}: no trainable assistant tokens")
            total = len(sample["input_ids"])
            masks = list(sample["loss_mask"])
            supervised = sum(bool(x) for x in masks)
            if not 0 < supervised < total:
                raise ValueError(f"{raw.get('id')}: invalid assistant-only loss mask")
            image_types = sample.get("mm_token_type_ids") or [0] * total
            if any(masks[i] and image_types[i + 1] for i in range(min(total - 1, len(image_types) - 1))):
                raise ValueError(f"{raw.get('id')}: image token contributes to assistant loss")
            target = "".join(p.get("text", "") for p in row["messages"][-1]["content"] if p.get("type") == "text")
            eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
            expected = tokenizer.encode(target, add_special_tokens=False) + [eos]
            learned = [token for token, mask in zip(sample["target_ids"], masks) if mask]
            if learned != expected:
                raise ValueError(f"{raw.get('id')}: assistant target/mask/tokenizer parity mismatch")
            record = {
                "id": raw["id"], "split": split,
                "source_kind": raw.get("source_kind", "existing"),
                "example_kind": raw.get("example_kind", "existing"),
                "source_group": raw.get("source_group", raw["id"]),
                "total_tokens": total, "assistant_tokens": supervised,
                "image_tokens": sum(bool(x) for x in image_types),
                "fits": total <= config.data.seq_len,
            }
            if not record["fits"]:
                raise ValueError(f"{row.get('id')}: {total} tokens exceed seq_len {config.data.seq_len}; refusing truncation")
            result.append(record)
    report = {
        "status": "passed", "seq_len": config.data.seq_len,
        "mask": "assistant_only_exact_target_plus_im_end", "no_truncation": True,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "processor_class": type(renderer._processor).__name__,
        "train_rows": sum(r["split"] == "train" for r in result),
        "validation_rows": sum(r["split"] == "validation" for r in result),
        "max_tokens": max(r["total_tokens"] for r in result),
        "max_assistant_tokens": max(r["assistant_tokens"] for r in result),
        "dataset_sha256": {
            split: hashlib.sha256(source_paths[split].read_bytes()).hexdigest()
            for split in ("train", "validation")
        },
        "examples": result,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--brush-root", type=Path, default=Path(os.environ.get("BRUSH_ROOT", "source/brush-rl")))
    args = p.parse_args()
    report = audit(args.config.resolve(), args.output.resolve(), args.brush_root.resolve())
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
