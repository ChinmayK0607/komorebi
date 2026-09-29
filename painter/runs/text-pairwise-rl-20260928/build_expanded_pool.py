"""Stage the remaining audited text briefs without changing the live RL run.

The existing 72-way split is immutable. New briefs are split before any model
evaluation, so a later scale experiment can keep its harder validation blind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HARD_VALIDATION_COUNT = 32
SPLIT_SEED = "text-paint-scale-20260928-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, original: Path, output: Path) -> dict:
    audited = json.loads(source.read_text())
    old = json.loads(original.read_text())["cases"]
    if not isinstance(audited, list) or not isinstance(old, list):
        raise ValueError("expected row lists")
    old_by_id = {row["id"]: row for row in old}
    text_rows = [row for row in audited if row.get("mode") == "text_to_image"]
    by_id = {row["audit_id"]: row for row in text_rows}
    if len(by_id) != len(text_rows) or len(old_by_id) != len(old):
        raise ValueError("duplicate brief identifiers")
    if not set(old_by_id) <= set(by_id):
        raise ValueError("original split is missing from source")
    for ident, row in old_by_id.items():
        prompt = by_id[ident].get("prompt")
        path = original.parent / row["reference"]
        if not isinstance(prompt, str) or prompt.strip() + "\n" != path.read_text():
            raise ValueError(f"changed original prompt: {ident}")
        if sha256(path) != row["reference_sha256"]:
            raise ValueError(f"changed original prompt hash: {ident}")
    new = [row for row in text_rows if row["audit_id"] not in old_by_id]
    for row in new:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError(f"invalid new prompt: {row['audit_id']}")
    if len({row["prompt"].strip() for row in text_rows}) != len(text_rows):
        raise ValueError("duplicate prompt text")
    ranked = sorted(new, key=lambda row: hashlib.sha256(
        (SPLIT_SEED + ":" + row["audit_id"]).encode()).hexdigest())
    hard_val = {row["audit_id"] for row in ranked[:HARD_VALIDATION_COUNT]}
    if len(hard_val) != HARD_VALIDATION_COUNT:
        raise ValueError("insufficient new briefs for blind validation")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in sorted(text_rows, key=lambda item: item["audit_id"]):
        ident = row["audit_id"]
        split = old_by_id[ident]["split"] if ident in old_by_id else (
            "validation" if ident in hard_val else "train")
        phase = "foundation" if ident in old_by_id else "complex"
        prompt_path = output / "prompts" / f"{ident}.txt"
        prompt_path.parent.mkdir(exist_ok=True)
        prompt_path.write_text(row["prompt"].strip() + "\n")
        cases.append({
            "id": ident, "source_group": ident, "mode": "text_to_image",
            "phase": phase, "family": old_by_id[ident]["family"] if ident in old_by_id else "complex",
            "split": split, "category": row.get("category"),
            "reference_kind": "text",
            "reference": prompt_path.relative_to(output).as_posix(),
            "reference_sha256": sha256(prompt_path),
        })
    manifest = {
        "schema": "painter-expanded-text-pool-v2",
        "source_audit_sha256": sha256(source),
        "original_pool_sha256": sha256(original),
        "split_seed": SPLIT_SEED,
        "cases": cases,
    }
    target = output / "pool.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "pool_sha256": sha256(target), "source_audit_sha256": sha256(source),
        "total": len(cases),
        "foundation": len(old), "complex_train": len(new) - len(hard_val),
        "complex_validation": len(hard_val),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.original, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
