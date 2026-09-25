#!/usr/bin/env python3
"""Build a finite first-paint/revision SFT curriculum with procedural retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate source IDs: {path}")
    if any([message.get("role") for message in row["messages"]] != ["system", "user", "assistant"] for row in rows):
        raise ValueError(f"unexpected chat shape: {path}")
    return rows


def emit(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build(reviewed: Path, foundation: Path, foundation_val: Path, output: Path, *, repeats: int, seed: int,
          checkpoint_multiple: int = 16) -> dict:
    if not 1 <= repeats <= 8:
        raise ValueError("reviewed-turn repeats must be 1..8")
    if not 1 <= checkpoint_multiple <= 128:
        raise ValueError("checkpoint_multiple must be 1..128")
    turns, retained, validation = read(reviewed), read(foundation), read(foundation_val)
    if any(row.get("split") != "train" for row in turns + retained):
        raise ValueError("training input has a non-train row")
    if any(row.get("split") not in {"dev", "validation"} for row in validation):
        raise ValueError("foundation validation has a non-validation row")
    seen_groups = {row["source_group"] for row in turns}
    if len(seen_groups) < 2 or any(row.get("source_kind") != "reviewed_teacher_revision" for row in turns):
        raise ValueError("reviewed source must cover multiple verified scenes")
    rng = random.Random(seed)
    staged = []
    for exposure in range(repeats):
        epoch = list(turns)
        if exposure == 0:
            epoch.sort(key=lambda row: (row["teacher_metadata"]["turn"], row["source_group"]))
        else:
            rng.shuffle(epoch)
        for row in epoch:
            staged.append((row, exposure))
    planned = len(staged)
    padding_pool = list(turns)
    rng.shuffle(padding_pool)
    padding_count = 0
    while len(staged) % 3 or (len(staged) // 3) % checkpoint_multiple:
        # Every full group contains three reviewed examples. Deterministically
        # cycle a shuffled pool for the small amount of checkpoint padding.
        staged.append((padding_pool[padding_count % len(padding_pool)], repeats + padding_count // len(padding_pool)))
        padding_count += 1
    retained_pool = list(retained)
    rng.shuffle(retained_pool)
    output.mkdir(parents=True, exist_ok=True)
    emissions = []
    for index in range(0, len(staged), 3):
        for row, exposure in staged[index:index + 3]:
            item = dict(row)
            item["id"] = f"{row['id']}__exposure_{exposure:02d}__{len(emissions):05d}"
            item["source_example_id"] = row["id"]
            emissions.append(item)
        source = retained_pool[(index // 3) % len(retained_pool)]
        item = dict(source)
        item["id"] = f"foundation__{source['id']}__{len(emissions):05d}"
        item["source_example_id"] = source["id"]
        item["source_kind"] = "procedural_retention"
        item["example_kind"] = "foundation"
        emissions.append(item)
    if len(emissions) % 4 or len({row["id"] for row in emissions}) != len(emissions):
        raise ValueError("finite SFT schedule is malformed")
    heldout = []
    for row in validation:
        item = dict(row)
        item["split"] = "validation"
        item["source_kind"] = "procedural_retention"
        item["example_kind"] = "foundation_validation"
        heldout.append(item)
    train_sha = emit(output / "train.jsonl", emissions)
    val_sha = emit(output / "validation.jsonl", heldout)
    report = {
        "schema": "painter.multiturn-sft-mix.v1", "seed": seed, "reviewed_repeats": repeats,
        "reviewed_rows": len(turns), "reviewed_scenes": len(seen_groups),
        "reviewed_exposures": len(staged), "retention_exposures": len(emissions) // 4,
        "checkpoint_multiple": checkpoint_multiple, "padding_exposures": len(staged) - planned,
        "padding_fraction_of_planned_reviewed": round((len(staged) - planned) / planned, 6),
        "optimizer_steps_at_batch_4": len(emissions) // 4, "validation_rows": len(heldout),
        "training_source_kinds": dict(Counter(row["source_kind"] for row in emissions)),
        "input_sha256": {"reviewed": sha(reviewed), "foundation": sha(foundation), "foundation_validation": sha(foundation_val)},
        "output_sha256": {"train": train_sha, "validation": val_sha},
        "order": "first pass sorts approved turns by turn index across scenes; later passes shuffle deterministically; each four-row optimizer group has three reviewed turns and one procedural row",
        "loss": "last assistant only; reviewed-turn history resides in user context",
        "validation_limit": "procedural validation is not a visual-quality score; compare the released adapter on the fixed photo holdout",
    }
    (output / "mix-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--foundation-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--checkpoint-multiple", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(args.reviewed, args.foundation, args.foundation_validation, args.output,
                           repeats=args.repeats, seed=args.seed, checkpoint_multiple=args.checkpoint_multiple), indent=2))


if __name__ == "__main__":
    main()
