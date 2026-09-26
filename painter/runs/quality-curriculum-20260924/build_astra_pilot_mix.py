#!/usr/bin/env python3
"""Build one finite, diverse Astra SFT pilot with procedural retention."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate source IDs: {path}")
    if any([message.get("role") for message in row["messages"]] != ["system", "user", "assistant"] for row in rows):
        raise ValueError(f"unexpected chat shape: {path}")
    return rows


def emit(path: Path, rows: list[dict]) -> str:
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def build(reviewed_path: Path, foundation_path: Path, validation_path: Path,
          reference_manifest_path: Path, output: Path, *, seed: int,
          stage: str = "all") -> dict:
    reviewed = read_rows(reviewed_path)
    foundation = read_rows(foundation_path)
    validation = read_rows(validation_path)
    if len(reviewed) != 60 or len({row["source_group"] for row in reviewed}) != 60:
        raise ValueError("pilot must contain 60 distinct reviewed references")
    if Counter(row["example_kind"] for row in reviewed) != {"initial_paint": 39, "multi_turn_revision": 21}:
        raise ValueError("unexpected first-paint/revision mix")
    manifest = json.loads(reference_manifest_path.read_text())
    train_ids = {row["id"] for row in manifest["references"] if row["split"] == "train"}
    if len(train_ids) != 100 or any(row["source_group"] not in train_ids for row in reviewed):
        raise ValueError("reviewed data includes a non-training reference")
    if any(row.get("split") != "train" or row.get("source_kind") != "reviewed_astra_teacher"
           or not row.get("complete_target") for row in reviewed):
        raise ValueError("reviewed rows have unexpected status")
    if any(row.get("split") != "train" for row in foundation):
        raise ValueError("foundation contains non-training rows")
    if any(row.get("split") not in {"dev", "validation"} for row in validation):
        raise ValueError("foundation validation contains training rows")
    if len(validation) != 36:
        raise ValueError("expected the frozen 36 procedural validation rows")

    rng = random.Random(seed)
    initial = [row for row in reviewed if row["example_kind"] == "initial_paint"]
    revisions = [row for row in reviewed if row["example_kind"] == "multi_turn_revision"]
    rng.shuffle(initial)
    rng.shuffle(revisions)
    if stage not in {"all", "first-paint"}:
        raise ValueError("stage must be all or first-paint")
    curriculum = initial + revisions if stage == "all" else initial
    retention_count = 60 if stage == "all" else 41
    # Keep distinct retention rows across all six frozen foundation families.
    families: dict[str, list[dict]] = defaultdict(list)
    for row in foundation:
        families[row["family"]].append(row)
    if set(families) != {"sphere", "box", "cylinder", "ribbons", "flower", "cup"}:
        raise ValueError("unexpected procedural retention families")
    retained = []
    for index, family in enumerate(sorted(families)):
        bucket = families[family]
        take = 10 if stage == "all" else (7 if index < 5 else 6)
        if len(bucket) < take:
            raise ValueError(f"too few retention examples for {family}")
        rng.shuffle(bucket)
        retained.extend(bucket[:take])
    rng.shuffle(retained)
    if len(retained) != retention_count:
        raise ValueError("wrong retention count")

    train = []
    steps = 30 if stage == "all" else 20
    for batch in range(steps):
        reviewed_slice = curriculum[2 * batch:2 * batch + 2]
        retention_slice = retained[2 * batch:2 * batch + (4 - len(reviewed_slice))]
        if len(reviewed_slice) + len(retention_slice) != 4:
            raise ValueError("incomplete batch")
        for row in reviewed_slice:
            train.append(row)
        for row in retention_slice:
            item = dict(row)
            item["id"] = f"retention__{row['id']}"
            item["source_example_id"] = row["id"]
            item["source_kind"] = "procedural_retention"
            item["example_kind"] = "foundation"
            train.append(item)
    if len(train) != steps * 4 or len({row["id"] for row in train}) != steps * 4:
        raise ValueError("finite SFT schedule is malformed")
    heldout = []
    for row in validation:
        item = dict(row)
        item["split"] = "validation"
        item["source_kind"] = "procedural_retention"
        item["example_kind"] = "foundation_validation"
        heldout.append(item)
    output.mkdir(parents=True, exist_ok=True)
    train_sha = emit(output / "train.jsonl", train)
    val_sha = emit(output / "validation.jsonl", heldout)
    report = {
        "schema": "painter.astra-pilot-sft-mix.v1", "seed": seed, "stage": stage,
        "reference_manifest_sha256": sha(reference_manifest_path),
        "source_reviewed_rows": 60, "reviewed_rows": len(curriculum),
        "reviewed_scenes": len({row["source_group"] for row in curriculum}),
        "initial_paint": len(initial),
        "multi_turn_revision": len(revisions) if stage == "all" else 0,
        "reviewed_exposures": len(curriculum), "retention_exposures": len(retained),
        "retention_family_counts": dict(sorted(Counter(row["family"] for row in retained).items())),
        "batch_size": 4, "optimizer_steps": steps, "checkpoint_steps": list(range(10, steps + 1, 10)),
        "validation_rows": len(heldout),
        "input_sha256": {"reviewed": sha(reviewed_path), "foundation": sha(foundation_path),
                         "foundation_validation": sha(validation_path)},
        "output_sha256": {"train": train_sha, "validation": val_sha},
        "order": "One exposure per selected reviewed scene; initial paintings before revisions, seeded within each phase; four-row batches contain two reviewed and two distinct retention rows except the first-paint final batch (one reviewed, three retention).",
        "validation_limit": "Procedural validation is not a visual-quality result; compare against the frozen 28-photo development set.",
    }
    (output / "mix-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--foundation-validation", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--stage", choices=("all", "first-paint"), default="all")
    args = parser.parse_args()
    print(json.dumps(build(args.reviewed, args.foundation, args.foundation_validation,
                           args.reference_manifest, args.output, seed=args.seed,
                           stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
