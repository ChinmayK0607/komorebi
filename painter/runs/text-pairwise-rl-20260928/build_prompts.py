"""Freeze a simple-to-intermediate text-only pool before SFT baseline sampling.

The 600-scene audit is used only as a source of authored briefs. Its teacher
paintings and labels are not RL opponents or student evaluation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SEED = 20260928
COUNTS = {"simple": (24, 8, 8), "intermediate": (16, 8, 8)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output: Path) -> dict:
    raw = json.loads(source.read_text())
    if not isinstance(raw, list):
        raise ValueError("teacher audit must be a row list")
    selected: list[dict] = []
    for tier, counts in COUNTS.items():
        candidates = [row for row in raw if row.get("mode") == "text_to_image"
                      and str(row.get("category", "")).startswith(tier + " /")]
        if len(candidates) != sum(counts):
            raise ValueError(f"expected {sum(counts)} {tier} prompts; found {len(candidates)}")
        candidates.sort(key=lambda row: row["audit_id"])
        random.Random(SEED + len(tier)).shuffle(candidates)
        for split, subset in zip(("train", "validation", "reserve"),
                                 (candidates[:counts[0]],
                                  candidates[counts[0]:sum(counts[:2])],
                                  candidates[sum(counts[:2]):]), strict=True):
            for row in subset:
                text = row.get("prompt")
                if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                    raise ValueError(f"invalid prompt: {row.get('audit_id')}")
                selected.append({"id": row["audit_id"], "family": tier,
                                 "source_group": row["audit_id"], "split": split,
                                 "category": row["category"], "brief": text.strip()})
    if len({row["id"] for row in selected}) != 72 or len({row["brief"] for row in selected}) != 72:
        raise ValueError("duplicate prompt identity or text")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in selected:
        path = output / "prompts" / (row["id"] + ".txt")
        path.parent.mkdir(exist_ok=True)
        path.write_text(row.pop("brief") + "\n", encoding="utf-8")
        cases.append({**row, "reference_kind": "text",
                      "reference": path.relative_to(output).as_posix(),
                      "reference_sha256": digest(path)})
    manifest = {"schema": "painter-text-pairwise-pool-v1",
                "source_audit_sha256": digest(source), "selection_seed": SEED,
                "cases": cases}
    target = output / "pool.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"pool_sha256": digest(target), "source_audit_sha256": digest(source),
            "counts": {split: sum(row["split"] == split for row in cases)
                       for split in ("train", "validation", "reserve")},
            "families": {tier: sum(row["family"] == tier for row in cases) for tier in COUNTS}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
