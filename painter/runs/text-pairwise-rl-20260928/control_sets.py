"""Build one SFT control eval for validation plus the fixed train probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def combine(validation: Path, probe: Path, output: Path) -> dict:
    val = json.loads(validation.read_text())["cases"]
    train = json.loads(probe.read_text())["cases"]
    rows = val + train
    if not val or not train or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("control sets must be nonempty and disjoint")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "painter-scaled-control-v1",
                                  "cases": rows}, indent=2, sort_keys=True) + "\n")
    return {"validation": len(val), "probe": len(train), "combined": len(rows)}


def subset(summary: Path, manifest: Path, output: Path) -> dict:
    source = json.loads(summary.read_text())
    ids = {row["id"] for row in json.loads(manifest.read_text())["cases"]}
    if not ids or len(ids) != len(json.loads(manifest.read_text())["cases"]):
        raise ValueError("empty or duplicate subset IDs")
    rows = [row for row in source["rows"] if row["id"] in ids]
    if len(rows) != len(ids):
        raise ValueError("control summary lacks subset IDs")
    comparable = [row for row in rows if row["status"] in {"scored", "model_invalid"}]
    result = {**source, "expected": len(rows), "comparable": len(comparable),
              "censored": sum(row["status"] == "censored" for row in rows),
              "valid_candidates": sum(row["status"] == "scored" for row in rows),
              "wins": sum(row["score"] > .5 for row in comparable),
              "ties": sum(row["score"] == .5 for row in comparable),
              "losses": sum(row["score"] < .5 for row in comparable),
              "mean_score": (sum(row["score"] for row in comparable) / len(comparable)
                             if comparable else None), "rows": rows,
              "subset_manifest": str(manifest)}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"expected": result["expected"], "comparable": result["comparable"],
            "mean_score": result["mean_score"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("combine")
    builder.add_argument("--validation", type=Path, required=True)
    builder.add_argument("--probe", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    splitter = sub.add_parser("subset")
    splitter.add_argument("--summary", type=Path, required=True)
    splitter.add_argument("--manifest", type=Path, required=True)
    splitter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.validation, args.probe, args.output) if args.command == "combine" else (
        subset(args.summary, args.manifest, args.output))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
