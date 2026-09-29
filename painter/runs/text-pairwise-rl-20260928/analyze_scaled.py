"""Compare matched SFT and RL visual scores without hiding invalid cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compare_validation import compare


def _subset(summary: dict, ids: set[str]) -> dict:
    rows = [row for row in summary["rows"] if row["id"] in ids]
    return {**summary, "expected": len(rows), "rows": rows}


def analyze(baseline: dict, candidate: dict) -> dict:
    all_cases = compare(baseline, candidate)
    a = {row["id"]: row for row in baseline["rows"]}
    b = {row["id"]: row for row in candidate["rows"]}
    families = sorted({row["family"] for row in a.values()})
    by_family = {}
    for family in families:
        ids = {ident for ident, row in a.items() if row["family"] == family}
        by_family[family] = compare(_subset(baseline, ids), _subset(candidate, ids))
    both_valid = {ident for ident in a if a[ident]["status"] == "scored" and
                  b[ident]["status"] == "scored"}
    visual_only = (compare(_subset(baseline, both_valid),
                           _subset(candidate, both_valid)) if both_valid else None)
    return {"schema": "painter-scaled-rl-analysis-v1", "all": all_cases,
            "both_valid": visual_only, "by_family": by_family,
            "both_valid_ids": sorted(both_valid)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.baseline.read_text()),
                     json.loads(args.candidate.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all": {k: v for k, v in result["all"].items() if k != "cases"},
                      "both_valid": ({k: v for k, v in result["both_valid"].items()
                                      if k != "cases"} if result["both_valid"] else None),
                      "families": {name: value["paired_mean_delta"]
                                   for name, value in result["by_family"].items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
