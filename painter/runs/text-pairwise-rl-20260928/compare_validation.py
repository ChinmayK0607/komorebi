"""Compare two fixed-prompt validation summaries on their paired cases.

Only cases comparable under both policies enter the paired quality estimate.
Service failures stay censored; they are never converted into visual losses.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


COMPARABLE = {"scored", "model_invalid"}


def _rows(summary: dict) -> dict[str, dict]:
    rows = summary["rows"]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("duplicate validation case ID")
    if len(rows) != summary["expected"]:
        raise ValueError("incomplete validation summary")
    for row in rows:
        if row["status"] in COMPARABLE and not isinstance(row.get("score"), (int, float)):
            raise ValueError(f"missing numeric score for {row['id']}")
    return by_id


def compare(baseline: dict, candidate: dict, *, samples: int = 10000) -> dict:
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("judge protocols differ")
    a, b = _rows(baseline), _rows(candidate)
    if set(a) != set(b):
        raise ValueError("validation case IDs differ")
    if any(a[ident]["family"] != b[ident]["family"] for ident in a):
        raise ValueError("validation case families differ")
    shared = [ident for ident in sorted(a) if a[ident]["status"] in COMPARABLE
              and b[ident]["status"] in COMPARABLE]
    differences = [float(b[ident]["score"]) - float(a[ident]["score"])
                   for ident in shared]
    interval = None
    if differences:
        rng = random.Random(20260928)
        means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                       for _ in range(samples))
        interval = [means[int(.025 * samples)], means[int(.975 * samples) - 1]]
    return {
        "schema": "painter-paired-validation-comparison-v1",
        "protocol": baseline["protocol"],
        "expected": len(a), "paired_comparable": len(shared),
        "baseline_comparable": sum(row["status"] in COMPARABLE for row in a.values()),
        "candidate_comparable": sum(row["status"] in COMPARABLE for row in b.values()),
        "baseline_valid": sum(row["status"] == "scored" for row in a.values()),
        "candidate_valid": sum(row["status"] == "scored" for row in b.values()),
        "baseline_censored": sum(row["status"] == "censored" for row in a.values()),
        "candidate_censored": sum(row["status"] == "censored" for row in b.values()),
        "paired_baseline_mean": (sum(float(a[i]["score"]) for i in shared) / len(shared)
                                 if shared else None),
        "paired_candidate_mean": (sum(float(b[i]["score"]) for i in shared) / len(shared)
                                  if shared else None),
        "paired_mean_delta": (sum(differences) / len(differences) if shared else None),
        "paired_bootstrap_95pct_interval": interval,
        "paired_improved": sum(d > 1e-9 for d in differences),
        "paired_tied": sum(abs(d) <= 1e-9 for d in differences),
        "paired_regressed": sum(d < -1e-9 for d in differences),
        "cases": [{"id": ident, "family": a[ident]["family"],
                   "baseline_status": a[ident]["status"],
                   "candidate_status": b[ident]["status"],
                   "baseline_score": a[ident].get("score"),
                   "candidate_score": b[ident].get("score"),
                   "delta": (float(b[ident]["score"]) - float(a[ident]["score"])
                             if ident in shared else None)} for ident in sorted(a)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()),
                     json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
