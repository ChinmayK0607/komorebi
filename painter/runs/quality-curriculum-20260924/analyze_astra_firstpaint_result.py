#!/usr/bin/env python3
"""Compare matched first-paint rollout validity without treating it as beauty."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(root: Path, policy: str) -> dict[str, dict]:
    paths = list((root / "eval" / policy / "results").glob("*/traces.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one {policy} traces.jsonl, got {len(paths)}")
    found = {}
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        for trace in json.loads(line).get("traces", []):
            info = trace.get("info") or {}
            task_id = info.get("task_id")
            if not task_id or task_id in found:
                raise ValueError(f"missing/duplicate {policy} task id: {task_id}")
            found[task_id] = info
    return found


def first_valid(info: dict) -> bool:
    turns = info.get("turns") or []
    return bool(turns and turns[0].get("valid") is True)


def final_valid(info: dict) -> bool:
    return info.get("final_canvas_valid") is True


def report(root: Path) -> dict:
    expected = {case["id"] for case in json.loads((root / "painter/eval-prep/eval-manifest.json").read_text())["cases"]}
    a, b = load(root, "baseline"), load(root, "trained")
    if set(a) != expected or set(b) != expected or len(expected) != 28:
        raise ValueError("incomplete or mismatched frozen evaluation")
    first_pairs = Counter()
    final_pairs = Counter()
    by_case = []
    for case_id in sorted(expected):
        x, y = a[case_id], b[case_id]
        if x.get("reference_sha256") != y.get("reference_sha256"):
            raise ValueError(f"reference mismatch: {case_id}")
        av, bv = first_valid(x), first_valid(y)
        af, bf = final_valid(x), final_valid(y)
        first_pairs[(av, bv)] += 1
        final_pairs[(af, bf)] += 1
        by_case.append({"case_id": case_id, "family": x.get("family"),
                        "baseline_first_valid": av, "trained_first_valid": bv,
                        "baseline_final_valid": af, "trained_final_valid": bf,
                        "baseline_turns": len(x.get("turns") or []),
                        "trained_turns": len(y.get("turns") or []),
                        "baseline_operational_cutoff": x.get("operational_cutoff"),
                        "trained_operational_cutoff": y.get("operational_cutoff")})
    def counts(pairs: Counter) -> dict:
        return {"both_valid": pairs[(True, True)], "baseline_only": pairs[(True, False)],
                "trained_only": pairs[(False, True)], "both_invalid": pairs[(False, False)]}
    return {"schema": "painter.astra-firstpaint-matched-analysis.v1", "cases": 28,
            "first_turn_validity_pairs": counts(first_pairs),
            "final_validity_pairs": counts(final_pairs),
            "quality_status": "unjudged; validity is not visual quality",
            "by_case": by_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = report(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "by_case"}))


if __name__ == "__main__":
    main()
