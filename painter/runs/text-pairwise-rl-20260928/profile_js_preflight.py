#!/usr/bin/env python3
"""Read-only test of Node syntax preflight on saved p5.brush programs.

Node --check parses without executing sketches. This profiler never changes
the renderer or active reward. A subsequent pixel-identical canary is still
required before any future integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def observed_syntax_error(receipt: dict) -> bool:
    errors = receipt.get("errors") or []
    text = " ".join(map(str, errors if isinstance(errors, list) else [errors])).lower()
    return "syntaxerror" in text or "unexpected end" in text


def analyze(rollouts: Path, node: Path, *, max_programs: int | None = None,
            sample_seed: int = 20260929) -> dict:
    if not node.is_file() or not node.stat().st_mode & 0o111:
        raise ValueError("bundled Node executable is missing")
    paths = sorted(rollouts.glob("**/turn-*/program.js"))
    if not paths:
        raise ValueError("no saved programs found")
    total = len(paths)
    if max_programs is not None:
        if max_programs < 1:
            raise ValueError("max_programs must be positive")
        paths = sorted(random.Random(sample_seed).sample(paths, min(max_programs, total)))
    rows = []
    for path in paths:
        receipt_path = path.with_name("painting.json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        started = time.perf_counter()
        try:
            checked = subprocess.run([str(node), "--check", str(path)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     timeout=2, check=False)
            check_status = "accept" if checked.returncode == 0 else "reject"
        except subprocess.TimeoutExpired:
            check_status = "timeout"
        elapsed = time.perf_counter() - started
        rows.append({"program_sha256": sha(path), "check_status": check_status,
                     "check_seconds": round(elapsed, 4),
                     "renderer_valid": receipt.get("valid") if receipt else None,
                     "renderer_syntax_error": observed_syntax_error(receipt) if receipt else None,
                     "renderer_elapsed_seconds": receipt.get("elapsed_seconds") if receipt else None})
    rejected = [row for row in rows if row["check_status"] == "reject"]
    with_receipt = [row for row in rows if row["renderer_valid"] is not None]
    false_rejects = [row for row in rejected if row["renderer_valid"] is True]
    true_syntax = [row for row in with_receipt if row["renderer_syntax_error"]]
    return {"schema": "painter.js-preflight-profile.v1",
            "denominator": "saved program.js files; may include speculative rollouts",
            "all_saved_programs": total, "sampled_programs": len(rows),
            "sample_seed": sample_seed if max_programs is not None else None,
            "receipt_count": len(with_receipt),
            "node_rejected": len(rejected),
            "node_timed_out": sum(row["check_status"] == "timeout" for row in rows),
            "renderer_syntax_errors": len(true_syntax),
            "node_rejected_renderer_syntax_errors": sum(row["renderer_syntax_error"] is True for row in rejected),
            "node_false_rejected_valid_renders": len(false_rejects),
            "false_reject_program_sha256": [row["program_sha256"] for row in false_rejects],
            "node_check_seconds_mean": statistics.mean(row["check_seconds"] for row in rows),
            "node_check_seconds_p90": pct([row["check_seconds"] for row in rows], .9),
            "rejected_renderer_elapsed_seconds_sum": sum(
                float(row["renderer_elapsed_seconds"] or 0) for row in rejected),
            "gross_renderer_seconds_potentially_avoided": sum(
                max(0.0, float(row["renderer_elapsed_seconds"] or 0) - row["check_seconds"])
                for row in rejected),
            "caveat": "Gross renderer seconds are overlapping work, not wall-clock speedup; only a later unchanged-pixel integration can justify adoption.",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-programs", type=int)
    args = parser.parse_args()
    result = analyze(args.rollouts, args.node, max_programs=args.max_programs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("all_saved_programs", "sampled_programs",
                                                  "node_rejected", "renderer_syntax_errors",
                                                  "node_false_rejected_valid_renders",
                                                  "gross_renderer_seconds_potentially_avoided")}))


if __name__ == "__main__":
    main()
