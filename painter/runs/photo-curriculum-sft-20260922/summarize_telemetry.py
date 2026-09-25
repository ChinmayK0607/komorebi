"""Summarize interval GPU utilization with an explicit denominator."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    with args.input.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                row["util"] = float(row["utilization.gpu"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    by_gpu = defaultdict(list)
    for row in rows:
        by_gpu[row["index"]].append(row["util"])
    summary = {
        "status": "measured" if rows else "no_samples",
        "measurement_window": {
            "start_utc": min((row["sample_utc"] for row in rows), default=None),
            "end_utc": max((row["sample_utc"] for row in rows), default=None),
        },
        "sample_count": len(rows),
        "denominator": "all parseable nvidia-smi GPU samples in the bounded training process window",
        "per_gpu": {
            gpu: {
                "samples": len(values), "mean_utilization_percent": sum(values) / len(values),
                "busy_fraction_at_least_10_percent": sum(value >= 10 for value in values) / len(values),
            } for gpu, values in sorted(by_gpu.items())
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
