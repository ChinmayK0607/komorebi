"""Bounded whole-run NVIDIA utilization sampler."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


QUERY = "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw"


def sample(output: Path, interval: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(QUERY.split(",") + ["sample_utc"])
        while True:
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.STDOUT,
                )
                now = datetime.now(timezone.utc).isoformat()
                for line in raw.splitlines():
                    writer.writerow([value.strip() for value in line.split(",")] + [now])
                stream.flush()
            except (OSError, subprocess.CalledProcessError):
                pass
            time.sleep(interval)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=10.0)
    args = p.parse_args()
    sample(args.output, max(1.0, args.interval))


if __name__ == "__main__":
    main()
