#!/usr/bin/env python3
"""Fetch a verified MiMo candidate prefix and build its offline review gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from prepare_mimo_cloud import BENCH, stage


def unpack_checked(archive: Path, root: Path) -> int:
    allowed_top = {"progress.json", "events.jsonl", "run-summary.json", "gallery.html", "restored-source.json"}
    count = 0
    total_bytes = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if path.parts[0] != "episodes" and member.name not in allowed_top:
                raise ValueError(f"unexpected archive member: {member.name}")
            total_bytes += member.size
            if total_bytes > 2_000_000_000:
                raise ValueError("archive expands beyond the 2 GB review limit")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("wb") as writer:
                assert reader is not None
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("easy", "hard"), required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    permitted = {"easy": (2, 4, 7), "hard": (2, 5, 10)}
    if args.count not in permitted[args.tier]:
        parser.error(f"count must be one of {permitted[args.tier]}")
    model = "xiaomi/mimo-v2.6-flash" if args.tier == "easy" else "xiaomi/mimo-v2.6-pro"
    root = stage(args.tier, model)
    run_id = f"mimo-quality-{args.tier}-{'flash' if args.tier == 'easy' else 'pro'}-20260924-n{args.count}"
    archive = root / f"{run_id}.tar.gz"
    subprocess.run([sys.executable, str(BENCH / "cloud/fetch_results.py"), run_id, str(archive)], check=True)
    count = unpack_checked(archive, root)
    gallery = root / "gallery.html"
    # Link verified renders directly; copying a blind packet can fill a small review disk.
    subprocess.run([sys.executable, str(BENCH / "review.py"), "--root", str(root),
                    "--no-blind", "--output", str(gallery)], check=True)
    print(json.dumps({"run_id": run_id, "verified_archive": str(archive), "files": count,
                      "gallery": str(gallery)}, sort_keys=True))


if __name__ == "__main__":
    main()
