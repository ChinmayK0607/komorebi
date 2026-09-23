#!/usr/bin/env python3
"""Restore a hash-verified public shard into an empty benchmark workspace."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import tarfile

from replay_renderer import public_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "episodes").exists():
        parser.error("episodes already exist; use an empty benchmark workspace")
    archive, receipt = public_bytes(args.source_run_id)
    restored = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle:
            if member.isdir():
                continue
            if not member.name.startswith("episodes/"):
                continue
            if not member.isfile() or not re.fullmatch(r"episodes/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", member.name):
                raise ValueError(f"unexpected archive member: {member.name}")
            if member.size > 10_000_000:
                raise ValueError(f"oversized episode file: {member.name}")
            target = root / member.name
            if target.exists():
                raise ValueError(f"refusing to overwrite {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
            restored += 1
    marker = {"schema": "painter.public-shard-restore.v1", "source_run_id": args.source_run_id,
              "source_archive_sha256": receipt["bundle_sha256"], "source_dataset_commit": receipt["dataset_commit"],
              "restored_episode_files": restored}
    (root / "restored-source.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
