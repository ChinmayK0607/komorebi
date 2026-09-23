#!/usr/bin/env python3
"""Fetch the public, fixed reference set and verify every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen


DATASET = "CK0607/komorebi-painter-teachers"
REVISION = "811564c415aac98601754ce6133de33bf25cd69d"


def fetch(root: Path, base_url: str) -> int:
    root = root.resolve()
    with urlopen(base_url + "/reference-manifest.json", timeout=60) as response:
        manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    rows = manifest.get("files")
    if manifest.get("count") != 40 or not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("expected the fixed 40-reference manifest")
    names: set[str] = set()
    for row in rows:
        name = row.get("path")
        if not isinstance(name, str) or PurePosixPath(name).parts[:1] != ("references",):
            raise ValueError("unsafe reference path")
        if len(PurePosixPath(name).parts) != 2 or name in names:
            raise ValueError("duplicate or nested reference path")
        names.add(name)
        target = root / name
        if target.is_file() and target.stat().st_size == row["bytes"]:
            if hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
                continue
        with urlopen(base_url + "/" + name + "?download=true", timeout=120) as response:
            data = response.read()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("reference hash mismatch: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (root / "references-manifest.json").write_bytes(manifest_bytes)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-url",
        default=f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}",
    )
    args = parser.parse_args()
    count = fetch(args.root, args.base_url.rstrip("/"))
    print(f"verified {count} public references at dataset revision {REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
