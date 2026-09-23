#!/usr/bin/env python3
"""Download a public benchmark archive and verify its receipt hash."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

DATASET = "CK0607/komorebi-painter-teachers"


def public_url(revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"


def checked_path(path: str, run_id: str) -> str:
    if not path.startswith(f"runs/{run_id}/") or ".." in Path(path).parts:
        raise ValueError("receipt contains a path outside the selected run")
    return path


def download(run_id: str, output: Path) -> dict[str, object]:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", run_id):
        raise ValueError("invalid run ID")
    receipt_path = f"runs/{run_id}/receipt.json"
    with urlopen(public_url("main", receipt_path), timeout=60) as response:
        receipt = json.load(response)
    if receipt.get("run_id") != run_id or receipt.get("dataset_repo") != DATASET:
        raise ValueError("public receipt does not match this run or dataset")
    if receipt.get("public_hash_verified") is not True:
        raise ValueError("public receipt is not marked hash-verified")
    revision = str(receipt["dataset_commit"])
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    digest = hashlib.sha256()
    count = 0

    def append(raw: bytes, stream) -> None:
        nonlocal count
        stream.write(raw)
        digest.update(raw)
        count += len(raw)

    try:
        with partial.open("wb") as stream:
            if isinstance(receipt.get("bundle_path"), str):
                path = checked_path(receipt["bundle_path"], run_id)
                with urlopen(public_url(revision, path), timeout=180) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        append(chunk, stream)
            else:
                paths = receipt.get("part_paths")
                if not isinstance(paths, list) or not paths:
                    raise ValueError("receipt has neither an archive nor text parts")
                if receipt.get("encoding") == "base64-per-part":
                    for path in paths:
                        checked_path(path, run_id)
                        with urlopen(public_url(revision, path), timeout=180) as response:
                            encoded = response.read()
                        append(base64.b64decode(b"".join(encoded.split()), validate=True), stream)
                elif receipt.get("encoding") == "base64":
                    encoded_parts = []
                    for path in paths:
                        checked_path(path, run_id)
                        with urlopen(public_url(revision, path), timeout=180) as response:
                            encoded_parts.append(response.read())
                    append(base64.b64decode(b"".join(b"".join(encoded_parts).split()), validate=True), stream)
                else:
                    raise ValueError("unsupported text-part encoding")
        if count != int(receipt["bundle_bytes"]) or digest.hexdigest() != receipt["bundle_sha256"]:
            raise ValueError("downloaded archive does not match public receipt")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "run_id": run_id, "output": str(output), "bytes": count,
        "sha256": digest.hexdigest(), "public_hash_verified": True,
        "representation": receipt.get("representation", "archive"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(download(args.run_id, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
