#!/usr/bin/env python3
"""Publish a small Astra candidate bundle as public hash-verified HF text."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924/astra-high-wave1"
DATASET = "CK0607/komorebi-painter-teachers"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_bytes(commit: str, path: str) -> bytes:
    with urlopen(f"https://huggingface.co/datasets/{DATASET}/resolve/{commit}/{path}?download=true", timeout=120) as source:
        return source.read()


def publish(shard: str) -> dict:
    source = COLLECTED / shard
    local = json.loads((source / "source-receipt.json").read_text())
    archive = Path(local["bundle"])
    raw = archive.read_bytes()
    if sha(raw) != local["bundle_sha256"]:
        raise ValueError("local source bundle hash mismatch")
    remote_path = f"curricula/astra-high-wave1/{shard}/source.b64.txt"
    encoded = base64.b64encode(raw) + b"\n"
    api = HfApi()
    commit = api.upload_file(path_or_fileobj=io.BytesIO(encoded), path_in_repo=remote_path,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message=f"Publish Astra painter candidate source {shard}")
    public_raw = base64.b64decode(public_bytes(commit.oid, remote_path).strip(), validate=True)
    if sha(public_raw) != local["bundle_sha256"]:
        raise ValueError("anonymous source download failed hash verification")
    receipt = {"schema": "painter.astra-high-wave1-public-source.v1", "shard": shard,
               "dataset_repo": DATASET, "dataset_commit": commit.oid,
               "bundle_sha256": local["bundle_sha256"], "bundle_bytes": local["bundle_bytes"],
               "bundle_path": remote_path, "count": local["count"],
               "reference_ids": local["reference_ids"], "public_hash_verified": True}
    receipt_raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path = f"curricula/astra-high-wave1/{shard}/source-receipt.json"
    receipt_commit = api.upload_file(path_or_fileobj=io.BytesIO(receipt_raw), path_in_repo=receipt_path,
                                     repo_id=DATASET, repo_type="dataset",
                                     commit_message=f"Record verified Astra painter source {shard}")
    if public_bytes(receipt_commit.oid, receipt_path) != receipt_raw:
        raise ValueError("anonymous source receipt verification failed")
    (source / "public-source-receipt.json").write_bytes(receipt_raw)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=("easy-a", "hard-a", "hard-b"))
    args = parser.parse_args()
    print(json.dumps(publish(args.shard), sort_keys=True))


if __name__ == "__main__":
    main()
