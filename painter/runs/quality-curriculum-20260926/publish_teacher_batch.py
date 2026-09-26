#!/usr/bin/env python3
"""Publish a teacher candidate source bundle and verify anonymous retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi


DATASET = "CK0607/komorebi-painter-teachers"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_public(url: str) -> str:
    h = hashlib.sha256()
    with urlopen(url, timeout=180) as response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(source: Path) -> dict:
    receipt = json.loads((source / "source-receipt.json").read_text())
    if receipt.get("schema") != "painter.teacher500-source-receipt.v1" or receipt.get("batch") != source.name:
        raise ValueError("invalid source receipt")
    archive = source / "source.tar.gz"
    if sha_file(archive) != receipt["archive_sha256"] or archive.stat().st_size != receipt["archive_bytes"]:
        raise ValueError("source archive hash/size mismatch")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset")
    if info.private:
        raise ValueError("teacher dataset is not public")
    remote = f"curricula/teacher500/{source.name}/source.tar.gz"
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=remote,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message=f"Add {source.name} teacher candidates")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit.oid}/{remote}?download=true"
    if sha_public(url) != receipt["archive_sha256"]:
        raise ValueError("anonymous download does not match the local archive")
    public = {"schema": "painter.teacher500-public-source.v1", "batch": source.name,
              "count": receipt["count"], "source_commit": receipt["source_commit"],
              "dataset": DATASET, "dataset_commit": commit.oid, "path": remote,
              "archive_bytes": receipt["archive_bytes"], "archive_sha256": receipt["archive_sha256"],
              "anonymous_hash_verified": True,
              "url": f"https://huggingface.co/datasets/{DATASET}/tree/{commit.oid}/curricula/teacher500/{source.name}"}
    payload = (json.dumps(public, indent=2, sort_keys=True) + "\n").encode()
    receipt_remote = f"curricula/teacher500/{source.name}/source-public.json"
    api.upload_file(path_or_fileobj=payload, path_in_repo=receipt_remote,
                    repo_id=DATASET, repo_type="dataset",
                    commit_message=f"Record {source.name} source verification")
    receipt_url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{receipt_remote}?download=true"
    if sha_public(receipt_url) != hashlib.sha256(payload).hexdigest():
        raise ValueError("anonymous public receipt hash mismatch")
    (source / "source-public.json").write_bytes(payload)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.source.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
