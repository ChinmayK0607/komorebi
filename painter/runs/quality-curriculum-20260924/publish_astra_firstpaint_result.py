#!/usr/bin/env python3
"""Publish a completed pilot evidence bundle and verify an anonymous download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi


REPO = "CK0607/komorebi-painter-evaluations"
PREFIX = "astra-firstpaint-sft-20260926-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_digest(commit: str, remote_path: str) -> str:
    h = hashlib.sha256()
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{remote_path}?download=true"
    with urlopen(url, timeout=180) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(archive: Path) -> dict:
    archive = archive.resolve()
    receipt_path = archive.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True or receipt.get("sha256") != digest(archive):
        raise ValueError("completed archive receipt or local hash mismatch")
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private:
        raise ValueError("evidence repository is private")
    remote_path = f"{PREFIX}/{archive.name}"
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=remote_path,
                             repo_id=REPO, repo_type="dataset",
                             commit_message="Publish Astra first-paint SFT matched evaluation")
    if public_digest(commit.oid, remote_path) != receipt["sha256"]:
        raise ValueError("anonymous evidence download failed SHA-256 verification")
    result = {"schema": "painter.astra-firstpaint-public-evidence.v1", "repo": REPO,
              "commit": commit.oid, "path": remote_path, "sha256": receipt["sha256"],
              "bytes": receipt["bytes"], "members": receipt["members"],
              "public_hash_verified": True,
              "url": f"https://huggingface.co/datasets/{REPO}/tree/{commit.oid}/{PREFIX}"}
    archive.with_suffix(".public.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.archive)))


if __name__ == "__main__":
    main()
