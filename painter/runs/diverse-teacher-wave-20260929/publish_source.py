#!/usr/bin/env python3
"""Publish the reviewed, hash-pinned source archive to public Hugging Face."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
DATASET = "CK0607/komorebi-painter-teachers"
REMOTE = "curricula/diverse-teacher-wave-20260929/source.tar.gz"


def main() -> None:
    receipt = json.loads((HERE / "source-receipt.json").read_text())
    local = HERE / "source.tar.gz"
    if hashlib.sha256(local.read_bytes()).hexdigest() != receipt["archive_sha256"]:
        raise ValueError("local source archive hash mismatch")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset")
    if info.private:
        raise ValueError("teacher dataset must be public")
    commit = api.upload_file(path_or_fileobj=str(local), path_in_repo=REMOTE,
        repo_id=DATASET, repo_type="dataset",
        commit_message="Publish 120 disjoint text and photo teacher candidate sources")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit.oid}/{REMOTE}?download=true"
    digest = hashlib.sha256()
    count = 0
    with urlopen(url, timeout=180) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    if count != receipt["archive_bytes"] or digest.hexdigest() != receipt["archive_sha256"]:
        raise ValueError("anonymous public source download did not match local archive")
    public = {"dataset": DATASET, "path": REMOTE, "revision": commit.oid,
              "archive_sha256": digest.hexdigest(), "archive_bytes": count,
              "anonymous_download_hash_verified": True}
    (HERE / "source-public.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(json.dumps(public))


if __name__ == "__main__":
    main()
