#!/usr/bin/env python3
"""Upload benchmark evidence to a public HF dataset and verify its hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

from huggingface_hub import HfApi

DATASET = "CK0607/komorebi-painter-teachers"
TOP_LEVEL = ("progress.json", "events.jsonl", "run-summary.json", "gallery.html")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(root: Path, output: Path) -> int:
    members = [root / name for name in TOP_LEVEL if (root / name).is_file()]
    episodes = root / "episodes"
    if episodes.is_dir():
        members.extend(sorted(path for path in episodes.rglob("*") if path.is_file() and not path.is_symlink()))
    if not any(path.is_relative_to(episodes) for path in members):
        raise ValueError("no episode evidence to publish")
    with tarfile.open(output, "w:gz") as archive:
        for path in members:
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    return len(members)


def public_sha256(url: str) -> str:
    digest = hashlib.sha256()
    with urlopen(url, timeout=180) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default=os.environ.get("PAINTER_RUN_ID"))
    args = parser.parse_args()
    root = args.root.resolve()
    run_id = args.run_id
    if not run_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", run_id):
        raise SystemExit("set a stable PAINTER_RUN_ID using letters, digits, dot, dash or underscore")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required for publishing")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    api = HfApi(token=os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory(prefix="painter-hf-publish-") as temp:
        archive = Path(temp) / "bundle.tar.gz"
        count = pack(root, archive)
        expected = sha256_file(archive)
        remote_path = f"runs/{run_id}/bundle.tar.gz"
        commit = api.upload_file(
            path_or_fileobj=str(archive), path_in_repo=remote_path,
            repo_id=DATASET, repo_type="dataset",
            commit_message=f"Publish painter benchmark {run_id}",
        )
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit.oid}/{remote_path}?download=true"
        if public_sha256(url) != expected:
            raise RuntimeError("public result archive failed SHA-256 verification")
        receipt = {
            "schema": "painter.hf-benchmark-result.v1",
            "run_id": run_id,
            "source_commit": source_commit,
            "dataset_repo": DATASET,
            "dataset_commit": commit.oid,
            "bundle_path": remote_path,
            "bundle_bytes": archive.stat().st_size,
            "bundle_sha256": expected,
            "file_count": count,
            "public_hash_verified": True,
        }
        receipt_path = Path(temp) / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        api.upload_file(
            path_or_fileobj=str(receipt_path),
            path_in_repo=f"runs/{run_id}/receipt.json",
            repo_id=DATASET, repo_type="dataset",
            commit_message=f"Record verified painter benchmark {run_id}",
        )
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
