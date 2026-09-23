#!/usr/bin/env python3
"""Upload benchmark evidence to a public HF dataset and verify its hash."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

from huggingface_hub import CommitOperationAdd, HfApi

DATASET = "CK0607/komorebi-painter-teachers"
TOP_LEVEL = ("progress.json", "events.jsonl", "run-summary.json", "gallery.html", "restored-source.json")
PART_RAW_BYTES = 384 * 1024
PARTS_PER_COMMIT = 8


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


def public_url(revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"


def publish_archive(api: HfApi, archive: Path, run_id: str, expected: str) -> dict[str, object]:
    remote_path = f"runs/{run_id}/bundle.tar.gz"
    commit = api.upload_file(
        path_or_fileobj=str(archive), path_in_repo=remote_path,
        repo_id=DATASET, repo_type="dataset",
        commit_message=f"Publish painter benchmark {run_id}",
    )
    if public_sha256(public_url(commit.oid, remote_path)) != expected:
        raise RuntimeError("public result archive failed SHA-256 verification")
    return {"dataset_commit": commit.oid, "bundle_path": remote_path, "representation": "archive"}


def publish_split(api: HfApi, archive: Path, run_id: str, expected: str, temp: Path) -> dict[str, object]:
    """Use regular-Git ASCII files when a cloud proxy blocks Xet/LFS hosts."""
    part_paths: list[str] = []
    local_parts: list[Path] = []
    with archive.open("rb") as source:
        for index, chunk in enumerate(iter(lambda: source.read(PART_RAW_BYTES), b"")):
            path = temp / f"part-{index:04d}.txt"
            path.write_bytes(base64.b64encode(chunk) + b"\n")
            local_parts.append(path)
            part_paths.append(f"runs/{run_id}/parts/{path.name}")
    if not part_paths:
        raise RuntimeError("empty result archive")
    commit = None
    for start in range(0, len(part_paths), PARTS_PER_COMMIT):
        operations = [
            CommitOperationAdd(path_in_repo=part_paths[i], path_or_fileobj=str(local_parts[i]))
            for i in range(start, min(start + PARTS_PER_COMMIT, len(part_paths)))
        ]
        commit = api.create_commit(
            repo_id=DATASET, repo_type="dataset", operations=operations,
            commit_message=f"Publish painter benchmark {run_id} text parts",
        )
    assert commit is not None
    digest = hashlib.sha256()
    byte_count = 0
    for path in part_paths:
        with urlopen(public_url(commit.oid, path), timeout=180) as response:
            encoded = response.read()
        raw = base64.b64decode(b"".join(encoded.split()), validate=True)
        digest.update(raw)
        byte_count += len(raw)
    if byte_count != archive.stat().st_size or digest.hexdigest() != expected:
        raise RuntimeError("public text parts failed archive SHA-256 verification")
    return {
        "dataset_commit": commit.oid,
        "representation": "split-base64-text",
        "encoding": "base64-per-part",
        "part_order": "lexical",
        "part_paths": part_paths,
        "public_parts_anonymously_fetched": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default=os.environ.get("PAINTER_RUN_ID"))
    parser.add_argument("--split", action="store_true", help="publish regular-Git Base64 text parts directly")
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
        if args.split:
            publication = publish_split(api, archive, run_id, expected, Path(temp))
        else:
            try:
                publication = publish_archive(api, archive, run_id, expected)
            except Exception as exc:
                # Never log a proxy exception: it may contain a signed upload
                # URL. Regular-Git text parts work through huggingface.co.
                print(json.dumps({"archive_upload_failed": type(exc).__name__, "retrying": "split-base64-text"}))
                publication = publish_split(api, archive, run_id, expected, Path(temp))
        receipt = {
            "schema": "painter.hf-benchmark-result.v1",
            "run_id": run_id,
            "source_commit": source_commit,
            "dataset_repo": DATASET,
            "bundle_bytes": archive.stat().st_size,
            "bundle_sha256": expected,
            "file_count": count,
            "public_hash_verified": True,
            **publication,
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
