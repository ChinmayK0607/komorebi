#!/usr/bin/env python3
"""Publish a rights-screened correction-state source archive with anonymous verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_XET_CACHE", str(Path(tempfile.gettempdir()) / "painter-hf-xet-cache"))
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "painter/collected/quality-curriculum-20260926/next-repair-eight-v1"
DATASET = "CK0607/komorebi-painter-teachers"
REMOTE = "curricula/teacher600/next-repair-eight-v1/source.tar.gz"
PUBLIC = "curricula/teacher600/next-repair-eight-v1/source-public.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_sha(revision: str, remote: str) -> str:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{remote}?download=true"
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=180) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
    except HTTPError as exc:
        if exc.code != 429:
            raise
        local = hf_hub_download(DATASET, remote, repo_type="dataset", revision=revision,
                                token=False, cache_dir=SOURCE / ".hf-public-cache")
        return sha(Path(local).read_bytes())
    return digest.hexdigest()


def publish() -> dict:
    archive = SOURCE / "source.tar.gz"
    receipt = json.loads((SOURCE / "source-receipt.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if (receipt.get("schema") != "painter.teacher600-repair-source-receipt.v1"
            or manifest.get("schema") != "painter.teacher600-repair-source.v1"
            or receipt["count"] != manifest["count"] != 8
            or not receipt["all_photo_sources_rights_complete"]
            or sha(archive.read_bytes()) != receipt["archive_sha256"]
            or archive.stat().st_size != receipt["archive_bytes"]
            or sha((SOURCE / "manifest.json").read_bytes()) != receipt["manifest_sha256"]):
        raise ValueError("repair source receipt/archive mismatch")
    for row in manifest["rows"]:
        if row["mode"] == "image_to_image":
            rights = row.get("rights") or {}
            if not all(rights.get(key) for key in ("name", "url", "source_url")):
                raise ValueError(f"missing photo rights for {row['audit_id']}")
    existing = SOURCE / "source-public.json"
    if existing.is_file():
        prior = json.loads(existing.read_text())
        if (prior.get("archive_sha256") != receipt["archive_sha256"]
                or prior.get("dataset") != DATASET or prior.get("path") != REMOTE
                or prior.get("anonymous_hash_verified") is not True
                or public_sha(prior["dataset_commit"], REMOTE) != receipt["archive_sha256"]):
            raise ValueError("existing public source receipt differs from this bundle")
        return prior
    api = HfApi()
    if api.repo_info(DATASET, repo_type="dataset").private:
        raise ValueError("teacher dataset must remain public")
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=REMOTE,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message="Publish inspected teacher600 repair states")
    if public_sha(commit.oid, REMOTE) != receipt["archive_sha256"]:
        raise ValueError("anonymous repair archive hash mismatch")
    result = {"schema": "painter.teacher600-repair-public-source.v1",
              "dataset": DATASET, "dataset_commit": commit.oid, "path": REMOTE,
              "archive_sha256": receipt["archive_sha256"],
              "archive_bytes": receipt["archive_bytes"],
              "plan_sha256": receipt["plan_sha256"], "count": 8,
              "status": "unreviewed_correction_states_not_sft_targets",
              "anonymous_hash_verified": True}
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    receipt_commit = api.upload_file(path_or_fileobj=raw, path_in_repo=PUBLIC,
                                     repo_id=DATASET, repo_type="dataset",
                                     commit_message="Record verified teacher600 repair source")
    if public_sha(receipt_commit.oid, PUBLIC) != sha(raw):
        raise ValueError("anonymous repair receipt hash mismatch")
    (SOURCE / "source-public.json").write_bytes(raw)
    return result


if __name__ == "__main__":
    print(json.dumps(publish(), sort_keys=True))
