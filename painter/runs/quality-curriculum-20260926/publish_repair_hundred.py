#!/usr/bin/env python3
"""Publish the 100-scene correction source with anonymous hash verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_XET_CACHE", str(Path(tempfile.gettempdir()) / "painter-hf-xet-cache"))
from huggingface_hub import HfApi  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "painter/collected/quality-curriculum-20260926/repair-hundred-v1"
DATASET = "CK0607/komorebi-painter-teachers"
REMOTE = "curricula/teacher600/repair-hundred-v1/source.tar.gz"
PUBLIC = "curricula/teacher600/repair-hundred-v1/source-public.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_sha(revision: str, remote: str) -> str:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{remote}?download=true"
    digest = hashlib.sha256()
    with urlopen(url, timeout=180) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish() -> dict:
    archive = SOURCE / "source.tar.gz"
    receipt = json.loads((SOURCE / "source-receipt.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if (receipt.get("schema") != "painter.teacher600-repair-hundred-receipt.v1"
            or manifest.get("schema") != "painter.teacher600-repair-hundred-source.v1"
            or receipt["count"] != manifest["count"] or manifest["count"] != 100
            or sha(archive.read_bytes()) != receipt["archive_sha256"]
            or sha((SOURCE / "manifest.json").read_bytes()) != receipt["manifest_sha256"]):
        raise ValueError("source receipt/archive mismatch")
    for row in manifest["rows"]:
        if row["mode"] == "image_to_image" and not all(
                (row.get("rights") or {}).get(key) for key in ("name", "url", "source_url")):
            raise ValueError(f"missing photo rights for {row['audit_id']}")
    existing = SOURCE / "source-public.json"
    if existing.is_file():
        prior = json.loads(existing.read_text())
        if (prior.get("archive_sha256") != receipt["archive_sha256"]
                or prior.get("dataset") != DATASET or prior.get("path") != REMOTE
                or prior.get("anonymous_hash_verified") is not True
                or public_sha(prior["dataset_commit"], REMOTE) != receipt["archive_sha256"]):
            raise ValueError("existing public receipt differs")
        return prior
    api = HfApi()
    if api.repo_info(DATASET, repo_type="dataset").private:
        raise ValueError("teacher dataset must remain public")
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=REMOTE,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message="Publish 100 rights-screened teacher repair states")
    if public_sha(commit.oid, REMOTE) != receipt["archive_sha256"]:
        raise ValueError("anonymous source archive hash mismatch")
    result = {"schema": "painter.teacher600-repair-hundred-public-source.v1",
              "dataset": DATASET, "dataset_commit": commit.oid, "path": REMOTE,
              "archive_sha256": receipt["archive_sha256"],
              "archive_bytes": receipt["archive_bytes"],
              "manifest_sha256": receipt["manifest_sha256"],
              "plan_sha256": receipt["plan_sha256"], "count": 100,
              "status": "unreviewed_correction_states_not_sft_targets",
              "anonymous_hash_verified": True}
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    receipt_commit = api.upload_file(path_or_fileobj=raw, path_in_repo=PUBLIC,
                                     repo_id=DATASET, repo_type="dataset",
                                     commit_message="Record verified 100-scene repair source")
    if public_sha(receipt_commit.oid, PUBLIC) != sha(raw):
        raise ValueError("anonymous source receipt hash mismatch")
    existing.write_bytes(raw)
    return result


if __name__ == "__main__":
    print(json.dumps(publish(), sort_keys=True))
