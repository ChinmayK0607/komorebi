#!/usr/bin/env python3
"""Download and verify the two public MiMo candidate archives without credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen


DATASET = "CK0607/komorebi-painter-teachers"
SOURCE_COMMIT = "862c28929151e62dfcedb2823fadc4e430ae7ad2"
RUN_IDS = ("mimo-pro-diverse-easy-20260925", "mimo-pro-diverse-photo-20260925")
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "painter/collected/quality-curriculum-20260924/mimo-cloud"


def get(path: str, revision: str = "main") -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"
    with urlopen(url, timeout=180) as response:
        return response.read()


def archive_bytes(receipt: dict) -> bytes:
    revision = receipt["dataset_commit"]
    if receipt["representation"] == "archive":
        raw = get(receipt["bundle_path"], revision)
    elif receipt["representation"] == "split-base64-text":
        parts = receipt["part_paths"]
        if parts != sorted(parts) or not parts:
            raise ValueError("unordered or empty public archive parts")
        raw = b"".join(base64.b64decode(b"".join(get(part, revision).split()), validate=True) for part in parts)
    else:
        raise ValueError("unknown public archive representation")
    if len(raw) != receipt["bundle_bytes"] or hashlib.sha256(raw).hexdigest() != receipt["bundle_sha256"]:
        raise ValueError("public candidate archive hash mismatch")
    return raw


def collect(run_id: str, output: Path) -> dict:
    if run_id not in RUN_IDS:
        raise ValueError("unreviewed run ID")
    receipt = json.loads(get(f"runs/{run_id}/receipt.json"))
    if (receipt.get("schema") != "painter.hf-benchmark-result.v1"
            or receipt.get("run_id") != run_id
            or receipt.get("source_commit") != SOURCE_COMMIT
            or receipt.get("dataset_repo") != DATASET
            or receipt.get("public_hash_verified") is not True):
        raise ValueError("candidate receipt provenance mismatch")
    raw = archive_bytes(receipt)
    if output.exists():
        previous = output / "public-receipt.json"
        if previous.is_file() and json.loads(previous.read_text()) == receipt:
            return {"run_id": run_id, "status": "already_collected", "output": str(output)}
        raise ValueError(f"output exists but has different receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="painter-candidates-", dir=output.parent) as temp:
        target = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != receipt["file_count"] or not members:
                raise ValueError("candidate archive member count mismatch")
            for member in members:
                name = member.name
                parts = Path(name).parts
                if (not member.isfile() or not parts or ".." in parts or Path(name).is_absolute()
                        or not (parts[0] == "episodes" or name in {
                            "progress.json", "events.jsonl", "run-summary.json", "gallery.html", "restored-source.json"})):
                    raise ValueError(f"unsafe or unexpected archive member: {name}")
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
        (target / "public-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(target, output)
    return {"run_id": run_id, "status": "collected", "output": str(output),
            "bundle_sha256": receipt["bundle_sha256"], "members": receipt["file_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=RUN_IDS, action="append", required=True)
    args = parser.parse_args()
    for run_id in args.run_id:
        tier = "easy" if "-easy-" in run_id else "photo"
        output = OUTPUT / f"diverse-pro-{tier}-20260925" / "evidence"
        print(json.dumps(collect(run_id, output)))


if __name__ == "__main__":
    main()
