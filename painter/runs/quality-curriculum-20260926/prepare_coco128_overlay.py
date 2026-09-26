#!/usr/bin/env python3
"""Publish only the 12 new programs, reusing already-public COCO128 source photos."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
OLDER = ROOT / "painter/collected/quality-curriculum-20260924"
DATASET = "CK0607/komorebi-painter-teachers"
BATCH = "astra-reference-seed-program-overlay-v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_bytes(commit: str, path: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit}/{path}?download=true"
    with urlopen(url, timeout=180) as response:
        return response.read(100_000_001)


def old_archive(receipt: dict) -> tarfile.TarFile:
    raw = public_bytes(receipt["dataset_commit"], receipt["bundle_path"])
    if receipt["bundle_path"].endswith(".b64.txt"):
        raw = base64.b64decode(b"".join(raw.split()), validate=True)
    if len(raw) != receipt["bundle_bytes"] or sha(raw) != receipt["bundle_sha256"]:
        raise ValueError("prior public source hash mismatch")
    return tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")


def make_overlay() -> tuple[Path, dict]:
    source = POOL / "astra-reference-seed" / "source.tar.gz"
    output = POOL / BATCH
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for path in OLDER.glob("astra-high-wave*/**/public-source-receipt.json"):
        data = json.loads(path.read_text())
        if data.get("dataset_repo") == DATASET and data.get("public_hash_verified") is True:
            receipts.append(data)
    archives: dict[str, tarfile.TarFile] = {}
    members: dict[str, bytes] = {}
    rows = []
    with tarfile.open(source, "r:gz") as original:
        manifest = json.load(original.extractfile("manifest.json"))
        for row in manifest["rows"]:
            if row["role"] != "first_paint_candidate" or row["mode"] != "image_to_image":
                raise ValueError("unexpected source role/mode")
            ident = row["id"]
            source_id = ident.removeprefix("astra-ref-seed-")
            image_name = f"references/coco128-{source_id}.jpg"
            candidates = sorted((r for r in receipts if f"coco128-{source_id}" in r.get("reference_ids", [])),
                                key=lambda r: (r["bundle_bytes"], r["bundle_sha256"]))
            found = None
            for prior in candidates:
                key = prior["bundle_sha256"]
                if key not in archives:
                    archives[key] = old_archive(prior)
                try:
                    photo = archives[key].extractfile(image_name).read()
                except KeyError:
                    continue
                if sha(photo) == row["input_sha256"]:
                    found = prior
                    break
            if found is None:
                raise ValueError(f"no exact previously-public photo for {ident}")
            program = original.extractfile(row["program"]).read()
            if sha(program) != row["program_sha256"]:
                raise ValueError(f"program hash mismatch: {ident}")
            program_path = f"programs/{ident}.js"
            members[program_path] = program
            rows.append({"id": ident, "mode": "image_to_image", "role": "first_paint_candidate",
                         "category": row["category"], "input_sha256": row["input_sha256"],
                         "program": program_path, "program_sha256": row["program_sha256"],
                         "prior_source": {key: found[key] for key in
                                          ("dataset_commit", "bundle_path", "bundle_sha256", "bundle_bytes")},
                         "prior_reference_member": image_name})
    if len(rows) != 12 or len({r["id"] for r in rows}) != 12:
        raise ValueError("expected 12 distinct reused-photo programs")
    overlay = {"schema": "painter.teacher600-program-overlay.v1", "batch": BATCH,
               "count": 12, "teacher_model": "gpt-6-astra", "teacher_reasoning_effort": "high",
               "photos_in_archive": False, "rows": rows}
    members["manifest.json"] = (json.dumps(overlay, indent=2, sort_keys=True) + "\n").encode()
    archive_path = output / "source.tar.gz"
    with archive_path.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name, raw in sorted(members.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mtime, entry.mode = len(raw), 0, 0o644
                archive.addfile(entry, io.BytesIO(raw))
    receipt = {"schema": "painter.teacher600-program-overlay-source.v1", "batch": BATCH,
               "count": 12, "archive_sha256": sha(archive_path.read_bytes()),
               "archive_bytes": archive_path.stat().st_size,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "previously_public_photo_hash_verified": True, "photos_in_archive": False}
    (output / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return archive_path, receipt


def publish(archive: Path, receipt: dict) -> dict:
    api = HfApi()
    if api.repo_info(DATASET, repo_type="dataset").private:
        raise ValueError("teacher dataset must be public")
    remote = f"curricula/teacher600/{BATCH}/source.tar.gz"
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=remote,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message=f"Add {BATCH} program-only source")
    if sha(public_bytes(commit.oid, remote)) != receipt["archive_sha256"]:
        raise ValueError("anonymous program-only archive hash mismatch")
    public = {"schema": "painter.teacher600-public-program-overlay.v1", "batch": BATCH,
              "count": 12, "archive_sha256": receipt["archive_sha256"],
              "archive_bytes": receipt["archive_bytes"], "dataset_commit": commit.oid,
              "path": remote, "photos_in_archive": False,
              "anonymous_hash_verified": True, "source_commit": receipt["source_commit"]}
    payload = (json.dumps(public, indent=2, sort_keys=True) + "\n").encode()
    receipt_remote = f"curricula/teacher600/{BATCH}/source-public.json"
    api.upload_file(path_or_fileobj=payload, path_in_repo=receipt_remote,
                    repo_id=DATASET, repo_type="dataset",
                    commit_message=f"Record {BATCH} source hash verification")
    if sha(public_bytes("main", receipt_remote)) != sha(payload):
        raise ValueError("anonymous program-only receipt hash mismatch")
    (archive.parent / "source-public.json").write_bytes(payload)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    archive, receipt = make_overlay()
    print(json.dumps(publish(archive, receipt) if args.publish else receipt, sort_keys=True))


if __name__ == "__main__":
    main()
