#!/usr/bin/env python3
"""Recover selected, already-paid MiMo episodes from public verified archives.

These are candidate paintings from the teacher-selection benchmark.  They are
never admitted automatically, and the benchmark references cease to be a
student holdout if any recovered turns enter training.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
DATASET = "CK0607/komorebi-painter-teachers"
SOURCE_COMMIT = "10f047ac91204bf1260e9325d4003f9a7d4e57a3"
SELECTIONS = {
    "full-quality-01-20260923": ("coco128-000000000136", "xiaomi-mimo-v2.6-pro"),
    "full-quality-02-20260923": ("coco128-000000000143", "xiaomi-mimo-v2.6-flash"),
    "full-quality-03-20260923": ("coco128-000000000520", "xiaomi-mimo-v2.6-pro"),
    "full-quality-07-20260923": ("coco128-000000000532", "xiaomi-mimo-v2.6-flash"),
}
OUTPUT = ROOT / "painter/collected/quality-curriculum-20260924/mimo-benchmark-reuse"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(revision: str, path: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"
    with urlopen(url, timeout=120) as response:
        return response.read()


def collect(run_id: str) -> dict:
    reference_id, model_slug = SELECTIONS[run_id]
    receipt = json.loads(get("main", f"runs/{run_id}/receipt.json"))
    if (receipt.get("schema") != "painter.hf-benchmark-result.v1"
            or receipt.get("run_id") != run_id
            or receipt.get("source_commit") != SOURCE_COMMIT
            or receipt.get("dataset_repo") != DATASET
            or receipt.get("public_hash_verified") is not True
            or receipt.get("representation") != "split-base64-text"):
        raise ValueError("public paid-episode receipt provenance mismatch")
    target = OUTPUT / run_id / "evidence"
    if target.exists():
        prior = target / "public-receipt.json"
        if prior.is_file() and json.loads(prior.read_text()) == receipt:
            return {"run_id": run_id, "status": "already_collected", "output": str(target)}
        raise ValueError(f"existing evidence has a different receipt: {target}")
    parts = receipt.get("part_paths") or []
    if not parts or parts != sorted(parts) or any(
            not part.startswith(f"runs/{run_id}/parts/part-") or not part.endswith(".txt")
            for part in parts):
        raise ValueError("unexpected archive part paths")
    with ThreadPoolExecutor(max_workers=12) as pool:
        chunks = list(pool.map(lambda part: base64.b64decode(
            b"".join(get(receipt["dataset_commit"], part).split()), validate=True), parts))
    raw = b"".join(chunks)
    if len(raw) != receipt["bundle_bytes"] or digest(raw) != receipt["bundle_sha256"]:
        raise ValueError("public archive byte count or hash mismatch")
    prefix = f"gateway--quality--{model_slug}-"
    selected = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != receipt["file_count"]:
            raise ValueError("public archive member count mismatch")
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe public archive member")
            if (len(path.parts) == 3 and path.parts[0] == "episodes"
                    and path.parts[1].startswith(prefix)
                    and f"--{reference_id}--" in path.parts[1]):
                selected.append(member)
        if not selected or not any(m.name.endswith("/episode.json") for m in selected):
            raise ValueError("selected paid episode absent from verified archive")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mimo-paid-", dir=target.parent) as temp:
            stage = Path(temp)
            for member in selected:
                path = stage / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
            episode_path = next(stage.glob("episodes/*/episode.json"))
            episode = json.loads(episode_path.read_text())
            if episode.get("reference_id") != reference_id or "mimo-v2.6" not in episode.get("model", ""):
                raise ValueError("selected episode identity mismatch")
            (stage / "public-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            os.replace(stage, target)
    return {"run_id": run_id, "status": "candidate_collected", "reference_id": reference_id,
            "model": episode["model"], "turns": len(episode["turns"]), "files": len(selected),
            "bundle_sha256": receipt["bundle_sha256"], "output": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=SELECTIONS, action="append", required=True)
    args = parser.parse_args()
    for run_id in args.run_id:
        print(json.dumps(collect(run_id), sort_keys=True))


if __name__ == "__main__":
    main()
