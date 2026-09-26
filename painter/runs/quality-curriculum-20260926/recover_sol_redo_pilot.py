#!/usr/bin/env python3
"""Recover a completed teacher redo from public split parts without API calls."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from urllib.error import HTTPError
from urllib.request import urlopen

from huggingface_hub import HfApi


DATASET = "CK0607/komorebi-painter-teachers"
HERE = Path(__file__).resolve().parent
POOL = HERE.parents[1] / "collected/quality-curriculum-20260926/rendered-cloud"
MODELS = {"t600-326": "xiaomi/mimo-v2.6-flash", "t600-492": "xiaomi/mimo-v2.6-pro"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(revision: str, path: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"
    with urlopen(url, timeout=180) as response:
        return response.read()


def recover(run_id: str, audit_id: str, *, publish: bool = True) -> dict:
    if not re.fullmatch(r"sol-redo-[a-z0-9-]{1,55}", run_id) or audit_id not in MODELS:
        raise ValueError("unexpected redo run or scene")
    public_id = f"{run_id}-{audit_id}"
    revision = HfApi(token=False).repo_info(DATASET, repo_type="dataset").sha
    prefix = f"runs/{public_id}/parts/"
    parts: list[str] = []
    archive = bytearray()
    for i in range(32):
        path = f"{prefix}part-{i:04d}.txt"
        try:
            encoded = fetch(revision, path)
        except HTTPError as error:
            if error.code == 404 and i > 0:
                break
            raise
        parts.append(path)
        archive.extend(base64.b64decode(b"".join(encoded.split()), validate=True))
    else:
        raise ValueError("more than 32 parts; refusing an unbounded recovery")

    expected = next(row for row in json.loads((HERE / "SOL_BRUSH_PILOT_8.json").read_text())["rows"]
                    if row["audit_id"] == audit_id)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        members = tf.getmembers()
        names = {member.name for member in members}
        if not all(member.isfile() and not Path(member.name).is_absolute()
                   and ".." not in Path(member.name).parts for member in members):
            raise ValueError("unsafe archive member")
        episodes = [name for name in names if re.fullmatch(r"episodes/[^/]+/episode.json", name)]
        if len(episodes) != 1:
            raise ValueError("expected exactly one episode")
        epdir = episodes[0].rsplit("/", 1)[0]

        def read(name: str) -> bytes:
            extracted = tf.extractfile(name)
            if extracted is None:
                raise ValueError(f"missing {name}")
            return extracted.read()

        summary = json.loads(read("run-summary.json"))
        source = json.loads(read("restored-source.json"))
        episode = json.loads(read(episodes[0]))
        if (source["audit_id"] != audit_id or source["model"] != MODELS[audit_id]
                or episode["model"] != MODELS[audit_id] or episode["reference_id"] != audit_id
                or summary["reference_ids_selected"] != [audit_id]
                or summary["episodes_selected"] != 1):
            raise ValueError("episode/source identity mismatch")
        pinned = {"first-paint.png": "original_canvas_sha256",
                  "first-paint.js": "source_program_sha256",
                  "source-prompt.txt": "input_sha256"}
        for file_name, key in pinned.items():
            if sha(read(f"{epdir}/{file_name}")) != expected[key]:
                raise ValueError(f"source hash mismatch: {file_name}")
        valid = 0
        for turn in episode["turns"]:
            rendered = turn.get("render") or {}
            if not rendered.get("valid"):
                continue
            canvas = read(f"{epdir}/turn-{turn['turn']:02d}.png")
            program = read(f"{epdir}/turn-{turn['turn']:02d}.program.js")
            receipt = rendered.get("receipt") or {}
            if (sha(canvas) != rendered.get("canvas_sha256")
                    or sha(canvas) != receipt.get("png_sha256")
                    or sha(program) != receipt.get("source_sha256")):
                raise ValueError(f"render hash mismatch: turn {turn['turn']}")
            valid += 1
        if not valid or len(episode["turns"]) != summary["response_turns_total"]:
            raise ValueError("incomplete episode or no valid turn")

    receipt = {
        "schema": "painter.hf-benchmark-result.v1", "run_id": public_id,
        "source_commit": "90474887457259590ca1d35dab1bcd054fbc93f7",
        "dataset_repo": DATASET, "bundle_bytes": len(archive),
        "bundle_sha256": sha(archive), "file_count": len(members),
        "public_hash_verified": True, "dataset_commit": revision,
        "representation": "split-base64-text", "encoding": "base64-per-part",
        "part_order": "lexical", "part_paths": parts,
        "public_parts_anonymously_fetched": True,
        "recovery_note": "Recovered completed paid episode from anonymously downloaded parts; original inputs and valid render hashes verified without model calls.",
    }
    output = POOL / run_id / audit_id
    output.mkdir(parents=True, exist_ok=True)
    path = output / "recovered-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if publish:
        HfApi(token=True).upload_file(path_or_fileobj=str(path),
                                      path_in_repo=f"runs/{public_id}/receipt.json",
                                      repo_id=DATASET, repo_type="dataset",
                                      commit_message=f"Recover verified painter result {public_id}")
        published = json.loads(fetch(HfApi(token=False).repo_info(DATASET, repo_type="dataset").sha,
                                     f"runs/{public_id}/receipt.json"))
        if published["bundle_sha256"] != receipt["bundle_sha256"]:
            raise ValueError("published receipt mismatch")
    return {"run_id": public_id, "archive_sha256": receipt["bundle_sha256"],
            "archive_bytes": len(archive), "parts": len(parts), "valid_turns": valid,
            "receipt": str(path), "published": publish}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("audit_id", choices=MODELS)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.run_id, args.audit_id, publish=not args.no_publish), sort_keys=True))
