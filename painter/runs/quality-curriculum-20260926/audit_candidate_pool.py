#!/usr/bin/env python3
"""Audit teacher source bundles without promoting unrendered candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit() -> dict:
    distinct = Counter()
    teachers = Counter()
    difficulties = Counter()
    alternatives = 0
    batches = 0
    errors = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    text_hashes: dict[str, list[str]] = defaultdict(list)
    for batch in sorted(POOL.iterdir()):
        receipt_path = batch / "source-receipt.json"
        archive_path = batch / "source.tar.gz"
        if not receipt_path.exists():
            continue
        batches += 1
        receipt = json.loads(receipt_path.read_text())
        raw = archive_path.read_bytes()
        if sha(raw) != receipt["archive_sha256"] or len(raw) != receipt["archive_bytes"]:
            errors.append(f"{batch.name}: source archive hash/size mismatch")
            continue
        with tarfile.open(archive_path, "r:gz") as archive:
            manifest = json.loads(archive.extractfile("manifest.json").read())
            if (manifest["batch"] != batch.name or manifest["count"] != receipt["count"]
                    or sha((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
                    != receipt["source_manifest_sha256"]):
                errors.append(f"{batch.name}: normalized manifest/receipt mismatch")
            for row in manifest["rows"]:
                key = f"{batch.name}/{row['id']}"
                inp = archive.extractfile(row["input"]).read()
                program = archive.extractfile(row["program"]).read()
                if sha(inp) != row["input_sha256"] or sha(program) != row["program_sha256"]:
                    errors.append(f"{key}: input/program hash mismatch")
                if row.get("prompt") and sha(archive.extractfile(row["prompt"]).read()) != row["prompt_sha256"]:
                    errors.append(f"{key}: prompt hash mismatch")
                if row["role"] == "unrendered_alternative_candidate":
                    alternatives += 1
                    if not row.get("baseline_batch") or not row.get("baseline_program_sha256"):
                        errors.append(f"{key}: alternative lacks baseline link")
                    continue
                distinct[row["mode"]] += 1
                teachers[row["model"]] += 1
                if row.get("difficulty"):
                    difficulties[f"{row['mode']}:{row['difficulty']}"] += 1
                hashes = image_hashes if row["mode"] == "image_to_image" else text_hashes
                hashes[row["input_sha256"]].append(key)
    for kind, hashes in (("image", image_hashes), ("text", text_hashes)):
        for digest, items in hashes.items():
            if len(items) > 1:
                errors.append(f"duplicate {kind} input hash {digest}: {items}")
    return {"schema": "painter.teacher-candidate-audit.v1",
            "batches": batches, "distinct": dict(distinct),
            "distinct_total": sum(distinct.values()), "alternatives": alternatives,
            "teachers": dict(teachers), "labelled_difficulty": dict(difficulties),
            "unique_image_hashes": len(image_hashes), "unique_text_hashes": len(text_hashes),
            "errors": errors, "status": "candidate_only_no_render_or_visual_admission"}


if __name__ == "__main__":
    result = audit()
    (POOL / "candidate-pool-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(bool(result["errors"]))
