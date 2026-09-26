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
    render_conditioned_corrections = 0
    batches = 0
    prior_campaign_reference_reuse = []
    missing_image_license = []
    errors = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    text_hashes: dict[str, list[str]] = defaultdict(list)
    image_source_urls: dict[str, list[str]] = defaultdict(list)
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
            author_raw = (batch / "manifest.json").read_bytes()
            if sha(author_raw) != manifest["author_manifest_sha256"]:
                errors.append(f"{batch.name}: author manifest changed after packaging")
            author = json.loads(author_raw)
            author_rows = {
                row["id"]: row for row in author.get("items", author.get("samples", author.get("entries", [])))
            }
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
                if row["role"] == "render_conditioned_correction_candidate":
                    render_conditioned_corrections += 1
                    if not row.get("baseline_batch") or not row.get("prior_run_id"):
                        errors.append(f"{key}: correction lacks first-turn link")
                    for member_field, digest_field in (("prior_canvas", "prior_canvas_sha256"),
                                                       ("baseline_program", "baseline_program_sha256")):
                        member = row.get(member_field)
                        if not member or sha(archive.extractfile(member).read()) != row.get(digest_field):
                            errors.append(f"{key}: correction prior evidence mismatch: {member_field}")
                    continue
                distinct[row["mode"]] += 1
                teachers[row["model"]] += 1
                if row["mode"] == "image_to_image":
                    if row.get("source_url"):
                        image_source_urls[row["source_url"]].append(key)
                    source_row = author_rows.get(row["id"], {})
                    source_path = source_row.get("reference_path", source_row.get("reference_file", ""))
                    if "quality-curriculum-20260924/" in source_path:
                        prior_campaign_reference_reuse.append(key)
                    license_info = row.get("license") or {}
                    if not license_info.get("id") and not license_info.get("name"):
                        missing_image_license.append(key)
                if row.get("difficulty"):
                    difficulties[f"{row['mode']}:{row['difficulty']}"] += 1
                hashes = image_hashes if row["mode"] == "image_to_image" else text_hashes
                hashes[row["input_sha256"]].append(key)
    for kind, hashes in (("image", image_hashes), ("text", text_hashes)):
        for digest, items in hashes.items():
            if len(items) > 1:
                errors.append(f"duplicate {kind} input hash {digest}: {items}")
    for url, items in image_source_urls.items():
        if len(items) > 1:
            errors.append(f"duplicate image source URL {url}: {items}")
    return {"schema": "painter.teacher-candidate-audit.v1",
            "batches": batches, "distinct": dict(distinct),
            "distinct_total": sum(distinct.values()), "alternatives": alternatives,
            "render_conditioned_corrections": render_conditioned_corrections,
            "teachers": dict(teachers), "labelled_difficulty": dict(difficulties),
            "unique_image_hashes": len(image_hashes), "unique_text_hashes": len(text_hashes),
            "unique_image_source_urls": len(image_source_urls),
            "prior_campaign_reference_reuse": prior_campaign_reference_reuse,
            "missing_image_license": missing_image_license,
            "new_input_coverage_total": sum(distinct.values()) - len(prior_campaign_reference_reuse),
            "errors": errors, "status": "candidate_only_no_render_or_visual_admission"}


if __name__ == "__main__":
    result = audit()
    (POOL / "candidate-pool-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(bool(result["errors"]))
