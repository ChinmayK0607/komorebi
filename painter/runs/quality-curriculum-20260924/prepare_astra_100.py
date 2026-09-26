#!/usr/bin/env python3
"""Stage the exact 100 COCO128 photos outside the fixed 28-photo evaluation set.

The image archive and staged references remain ignored. The small manifest is
reviewable and pins every source by hash; it admits no teacher painting to SFT.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924"
ARCHIVE = COLLECTED / "coco128-source.zip"
STAGED = COLLECTED / "astra-high-100/references"
OUTPUT = RUN / "reference-manifest-astra-100.json"
DEV = ROOT / "painter/runs/photo-curriculum-sft-20260922/eval-prep/eval-manifest.json"
WAVE5 = RUN / "reference-manifest-wave5.json"
ARCHIVE_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    if sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("COCO128 source archive hash mismatch")
    dev_raw = DEV.read_bytes()
    dev = json.loads(dev_raw)["cases"]
    dev_ids = {row["source_id"] for row in dev}
    dev_hashes = {row["reference_sha256"] for row in dev}
    wave5 = {row["source_id"]: row for row in json.loads(WAVE5.read_bytes())["references"]}
    STAGED.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        images = sorted(name for name in archive.namelist()
                        if name.startswith("coco128/images/train2017/") and name.endswith(".jpg"))
        if len(images) != 128 or len(dev_ids) != 28:
            raise ValueError(f"expected 128 source and 28 evaluation images, found {len(images)}, {len(dev_ids)}")
        for name in images:
            source_id = Path(name).stem
            raw = archive.read(name)
            digest = sha(raw)
            if source_id in dev_ids:
                if digest not in dev_hashes:
                    raise ValueError(f"evaluation photo changed: {source_id}")
                continue
            if digest in dev_hashes:
                raise ValueError(f"evaluation hash leaked into train: {source_id}")
            label_name = f"coco128/labels/train2017/{source_id}.txt"
            labels = archive.read(label_name).decode().splitlines() if label_name in archive.namelist() else []
            classes = sorted({int(line.split()[0]) for line in labels if line.strip()})
            width, height = Image.open(io.BytesIO(raw)).size
            staged = STAGED / f"{source_id}.jpg"
            if not staged.is_file() or sha(staged.read_bytes()) != digest:
                staged.write_bytes(raw)
            prior = wave5.get(source_id)
            if prior and prior["sha256"] != digest:
                raise ValueError(f"wave 5 image mismatch: {source_id}")
            records.append({
                "id": f"coco128-{source_id}", "source_id": source_id,
                "sha256": digest, "bytes": len(raw),
                "width": width, "height": height,
                "class_ids": classes, "object_count": len(labels),
                "path": f"references/{source_id}.jpg", "split": "train",
                "wave5_candidate": bool(prior),
                "difficulty_proxy": "few_annotations" if len(labels) <= 3 else "many_annotations",
            })
    if len(records) != 100 or len({r["sha256"] for r in records}) != 100:
        raise ValueError(f"expected 100 unique train photos; got {len(records)}")
    result = {
        "schema": "painter.astra-high-100-references.v1",
        "status": "references_staged_teacher_output_unreviewed",
        "official_archive_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "official_archive_sha256": ARCHIVE_SHA256,
        "excluded_evaluation_manifest_sha256": sha(dev_raw),
        "existing_wave5_manifest_sha256": sha(WAVE5.read_bytes()),
        "count": len(records), "already_in_wave5": sum(r["wave5_candidate"] for r in records),
        "references": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(OUTPUT), "sha256": sha(OUTPUT.read_bytes()),
                      "count": len(records), "already_in_wave5": result["already_in_wave5"],
                      "staged": str(STAGED)}, sort_keys=True))


if __name__ == "__main__":
    main()
