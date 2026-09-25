#!/usr/bin/env python3
"""Select disjoint, category-diverse COCO128 photos for the next finite teacher wave.

The source tar is an ignored upload artifact. The small manifest is reviewable in Git.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
COCO = ROOT / "painter/runs/scaled-curriculum-20260921/real-pool/raw/coco128"
DEV = ROOT / "painter/collected/quality-curriculum-20260924/multiturn-sft-wave4/eval-collected/evidence/painter/eval-prep/eval-manifest.json"
PRIOR = RUN / "reference-manifest-wave3.json"
COCO_ARCHIVE_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select(candidates: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    remaining = list(candidates)
    chosen: list[dict] = []
    seen_classes: set[int] = set()
    seen_aspects: set[str] = set()
    while len(chosen) < count:
        if not remaining:
            raise ValueError(f"only {len(chosen)} candidates available; expected {count}")
        scored = []
        for item in remaining:
            score = 8 * len(set(item["class_ids"]) - seen_classes)
            score += 3 * (item["aspect"] not in seen_aspects)
            score += len(set(item["class_ids"])) / 10
            score += rng.random() / 1000
            scored.append((score, item["id"], item))
        _, _, picked = max(scored)
        chosen.append(picked)
        seen_classes.update(picked["class_ids"])
        seen_aspects.add(picked["aspect"])
        remaining.remove(picked)
    return chosen


def build(output: Path, manifest_path: Path) -> dict:
    from PIL import Image

    dev = json.loads(DEV.read_text())
    prior = json.loads(PRIOR.read_text())
    excluded_ids = {case["source_id"] for case in dev["cases"]}
    excluded_hashes = {case["reference_sha256"] for case in dev["cases"]}
    for row in prior["references"]:
        excluded_ids.add(row["id"].removeprefix("coco128-"))
        excluded_hashes.add(row["sha256"])
    files = sorted((COCO / "images/train2017").glob("*.jpg"))
    if len(files) != 128:
        raise ValueError(f"expected 128 local COCO128 train photos, found {len(files)}")
    candidates = []
    for image in files:
        raw = image.read_bytes()
        digest = sha(raw)
        if image.stem in excluded_ids or digest in excluded_hashes:
            continue
        label = COCO / "labels/train2017" / (image.stem + ".txt")
        lines = label.read_text().splitlines() if label.is_file() else []
        classes = sorted({int(line.split()[0]) for line in lines if line.strip()})
        width, height = Image.open(io.BytesIO(raw)).size
        aspect = "wide" if width / height > 1.3 else "tall" if height / width > 1.3 else "square"
        candidates.append({"id": f"coco128-{image.stem}", "source_id": image.stem,
                           "file": image, "sha256": digest, "bytes": len(raw),
                           "object_count": len(lines), "class_ids": classes,
                           "width": width, "height": height, "aspect": aspect})
    easier = select([r for r in candidates if 1 <= r["object_count"] <= 3], 24, 2509251)
    harder = select([r for r in candidates if r["object_count"] >= 4], 24, 2509252)
    selected = easier + harder
    if len({r["sha256"] for r in selected}) != 48:
        raise ValueError("duplicate reference hashes in wave 5")
    rows = []
    for idx, row in enumerate(selected):
        shard = ("easy-a" if idx < 12 else "easy-b" if idx < 24 else
                 "hard-a" if idx < 36 else "hard-b")
        rows.append({k: v for k, v in row.items() if k != "file"} | {
            "shard": shard, "tier": "less_cluttered_photo" if idx < 24 else "more_complex_photo",
            "model": "xiaomi/mimo-v2.6-flash" if idx < 24 else "xiaomi/mimo-v2.6-pro",
            "max_turns": 4 if idx < 24 else 6,
            "path": f"references/{row['source_id']}.jpg",
            "split": "train",
        })
    manifest = {"schema": "painter.mimo-wave5-source.v1", "status": "candidate_generation_not_training_admission",
                "selection": "deterministic greedy category/aspect coverage in two annotation-count buckets, 24 each",
                "official_coco128_archive_sha256": COCO_ARCHIVE_SHA256,
                "excluded_dev_manifest_sha256": sha(DEV.read_bytes()),
                "excluded_prior_manifest_sha256": sha(PRIOR.read_bytes()),
                "count": len(rows), "shards": {s: 12 for s in ("easy-a", "easy-b", "hard-a", "hard-b")},
                "references": rows}
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(raw_manifest)
    with tarfile.open(output, "w:gz") as archive:
        meta = tarfile.TarInfo("manifest.json")
        meta.size, meta.mtime, meta.mode = len(raw_manifest), 0, 0o644
        archive.addfile(meta, io.BytesIO(raw_manifest))
        for row in selected:
            raw = row["file"].read_bytes()
            meta = tarfile.TarInfo(f"references/{row['source_id']}.jpg")
            meta.size, meta.mtime, meta.mode = len(raw), 0, 0o644
            archive.addfile(meta, io.BytesIO(raw))
    receipt = {"schema": "painter.mimo-wave5-source-receipt.v1", "manifest_sha256": sha(raw_manifest),
               "archive_sha256": sha(output.read_bytes()), "archive_bytes": output.stat().st_size,
               "source_photos": len(rows), "shards": manifest["shards"]}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.manifest)))


if __name__ == "__main__":
    main()
