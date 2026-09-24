#!/usr/bin/env python3
"""Freeze visually screened easy and manual-teacher reference bytes.

This only copies source images without transforming them. Generated assets and
COCO photos remain ignored; a small hash-bound manifest is the tracked record.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
ASSETS = ROOT / "painter/collected/quality-curriculum-20260924/references"
COCO_MANIFEST = ROOT / "painter/runs/scaled-curriculum-20260921/real-pool/reference-manifest.json"
BENCHMARK = ROOT / "painter/benchmarks/openrouter-teachers-20260922/refs.json"

EASY = (
    ("simple-cup", "clean form", "single cream ceramic cup, quiet blue wall"),
    ("simple-pear", "clean form", "single golden pear, green leaf, linen cloth"),
    ("simple-poppy", "clean form", "single red poppy, clear vase, windowsill"),
    ("simple-sailboat", "clean form", "blue wooden toy sailboat, cream sails"),
    ("simple-orange", "clean form", "single orange and leaf on pale blue plate"),
    ("simple-cottage", "recognizable subject", "white cottage, clay roof, landscape"),
    ("simple-fern", "recognizable subject", "potted fern by a window"),
)

# These exact train-split COCO128 scenes were inspected visually. They are
# structurally harder than the seven clean-form images and are excluded from
# the 40-reference teacher benchmark. They are candidates, not accepted SFT.
MANUAL_COCO_IDS = ("000000000025", "000000000034", "000000000092",
                   "000000000263", "000000000394", "000000000400",
                   "000000000471", "000000000491", "000000000605",
                   "000000000650")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for ident, tier, description in EASY:
        source = ASSETS / f"{ident}.png"
        if not source.is_file():
            raise FileNotFoundError(f"generated reference missing: {source}")
        rows.append({"id": ident, "split": "train", "tier": tier,
                     "source_kind": "codex_imagegen_generated_reference",
                     "reference_path": str(source.relative_to(ROOT)),
                     "sha256": sha(source), "description": description,
                     "visual_review": "accepted_reference_only",
                     "teacher_status": "not_generated"})

    pool = json.loads(COCO_MANIFEST.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in pool["references"]}
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    benchmark_ids = {row["id"] for row in benchmark["references"]}
    for number in MANUAL_COCO_IDS:
        ident = f"coco128-{number}"
        if ident in benchmark_ids:
            raise ValueError(f"benchmark reference selected for training: {ident}")
        row = by_id[ident]
        if row["split"] != "train":
            raise ValueError(f"non-training COCO source selected: {ident}")
        source = ROOT / row["reference_path"]
        if sha(source) != row["image_sha256"]:
            raise ValueError(f"source hash mismatch: {ident}")
        target = ASSETS / f"{ident}.jpg"
        if target.is_file() and sha(target) != row["image_sha256"]:
            raise ValueError(f"existing copied image differs: {ident}")
        if not target.is_file():
            shutil.copyfile(source, target)
        rows.append({"id": ident, "split": "train", "tier": "manual hard",
                     "source_kind": "coco128_train_photo",
                     "source_group": row["source_group"],
                     "reference_path": str(target.relative_to(ROOT)),
                     "sha256": row["image_sha256"],
                     "description": ", ".join(row.get("annotation_classes") or []),
                     "per_image_license_status": row["per_image_license_status"],
                     "visual_review": "accepted_reference_only",
                     "teacher_status": "not_generated"})

    output = {"schema": "painter.quality-curriculum.references.v1",
              "purpose": "teacher candidate references; no painting accepted yet",
              "reference_count": len(rows), "easy_count": len(EASY),
              "manual_hard_count": len(MANUAL_COCO_IDS), "references": rows}
    path = RUN / "reference-manifest.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cards = []
    for row in rows:
        img = (ASSETS / Path(str(row["reference_path"])).name).as_uri()
        cards.append(f'<article><img src="{html.escape(img)}" alt="{html.escape(str(row["description"]))}">'
                     f'<h2>{html.escape(str(row["id"]))}</h2><p>{html.escape(str(row["tier"]))} · '
                     f'{html.escape(str(row["description"]))}</p></article>')
    gallery = ROOT / "painter/collected/quality-curriculum-20260924/reference-review.html"
    gallery.write_text('<!doctype html><meta charset="utf-8"><title>Quality curriculum references</title>'
                       '<style>body{font:16px system-ui;background:#f6f3ec;color:#25231f;margin:24px}'
                       'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}'
                       'article{background:white;padding:10px;border-radius:8px;box-shadow:0 1px 6px #0002}'
                       'img{width:100%;height:280px;object-fit:contain;background:#e6e3db}'
                       'h2{font-size:15px;margin:8px 0 2px}p{font-size:13px;margin:0;color:#555}</style>'
                       '<h1>17 candidate references</h1><p>Seven clean generated inputs; ten harder COCO training '
                       'photos reserved for manual painting loops. These are references, not teacher outputs.</p>'
                       '<main>' + ''.join(cards) + '</main>\n', encoding="utf-8")
    print(f"{len(rows)} hash-bound reference candidates; review {gallery}")


if __name__ == "__main__":
    main()
