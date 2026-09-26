#!/usr/bin/env python3
"""Select disjoint, caption-diverse COCO val2017 references for teacher authoring.

The source metadata is pinned by hash. Downloaded photos stay outside Git; the
manifest keeps their source and license so a later dataset can attribute them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "painter/collected/quality-curriculum-20260926/coco-val2017"
METADATA_URL = (
    "https://huggingface.co/datasets/HamGangster/coco_2017_caption_validation/"
    "resolve/57b44af916dcc91ec05b8c1a86977e7a5a65764e/captions_val2017.json"
)
METADATA_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/"
ALLOWED_LICENSES = {4, 5, 7, 8}
PATTERNS = {
    "animals": r"\b(dog|cat|horse|bird|cow|sheep|elephant|bear|zebra|giraffe|duck|goat|rabbit)\b",
    "vehicles": r"\b(car|bus|train|truck|airplane|plane|bicycle|bike|motorcycle|boat|ship|skateboard)\b",
    "food": r"\b(food|pizza|sandwich|cake|apple|banana|orange|donut|doughnut|meal|plate|bowl|bread|fruit)\b",
    "people": r"\b(man|woman|person|people|boy|girl|child|children|family|crowd)\b",
    "interiors": r"\b(room|kitchen|bed|sofa|couch|toilet|bathroom|living room|office)\b",
    "nature": r"\b(tree|mountain|river|beach|ocean|snow|lake|forest|flower|grass|garden)\b",
    "streets": r"\b(street|building|city|sidewalk|traffic|bridge|road|store|shop)\b",
    "sports": r"\b(baseball|tennis|soccer|ski|skate|surf|frisbee|football|basketball)\b",
    "objects": r"\b(chair|table|clock|lamp|bottle|vase|book|umbrella|luggage|computer|television)\b",
    "other": r".",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_metadata(path: Path | None) -> tuple[dict, str]:
    raw = path.read_bytes() if path else urlopen(METADATA_URL, timeout=120).read()
    digest = sha(raw)
    if digest != METADATA_SHA256:
        raise ValueError(f"COCO captions metadata hash mismatch: {digest}")
    return json.loads(raw), digest


def choose(metadata: dict, digest: str, per_category: int = 25, heldout_per_category: int = 5) -> dict:
    captions: dict[int, list[str]] = defaultdict(list)
    for row in metadata["annotations"]:
        captions[row["image_id"]].append(row["caption"])
    licenses = {row["id"]: row for row in metadata["licenses"]}
    images = [row for row in metadata["images"] if row["license"] in ALLOWED_LICENSES]
    images_by_id = {row["id"]: row for row in images}
    if len(images_by_id) != len(images):
        raise ValueError("duplicate source image ID")
    tagged: dict[str, list[int]] = {name: [] for name in PATTERNS}
    for row in images:
        words = " ".join(captions[row["id"]]).lower()
        for name, pattern in PATTERNS.items():
            if name == "other" or re.search(pattern, words):
                tagged[name].append(row["id"])

    used: set[int] = set()
    selected: dict[str, dict[str, list[int]]] = {"train": {}, "heldout": {}}
    for split, count in (("train", per_category), ("heldout", heldout_per_category)):
        # Prefer low IDs only after the seeded hash: ordering is independent of
        # JSON storage order and repeatable on any machine.
        for category in PATTERNS:
            pool = sorted(tagged[category], key=lambda ident: (sha(f"20260926:{ident}".encode()), ident))
            picks = [ident for ident in pool if ident not in used][:count]
            if len(picks) != count:
                raise ValueError(f"insufficient {split} source images in {category}")
            selected[split][category] = picks
            used.update(picks)

    entries = []
    for split, groups in selected.items():
        # Interleave categories so --download-count 20 provides broad coverage.
        for offset in range(per_category if split == "train" else heldout_per_category):
            for category, ids in groups.items():
                source = images_by_id[ids[offset]]
                license_row = licenses[source["license"]]
                entries.append({
                    "id": f"coco-val2017-{source['id']:012d}",
                    "source_id": f"{source['id']:012d}",
                    "split": split,
                    "selection_category": category,
                    "captions": captions[source["id"]],
                    "source_url": IMAGE_URL + source["file_name"],
                    "flickr_url": source["flickr_url"],
                    "license_id": license_row["id"],
                    "license_name": license_row["name"],
                    "license_url": license_row["url"],
                    "width": source["width"], "height": source["height"],
                    "local_reference": f"references/{source['file_name']}",
                    "review_status": "unreviewed_source",
                })
    return {
        "schema": "painter.teacher-reference-pool.v1",
        "metadata_url": METADATA_URL, "metadata_sha256": digest,
        "license_filter": sorted(ALLOWED_LICENSES),
        "selection_seed": "20260926", "train_count": 10 * per_category,
        "heldout_count": 10 * heldout_per_category,
        "entries": entries,
    }


def download(manifest: dict, count: int, output: Path) -> dict:
    selected = [row for row in manifest["entries"] if row["split"] == "train"][:count]
    refs = output / "references"
    refs.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, row in enumerate(selected, 1):
        target = refs / f"{row['source_id']}.jpg"
        if target.exists():
            raw = target.read_bytes()
        else:
            with urlopen(row["source_url"], timeout=120) as response:
                raw = response.read(5_000_001)
        if len(raw) > 5_000_000 or not raw.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"invalid JPEG source: {row['id']}")
        if not target.exists():
            target.write_bytes(raw)
        receipts.append({"id": row["id"], "source_url": row["source_url"],
                         "path": row["local_reference"], "bytes": len(raw), "sha256": sha(raw)})
        print(f"reference {index}/{len(selected)} {row['id']} {len(raw)} bytes", flush=True)
    receipt = {"schema": "painter.teacher-reference-download.v1", "count": len(receipts), "files": receipts}
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (output / f"download-receipt-n{len(receipts)}.json").write_text(payload)
    (output / "download-receipt.json").write_text(payload)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="already downloaded source metadata")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--download-count", type=int, default=0,
                        help="fetch this many diverse train references, up to 250")
    args = parser.parse_args()
    if not 0 <= args.download_count <= 250:
        parser.error("--download-count must be 0..250")
    metadata, digest = source_metadata(args.metadata)
    manifest = choose(metadata, digest)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference-pool.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.download_count:
        download(manifest, args.download_count, args.output)
    print(json.dumps({"train": manifest["train_count"], "heldout": manifest["heldout_count"],
                      "downloaded": args.download_count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
