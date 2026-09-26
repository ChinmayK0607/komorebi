#!/usr/bin/env python3
"""Collect newly published finite render batches and refresh the review gallery."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from build_firstpaint_gallery import DEST, build
from collect_teacher_render import OUT, collect, gallery


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "painter/runs/quality-curriculum-20260926"
DATASET = "CK0607/komorebi-painter-teachers"
EXTRA = [
    {"batch": "astra-text-eight-v1", "count": 8,
     "run_id": "teacher600-astra-text-eight-20260927",
     "source_archive_sha256": "caaa093bc11eae390e04750f36632531f2e06692f8888a438a154f42270cd91b"},
    {"batch": "astra-reference-seed-program-overlay-v1", "count": 12,
     "run_id": "teacher600-astra-coco128-overlay-20260927",
     "source_archive_sha256": "d521ef151de14c3410dddf9f7e7d4844fc36c75e2de853173cf59a12a961e717"},
]


def public_receipt_exists(run_id: str) -> bool:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/runs/{run_id}/receipt.json?download=true"
    try:
        with urlopen(url, timeout=30) as response:
            receipt = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    if receipt.get("run_id") != run_id or receipt.get("public_hash_verified") is not True:
        raise ValueError(f"unverified public render receipt: {run_id}")
    return True


def main() -> None:
    plan = json.loads((RUN / "REMAINING_FIRSTPAINT_RENDER_PLAN.json").read_text())
    rows = [row for group in plan["groups"].values() for row in group] + EXTRA
    for row in rows:
        output = OUT / row["run_id"]
        summary_path = output / "run-summary.json"
        if summary_path.is_file() and (output / "collection-receipt.json").is_file():
            summary = json.loads(summary_path.read_text())
        elif public_receipt_exists(row["run_id"]):
            output, summary = collect(row["run_id"])
            gallery(output, summary)
            print(f"collected {row['run_id']}: {summary['valid']}/{summary['count']}", flush=True)
        else:
            continue
        if (summary.get("batch") != row["batch"] or summary.get("count") != row["count"]
                or summary.get("source_bundle_sha256") != row["source_archive_sha256"]):
            raise ValueError(f"collected render does not match plan: {row['run_id']}")
    path = build()
    coverage = json.loads((DEST / "coverage.json").read_text())
    coverage["raw_target"] = 600
    coverage["remaining_uncollected"] = 600 - coverage["collected"]
    (DEST / "progress.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gallery": str(path), **coverage}, sort_keys=True))


if __name__ == "__main__":
    main()
