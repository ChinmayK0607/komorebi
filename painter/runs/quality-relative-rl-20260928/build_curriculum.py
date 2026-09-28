"""Build source-matched four-turn SFT opponents from the frozen step-four eval.

The old eval's split names are historical: its six held-out photos become
training tasks here. New held-out photos must be selected independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PHOTO_IDS = (
    "photo-000000000049", "photo-000000000322", "photo-000000000572",
    "photo-000000000562", "photo-000000000431", "photo-000000000634",
)
SHAPE_IDS = (
    "sc-v1-train-sphere-0099", "sc-v1-train-bus-0189",
    "sc-v1-train-flower-0199",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(review: Path, output: Path) -> dict:
    report_path = review / "report.json"
    report = json.loads(report_path.read_text())
    by_id = {row["id"]: row for row in report["rows"]}
    selected = PHOTO_IDS + SHAPE_IDS
    if len(by_id) != len(report["rows"]) or not set(selected) <= set(by_id):
        raise ValueError("frozen evaluation is missing or duplicates a selected case")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in selected:
        source = by_id[case_id]
        sft = source["sft-step512"]
        if not sft["final"] or not sft["valid_turns"]:
            raise ValueError(f"no valid SFT final for {case_id}")
        if sft["valid_turns"][-1]["sha256"] != sft["final_sha256"]:
            raise ValueError(f"SFT final receipt mismatch for {case_id}")
        reference_matches = list(review.glob(f"{case_id}-reference.*"))
        if len(reference_matches) != 1:
            raise ValueError(f"reference selection ambiguous for {case_id}")
        reference, anchor = reference_matches[0], review / sft["final"]
        if digest(reference) != source["reference_sha256"] or digest(anchor) != sft["final_sha256"]:
            raise ValueError(f"image digest mismatch for {case_id}")
        if source["reference_sha256"] == sft["final_sha256"]:
            raise ValueError(f"reference reused as opponent for {case_id}")
        reference_dst = output / "references" / reference.name
        anchor_dst = output / "anchors" / (case_id + ".png")
        reference_dst.parent.mkdir(exist_ok=True)
        anchor_dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(reference, reference_dst)
        shutil.copyfile(anchor, anchor_dst)
        rows.append({
            "id": case_id, "source_group": case_id,
            "family": "photo" if case_id in PHOTO_IDS else "procedural",
            "mode": "fresh", "split": "train",
            "reference": reference_dst.relative_to(output).as_posix(),
            "reference_sha256": digest(reference_dst),
            "opponent": anchor_dst.relative_to(output).as_posix(),
            "opponent_sha256": digest(anchor_dst),
            "opponent_source_group": case_id,
        })
    manifest = {"schema": "painter-four-turn-sft-anchors-v1", "rows": rows}
    manifest_path = output / "train-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    provenance = {
        "source_report_sha256": digest(report_path),
        "baseline": "Qwen3.8-27B photo SFT step 512; frozen four-turn matched evaluation",
        "selected_ids": list(selected), "manifest_sha256": digest(manifest_path),
        "excluded_weak_sft_photo": "photo-000000000077",
        "note": "New train split; previous-run heldout labels do not transfer to this run.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"rows": len(rows), **provenance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.review, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
