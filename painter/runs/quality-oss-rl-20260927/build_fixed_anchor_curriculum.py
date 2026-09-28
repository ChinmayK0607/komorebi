"""Build an auditable mixed-reference RL set from existing rendered evidence.

This is deliberately offline: no model, judge, renderer, or GPU is started.
The output images are generated data and should be published outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PHOTO_TRAIN = (
    "000000000127", "000000000009", "000000000419", "000000000599",
    "000000000089", "000000000208", "000000000357", "000000000078",
    "000000000072", "000000000113", "000000000359", "000000000584",
)
PHOTO_HELDOUT = (
    "000000000049", "000000000077", "000000000322", "000000000572",
    "000000000562", "000000000431", "000000000634",
)
PROCEDURAL_TRAIN = (
    "sc-v1-train-sphere-0099", "sc-v1-train-bus-0189",
    "sc-v1-train-flower-0199",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_copy(source: Path, expected: str, destination: Path) -> str:
    if not source.is_file() or digest(source) != expected:
        raise ValueError(f"source missing or hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != expected:
        raise ValueError(f"copied image hash mismatch: {destination}")
    return destination.name


def build(photo_gallery: Path, window_evidence: Path, output: Path) -> dict:
    gallery = json.loads((photo_gallery / "manifest.json").read_text())
    photos = {r["id"].rsplit("-", 1)[-1]: r for r in gallery["records"]}
    prior = window_evidence / "matched-eval-step-4" / "review"
    procedural = {r["id"]: r for r in json.loads((prior / "report.json").read_text())["rows"]}
    output.mkdir(parents=True, exist_ok=True)
    train: list[dict] = []
    heldout: list[dict] = []
    provenance: list[dict] = []

    for name in PHOTO_TRAIN + PHOTO_HELDOUT:
        record = photos[name]
        reference = record["reference"]["asset"]
        baseline = record["policies"]["midpoint512"]
        if baseline["status"] != "valid" or not baseline["asset"]:
            raise ValueError(f"SFT step-512 anchor unavailable: {name}")
        baseline_asset = baseline["asset"]
        task_id = "photo-" + name
        ref_dst = output / "references" / f"{task_id}.jpg"
        anchor_dst = output / "anchors" / f"{task_id}.png"
        checked_copy(photo_gallery / reference["path"], reference["sha256"], ref_dst)
        checked_copy(photo_gallery / baseline_asset["path"], baseline_asset["sha256"], anchor_dst)
        base = {
            "id": task_id, "source_group": task_id, "family": "photo",
            "reference": str(ref_dst.relative_to(output)),
            "reference_sha256": reference["sha256"],
        }
        if name in PHOTO_TRAIN:
            train.append({**base, "split": "train", "mode": "fresh",
                          "opponent": str(anchor_dst.relative_to(output)),
                          "opponent_sha256": baseline_asset["sha256"],
                          "opponent_source_group": task_id})
        else:
            heldout.append(base)
        provenance.append({"id": task_id, "source": "photo-sft-matched-gallery",
                           "anchor_policy": "sft-step512-midpoint512-one-turn",
                           "source_manifest_sha256": digest(photo_gallery / "manifest.json"),
                           "reference_sha256": reference["sha256"],
                           "anchor_sha256": baseline_asset["sha256"]})

    for task_id in PROCEDURAL_TRAIN:
        record = procedural[task_id]
        ref_source = prior / f"{task_id}-reference.png"
        anchor = record["sft-step512"]
        if not anchor["final"]:
            raise ValueError(f"SFT step-512 anchor unavailable: {task_id}")
        anchor_source = prior / anchor["final"]
        ref_dst = output / "references" / f"{task_id}.png"
        anchor_dst = output / "anchors" / f"{task_id}.png"
        checked_copy(ref_source, record["reference_sha256"], ref_dst)
        checked_copy(anchor_source, anchor["final_sha256"], anchor_dst)
        train.append({"id": task_id, "source_group": task_id,
                      "family": task_id.split("-")[3], "split": "train",
                      "mode": "fresh", "reference": str(ref_dst.relative_to(output)),
                      "reference_sha256": record["reference_sha256"],
                      "opponent": str(anchor_dst.relative_to(output)),
                      "opponent_sha256": anchor["final_sha256"],
                      "opponent_source_group": task_id})
        provenance.append({"id": task_id, "source": "window-run-matched-eval",
                           "anchor_policy": "sft-step512-four-turn-cap",
                           "source_report_sha256": digest(prior / "report.json"),
                           "reference_sha256": record["reference_sha256"],
                           "anchor_sha256": anchor["final_sha256"]})

    if len(train) != 15 or len(heldout) != 7 or len({r["id"] for r in train + heldout}) != 22:
        raise AssertionError("curriculum count or uniqueness changed")
    train_doc = {"schema": "native-painting-rl-mixed-fixed-anchors-v1",
                 "run_id": "quality-mixed-anchors-20260928",
                 "training_scope": {"photos": 12, "procedural": 3,
                                    "anchor_note": "Photo anchors are selected one-turn SFT outputs; compare full four-turn SFT in evaluation."},
                 "rows": train}
    heldout_doc = {"schema": "native-painting-heldout-cases-v1", "cases": heldout}
    for name, data in (("train-manifest.json", train_doc),
                       ("heldout-cases.json", heldout_doc),
                       ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {"train": len(train), "heldout": len(heldout),
            "train_manifest_sha256": digest(output / "train-manifest.json"),
            "heldout_cases_sha256": digest(output / "heldout-cases.json"),
            "provenance_sha256": digest(output / "provenance.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-gallery", type=Path, required=True)
    parser.add_argument("--window-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.photo_gallery, args.window_evidence, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
