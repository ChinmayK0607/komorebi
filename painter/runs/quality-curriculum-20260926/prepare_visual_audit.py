#!/usr/bin/env python3
"""Pin the exact 600-task first-paint visual review inputs and repaired canvases."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from build_firstpaint_gallery import collect, RENDERS


OUT = RENDERS / "teacher600-visual-audit-v1"
REPAIR_RUNS = (
    "teacher600-sol-runtime-repair-20260927",
    "teacher600-sol-runtime-repair-v2-20260927",
    "teacher600-sol-runtime-repair-v3-20260927",
    "teacher600-sol-timeout-repair-20260927",
    "teacher600-astra-timeout-repair-20260927",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    originals = collect()
    if len(originals) != 600 or len({(r["mode"], r["input_sha256"]) for r in originals}) != 600:
        raise ValueError("first-paint pool is not exactly 600 distinct inputs")
    repairs = {}
    for run_id in REPAIR_RUNS:
        summary = json.loads((RENDERS / run_id / "run-summary.json").read_text())
        for status in summary["statuses"]:
            if not status["valid"]:
                continue
            key = (status["mode"], status["input_sha256"])
            if key in repairs:
                raise ValueError(f"multiple valid repairs on one input: {key}")
            canvas = RENDERS / run_id / "episodes" / status["id"] / "canvas.png"
            if sha(canvas) != status["canvas_sha256"]:
                raise ValueError(f"repair canvas hash mismatch: {run_id}/{status['id']}")
            repairs[key] = (run_id, status, canvas)
    rows = []
    censored = []
    for original in originals:
        key = (original["mode"], original["input_sha256"])
        run_id = original["run_id"]
        episode = RENDERS / run_id / "episodes" / original["id"]
        source = episode / ("input.txt" if original["mode"] == "text_to_image" else "input.jpg")
        if sha(source) != original["input_sha256"]:
            raise ValueError(f"input hash mismatch: {original['id']}")
        if original["valid"]:
            canvas = episode / "canvas.png"
            canvas_sha = original["canvas_sha256"]
            chosen_run = run_id
            route = "original"
        elif key in repairs:
            chosen_run, status, canvas = repairs[key]
            canvas_sha = status["canvas_sha256"]
            route = "static_repair"
        else:
            censored.append({"id": original["id"], "mode": original["mode"],
                             "input_sha256": original["input_sha256"],
                             "original_run_id": run_id, "error": original["error"]})
            continue
        if sha(canvas) != canvas_sha:
            raise ValueError(f"canvas hash mismatch: {original['id']}")
        rows.append({"id": original["id"], "mode": original["mode"],
                     "category": original["category"], "prompt": original["prompt"],
                     "reference_path": str(source) if original["mode"] == "image_to_image" else None,
                     "reference_sha256": original["input_sha256"] if original["mode"] == "image_to_image" else None,
                     "canvas_path": str(canvas), "canvas_sha256": canvas_sha,
                     "input_sha256": original["input_sha256"],
                     "original_run_id": run_id, "chosen_run_id": chosen_run,
                     "canvas_route": route})
    if len(rows) != 595 or len(censored) != 5 or len(repairs) != 17:
        raise ValueError(f"unexpected audit scope: {len(rows)} valid, {len(censored)} censored, {len(repairs)} repairs")
    rows.sort(key=lambda r: (r["mode"], r["category"], r["id"]))
    for index, row in enumerate(rows):
        row["audit_id"] = f"t600-{index+1:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(6)]
    for index, row in enumerate(rows):
        shards[index % len(shards)].append(row)
    for i, shard in enumerate(shards):
        (OUT / f"shard-{i}.json").write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n")
    summary = {"schema": "painter.teacher600-visual-audit.v1", "raw_tasks": len(originals),
               "reviewable": len(rows), "censored": censored,
               "routes": dict(Counter(r["canvas_route"] for r in rows)),
               "shards": [{"path": str(OUT / f"shard-{i}.json"), "count": len(shard)}
                          for i, shard in enumerate(shards)],
               "rubric": {"A": "Direct SFT candidate: attractive, coherent, recognizable and, for photos, faithful to important reference structure.",
                          "B": "Useful correction seed: readable but has a concrete visual or fidelity defect.",
                          "C": "Reject: confusing, ugly or substantially wrong; little value as a starting canvas.",
                          "U": "Uncertain: image/reference cannot be judged confidently; requires parent review."},
               "status": "review_packet_not_labels_or_sft_admission"}
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    result = build()
    print(json.dumps({key: result[key] for key in ("raw_tasks", "reviewable", "routes", "shards")}, sort_keys=True))
