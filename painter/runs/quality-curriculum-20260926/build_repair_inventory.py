#!/usr/bin/env python3
"""Freeze all 600 teacher tasks as correction states without quality promotion."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud/teacher600-visual-audit-v1"
BRUSH = re.compile(r"\bbrush\s*\.\s*(?:fill|wash|hatch|line|flowLine|spline|rect|circle|arc|beginShape|polygon|beginStroke|move|endStroke)\s*\(")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"evidence path leaves checkout: {resolved}")
    return resolved.relative_to(ROOT).as_posix()


def build() -> dict:
    audit_manifest = AUDIT / "manifest.json"
    manifest = json.loads(audit_manifest.read_text())
    if manifest["raw_tasks"] != 600 or manifest["reviewable"] != 595 or len(manifest["censored"]) != 5:
        raise ValueError("unexpected 600-task audit scope")
    rows = [r for i in range(6) for r in json.loads((AUDIT / f"shard-{i}.json").read_text())]
    if len(rows) != 595 or len({r["audit_id"] for r in rows}) != 595:
        raise ValueError("duplicate or missing valid canvas rows")
    review = json.loads((HERE / "SOL_BRUSH_PILOT_REVIEW.json").read_text())
    pilot_ids = {r["audit_id"] for r in review["rows"]}
    output = []
    counts = Counter()
    for row in rows:
        canvas = Path(row["canvas_path"])
        if sha(canvas.read_bytes()) != row["canvas_sha256"]:
            raise ValueError(f"changed canvas: {row['audit_id']}")
        program = canvas.parent / "program.js"
        code = program.read_bytes()
        input_path = Path(row["reference_path"]) if row["mode"] == "image_to_image" else canvas.parent / "input.txt"
        if sha(input_path.read_bytes()) != row["input_sha256"]:
            raise ValueError(f"changed reference/prompt: {row['audit_id']}")
        brush = bool(BRUSH.search(code.decode("utf-8")))
        route = "regenerate_from_prior" if not brush else "pairwise_review_then_correct_if_needed"
        if row["audit_id"] in pilot_ids:
            route = "translation_rejected_redraw_from_original"
        record = {
            "audit_id": row["audit_id"], "id": row["id"], "mode": row["mode"],
            "category": row["category"], "source_run_id": row["chosen_run_id"],
            "canvas_route": row["canvas_route"], "input_sha256": row["input_sha256"],
            "program_sha256": sha(code), "canvas_sha256": row["canvas_sha256"],
            "input_path": rel(input_path), "program_path": rel(program),
            "canvas_path": rel(canvas), "actual_brush_paint_call": brush,
            "action": route, "quality_status": "unreviewed_candidate_or_correction_state",
        }
        output.append(record)
        counts.update([route, row["mode"], "brush_paint" if brush else "no_brush_paint"])
    for row in manifest["censored"]:
        output.append({
            "audit_id": None, "id": row["id"], "mode": row["mode"],
            "source_run_id": row["original_run_id"], "input_sha256": row["input_sha256"],
            "action": "repair_render_timeout_before_visual_review", "quality_status": "censored_no_canvas",
        })
        counts.update(["censored", row["mode"]])
    if len(output) != 600 or counts["no_brush_paint"] != 75 or counts["brush_paint"] != 520:
        raise ValueError("repair inventory medium or task count differs from prior audit")
    return {
        "schema": "painter.teacher600-repair-inventory.v1",
        "source_audit_sha256": sha(audit_manifest.read_bytes()),
        "translation_review_sha256": sha((HERE / "SOL_BRUSH_PILOT_REVIEW.json").read_bytes()),
        "hypothesis": "Prior-canvas-conditioned teacher revisions improve visual quality over the matched first paint.",
        "matched_baseline": "The exact hash-pinned first painting of each of 595 rendered tasks; five timeouts are censored.",
        "training_admission": "none_from_this_inventory",
        "selection_policy": "Counts and brush-use are routing metadata, never visual quality labels. Only rendered pairwise wins may become SFT targets.",
        "counts": dict(sorted(counts.items())),
        "rows": output,
    }


if __name__ == "__main__":
    result = build()
    destination = ROOT / "painter/collected/quality-curriculum-20260926/teacher600-repair-inventory-v1/manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {"schema": "painter.teacher600-repair-inventory-summary.v1",
               "manifest_sha256": sha(destination.read_bytes()),
               "source_audit_sha256": result["source_audit_sha256"],
               "translation_review_sha256": result["translation_review_sha256"],
               "row_count": len(result["rows"]), "counts": result["counts"],
               "training_admission": result["training_admission"]}
    (HERE / "REPAIR_INVENTORY_600_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(destination), **summary}, sort_keys=True))
