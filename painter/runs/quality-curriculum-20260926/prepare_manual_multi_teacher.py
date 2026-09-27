#!/usr/bin/env python3
"""Make three immutable, source-grounded candidate batches for Linux rendering.

Run this before the existing package_teacher_batch.py and publish_teacher_batch.py.
No generated programs or reference images are added to Git.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
PLAN = HERE / "MULTI_TEACHER_REPAIR_PLAN.json"
MANUAL = POOL / "manual-multi-teacher-20260927"
sys.path.insert(0, str(HERE))
from prepare_repair_hundred import rights_index  # noqa: E402

BATCHES = {
    "sol_5_6_high": ("sol", "manual-repair-sol-five-20260927", "gpt-5.6-sol", "high"),
    "astra_high": ("astra", "manual-repair-astra-five-20260927", "gpt-6-astra", "high"),
    "parent": ("parent", "manual-repair-parent-five-20260927", "codex-parent", "not-exposed"),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare() -> list[dict]:
    plan = json.loads(PLAN.read_text())
    rows = plan["rows"]
    if len(rows) != 15 or len({r["audit_id"] for r in rows}) != 15:
        raise ValueError("expected 15 distinct repair scenes")
    rights = rights_index()
    result = []
    for teacher, (folder, batch, model, effort) in BATCHES.items():
        items = []
        subset = [r for r in rows if r["teacher"] == teacher]
        if len(subset) != 5 or sum(r["mode"] == "text_to_image" for r in subset) != 3:
            raise ValueError(f"unexpected teacher split: {teacher}")
        for row in subset:
            ident = row["audit_id"]
            program = MANUAL / folder / ident / "program.js"
            critique_path = MANUAL / folder / ident / "critique.json"
            critique = json.loads(critique_path.read_text())
            if critique.get("audit_id") != ident:
                raise ValueError(f"critique mismatch: {ident}")
            for field in ("input", "first_paint", "first_program"):
                if sha((ROOT / row[field + "_path"]).read_bytes()) != row[field + "_sha256"]:
                    raise ValueError(f"source changed: {ident} {field}")
            photo_rights = None
            source_url = None
            if row["mode"] == "image_to_image":
                source_id = Path(row["input_path"]).parent.name
                photo_rights = rights.get(source_id)
                if not photo_rights or not all(photo_rights.get(k) for k in ("name", "url", "source_url")):
                    raise ValueError(f"photo rights incomplete: {ident}")
                source_url = photo_rights["source_url"]
            item = {
                "id": ident, "mode": row["mode"], "category": row["category"],
                "new_program_path": str(program.relative_to(ROOT)),
                "new_program_sha256": sha(program.read_bytes()),
                "input_path": row["input_path"], "input_sha256": row["input_sha256"],
                "prior_canvas_path": row["first_paint_path"],
                "prior_canvas_sha256": row["first_paint_sha256"],
                "prior_program_path": row["first_program_path"],
                "prior_program_sha256": row["first_program_sha256"],
                "source_batch": row["source_run_id"], "prior_run_id": row["source_run_id"],
                "rationale": (critique.get("revision_intent") or critique.get("intended_correction")),
                "source_url": source_url,
                "source_metadata": photo_rights,
                "source_visual_type": "photograph" if photo_rights else "text_brief",
                "license": ({"name": photo_rights["name"], "url": photo_rights["url"]}
                            if photo_rights else None),
                "turn_count": 2,
            }
            items.append(item)
        source = POOL / batch
        source.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "painter.manual-repair-author.v1",
            "modality": "render-conditioned-correction",
            "model": model, "reasoning_effort": effort,
            "count": 5, "items": items,
            "plan_sha256": sha(PLAN.read_bytes()),
            "hypothesis": plan["hypothesis"],
            "matched_baseline": plan["matched_baseline"],
            "render_status": "unrendered", "sft_admitted_count": 0,
        }
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        path = source / "manifest.json"
        if path.is_file() and path.read_bytes() != raw:
            raise ValueError(f"existing batch differs: {batch}")
        path.write_bytes(raw)
        result.append({"batch": batch, "model": model, "manifest_sha256": sha(raw), "count": 5})
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
