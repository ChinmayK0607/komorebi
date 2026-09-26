#!/usr/bin/env python3
"""Freeze a visually inspected, mixed eight-scene correction wave."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INVENTORY = ROOT / "painter/collected/quality-curriculum-20260926/teacher600-repair-inventory-v1/manifest.json"

# Parent inspected both source photo and current canvas for each photo task,
# and the actual canvas plus exact prompt for each text task.
SELECTION = [
    ("t600-511", "easy", "The duck silhouette is washed out; deepen the honey wood body, sharpen the beak and tail, show carved wing relief and a soft contact shadow."),
    ("t600-427", "easy", "The avocado halves are pale and flat; establish pebbled dark skins, yellow-green flesh, one brown stone, the empty cavity and a convincing oblique view."),
    ("t600-181", "easy", "The silver pot is nearly invisible; rebuild its concave neck, long rising spout, dark wooden right handle, metal reflections and contact shadow."),
    ("t600-331", "middle", "The hedgehog reads as a simple pale oval; build dense varied brown spines, a narrow cream face, four short feet and depth in the gravel path."),
    ("t600-085", "middle", "The noodles are loose loops on a flat disk and broccoli is green dots; show glossy overlapping noodles, real broccoli florets and the plate's elliptical depth."),
    ("t600-145", "hard", "The boat, lake, mountain and shore are schematic. Use the actual monochrome photo to fix boat scale and placement, shore architecture, wooded slopes and distant peaks."),
    ("t600-457", "hard", "The basalt columns read as pale blocks; restore black rock masses, cobalt sea, rust beach, curling white surf and the distant island."),
    ("t600-013", "hard", "The black dog becomes a mint-colored cartoon. Restore a dark furry head with expressive glossy eyes, broad muzzle, floppy ears and snow flecks against a light background."),
]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    summary = json.loads((HERE / "REPAIR_INVENTORY_600_SUMMARY.json").read_text())
    if sha(INVENTORY.read_bytes()) != summary["manifest_sha256"]:
        raise ValueError("repair inventory changed")
    by_id = {r["audit_id"]: r for r in inventory["rows"] if r["audit_id"]}
    rows = []
    for audit_id, difficulty, weakness in SELECTION:
        source = by_id[audit_id]
        if source["quality_status"] != "unreviewed_candidate_or_correction_state":
            raise ValueError(f"unexpected source status: {audit_id}")
        rows.append({key: source[key] for key in (
            "audit_id", "id", "mode", "category", "source_run_id", "input_sha256",
            "program_sha256", "canvas_sha256", "input_path", "program_path", "canvas_path"
        )} | {"difficulty": difficulty, "parent_visual_weakness": weakness,
             "status": "inspected_correction_state_not_training_target"})
    if len(rows) != 8 or len({r["audit_id"] for r in rows}) != 8:
        raise ValueError("unexpected next-wave size")
    result = {"schema": "painter.teacher600-next-repair-wave.v1",
              "source_inventory_sha256": summary["manifest_sha256"],
              "selection": "Four text, four photo, visually inspected and ordered easy-to-hard",
              "hypothesis": "Actual-canvas-conditioned revisions improve beauty and subject fidelity over these eight exact first paints.",
              "matched_baseline": "Exact hash-pinned first painting for each row, with the same source prompt or photo.",
              "training_admission": "none_until_rendered_pairwise_review",
              "rows": rows}
    target = HERE / "NEXT_REPAIR_8.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(target), "sha256": sha(target.read_bytes()),
                      "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
