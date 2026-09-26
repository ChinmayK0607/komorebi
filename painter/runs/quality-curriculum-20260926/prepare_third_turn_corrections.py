#!/usr/bin/env python3
"""Bind selected third-turn programs to the exact second-turn canvases inspected."""

from __future__ import annotations

import json

from prepare_render_conditioned_corrections import POOL, normalized_row, source_rows


def prepare() -> list[str]:
    sources = source_rows()
    astra = json.loads((POOL / "astra-shell-turn3-v1/audit.json").read_text())
    sol = json.loads((POOL / "sol-can-turn3-v1/audit.json").read_text())
    if (astra["id"] != "openverse-a698f10d-291e-4474-9dab-7511a0ae5280"
            or astra["teacher_model"] != "gpt-6-astra" or astra["reasoning_effort"] != "high"
            or sol["source_id"] != "openverse-db1a60b0-83db-4115-939f-4e97606a04aa"
            or sol["model"] != "gpt-5.6-sol" or sol["reasoning_effort"] != "high"):
        raise ValueError("unexpected third-turn teacher identity")
    ae = {
        "reference": (astra["reference"]["path"], astra["reference"]["sha256"]),
        "prior_canvas": (astra["observed_turn2_canvas"]["path"],
                         astra["observed_turn2_canvas"]["sha256"]),
        "prior_program": (astra["prior_full_program"]["path"],
                          astra["prior_full_program"]["sha256"]),
        "prompt": (astra["original_prompt"]["path"], astra["original_prompt"]["sha256"]),
        "new_program": (astra["new_full_program"]["path"], astra["new_full_program"]["sha256"]),
    }
    se = sol["evidence"]
    so = {
        "reference": (se["reference_path"], se["reference_sha256"]),
        "prior_canvas": (se["observed_turn2_canvas_path"], se["observed_turn2_canvas_sha256"]),
        "prior_program": (se["turn2_full_program_path"], se["turn2_full_program_sha256"]),
        "prompt": (se["original_prompt_path"], se["original_prompt_sha256"]),
        "new_program": (sol["turn3_full_program_path"], sol["turn3_full_program_sha256"]),
    }
    cases = [
        ("astra-shell-render-conditioned-turn3-v1", "gpt-6-astra", astra["id"],
         ae, astra["rationale"], astra["prior_run_id"]),
        ("sol-can-render-conditioned-turn3-v1", "gpt-5.6-sol", sol["source_id"],
         so, sol["correction_rationale"], "teacher500-sol-turn2-20260926"),
    ]
    paths = []
    for name, model, ident, evidence, rationale, prior_run in cases:
        row = normalized_row(ident, sources[ident], evidence, rationale,
                             prior_run=prior_run, turn_count=3)
        directory = POOL / name
        directory.mkdir(exist_ok=True)
        path = directory / "manifest.json"
        content = {"schema": "painter.render-conditioned-corrections.v1",
                   "modality": "image-to-painting-render-conditioned-correction",
                   "model": model, "reasoning_effort": "high", "source_batch": row["source_batch"],
                   "prior_run_id": prior_run, "count": 1, "items": [row],
                   "render_status": "awaiting_third_turn_render", "sft_admitted_count": 0}
        raw = json.dumps(content, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != raw:
            raise ValueError(f"existing third-turn manifest differs: {path}")
        path.write_text(raw)
        paths.append(str(directory))
    return paths


if __name__ == "__main__":
    print(json.dumps(prepare()))
