#!/usr/bin/env python3
"""Bind authored next-turn programs to the exact observed photo/canvas state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_diverse_second_turns import bind, original_rows
from prepare_render_conditioned_corrections import POOL


def prepare(audit_directory: str, source_batch: str, output_batch: str) -> Path:
    audit_path = POOL / audit_directory / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("schema") != "painter.render-conditioned-corrections-audit.v1":
        raise ValueError("unexpected agent correction audit schema")
    prior_run = audit["prior_run_id"]
    source = original_rows(source_batch)
    entries = audit["entries"]
    if not entries or len({row["id"] for row in entries}) != len(entries):
        raise ValueError("empty or duplicate correction IDs")
    rows = []
    for entry in entries:
        ident = entry["id"]
        if ident not in source or entry["prior_run_id"] != prior_run:
            raise ValueError(f"unknown input or prior run: {ident}")
        evidence = {}
        for key, name in (("input", "reference"), ("prior_canvas", "prior_canvas"),
                          ("prior_program", "prior_full_program"),
                          ("prompt", "original_prompt"),
                          ("new_program", "new_full_program")):
            item = entry[name]
            evidence[key] = (item["path"], item["sha256"])
        rows.append(bind(ident, source[ident], source_batch, prior_run,
                         evidence, entry["rationale"]))
    manifest = {
        "schema": "painter.render-conditioned-corrections.v1",
        "modality": "render-conditioned-correction",
        "model": audit["model"],
        "reasoning_effort": audit["reasoning_effort"],
        "count": len(rows),
        "items": rows,
        "render_status": "awaiting_second_turn_render",
        "sft_admitted_count": 0,
    }
    target = POOL / output_batch / "manifest.json"
    target.parent.mkdir(exist_ok=True)
    raw = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text() != raw:
        raise ValueError(f"existing correction manifest differs: {target}")
    target.write_text(raw)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_directory")
    parser.add_argument("source_batch")
    parser.add_argument("output_batch")
    args = parser.parse_args()
    print(prepare(args.audit_directory, args.source_batch, args.output_batch))
