#!/usr/bin/env python3
"""Normalize reviewed text/photo corrections against actual prior canvases."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile

from prepare_render_conditioned_corrections import POOL, relative


def original_rows(batch: str) -> dict[str, dict]:
    with tarfile.open(POOL / batch / "source.tar.gz", "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
    return {row["id"]: row for row in manifest["rows"]}


def file_pair(value: dict) -> tuple[str, str]:
    return value["path"], value["sha256"]


def bind(ident: str, source: dict, source_batch: str, prior_run: str,
         evidence: dict[str, tuple[str, str]], rationale: str) -> dict:
    mode = source["mode"]
    if mode not in {"image_to_image", "text_to_image"}:
        raise ValueError(f"unknown correction mode: {ident}")
    if source["input_sha256"] != evidence["input"][1]:
        raise ValueError(f"correction input differs from first turn: {ident}")
    if source["program_sha256"] != evidence["prior_program"][1]:
        raise ValueError(f"correction program differs from first turn: {ident}")
    if mode == "image_to_image" and source["prompt_sha256"] != evidence["prompt"][1]:
        raise ValueError(f"correction photo prompt differs from first turn: {ident}")
    episode = POOL / "rendered-cloud" / prior_run / "episodes" / ident
    status = json.loads((episode / "render-status.json").read_text())
    if (status.get("valid") is not True or status.get("input_sha256") != evidence["input"][1]
            or status.get("program_sha256") != evidence["prior_program"][1]
            or status.get("canvas_sha256") != evidence["prior_canvas"][1]):
        raise ValueError(f"correction prior render mismatch: {ident}")
    output = {"id": ident, "mode": mode, "category": source["category"],
              "source_batch": source_batch, "prior_run_id": prior_run,
              "turn_count": 2, "rationale": rationale,
              "source_url": source.get("source_url"),
              "source_metadata": source.get("source_metadata"),
              "source_visual_type": source.get("source_visual_type"),
              "license": source.get("license"),
              "improvement_claim": False, "sft_admitted": False}
    for key, (path, expected) in evidence.items():
        output[f"{key}_path"] = relative(path, expected)
        output[f"{key}_sha256"] = expected
    return output


def prepare() -> list[str]:
    photo_batch = "astra-curated-coco-v1"
    text_batch = "sol-text-curated-v1"
    photo = original_rows(photo_batch)
    text = original_rows(text_batch)
    groups: list[tuple[str, str, list[dict]]] = []
    for audit_dir, output_batch in (
        ("astra-coco-objects-turn2-v1", "astra-coco-objects-render-conditioned-v1"),
        ("astra-coco-scenes-turn2-v1", "astra-coco-scenes-render-conditioned-v1"),
    ):
        audit = json.loads((POOL / audit_dir / "audit.json").read_text())
        reviewed = audit.get("rows", audit.get("cases", []))
        selected = [row for row in reviewed if row.get("new_program") or row.get("new_full_program")]
        if len(reviewed) != 3 or len(selected) != 2:
            raise ValueError(f"unexpected photo review/selection count: {audit_dir}")
        rows = []
        for row in selected:
            ident = row["id"]
            prior_run = row["prior_run_id"]
            if prior_run != "teacher500-astra-coco-curated-20260926":
                raise ValueError("photo correction has wrong prior run")
            evidence = {
                "input": file_pair(row["reference"]),
                "prior_canvas": file_pair(row.get("prior_canvas", row.get("observed_prior_canvas"))),
                "prior_program": file_pair(row.get("prior_program", row.get("prior_full_program"))),
                "prompt": file_pair(row.get("prompt", row.get("original_prompt"))),
                "new_program": file_pair(row.get("new_program", row.get("new_full_program"))),
            }
            rows.append(bind(ident, photo[ident], photo_batch, prior_run, evidence,
                             row["rationale"]))
        groups.append((output_batch, "gpt-6-astra", rows))
    audit = json.loads((POOL / "sol-text-turn2-v1/audit.json").read_text())
    selected = audit["selected_revisions"]
    if audit["audited_count"] != 4 or len(selected) != 2:
        raise ValueError("unexpected text review/selection count")
    text_rows = []
    for row in selected:
        ident, prior_run = row["id"], row["prior_run_id"]
        if prior_run != "teacher500-sol-text-curated-20260926":
            raise ValueError("text correction has wrong prior run")
        evidence = {
            "input": (row["input_prompt_path"], row["input_prompt_sha256"]),
            "prior_canvas": (row["prior_canvas_path"], row["prior_canvas_sha256"]),
            "prior_program": (row["prior_full_program_path"], row["prior_full_program_sha256"]),
            "new_program": (row["new_full_program_path"], row["new_full_program_sha256"]),
        }
        text_rows.append(bind(ident, text[ident], text_batch, prior_run, evidence,
                              row["rationale"]))
    groups.append(("sol-text-render-conditioned-v1", "gpt-5.6-sol", text_rows))
    paths = []
    for batch, model, rows in groups:
        directory = POOL / batch
        directory.mkdir(exist_ok=True)
        path = directory / "manifest.json"
        content = {"schema": "painter.render-conditioned-corrections.v1",
                   "modality": "render-conditioned-correction", "model": model,
                   "reasoning_effort": "high", "count": len(rows), "items": rows,
                   "render_status": "awaiting_second_turn_render", "sft_admitted_count": 0}
        raw = json.dumps(content, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != raw:
            raise ValueError(f"existing correction manifest differs: {path}")
        path.write_text(raw)
        paths.append(str(directory))
    return paths


if __name__ == "__main__":
    print(json.dumps(prepare()))
