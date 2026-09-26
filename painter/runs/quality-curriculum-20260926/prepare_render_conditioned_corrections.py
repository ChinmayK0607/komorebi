#!/usr/bin/env python3
"""Normalize six inspected-canvas corrections without calling them improvements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
PRIOR_RUN = "teacher500-simple-cc0-pilot-20260926"
SOURCE_BATCH = "astra-simple-cc0-select-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: str, expected: str) -> str:
    candidate = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not candidate.is_relative_to(ROOT) or not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"unsafe or missing correction evidence: {path}")
    if sha(candidate) != expected:
        raise ValueError(f"correction evidence hash mismatch: {path}")
    return str(candidate.relative_to(ROOT))


def source_rows() -> dict[str, dict]:
    with tarfile.open(POOL / SOURCE_BATCH / "source.tar.gz", "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
    return {row["id"]: row for row in manifest["rows"]}


def normalized_row(ident: str, source: dict, paths: dict, rationale: str,
                   prior_run: str = PRIOR_RUN, turn_count: int = 2) -> dict:
    prior = POOL / "rendered-cloud" / prior_run / "episodes" / ident
    # The uploaded first-turn source is authoritative for rights and input identity.
    if source["input_sha256"] != paths["reference"][1]:
        raise ValueError(f"correction reference differs from first turn: {ident}")
    if turn_count == 2 and source["program_sha256"] != paths["prior_program"][1]:
        raise ValueError(f"correction prior program differs from first turn: {ident}")
    if source["prompt_sha256"] != paths["prompt"][1]:
        raise ValueError(f"correction prompt differs from first turn: {ident}")
    status = json.loads((prior / "render-status.json").read_text())
    if (status.get("valid") is not True or status.get("canvas_sha256") != paths["prior_canvas"][1]
            or status.get("program_sha256") != paths["prior_program"][1]):
        raise ValueError(f"correction prior render identity mismatch: {ident}")
    output = {"id": ident, "source_batch": SOURCE_BATCH, "prior_run_id": prior_run,
              "turn_count": turn_count,
              "category": source["category"], "rationale": rationale,
              "source_url": source["source_url"], "source_metadata": source["source_metadata"],
              "license": source["license"], "improvement_claim": False,
              "sft_admitted": False}
    for name, (path, expected) in paths.items():
        output[f"{name}_path"] = relative(path, expected)
        output[f"{name}_sha256"] = expected
    return output


def prepare() -> list[Path]:
    sources = source_rows()
    astra: list[dict] = []
    sol: list[dict] = []
    first = json.loads((POOL / "astra-visual-correction-pilot-v1/manifest.json").read_text())
    for row in first["entries"]:
        ident = row["id"]
        astra.append(normalized_row(ident, sources[ident], {
            "reference": (row["reference_path"], row["reference_sha256"]),
            "prior_canvas": (row["prior_canvas_path"], row["prior_canvas_sha256"]),
            "prior_program": (row["prior_program_path"], row["prior_program_sha256"]),
            "prompt": (row["original_prompt_path"], row["original_prompt_sha256"]),
            "new_program": (row["new_program_path"], row["new_program_sha256"]),
        }, row["rationale"]))
    second = json.loads((POOL / "astra-visual-correction-pilot-v2/audit.json").read_text())
    for row in second["entries"]:
        if row.get("correction_decision") != "author_next_full_program":
            continue
        ident = row["original_source_id"]
        astra.append(normalized_row(ident, sources[ident], {
            "reference": (row["reference"]["path"], row["reference"]["sha256"]),
            "prior_canvas": (row["prior_canvas"]["path"], row["prior_canvas"]["sha256"]),
            "prior_program": (row["prior_full_program"]["path"], row["prior_full_program"]["sha256"]),
            "prompt": (row["original_prompt"]["path"], row["original_prompt"]["sha256"]),
            "new_program": (row["new_full_program"]["path"], row["new_full_program"]["sha256"]),
        }, row["rationale"]))
    third = json.loads((POOL / "sol-visual-correction-pilot-v1/audit.json").read_text())
    for row in third["corrections"]:
        ident = row["source_id"]
        sol.append(normalized_row(ident, sources[ident], {
            "reference": (row["reference_path"], row["reference_sha256"]),
            "prior_canvas": (row["prior_canvas_path"], row["prior_canvas_sha256"]),
            "prior_program": (row["prior_full_program_path"], row["prior_full_program_sha256"]),
            "prompt": (row["original_prompt_path"], row["original_prompt_sha256"]),
            "new_program": (row["new_full_program_path"], row["new_full_program_sha256"]),
        }, row["rationale"]))
    if len(astra) != 4 or len(sol) != 2 or len({r["id"] for r in astra + sol}) != 6:
        raise ValueError("expected four Astra and two Sol distinct corrections")
    paths = []
    for name, model, rows in (
        ("astra-render-conditioned-correction-v1", "gpt-6-astra", astra),
        ("sol-render-conditioned-correction-v1", "gpt-5.6-sol", sol),
    ):
        directory = POOL / name
        directory.mkdir(exist_ok=True)
        path = directory / "manifest.json"
        content = {"schema": "painter.render-conditioned-corrections.v1",
                   "modality": "image-to-painting-render-conditioned-correction",
                   "model": model, "reasoning_effort": "high", "source_batch": SOURCE_BATCH,
                   "prior_run_id": PRIOR_RUN, "count": len(rows), "items": rows,
                   "render_status": "awaiting_second_turn_render", "sft_admitted_count": 0}
        raw = json.dumps(content, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != raw:
            raise ValueError(f"existing correction manifest differs: {path}")
        path.write_text(raw)
        paths.append(directory)
    return paths


if __name__ == "__main__":
    print(json.dumps([str(path) for path in prepare()]))
