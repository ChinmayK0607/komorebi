#!/usr/bin/env python3
"""Make auditable static alternatives for observed p5 runtime failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import argparse

from package_teacher_batch import package


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "painter/runs/quality-curriculum-20260926"
RENDERS = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def corrected(program: str, fix: str) -> str:
    if fix == "curveVertex_to_vertex":
        count = len(re.findall(r"(?<![\w.])curveVertex\s*\(", program))
        if not count:
            raise ValueError("selected curveVertex failure has no global call")
        result = re.sub(r"(?<![\w.])curveVertex\s*\(", "vertex(", program)
    elif fix == "rename_box_helper":
        if len(re.findall(r"\bfunction\s+box\s*\(", program)) != 1:
            raise ValueError("selected box failure does not define exactly one helper")
        result = re.sub(r"(?<![\w.])box\s*\(", "paintBox(", program)
    else:
        raise ValueError(f"unknown repair: {fix}")
    if result == program:
        raise ValueError("repair did not change program")
    return result


def prepare(selection_name: str = "RUNTIME_REPAIR_V1.json") -> dict:
    selection_path = RUN / selection_name
    selection = json.loads(selection_path.read_text())
    if selection.get("schema") != "painter.teacher600-runtime-repair-selection.v1" or selection["count"] != len(selection["rows"]):
        raise ValueError("unexpected static-repair selection")
    output_name = selection.get("output_batch", "sol-runtime-repair-v1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", output_name):
        raise ValueError("unsafe output batch")
    output = ROOT / "painter/collected/quality-curriculum-20260926" / output_name
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for row in selection["rows"]:
        episode = RENDERS / row["run_id"] / "episodes" / row["id"]
        status = json.loads((episode / "render-status.json").read_text())
        receipt = json.loads((episode / "canvas.json").read_text())
        observed = receipt.get("errors") or []
        if (status["valid"] or status["id"] != row["id"]
                or status["input_sha256"] != row["input_sha256"]
                or status["program_sha256"] != row["program_sha256"]
                or row["observed_error"] not in observed):
            raise ValueError(f"failure evidence mismatch: {row['id']}")
        prompt = (episode / "input.txt").read_bytes()
        program = (episode / "program.js").read_bytes()
        if sha(prompt) != row["input_sha256"] or sha(program) != row["program_sha256"]:
            raise ValueError(f"source bytes differ: {row['id']}")
        repaired = corrected(program.decode("utf-8"), row["fix"]).encode()
        folder = output / row["id"]
        folder.mkdir(exist_ok=True)
        (folder / "prompt.txt").write_bytes(prompt)
        (folder / "program.js").write_bytes(repaired)
        items.append({"id": row["id"], "category": "runtime-repair",
                      "role": "unrendered_alternative_candidate",
                      "difficulty": "mixed", "prompt_file": f"{row['id']}/prompt.txt",
                      "prompt_sha256": sha(prompt),
                      "program_file": f"{row['id']}/program.js",
                      "program_sha256": sha(repaired),
                      "source_run_id": row["run_id"],
                      "original_program_sha256": sha(program),
                      "observed_error": row["observed_error"],
                      "repair_note": f"Static runtime repair {row['fix']} after exact observed canvas error; visual quality unreviewed."})
    manifest = {"schema": "painter.teacher600-runtime-repair.v1",
                "batch": output.name, "created": "2026-09-27",
                "model": "gpt-5.6-sol", "reasoning_effort": "high",
                "count": len(items), "distinct_new_prompts": 0,
                "source_selection_sha256": sha(selection_path.read_bytes()),
                "status": "unrendered_alternatives_not_sft_admitted",
                "hypothesis": f"Replacing unsupported p5 calls can recover canvases on {len(items)} exact failed text prompts.",
                "matched_baseline": "Original failed render on the same prompt; this is runtime recovery, not aesthetic improvement.",
                "items": items}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return package(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="RUNTIME_REPAIR_V1.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection), sort_keys=True))
