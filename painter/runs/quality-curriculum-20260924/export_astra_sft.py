#!/usr/bin/env python3
"""Export visually selected Astra canvases as reference/correction SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "painter/collected/quality-curriculum-20260924"
sys.path.insert(0, str(ROOT / "painter"))
sys.path.insert(0, str(HERE))
from contract import NEXT_VERSION, SYSTEM, identity, paint_target  # noqa: E402
from export_turn_sft import image_part  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(admissions: Path) -> tuple[list[dict], list[dict]]:
    selection = json.loads(admissions.read_text())
    if selection.get("schema") != "painter.astra-high-admissions.v1":
        raise ValueError("unexpected admission schema")
    reference_file = HERE / "reference-manifest-astra-100.json"
    if selection.get("reference_manifest_sha256") != sha(reference_file):
        raise ValueError("reference manifest revision mismatch")
    references = {r["id"]: r for r in json.loads(reference_file.read_text())["references"]}
    rows, audit, seen = [], [], set()
    for chosen in selection["selections"]:
        if chosen["decision"] != "sft":
            continue
        wave, shard, ident = chosen["wave"], chosen["shard"], chosen["reference_id"]
        key = (wave, shard, ident)
        if (not isinstance(wave, int) or not 1 <= wave <= 99 or
                not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", shard) or
                key in seen or ident not in references or references[ident]["split"] != "train"):
            raise ValueError(f"duplicate or non-training admission: {key}")
        seen.add(key)
        episode = BASE / f"astra-high-wave{wave}-results" / shard / "episodes" / ident
        summary = episode.parents[1] / "run-summary.json"
        status_file = episode / "render-status.json"
        status = json.loads(status_file.read_text())
        run = json.loads(summary.read_text())
        if (not status["valid"] or status["reference_id"] != ident or
                run.get("wave", 1) != wave or run["shard"] != shard or
                status not in run["statuses"]):
            raise ValueError(f"invalid or mismatched render: {key}")
        ref, canvas, program = (episode / "reference.jpg", episode / "turn-01.png",
                                episode / "turn-01.program.js")
        for path, digest in ((ref, status["reference_sha256"]), (canvas, status["canvas_sha256"]),
                             (program, status["program_sha256"])):
            if sha(path) != digest:
                raise ValueError(f"render evidence hash mismatch: {path}")
        if sha(ref) != references[ident]["sha256"]:
            raise ValueError(f"reference manifest hash mismatch: {ident}")
        content = [{"type": "text", "text": "REFERENCE"}, image_part(ref)]
        prior = episode / "prior.png"
        if status.get("prior_canvas_sha256"):
            if sha(prior) != status["prior_canvas_sha256"]:
                raise ValueError(f"prior canvas hash mismatch: {ident}")
            content.extend([{"type": "text", "text": "CURRENT CANVAS"}, image_part(prior)])
            feedback = chosen.get("prior_feedback", "").strip()
            if not feedback:
                raise ValueError(f"correction admission lacks targeted feedback: {key}")
            content.append({"type": "text", "text":
                            f"Inspect the current canvas. {feedback}\n{NEXT_VERSION}"})
            kind = "multi_turn_revision"
        else:
            content.append({"type": "text", "text": NEXT_VERSION})
            kind = "initial_paint"
        plan = chosen.get("visual_plan", "").strip()
        reason, weaknesses = chosen.get("reason", "").strip(), chosen.get("chosen_weaknesses", [])
        if not plan or not reason or not isinstance(weaknesses, list) or not weaknesses:
            raise ValueError(f"incomplete visual admission: {key}")
        target = paint_target(plan, program.read_text())
        rows.append({
            "id": f"astra__wave{wave}__{shard}__{ident}", "source_group": ident,
            "split": "train", "source_kind": "reviewed_astra_teacher",
            "example_kind": kind, "complete_target": True, "contract": identity(),
            "reference_sha256": sha(ref), "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "teacher_metadata": {
                "model": run["teacher_model"], "reasoning_effort": run["teacher_reasoning_effort"],
                "source_commit": run["source_commit"], "source_dataset_commit": run["source_dataset_commit"],
                "source_bundle_sha256": run["source_bundle_sha256"],
                "render_summary_sha256": sha(summary), "program_sha256": sha(program),
                "canvas_sha256": sha(canvas), "selection_reason": reason,
                "chosen_weaknesses": weaknesses,
            },
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ],
        })
        audit.append({"id": rows[-1]["id"], "example_kind": kind,
                      "program_sha256": sha(program), "canvas_sha256": sha(canvas),
                      "render_status_sha256": sha(status_file)})
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = export(args.admissions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    args.output.write_bytes(raw)
    args.output.with_suffix(".audit.json").write_text(json.dumps({
        "schema": "painter.astra-high-sft-export.v1", "rows": len(rows),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "admissions_sha256": sha(args.admissions), "admissions": audit,
        "training_note": "Run exact Qwen processor-length audit before GPU training",
    }, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output_sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
