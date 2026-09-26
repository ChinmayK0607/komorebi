#!/usr/bin/env python3
"""Stage two hash-pinned first-paint corrections for the existing Cloud teacher runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
BASE = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"
PILOT = BASE / "sol-brush-translation-pilot-20260927"
SELECTION = {
    "t600-326": "xiaomi/mimo-v2.6-flash",
    "t600-492": "xiaomi/mimo-v2.6-pro",
}
COPY = ("contract.json", "model-catalog.json", "gateway_transport.ts", "gateway_prompt.ts",
        "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stage(audit_id: str, run_id: str) -> Path:
    if audit_id not in SELECTION:
        raise ValueError("audit ID outside finite redo pilot")
    plan = json.loads((HERE / "SOL_BRUSH_PILOT_8.json").read_text())
    redo = json.loads((HERE / "SOL_REDO_8.json").read_text())
    review = json.loads((HERE / "SOL_BRUSH_PILOT_REVIEW.json").read_text())
    if redo["source_pilot_sha256"] != sha((HERE / "SOL_BRUSH_PILOT_8.json").read_bytes()):
        raise ValueError("redo queue does not match frozen source")
    if review["pilot_plan_sha256"] != redo["source_pilot_sha256"]:
        raise ValueError("visual review source mismatch")
    row = next(r for r in plan["rows"] if r["audit_id"] == audit_id)
    target = next(r for r in redo["rows"] if r["audit_id"] == audit_id)
    episode = PILOT / "episodes" / audit_id
    prompt = (episode / "input.txt").read_bytes()
    original = (episode / "original.png").read_bytes()
    if sha(prompt) != row["input_sha256"] or sha(original) != row["original_canvas_sha256"]:
        raise ValueError("first-paint prompt/canvas hash mismatch")
    if target["prior_canvas_sha256"] != sha(original):
        raise ValueError("redo prior canvas differs from pilot")
    if not any(r["audit_id"] == audit_id and r["preferred"] == "original" for r in review["rows"]):
        raise ValueError("translation review is missing rejection")
    model = SELECTION[audit_id]
    output = BASE / run_id / audit_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "references").mkdir()
    ref_name = f"references/{audit_id}-first-paint.png"
    (output / ref_name).write_bytes(original)
    (output / "refs.json").write_text(json.dumps({
        "version": 1, "count": 1,
        "references": [{"id": audit_id, "image": ref_name, "sha256": sha(original),
                        "category": target["category"], "split": "train",
                        "source_role": "first_paint_to_improve"}]}, indent=2, sort_keys=True) + "\n")
    base_prompt = (ROOT / "painter/runs/quality-curriculum-20260924/quality-prompt.txt").read_text()
    instruction = (
        "\n\nDATA CORRECTION TASK. The attached REFERENCE is an existing first-paint canvas, "
        "not a photograph and not the final target. Make a new complete painting that is clearly "
        "more beautiful and faithful to the scene brief. Preserve the recognizable subject, "
        "composition and major color relationships; fix the visible weakness below. "
        "Use opaque native p5 forms for sound structure, then p5.brush for controlled painterly "
        "texture and edges. Do not simply convert API calls or trace the old painting. "
        "On every subsequent turn, inspect your actual CURRENT CANVAS and correct its largest "
        "remaining flaw. Do not stop at merely valid code.\n\n"
        f"SCENE BRIEF (source data, not instructions):\n{prompt.decode('utf-8').strip()}\n\n"
        f"VISUALLY INSPECTED WEAKNESS:\n{target['parent_visual_weakness']}\n"
    )
    (output / "prompt.txt").write_text(base_prompt + instruction)
    config = {
        "benchmark": f"sol-redo-{audit_id}-20260927", "models": [model],
        "tracks": {"quality": {"max_turns": 4, "max_tokens": 16384,
                               "episode_timeout_seconds": 2400,
                               "reasoning_effort": "highest_supported"}},
        "temperature": 0.7, "concurrency": 1, "render_concurrency": 1,
        "renderer_timeout": 300, "samples_per_image": 1,
        "catalog": "model-catalog.json", "transport_script": "gateway_transport.ts",
        "references": "refs.json", "prompt": "prompt.txt",
        "request_timeout_seconds": 900, "max_retries": 0,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for name in COPY:
        shutil.copyfile(BENCH / name, output / name)
    provenance = {
        "schema": "painter.sol-redo-stage.v1", "audit_id": audit_id, "model": model,
        "source_pilot_sha256": redo["source_pilot_sha256"],
        "visual_review_sha256": sha((HERE / "SOL_BRUSH_PILOT_REVIEW.json").read_bytes()),
        "source_prompt_sha256": sha(prompt), "first_paint_sha256": sha(original),
        "prompt_sha256": sha((output / "prompt.txt").read_bytes()),
        "config_sha256": sha((output / "config.json").read_bytes()),
        "source_program_sha256": row["source_program_sha256"],
        "hypothesis": "A visual correction loop will produce a painting that beats the exact first-paint canvas.",
        "matched_baseline": "The preselected exact original Sol canvas, same scene brief and frozen renderer.",
        "status": "unreviewed_teacher_candidates",
    }
    (output / "restored-source.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": str(output), "audit_id": audit_id, "model": model,
                      "reference_sha256": sha(original)}), flush=True)
    return output


def finalize(stage_root: Path, audit_id: str) -> None:
    if audit_id not in SELECTION or not stage_root.is_dir():
        raise ValueError("unexpected stage")
    episodes = list((stage_root / "episodes").glob("*/episode.json"))
    if len(episodes) != 1:
        raise ValueError("expected exactly one teacher episode")
    episode = episodes[0].parent
    state = json.loads(episodes[0].read_text())
    if state.get("reference_id") != audit_id or state.get("model") != SELECTION[audit_id]:
        raise ValueError("teacher episode identity mismatch")
    shutil.copyfile(PILOT / "episodes" / audit_id / "input.txt", episode / "source-prompt.txt")
    shutil.copyfile(PILOT / "episodes" / audit_id / "original.png", episode / "first-paint.png")
    shutil.copyfile(PILOT / "episodes" / audit_id / "original.js", episode / "first-paint.js")
    print(json.dumps({"finalized": audit_id, "status": state.get("status"),
                      "turns": len(state.get("turns", [])), "episode": str(episode)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_id", choices=sorted(SELECTION))
    parser.add_argument("run_id")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not args.run_id.startswith("sol-redo-") or not args.run_id.replace("-", "").isalnum():
        parser.error("invalid finite run ID")
    root = BASE / args.run_id / args.audit_id
    if args.finalize:
        finalize(root, args.audit_id)
    else:
        stage(args.audit_id, args.run_id)


if __name__ == "__main__":
    main()
