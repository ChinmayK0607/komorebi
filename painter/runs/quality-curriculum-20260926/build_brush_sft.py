#!/usr/bin/env python3
"""Export hash-bound p5.brush first paints and rendered revisions for LoRA SFT.

The source images/programs remain in their collected evidence bundles. This
export embeds small image views and keeps every assistant target a complete
program; invalid renders, native-only sketches, incompatible coordinate
contracts, and the frozen evaluation references cannot become loss targets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RENDERS = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"
sys.path[:0] = [str(ROOT / "painter"), str(ROOT / "painter/runs/quality-curriculum-20260924")]
from contract import NEXT_VERSION, SYSTEM, identity, paint_target, validate_teacher_program  # noqa: E402
from export_turn_sft import image_part  # noqa: E402

BRUSH_MARK = re.compile(r"\bbrush\.(?:line|polygon|rect|ellipse|stroke|spline|hatch)\s*\(")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, digest: str) -> Path:
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"missing or changed evidence: {path}")
    return path


def accepted_program(path: Path, rejected: Counter) -> str | None:
    program = path.read_text()
    if not BRUSH_MARK.search(program):
        rejected["native_only"] += 1
        return None
    try:
        validate_teacher_program(program)
    except ValueError:
        rejected["coordinate_or_program_contract"] += 1
        return None
    return program


def row(*, ident: str, group: str, source: Path, mode: str, program: str,
        program_sha: str, canvas_sha: str, plan: str, origin: str,
        prior_canvas: Path | None = None, prior_program: Path | None = None,
        feedback: str = "") -> dict:
    content = [{"type": "text", "text": "REFERENCE"}]
    if mode == "text_to_image":
        content.append({"type": "text", "text": source.read_text()})
    else:
        content.append(image_part(source))
    if prior_canvas:
        content.extend([{"type": "text", "text": "CURRENT CANVAS"}, image_part(prior_canvas)])
        if prior_program:
            content.append({"type": "text", "text": "Current complete program:\n" + prior_program.read_text()})
        content.append({"type": "text", "text": "Inspect this actual render. " + feedback.strip()})
    content.append({"type": "text", "text": NEXT_VERSION})
    target = paint_target(plan.strip() or "Paint the scene with clear structure and restrained brush texture.", program)
    return {
        "id": ident, "source_group": group, "split": "train",
        "source_kind": origin, "example_kind": "multi_turn_revision" if prior_canvas else "initial_paint",
        "complete_target": True, "contract": identity(), "reference_sha256": sha(source),
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "teacher_metadata": {"program_sha256": program_sha, "canvas_sha256": canvas_sha,
                             "source_sha256": sha(source), "origin": origin},
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


def first_paints(holdout: set[str], rejected: Counter) -> list[dict]:
    labels = json.loads((RENDERS / "teacher600-visual-audit-v1/labels-all.json").read_text())
    result = []
    for item in labels:
        if item["label"] not in {"A", "B"}:
            rejected["firstpaint_visual_C"] += 1
            continue
        episode = RENDERS / item["chosen_run_id"] / "episodes" / item["id"]
        source = episode / ("input.txt" if item["mode"] == "text_to_image" else "input.jpg")
        verified(source, item["input_sha256"])
        if sha(source) in holdout:
            rejected["frozen_holdout"] += 1
            continue
        verified(episode / "canvas.png", item["canvas_sha256"])
        program_path = episode / "program.js"
        program = accepted_program(program_path, rejected)
        if program is None:
            continue
        result.append(row(ident="first__" + item["audit_id"], group=item["audit_id"],
                          source=source, mode=item["mode"], program=program,
                          program_sha=sha(program_path), canvas_sha=item["canvas_sha256"],
                          plan="Paint the reference with a coherent silhouette, color, and brush texture.",
                          origin="teacher600_firstpaint_" + item["label"]))
    return result


def gateway_revisions(root: Path, holdout: set[str], source_hashes: dict[str, str],
                      rejected: Counter) -> list[dict]:
    result = []
    for episode_path in sorted(root.glob("**/episodes/*/episode.json")):
        episode = json.loads(episode_path.read_text())
        ep = episode_path.parent
        scene = episode["reference_id"]
        source = ep / "source-input.jpg"
        mode = "image_to_image"
        if not source.is_file():
            source = ep / "source-input.txt"
            mode = "text_to_image"
        if not source.is_file():
            # Brush-polish bundles store the source in source-staging.
            source = root / "source-staging/source" / (scene + (".jpg" if (root / "source-staging/source" / (scene + ".jpg")).is_file() else ".txt"))
            mode = "image_to_image" if source.suffix == ".jpg" else "text_to_image"
        if not source.is_file():
            raise ValueError(f"missing scene source: {episode_path}")
        verified(source, source_hashes[scene])
        if sha(source) in holdout:
            rejected["frozen_holdout"] += 1
            continue
        prior_canvas = ep / "first-paint.png"
        prior_program = ep / "first-paint.js"
        if not prior_canvas.is_file():
            prior_canvas = root / "source-staging/current" / (scene + ".png")
            prior_program = root / "source-staging/programs" / (scene + ".js")
        if not prior_canvas.is_file() or not prior_program.is_file():
            raise ValueError(f"missing first-paint state: {episode_path}")
        for turn_file in sorted(ep.glob("turn-*.json")):
            turn = json.loads(turn_file.read_text())
            render = turn.get("render") or {}
            if render.get("valid") is not True or not turn.get("program") or not turn.get("canvas"):
                rejected["invalid_or_finish_turn"] += 1
                continue
            number = turn["turn"]
            program_path = ep / f"turn-{number:02d}.program.js"
            canvas_path = ep / f"turn-{number:02d}.png"
            if not program_path.is_file() or not canvas_path.is_file():
                rejected["missing_render_artifact"] += 1
                continue
            receipt = render.get("receipt") or {}
            verified(program_path, receipt["source_sha256"])
            verified(canvas_path, receipt["png_sha256"])
            program = accepted_program(program_path, rejected)
            if program is not None and sha(program_path) != sha(prior_program):
                result.append(row(ident=f"gateway__{root.name}__{scene}__{number:02d}",
                                  group=scene, source=source, mode=mode,
                                  program=program, program_sha=sha(program_path),
                                  canvas_sha=sha(canvas_path),
                                  plan=str(turn.get("plan") or "Improve the painting's structure and visual finish."),
                                  origin=root.name, prior_canvas=prior_canvas,
                                  prior_program=prior_program,
                                  feedback=str(turn.get("request_render_feedback") or "Improve the painting while preserving its correct forms.")))
            elif program is not None:
                rejected["unchanged_program"] += 1
            prior_canvas, prior_program = canvas_path, program_path
    return result


def manual_revisions(holdout: set[str], source_hashes: dict[str, str],
                     rejected: Counter) -> list[dict]:
    result = []
    for ep in sorted(RENDERS.glob("manual-repair-*-render-20260927/episodes/*")):
        status = json.loads((ep / "render-status.json").read_text())
        if not status.get("valid"):
            rejected["invalid_manual_render"] += 1
            continue
        source = ep / "input.jpg"
        mode = "image_to_image"
        if not source.is_file():
            source = ep / "input.txt"
            mode = "text_to_image"
        verified(source, source_hashes[ep.name])
        if sha(source) in holdout:
            rejected["frozen_holdout"] += 1
            continue
        program_path, canvas_path = ep / "program.js", ep / "canvas.png"
        verified(program_path, status["program_sha256"])
        verified(canvas_path, status["canvas_sha256"])
        program = accepted_program(program_path, rejected)
        if program is None:
            continue
        result.append(row(ident=f"manual__{ep.parent.parent.name}__{ep.name}", group=ep.name,
                          source=source, mode=mode, program=program,
                          program_sha=sha(program_path), canvas_sha=sha(canvas_path),
                          plan="Improve the rendered draft with clearer forms and painterly detail.",
                          origin=ep.parent.parent.name, prior_canvas=ep / "prior-canvas.png",
                          prior_program=ep / "prior-program.js",
                          feedback="Preserve correct parts of the current draft and fix its visible weaknesses."))
    return result


def write(path: Path, rows: list[dict]) -> str:
    raw = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def build(output: Path, seed: int) -> dict:
    eval_manifest = ROOT / "painter/runs/photo-curriculum-sft-20260922/eval-prep/eval-manifest.json"
    holdout = {case["reference_sha256"] for case in json.loads(eval_manifest.read_text())["cases"]}
    rejected: Counter = Counter()
    audit_labels = json.loads((RENDERS / "teacher600-visual-audit-v1/labels-all.json").read_text())
    source_hashes = {item["audit_id"]: item["input_sha256"] for item in audit_labels}
    paintings = first_paints(holdout, rejected)
    hundred = RENDERS / "teacher600-repair-hundred-20260927"
    eight = RENDERS / "teacher600-repair-eight-v2-20260927"
    polish = RENDERS / "brush-polish-four-20260927"
    for root in [hundred / name for name in ("flash-00", "flash-01", "flash-02", "flash-03", "flash-04", "flash-05", "flash-06", "pro-00", "pro-01", "pro-02")]:
        paintings.extend(gateway_revisions(root, holdout, source_hashes, rejected))
    for root in (eight / "flash", eight / "pro", polish):
        paintings.extend(gateway_revisions(root, holdout, source_hashes, rejected))
    paintings.extend(manual_revisions(holdout, source_hashes, rejected))
    if len({item["id"] for item in paintings}) != len(paintings):
        raise ValueError("duplicate SFT row IDs")
    # One exposure per audited painting; retain procedural format/validity with
    # one foundation example per four-row batch, without repeating weak photos.
    foundation = ROOT / "painter/collected/quality-curriculum-20260924/astra-high-100/pilot-input/foundation.jsonl"
    validation = ROOT / "painter/collected/quality-curriculum-20260924/astra-high-100/pilot-input/foundation-validation.jsonl"
    retention = [json.loads(line) for line in foundation.read_text().splitlines() if line.strip()]
    heldout_rows = [json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(paintings)
    rng.shuffle(retention)
    scheduled: list[dict] = []
    for index, item in enumerate(paintings):
        scheduled.append(item)
        if (index + 1) % 3 == 0:
            retained = dict(retention[(index // 3) % len(retention)])
            retained["id"] = f"retention__{index // 3:04d}__{retained['id']}"
            retained["source_kind"] = "procedural_retention"
            scheduled.append(retained)
    while len(scheduled) % 4 or (len(scheduled) // 4) % 40:
        retained = dict(retention[len(scheduled) % len(retention)])
        retained["id"] = f"retention__pad_{len(scheduled)}__{retained['id']}"
        retained["source_kind"] = "procedural_retention"
        scheduled.append(retained)
    output.mkdir(parents=True, exist_ok=True)
    train_sha = write(output / "train.jsonl", scheduled)
    val_sha = write(output / "validation.jsonl", heldout_rows)
    report = {"schema": "painter.brush-sft-data.v1", "seed": seed,
              "hypothesis": "A conservative LoRA pass over diverse rendered p5.brush first paints and state-aware revisions improves held-out beauty and fidelity over the step-512 initializer.",
              "matched_baseline": "Qwen3.8-27B photo SFT step-512 on frozen 28-photo two-turn evaluation",
              "counts": {"painting_targets": len(paintings), "train_rows": len(scheduled),
                         "retention_exposures": len(scheduled) - len(paintings),
                         "checkpoint_padding_exposures": len(scheduled) - (len(paintings) + len(paintings) // 3 + ((-len(paintings) - len(paintings) // 3) % 4)),
                         "validation_rows": len(heldout_rows),
                         "by_origin": dict(Counter(x["source_kind"] for x in paintings)),
                         "by_kind": dict(Counter(x["example_kind"] for x in paintings)),
                         "rejected": dict(rejected)},
              "optimizer_steps": len(scheduled) // 4, "batch_size": 4,
              "checkpoint_interval": 40, "learning_rate": 1e-6,
              "output_sha256": {"train": train_sha, "validation": val_sha},
              "input_sha256": {"eval_manifest": sha(eval_manifest), "foundation": sha(foundation),
                               "foundation_validation": sha(validation)},
              "quality_note": "A/B first paints are provisional Luna labels; repair turns are renderer-valid and contract-compatible candidates, not proven visually superior.",
              "processor_audit_required": True}
    (output / "mix-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.seed), indent=2))
