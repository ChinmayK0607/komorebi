#!/usr/bin/env python3
"""Export reviewed teacher revisions as visual-state, assistant-only SFT rows.

The pinned Prime trainer supervises *all* assistant messages. Therefore each
approved turn is a separate system/user/assistant example: the previous
rendered canvas and feedback are user context, and only the next complete
program is an assistant target. The released wave-4 export omitted the full
prior program to fit 16k. Future exports can include the current program or
the last failed attempt with explicit flags, but must pass the exact processor
length audit before training.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contract import NEXT_VERSION, SYSTEM, identity, paint_target, validate_teacher_program


# Two-image correction rows otherwise exceed the pinned 16k processor limit.
# Resize only the embedded training view; keep source images and their hashes intact.
MAX_IMAGE_EDGE = 448


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_part(path: Path) -> dict:
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if not mime:
        raise ValueError(f"unsupported image extension: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG", optimize=False)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()}}


def under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"missing or escaping evidence file: {relative}")
    return candidate


def load_episode(evidence: Path, reference_id: str) -> tuple[dict, Path]:
    matches = []
    for path in (evidence / "episodes").glob("*/episode.json"):
        episode = json.loads(path.read_text())
        if episode.get("reference_id") == reference_id:
            matches.append((episode, path))
    if len(matches) != 1:
        raise ValueError(f"expected one episode for {reference_id} in {evidence}; found {len(matches)}")
    return matches[0]


def make_rows(annotations: Path, references: Path, *, include_prior_program: bool = False,
              include_last_attempt_program: bool = False) -> tuple[list[dict], list[dict]]:
    cfg = json.loads(annotations.read_text())
    manifest = json.loads(references.read_text())
    ref_by_id = {row["id"]: row for row in manifest["references"]}
    rows, audit = [], []
    for selection in cfg["episodes"]:
        evidence = Path(selection["evidence_root"]).resolve()
        reference_id = selection["reference_id"]
        episode, episode_path = load_episode(evidence, reference_id)
        ref_meta = ref_by_id[reference_id]
        if ref_meta["split"] != "train" or episode["reference_sha256"] != ref_meta["sha256"]:
            raise ValueError(f"reference split/hash mismatch: {reference_id}")
        reference = Path(ref_meta["reference_path"]).resolve()
        if sha(reference.read_bytes()) != ref_meta["sha256"]:
            raise ValueError(f"reference file hash mismatch: {reference_id}")
        approved = set(selection["supervise_turns"])
        if not approved or any(not isinstance(t, int) or t < 1 for t in approved):
            raise ValueError(f"invalid approved turns: {reference_id}")
        previous_valid = None
        previous_attempt_program = None
        previous_history = []
        found = set()
        for turn in sorted(episode["turns"], key=lambda item: item["turn"]):
            number = turn["turn"]
            receipt = json.loads(under(evidence, f"episodes/{episode_path.parent.name}/turn-{number:02d}.json").read_text())
            if receipt.get("turn") != number or receipt.get("request_binding", {}).get("reference_sha256") != ref_meta["sha256"]:
                raise ValueError(f"turn receipt mismatch: {reference_id} T{number}")
            render = receipt.get("render") or {}
            attempt_program = under(evidence, receipt["program"]) if receipt.get("program") else None
            valid = render.get("valid") is True and bool(receipt.get("program")) and bool(receipt.get("canvas"))
            program = canvas = None
            if valid:
                program = under(evidence, receipt["program"])
                canvas = under(evidence, receipt["canvas"])
                render_receipt = render.get("receipt") or {}
                if render_receipt.get("source_sha256") != sha(program.read_bytes()) or render_receipt.get("png_sha256") != sha(canvas.read_bytes()):
                    raise ValueError(f"render hash mismatch: {reference_id} T{number}")
            if number in approved:
                found.add(number)
                if not valid:
                    raise ValueError(f"approved turn has no valid canvas: {reference_id} T{number}")
                if previous_valid is not None:
                    previous_program, previous_canvas = previous_valid
                    if (sha(previous_program.read_bytes()) == sha(program.read_bytes())
                            or sha(previous_canvas.read_bytes()) == sha(canvas.read_bytes())):
                        raise ValueError(f"approved revision made no change: {reference_id} T{number}")
                target_program = program.read_text()
                try:
                    validate_teacher_program(target_program)
                except ValueError as exc:
                    raise ValueError(f"approved program contract mismatch: {reference_id} T{number}: {exc}") from exc
                plan = str(receipt.get("plan") or "Inspect the image and improve the complete painting.").strip()
                target = paint_target(plan, target_program)
                content = [{"type": "text", "text": "REFERENCE"}, image_part(reference)]
                if previous_valid is not None:
                    old_program, old_canvas = previous_valid
                    content.extend([
                        {"type": "text", "text": "CURRENT CANVAS"}, image_part(old_canvas),
                        {"type": "text", "text": "Inspect the current rendered canvas and output a complete replacement program."},
                    ])
                    if include_prior_program:
                        content.append({"type": "text", "text": "Current program:\n" + old_program.read_text()})
                if (include_last_attempt_program and previous_attempt_program is not None and
                        not (include_prior_program and previous_valid is not None and
                             previous_attempt_program == previous_valid[0])):
                    content.append({"type": "text", "text":
                                    "Most recent attempted program (context; replace it completely):\n" +
                                    previous_attempt_program.read_text()})
                if previous_history:
                    content.append({"type": "text", "text": "Prior turn outcomes (context, not examples to imitate):\n" + "\n".join(previous_history[-6:])})
                feedback = str(receipt.get("request_render_feedback") or "").strip()
                content.append({"type": "text", "text": (feedback + "\n" if feedback else "") + NEXT_VERSION})
                ident = f"mimo__{reference_id}__turn_{number:02d}"
                rows.append({
                    "id": ident, "source_group": reference_id, "split": "train", "family": ref_meta.get("description", reference_id),
                    "source_kind": "reviewed_teacher_revision", "example_kind": "initial_paint" if previous_valid is None else "multi_turn_revision",
                    "complete_target": True, "contract": identity(), "reference_sha256": ref_meta["sha256"],
                    "target_sha256": sha(target.encode()), "teacher_metadata": {
                        "model": episode["model"], "job_id": episode["job_id"], "turn": number,
                        "episode_receipt_sha256": sha(episode_path.read_bytes()),
                        "program_sha256": sha(program.read_bytes()), "canvas_sha256": sha(canvas.read_bytes()),
                        "selection_reason": selection["reason"],
                    },
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": [{"type": "text", "text": target}]},
                    ],
                })
            outcome = "valid render" if valid else "invalid: " + str(receipt.get("render_feedback") or receipt.get("render_error") or "no canvas")[:220]
            previous_history.append(f"Turn {number}: {outcome}")
            if valid:
                previous_valid = (program, canvas)
            previous_attempt_program = attempt_program
        if found != approved:
            raise ValueError(f"approved turn absent: {reference_id}: {sorted(approved - found)}")
        audit.append({"reference_id": reference_id, "selected_turns": sorted(approved), "all_turns": len(episode["turns"]),
                      "source_episode_sha256": sha(episode_path.read_bytes())})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate training ID")
    return rows, audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-prior-program", action="store_true",
                   help="match the evaluator's current-program prompt; requires exact length audit")
    p.add_argument("--include-last-attempt-program", action="store_true",
                   help="show the immediately preceding code, including a failed attempt; requires exact length audit")
    args = p.parse_args()
    rows, audit = make_rows(args.annotations, args.references,
                            include_prior_program=args.include_prior_program,
                            include_last_attempt_program=args.include_last_attempt_program)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode()
    args.output.write_bytes(raw)
    report = {"schema": "painter.reviewed-turn-sft.v1", "rows": len(rows), "output_sha256": sha(raw),
              "annotations_sha256": sha(args.annotations.read_bytes()), "references_sha256": sha(args.references.read_bytes()),
              "embedded_image_max_edge": MAX_IMAGE_EDGE,
              "include_prior_program": args.include_prior_program,
              "include_last_attempt_program": args.include_last_attempt_program,
              "admissions": audit, "loss_target": "last assistant only; previous turns are user context"}
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "sha256": sha(raw), "audit": str(args.output.with_suffix('.audit.json'))}))


if __name__ == "__main__":
    main()
