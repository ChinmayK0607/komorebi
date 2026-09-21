"""Prepare the aesthetic RL state and same-reference anchor pool.

The source of truth is the reviewed portable state manifest from the prior
step-24 run plus the contract-quality training-state evidence.  This module
does not render, call a model, or assign quality tiers.  It only copies
already observed student paintings and writes a small, hash-addressed
manifest for the new taskset.

Generated PNG/JS assets are intentionally kept under the run directory as
packaging inputs.  They are ignored by the repository's tracked-surface
policy; ``package_run.py --overlay`` includes them in the overlay archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RUN_ROOT = Path(__file__).resolve().parent
SOURCE_RUN = RUN_ROOT.parent / "revision-rl-20260918"
SOURCE_INPUTS = SOURCE_RUN / "inputs"
EVIDENCE_RUN = RUN_ROOT.parent / "contract-quality-20260918" / "collected-final" / "evidence"
ANCHOR_DIR_NAME = "aesthetic-anchor-pool"
MANIFEST_NAME = "aesthetic-states.json"
RUNTIME_STATES_NAME = "rl-states.json"
RECEIPT_NAME = "aesthetic-data-receipt.json"

# These are the two parent-inspected weak cases.  The archived render is a
# same-source-group, non-reference teacher program render with an immutable
# renderer receipt.  It supplements the actual student anchor; it does not
# replace or relabel that student artifact as a quality tier.
CURATED_ARCHIVED_ANCHORS = {
    "v2-bus-000": {"turn": 2, "reason": "parent-inspected visible bus body and windows"},
    "v2-flower-000": {"turn": 2, "reason": "parent-inspected visible flower structure"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_evidence_path(path: Path) -> str:
    """Return a durable source identifier without local absolute paths."""

    try:
        relative = path.relative_to(EVIDENCE_RUN)
    except ValueError:
        relative = path
    return f"contract-quality-20260918/collected-final/evidence/{relative.as_posix()}"


def _evidence_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("training-state canvas path must be a non-empty string")
    # Evidence summaries were written on the node with /root/painter paths.
    # Resolve only the known prefix; never accept arbitrary absolute paths.
    prefix = "/root/painter/"
    if raw.startswith(prefix):
        return EVIDENCE_RUN / raw[len(prefix) :]
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"unexpected absolute evidence path: {raw}")
    return EVIDENCE_RUN / candidate


def _paint_turns(summary: dict[str, Any], evidence_dir: Path) -> list[dict[str, Any]]:
    """Return valid paint turns whose actual canvas and program are present."""

    turns = summary.get("turns")
    if not isinstance(turns, list):
        raise ValueError("training-state summary has no turns list")
    selected: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict) or turn.get("valid") is not True or turn.get("action") != "paint":
            continue
        canvas = _evidence_path(turn.get("canvas"))
        program = canvas.with_name("program.js")
        if not canvas.is_file() or not program.is_file():
            continue
        item = dict(turn)
        item["canvas_path"] = canvas
        item["program_path"] = program
        selected.append(item)
    return selected


def _distinct_turns(turns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate no-change turns by actual canvas bytes, preserving order."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for turn in turns:
        digest = sha256(turn["canvas_path"])
        if digest in seen:
            continue
        seen.add(digest)
        item = dict(turn)
        item["canvas_sha256"] = digest
        item["program_sha256"] = sha256(turn["program_path"])
        result.append(item)
    return result


def select_anchor_turns(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Select first/last distinct observed paintings and a selection note.

    The function is intentionally independent of visual quality.  ``early``
    and ``final`` are temporal roles only; they must not be read as quality
    labels.  The returned records are suitable for manifest-only unit tests.
    """

    turns = _distinct_turns(_paint_turns(summary, EVIDENCE_RUN))
    if not turns:
        return [], "no valid rendered student painting with a paired program"
    if len(turns) == 1:
        return turns, "only one unique actual student painting was available"
    return [turns[0], turns[-1]], "first and last distinct actual student paintings"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc


def _validate_source_states(states: list[dict[str, Any]], heldout: set[str]) -> None:
    if len(states) != 24:
        raise ValueError(f"expected 24 training states, found {len(states)}")
    if any(row.get("split") != "train" for row in states):
        raise ValueError("aesthetic data may only use train states")
    groups = {row.get("source_group") for row in states}
    if None in groups or len(groups) != 24:
        raise ValueError("training states must have unique source_group values")
    if groups & heldout:
        raise ValueError(f"training/heldout source-group overlap: {sorted(groups & heldout)}")
    families = Counter(row.get("family") for row in states)
    if len(families) != 12 or set(families.values()) != {2}:
        raise ValueError(f"expected two train states for each of 12 families, got {dict(families)}")
    required = {"id", "source_group", "family", "reference", "canvas", "program"}
    for row in states:
        if not required <= set(row):
            raise ValueError(f"state missing required fields: {row.get('id')}")


def _copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build(root: Path = RUN_ROOT, *, source_inputs: Path = SOURCE_INPUTS,
          evidence_run: Path = EVIDENCE_RUN) -> dict[str, Any]:
    """Build ``aesthetic-states.json`` and copied anchor assets."""

    global EVIDENCE_RUN
    old_evidence = EVIDENCE_RUN
    EVIDENCE_RUN = evidence_run.resolve()
    try:
        source_inputs = source_inputs.resolve()
        root = root.resolve()
        state_rows = _load_json(source_inputs / "rl-states.json")
        heldout_rows = _load_json(source_inputs / "evaluation-tasks.json")
        if not isinstance(state_rows, list) or not all(isinstance(row, dict) for row in state_rows):
            raise ValueError("rl-states.json must be a list of objects")
        if not isinstance(heldout_rows, list):
            raise ValueError("evaluation-tasks.json must be a list")
        heldout = {row.get("source_group") for row in heldout_rows if isinstance(row, dict)}
        _validate_source_states(state_rows, heldout)

        pool = root / ANCHOR_DIR_NAME
        pool.mkdir(parents=True, exist_ok=True)
        output_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for row in sorted(state_rows, key=lambda value: value["id"]):
            state_id = row["id"]
            task = row.get("provenance", {}).get("task")
            if not isinstance(task, str) or not task:
                # The portable manifest is expected to carry the provenance;
                # do not guess a task from a source-group string.
                missing.append(f"{state_id}:missing provenance.task")
                continue
            summary_path = EVIDENCE_RUN / "training-states" / "parent" / task / "sample-00" / "summary.json"
            if not summary_path.is_file():
                missing.append(f"{state_id}:missing {summary_path}")
                continue
            summary = _load_json(summary_path)
            turns = _distinct_turns(_paint_turns(summary, EVIDENCE_RUN))
            if not turns:
                missing.append(f"{state_id}:no valid student render")
                continue
            selected = turns if len(turns) == 1 else [turns[0], turns[-1]]
            selection_note = (
                "only one unique actual student painting was available"
                if len(turns) == 1 else
                "first and last distinct actual student paintings"
            )

            anchor_rows: list[dict[str, Any]] = []
            for index, turn in enumerate(selected):
                role = "observed" if len(selected) == 1 else ("early" if index == 0 else "final")
                target_base = pool / state_id / role
                canvas_target = target_base / "painting.png"
                program_target = target_base / "program.js"
                _copy(turn["canvas_path"], canvas_target)
                _copy(turn["program_path"], program_target)
                anchor_rows.append({
                    "id": f"{state_id}-anchor-{role}",
                    "role": role,
                    "turn": turn.get("turn"),
                    "canvas": str(canvas_target.relative_to(root)),
                    "program": str(program_target.relative_to(root)),
                    "canvas_sha256": sha256(canvas_target),
                    "program_sha256": sha256(program_target),
                    "source_artifact": _relative_evidence_path(turn["canvas_path"]),
                    "source_program_artifact": _relative_evidence_path(turn["program_path"]),
                })

            # For the two parent-identified near-blank cases, add one archived
            # teacher render from the exact same source_group.  This is
            # deliberately opt-in and receipt-verified; it does not infer a
            # quality score for any other state or turn.
            archived = CURATED_ARCHIVED_ANCHORS.get(row["source_group"])
            if archived is not None:
                turn_number = archived["turn"]
                teacher_canvas = source_inputs / "data" / row["source_group"] / f"turn-{turn_number}.png"
                teacher_program = source_inputs / "data" / row["source_group"] / f"turn-{turn_number}.js"
                teacher_receipt = EVIDENCE_RUN / "data" / row["source_group"] / f"turn-{turn_number}.json"
                if not all(path.is_file() for path in (teacher_canvas, teacher_program, teacher_receipt)):
                    raise ValueError(
                        f"curated archived anchor is incomplete for {row['source_group']}: "
                        f"{teacher_canvas}, {teacher_program}, {teacher_receipt}"
                    )
                receipt = _load_json(teacher_receipt)
                if receipt.get("valid") is not True:
                    raise ValueError(f"curated archived anchor receipt is not valid: {teacher_receipt}")
                if receipt.get("source_sha256") != sha256(teacher_program):
                    raise ValueError(f"teacher source hash mismatch: {teacher_program}")
                if receipt.get("png_sha256") != sha256(teacher_canvas):
                    raise ValueError(f"teacher PNG hash mismatch: {teacher_canvas}")
                reference_source = source_inputs / row["reference"]
                if reference_source.is_file() and sha256(reference_source) == sha256(teacher_canvas):
                    raise ValueError(f"curated archived anchor aliases reference: {teacher_canvas}")
                role = "archived-teacher"
                target_base = pool / state_id / role
                canvas_target = target_base / "painting.png"
                program_target = target_base / "program.js"
                _copy(teacher_canvas, canvas_target)
                _copy(teacher_program, program_target)
                anchor_rows.append({
                    "id": f"{state_id}-anchor-{role}",
                    "role": role,
                    "turn": turn_number,
                    "canvas": str(canvas_target.relative_to(root)),
                    "program": str(program_target.relative_to(root)),
                    "canvas_sha256": sha256(canvas_target),
                    "program_sha256": sha256(program_target),
                    "source_artifact": _relative_evidence_path(teacher_canvas),
                    "source_program_artifact": _relative_evidence_path(teacher_program),
                    "render_receipt_artifact": _relative_evidence_path(teacher_receipt),
                    "render_receipt_sha256": sha256(teacher_receipt),
                    "verified_source_sha256": receipt["source_sha256"],
                    "verified_png_sha256": receipt["png_sha256"],
                    "renderer_sha256": receipt.get("renderer_sha256"),
                    "source_kind": "archived_teacher_render",
                    "selection_reason": archived["reason"],
                })

            # Reference/initial canvas/program remain the reviewed portable
            # state assets.  They are copied into the full bundle from the old
            # input archive; overlay mode relies on its already bootstrapped
            # rl-states tree.
            output_rows.append({
                "id": state_id,
                "source_group": row["source_group"],
                "family": row["family"],
                "split": "train",
                "reference": row["reference"],
                "canvas": row["canvas"],
                "program": row["program"],
                "initial_canvas": row["canvas"],
                "initial_program": row["program"],
                "anchors": anchor_rows,
                "selection_note": selection_note,
                "anchor_source": {
                    "run": "contract-quality-20260918",
                    "policy": "parent",
                    "task": task,
                    "sample": "sample-00",
                    "summary": _relative_evidence_path(summary_path),
                },
            })

        if missing:
            raise ValueError("missing required same-reference training evidence: " + "; ".join(missing))

        manifest = {
            "schema": "painter.aesthetic-states.v1",
            "purpose": "same-reference pairwise anchor pool for aesthetic RL",
            "anchor_policy": "Anchors are same-source_group observed paintings. Student early/final/observed roles are temporal roles, not quality tiers. Two parent-inspected archived teacher renders are explicitly marked archived-teacher and receipt-verified; consumers may mix the stable pool deterministically.",
            "initial_canvas_policy": "initial_canvas and initial_program are the reviewed portable train-state assets; they may correspond to an observed early or final student render.",
            "training_states": output_rows,
            "counts": {
                "states": len(output_rows),
                "families": len({row["family"] for row in output_rows}),
                "anchors": sum(len(row["anchors"]) for row in output_rows),
                "one_anchor_states": sum(len(row["anchors"]) == 1 for row in output_rows),
                "archived_teacher_anchors": sum(
                    sum(anchor.get("source_kind") == "archived_teacher_render" for anchor in row["anchors"])
                    for row in output_rows
                ),
                "archived_teacher_states": sum(
                    any(anchor.get("source_kind") == "archived_teacher_render" for anchor in row["anchors"])
                    for row in output_rows
                ),
            },
        }
        manifest_path = root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        # The native taskset keeps the historical ``rl-states.json`` loader
        # contract.  Its ``anchors`` field must therefore be a list of
        # relative canvas strings; the detailed records remain in
        # ``aesthetic-states.json`` for provenance and hash inspection.
        runtime_rows = []
        for row in output_rows:
            runtime = {
                key: row[key]
                for key in (
                    "id", "source_group", "family", "split", "reference", "canvas",
                    "program", "initial_canvas", "initial_program",
                )
            }
            runtime["anchors"] = [anchor["canvas"] for anchor in row["anchors"]]
            runtime["anchor_records"] = row["anchors"]
            runtime["anchor_source"] = row["anchor_source"]
            runtime["selection_note"] = row["selection_note"]
            runtime_rows.append(runtime)
        runtime_path = root / RUNTIME_STATES_NAME
        runtime_path.write_text(json.dumps(runtime_rows, indent=2) + "\n")
        files = [manifest_path, runtime_path]
        files.extend(sorted(pool.rglob("*")))
        file_receipts = {
            str(path.relative_to(root)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files if path.is_file() and not path.is_symlink()
        }
        receipt = {
            "schema": "painter.aesthetic-data-receipt.v1",
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "runtime_states": str(runtime_path.relative_to(root)),
            "runtime_states_sha256": sha256(runtime_path),
            "source_inputs": "painter/runs/revision-rl-20260918/inputs/rl-states.json",
            "source_inputs_sha256": sha256(source_inputs / "rl-states.json"),
            "source_evidence": "painter/runs/contract-quality-20260918/collected-final/evidence/training-states/parent",
            "files": file_receipts,
            "counts": manifest["counts"],
            "heldout_overlap": [],
        }
        (root / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        EVIDENCE_RUN = old_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN_ROOT)
    parser.add_argument("--source-inputs", type=Path, default=SOURCE_INPUTS)
    parser.add_argument("--evidence-run", type=Path, default=EVIDENCE_RUN)
    args = parser.parse_args()
    receipt = build(args.root, source_inputs=args.source_inputs, evidence_run=args.evidence_run)
    print(json.dumps({"manifest": receipt["manifest"], "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
