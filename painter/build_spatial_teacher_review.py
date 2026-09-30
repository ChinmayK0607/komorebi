#!/usr/bin/env python3
"""Build a read-only, hash-verified review gallery for spatial teacher candidates.

The exporter only reads candidate metadata and collected render receipts.  It
copies verified PNGs into a self-contained review directory, writes a small JSON
manifest, and creates an HTML page with the initial render on the left and the
explicitly selected render on the right.  Teacher observations remain pending
human review and are never exported as training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "painter.spatial-teacher-review.v1"
# Explicit curation supplied with the experiment record.  Selected revisions
# remain pending human review even when their collected render is verified.
SELECTED_TURNS: dict[tuple[str, str], int | None] = {
    ("astra", "01"): 3,
    ("astra", "02"): 3,
    ("astra", "03"): 3,
    ("astra", "04"): 1,
    ("sol", "01"): 2,
    ("sol", "02"): 1,
    ("sol", "03"): 1,
    ("sol", "04"): 3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def safe(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown"))
    return value.strip(".") or "unknown"


def as_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def candidate_dirs(data_dir: Path) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    for model_dir in sorted(data_dir.iterdir() if data_dir.is_dir() else []):
        if not model_dir.is_dir() or model_dir.name not in {"astra", "sol"}:
            continue
        for number_dir in sorted(model_dir.iterdir()):
            if not number_dir.is_dir() or not re.fullmatch(r"\d{2}", number_dir.name):
                continue
            if (number_dir / "candidate.json").is_file():
                result.append((model_dir.name, number_dir.name, number_dir))
    return result


def render_id(model: str, number: str, turn: int) -> str:
    return f"{model}-{number}-turn-{turn}"


def source_relpath(collected_dir: Path, path: Path) -> str | None:
    try:
        return path.relative_to(collected_dir).as_posix()
    except ValueError:
        return None


def summarize_observation(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"present": False}
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"present": False, "error": "invalid observation JSON"}
    if not isinstance(raw, dict):
        return {"present": False, "error": "observation is not an object"}
    # Keep only review text and statuses.  In particular, do not carry source
    # paths or arbitrary receipt fields into the archive manifest.
    result: dict[str, Any] = {"present": True}
    for key in ("observation_type", "status", "valid", "action", "quality_acceptance", "style_observed", "timing_note"):
        if key in raw and isinstance(raw[key], (str, bool, int, float)):
            result[key] = raw[key]
    for key in ("good", "weak"):
        values = as_texts(raw.get(key))
        if values:
            result[key] = values
    return result


def find_observation(candidate_dir: Path, selected_turn: int | None) -> Path | None:
    choices: list[Path] = []
    if selected_turn is not None:
        choices.extend([
            candidate_dir / f"turn-{selected_turn}" / "final-observation.json",
            candidate_dir / f"turn-{selected_turn}" / "critique.json",
        ])
    choices.extend([candidate_dir / "final-observation.json", candidate_dir / "critique.json"])
    # Root critique is often the record for a first-turn observation.  A
    # turn-specific final observation is preferred for selected revisions.
    for path in choices:
        if path.is_file():
            return path
    return None


def style_class(style: Any) -> str:
    value = str(style or "").lower()
    return "hybrid" if "hybrid" in value else "brush"


def receipt_error_values(receipt: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("error", "error_code"):
        if receipt.get(key) not in (None, ""):
            values.append(receipt[key])
    errors = receipt.get("errors")
    if isinstance(errors, list):
        values.extend(errors)
    elif errors not in (None, ""):
        values.append(errors)
    return [str(value) for value in values]


def inspect_turn(collected_dir: Path, model: str, number: str, turn_dir: Path) -> dict[str, Any]:
    match = re.fullmatch(r"turn-(\d+)", turn_dir.name)
    if not match:
        raise ValueError(f"unexpected turn directory: {turn_dir}")
    turn = int(match.group(1))
    rid = render_id(model, number, turn)
    receipt_path = turn_dir / "painting.json"
    source_path = turn_dir / "program.js"
    png_path = turn_dir / "painting.png"
    record: dict[str, Any] = {
        "render_id": rid,
        "turn": turn,
        "source_file": source_relpath(collected_dir, source_path),
        "receipt_file": source_relpath(collected_dir, receipt_path),
        "png_file": source_relpath(collected_dir, png_path) if png_path.is_file() else None,
        "source_sha256": sha256(source_path) if source_path.is_file() else None,
        "receipt_sha256": sha256(receipt_path) if receipt_path.is_file() else None,
        "png_sha256": sha256(png_path) if png_path.is_file() else None,
        "receipt_source_sha256": None,
        "receipt_png_sha256": None,
        "seed": None,
        "valid": False,
        "finished": None,
        "timed_out": None,
        "error_code": None,
        "elapsed_seconds": None,
        "verification": "missing_receipt",
        "asset_file": None,
        "errors": [],
    }
    try:
        receipt = read_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record["errors"] = [f"invalid receipt JSON: {type(exc).__name__}"]
        return record
    if not isinstance(receipt, dict):
        record["errors"] = ["receipt is not an object"]
        return record
    record.update(
        receipt_source_sha256=first(receipt, "source_sha256", "sourceHash"),
        receipt_png_sha256=first(receipt, "png_sha256", "render_sha256", "output_sha256", "image_sha256"),
        seed=receipt.get("seed"),
        valid=receipt.get("valid") is True,
        finished=receipt.get("finished"),
        timed_out=receipt.get("timed_out"),
        error_code=receipt.get("error_code"),
        elapsed_seconds=receipt.get("elapsed_seconds"),
        errors=receipt_error_values(receipt),
    )
    reasons: list[str] = []
    if record["valid"] is not True:
        reasons.append("receipt_invalid")
    if record["finished"] is False:
        reasons.append("not_finished")
    if record["timed_out"] is True or record["error_code"] == "render_timeout":
        reasons.append("timed_out")
    if record["errors"]:
        reasons.append("receipt_error")
    if not source_path.is_file():
        reasons.append("missing_source")
    elif not record["receipt_source_sha256"] or record["source_sha256"] != record["receipt_source_sha256"]:
        reasons.append("source_hash_mismatch")
    if not png_path.is_file():
        reasons.append("missing_png")
    elif not record["receipt_png_sha256"] or record["png_sha256"] != record["receipt_png_sha256"]:
        reasons.append("png_hash_mismatch")
    if reasons:
        record["verification"] = "failed:" + ",".join(reasons)
    else:
        record["verification"] = "verified"
    return record


def collect_turns(collected_dir: Path, model: str, number: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    roots = (collected_dir / "linux-turn1", collected_dir / "linux-revisions")
    for root in roots:
        task_dir = root / model / number
        if not task_dir.is_dir():
            continue
        for turn_dir in sorted(task_dir.glob("turn-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 10**9):
            if turn_dir.is_dir() and re.fullmatch(r"turn-\d+", turn_dir.name):
                turns.append(inspect_turn(collected_dir, model, number, turn_dir))
    turns.sort(key=lambda row: (int(row["turn"]), str(row["source_file"])))
    # If a copied evidence root ever contains a duplicate turn, keep the first
    # deterministic record and expose the duplicate as a failed evidence row.
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in turns:
        if row["turn"] in seen:
            row["verification"] = "failed:duplicate_turn"
        else:
            unique.append(row)
            seen.add(row["turn"])
    return unique


def copy_verified_assets(output_dir: Path, turns: list[dict[str, Any]], collected_dir: Path) -> None:
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for row in turns:
        if row["verification"] != "verified":
            continue
        png_file = row.get("png_file")
        if not isinstance(png_file, str):
            continue
        source = collected_dir / Path(png_file)
        if not source.is_file() or sha256(source) != row.get("receipt_png_sha256"):
            # Recheck immediately before copying; this keeps an output asset
            # tied to the receipt even if evidence is edited between scans.
            row["verification"] = "failed:png_hash_mismatch"
            continue
        name = safe(row["render_id"]) + ".png"
        destination = assets / name
        shutil.copyfile(source, destination)
        if sha256(destination) != row.get("receipt_png_sha256"):
            destination.unlink(missing_ok=True)
            row["verification"] = "failed:copied_png_hash_mismatch"
            continue
        row["asset_file"] = f"assets/{name}"


def task_record(data_dir: Path, collected_dir: Path, model: str, number: str, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw = read_json(candidate_dir / "candidate.json")
    if not isinstance(raw, dict):
        raise ValueError(f"{candidate_dir / 'candidate.json'} must be an object")
    candidate_id = raw.get("candidate_id") or f"spatial-readability-20260930-{model}-{number}"
    selected_turn = SELECTED_TURNS.get((model, number))
    # Candidate metadata is useful evidence but does not override the explicit
    # curation map or silently turn a missing revision into a final.
    turns = collect_turns(collected_dir, model, number)
    copy_verified_assets(output_dir, turns, collected_dir)
    initial = next((row for row in turns if row["turn"] == 1), None)
    selected = next((row for row in turns if selected_turn is not None and row["turn"] == selected_turn), None)
    selected_final_id: str | None = None
    if selected is not None and selected["verification"] == "verified":
        selected_final_id = selected["render_id"]
    if selected_turn is None:
        selection_status = "pending_revision"
    elif selected is None:
        selection_status = "missing_selected_render"
    elif selected["verification"] != "verified":
        selection_status = "failed_selected_render"
    elif raw.get("selected_turn") != selected_turn or not isinstance(raw.get("acceptance"), dict):
        # A valid render can be shown for review while its candidate record is
        # still stale.  Keep this explicitly pending instead of treating a
        # critique receipt as an acceptance decision.
        selection_status = "pending_final_metadata"
    else:
        observation_path = find_observation(candidate_dir, selected_turn)
        selection_status = "pending_human_review" if observation_path else "pending_final_metadata"
    observation = summarize_observation(find_observation(candidate_dir, selected_turn))
    failed = [row["render_id"] for row in turns if row["verification"] != "verified"]
    if not isinstance(raw.get("prompt"), str):
        prompt_path = candidate_dir / "prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
    else:
        prompt = raw["prompt"]
    style = raw.get("style") or raw.get("initial_requested_style") or ""
    return {
        "candidate_id": str(candidate_id),
        "model": raw.get("model") or model,
        "model_family": model,
        "name": raw.get("name") or candidate_dir.name,
        "prompt": prompt,
        "requested_style": style,
        "style_class": style_class(style),
        "focus": raw.get("focus") if isinstance(raw.get("focus"), (str, list)) else [],
        "actual_seeds": sorted({row["seed"] for row in turns if row.get("seed") is not None}),
        "candidate_status": raw.get("status"),
        "review_status": "pending_human" if selection_status not in {"pending_revision", "failed_selected_render"} else selection_status,
        "selection_status": selection_status,
        "initial_render_id": initial["render_id"] if initial else None,
        "selected_turn": selected_turn,
        "selected_final_render_id": selected_final_id,
        "selected_final_accepted": False,
        "teacher_observation": observation,
        "human_label": None,
        "training_benefit_claim": False,
        "training_export": False,
        "preserved_failed_turns": failed,
        "turns": turns,
    }


def render_image(row: dict[str, Any] | None, label: str) -> str:
    if row is None:
        return f'<div class="missing"><strong>{html.escape(label)}</strong><br>Evidence missing.</div>'
    asset = row.get("asset_file")
    if isinstance(asset, str):
        return f'<figure><img loading="lazy" src="{html.escape(asset)}" alt="{html.escape(label)}"><figcaption>{html.escape(label)} · {html.escape(str(row["render_id"]))}</figcaption></figure>'
    return f'<div class="missing"><strong>{html.escape(label)}</strong><br>{html.escape(str(row.get("verification", "unavailable")))}</div>'


def render_list(values: Any) -> str:
    texts = as_texts(values)
    if not texts:
        return "<p class=muted>None recorded.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in texts) + "</ul>"


def render_html(manifest: dict[str, Any]) -> str:
    cards: list[str] = []
    for task in manifest["tasks"]:
        turns = task["turns"]
        by_id = {row["render_id"]: row for row in turns}
        initial = by_id.get(task["initial_render_id"])
        selected = by_id.get(task["selected_final_render_id"])
        observation = task["teacher_observation"]
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(task['name']))}</h2>"
            f"<p class=meta><code>{html.escape(task['candidate_id'])}</code> · {html.escape(str(task['model']))} · style class: <b>{html.escape(task['style_class'])}</b> · review: <b>{html.escape(task['review_status'])}</b></p>"
            "<div class=compare>"
            f"<section><h3>Initial render</h3>{render_image(initial, 'Initial · turn 1')}</section>"
            f"<section><h3>Selected final render</h3>{render_image(selected, 'Selected final · explicit curation')}</section>"
            "</div>"
            f"<section class=prompt><h3>Prompt</h3><p>{html.escape(task['prompt'])}</p></section>"
            f"<section><h3>Rubric review</h3><p>Focus: {html.escape(', '.join(as_texts(task['focus'])) or 'not recorded')}</p>"
            f"<h4>Observed strengths</h4>{render_list(observation.get('good'))}"
            f"<h4>Observed weaknesses</h4>{render_list(observation.get('weak'))}</section>"
            f"<section><h3>Validity</h3><p>Selection status: <b>{html.escape(task['selection_status'])}</b>. Failed or unavailable turns are preserved in the manifest: {html.escape(', '.join(task['preserved_failed_turns']) or 'none')}.</p></section>"
            f"<section><h3>Style</h3><p>Requested: {html.escape(str(task['requested_style']))}</p><p>Class: <b>{html.escape(task['style_class'])}</b>. Teacher observations are evidence for review, not human labels.</p></section>"
            "</article>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial teacher candidate review</title><style>
:root{color-scheme:light}body{font:16px/1.5 system-ui,sans-serif;background:#f4f0e8;color:#25231e;margin:0;padding:28px}main{max-width:1500px;margin:auto}article{background:#fff;border:1px solid #d8d0c2;border-radius:12px;padding:22px;margin:28px 0;box-shadow:0 3px 14px #0000000d}.compare{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.compare section{min-width:0}figure{margin:0}img{display:block;width:100%;height:auto;background:#eee;border-radius:6px}figcaption{font-size:.9rem;color:#5b554c;margin-top:7px}.missing{min-height:360px;display:grid;place-content:center;text-align:center;background:#eee8dd;border:1px dashed #a89d8e;border-radius:6px;color:#655d52;padding:20px}.meta{color:#655d52}.prompt{background:#faf8f4;padding:12px 16px;border-radius:8px}.prompt p{white-space:pre-wrap}.muted{color:#746d63}h1{font-size:2rem}h2{margin-top:0}h3{margin-bottom:8px}h4{margin:8px 0 0}@media(max-width:850px){body{padding:14px}.compare{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Spatial teacher candidate review</h1>
<p>Eight assistant-authored candidate trajectories are shown for human review. The left image is always the collected initial turn; the right image is the explicitly selected final render when its receipt, source hash, and PNG hash verify. Teacher observations are not human labels. No training export or training-benefit claim is made.</p>
""" + "".join(cards) + "</main></body></html>\n"


def build_review(data_dir: Path, collected_dir: Path, output_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    collected_dir = Path(collected_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task_record(data_dir, collected_dir, model, number, candidate_dir, output_dir) for model, number, candidate_dir in candidate_dirs(data_dir)]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_root": data_dir.name,
        "evidence_root": collected_dir.name,
        "review_status": "pending_human",
        "human_labels_present": False,
        "training_export": False,
        "training_benefit_claim": False,
        "task_count": len(tasks),
        "summary": {
            "verified_render_count": sum(1 for task in tasks for row in task["turns"] if row["verification"] == "verified"),
            "failed_or_unavailable_render_count": sum(1 for task in tasks for row in task["turns"] if row["verification"] != "verified"),
            "selected_final_count": sum(1 for task in tasks if task["selected_final_render_id"] is not None),
            "pending_revision_count": sum(1 for task in tasks if task["selection_status"] == "pending_revision"),
        },
        "tasks": tasks,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="candidate directory")
    parser.add_argument("--collected-dir", type=Path, required=True, help="collected render root")
    parser.add_argument("--output", type=Path, required=True, help="ignored self-contained review output directory")
    args = parser.parse_args()
    manifest = build_review(args.data_dir, args.collected_dir, args.output)
    print(json.dumps({"output": str(args.output), "task_count": manifest["task_count"], "summary": manifest["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
