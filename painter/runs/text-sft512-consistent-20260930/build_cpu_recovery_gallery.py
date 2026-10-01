#!/usr/bin/env python3
"""Build a static review page for the bounded CPU-recovery evaluation.

This is a file-only presentation step. It does not render, judge, contact an
API, or infer model failures from an incomplete recovery. Verified replay PNGs
whose hash differs from the recorded target are labelled ``new CPU replay``;
renderer failures and missing files remain ``recovery failure``.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ORIGINAL_NO_CANVAS = 2
DEFAULT_UNOBSERVED = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: Any) -> str:
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe recovery case id: {value!r}")
    return value


def _relative_file(root: Path, value: Any, default: str | None = None) -> Path | None:
    """Resolve a manifest-relative file while refusing path traversal."""
    if not isinstance(value, str) or not value:
        value = default
    if not value:
        return None
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest file escapes result root: {value!r}") from exc
    return path


def _copy_used(source: Path, destination: Path) -> bool:
    if not source.is_file() or source.is_symlink():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _judgment_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "judgment pending"
    status = value.get("status")
    if status == "uncertain":
        return "Judge disagreed (uncertain)"
    label = value.get("label") or value.get("verdict") or value.get("status")
    score = value.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        if score > 0.5:
            preference = "RL preferred"
        elif score < 0.5:
            preference = "SFT preferred"
        else:
            preference = "Tie"
        return f"{preference} ({score:.3f})"
    return str(label or "receipt present")


def _classify(case: dict[str, Any], result_root: Path) -> tuple[str, str, Path | None]:
    status = case.get("status")
    comparison = case.get("comparison")
    replay_path = _relative_file(
        result_root,
        case.get("png_file") or case.get("replay_png_file"),
        f"cases/{case.get('id')}/painting.png",
    )
    replay_hash = case.get("replay_png_sha256")
    target_hash = case.get("target_png_sha256")
    if replay_hash and target_hash:
        comparison = "recovered-identical" if replay_hash == target_hash else "new-replay"
    if status == "verified" and comparison == "recovered-identical" and replay_path and replay_path.is_file():
        return "recovered-identical", "recovered identical", replay_path
    if status == "verified" and comparison in {"new-replay", "new CPU replay"} and replay_path and replay_path.is_file():
        return "new-replay", "new CPU replay", replay_path
    reasons = case.get("failure_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    detail = ", ".join(str(reason) for reason in reasons if reason) or "missing or failed recovery artifact"
    return "recovery-failure", f"recovery failure: {detail} (not model failure)", None


def _case_row(
    case: dict[str, Any],
    *,
    result_root: Path,
    anchors_root: Path,
    prompts_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_id = _safe_id(case.get("id") or case.get("task_id"))
    anchor = anchors_root / f"{case_id}.png"
    prompt = prompts_root / f"{case_id}.txt"
    anchor_dest = output_root / "assets" / "anchors" / f"{case_id}.png"
    prompt_dest = output_root / "assets" / "prompts" / f"{case_id}.txt"
    anchor_used = _copy_used(anchor, anchor_dest)
    prompt_used = _copy_used(prompt, prompt_dest)
    prompt_text = None
    if prompt_used:
        prompt_text = prompt.read_text(encoding="utf-8").strip()
    classification, label, replay = _classify(case, result_root)
    receipt = _relative_file(
        result_root,
        case.get("replay_receipt_file") or case.get("receipt_file"),
        f"cases/{case_id}/replay-receipt.json",
    )
    renderer_receipt = _relative_file(result_root, case.get("renderer_receipt_file"))
    judgment = _relative_file(result_root, f"judgments/{case_id}.json")
    judgment_value = _load_optional_json(judgment)
    def href(path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        return Path(os.path.relpath(path, output_root)).as_posix()

    # Do not expose absolute source paths from the downloaded result bundle.
    return {
        "id": case_id,
        "family": case.get("family"),
        "source_turn": case.get("source_turn"),
        "source_sha256": case.get("source_sha256"),
        "target_png_sha256": case.get("target_png_sha256"),
        "replay_png_sha256": case.get("replay_png_sha256"),
        "classification": classification,
        "label": label,
        "anchor": f"assets/anchors/{case_id}.png" if anchor_used else None,
        "prompt": f"assets/prompts/{case_id}.txt" if prompt_used else None,
        "prompt_text": prompt_text,
        "replay": href(replay),
        "receipt": href(receipt),
        "renderer_receipt": href(renderer_receipt),
        "judgment": href(judgment),
        "judgment_text": _judgment_text(judgment_value),
    }


def _esc(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _image_or_missing(path: str | None, alt: str) -> str:
    if not path:
        return f'<div class="missing">{html.escape(alt)} unavailable</div>'
    return f'<img src="{html.escape(path)}" alt="{html.escape(alt)}" loading="lazy">'


def _link(path: str | None, label: str) -> str:
    if not path:
        return f'<span class="muted">{html.escape(label)} unavailable</span>'
    return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'


def _render_row(row: dict[str, Any]) -> str:
    provenance = [
        f"source turn {_esc(row.get('source_turn'))}",
        f"source {_esc(row.get('source_sha256'))}",
        f"target {_esc(row.get('target_png_sha256'))}",
        f"replay {_esc(row.get('replay_png_sha256'))}",
    ]
    links = " · ".join([
        _link(row.get("prompt"), "prompt"),
        _link(row.get("receipt"), "replay receipt"),
        _link(row.get("renderer_receipt"), "renderer receipt"),
        _link(row.get("judgment"), row.get("judgment_text", "judgment")),
    ])
    return (
        f'<article class="case {html.escape(row["classification"])}">'
        f'<h2>{_esc(row["id"])} <span>{_esc(row["label"])}</span></h2>'
        f'<p class="family">family: {_esc(row.get("family"))}</p>'
        f'<p class="prompt"><b>Prompt:</b> {_esc(row.get("prompt_text"))}</p>'
        '<div class="pair">'
        f'<figure>{_image_or_missing(row.get("anchor"), "Frozen SFT512")}'
        '<figcaption>Frozen SFT512</figcaption></figure>'
        f'<figure>{_image_or_missing(row.get("replay"), "Replayed RL step16")}'
        '<figcaption>Replayed RL step16</figcaption></figure>'
        '</div>'
        f'<details><summary>Provenance and receipts</summary><p class="provenance">{" · ".join(provenance)}</p><p>{links}</p></details>'
        '</article>'
    )


def build_gallery(
    manifest_path: Path,
    anchors_root: Path,
    prompts_root: Path,
    output_root: Path | None = None,
    *,
    original_no_canvas: int = DEFAULT_ORIGINAL_NO_CANVAS,
    unobserved: int = DEFAULT_UNOBSERVED,
) -> dict[str, Any]:
    """Build the ignored review page and return its bounded summary."""
    manifest_path = manifest_path.expanduser().resolve()
    result_root = manifest_path.parent
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"recovery manifest is missing or symlinked: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("recovery manifest must contain a cases list")
    output_root = (output_root or result_root / "review").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    anchors_root = anchors_root.expanduser().resolve()
    prompts_root = prompts_root.expanduser().resolve()
    rows = []
    for case in payload["cases"]:
        if not isinstance(case, dict):
            raise ValueError("recovery case is not an object")
        rows.append(_case_row(
            case,
            result_root=result_root,
            anchors_root=anchors_root,
            prompts_root=prompts_root,
            output_root=output_root,
        ))
    rows.sort(key=lambda row: row["id"])
    counts = {
        "recoverable": len(rows),
        "recovered_identical": sum(row["classification"] == "recovered-identical" for row in rows),
        "new_cpu_replay": sum(row["classification"] == "new-replay" for row in rows),
        "recovery_failures": sum(row["classification"] == "recovery-failure" for row in rows),
        "original_no_canvas": original_no_canvas,
        "unobserved": unobserved,
    }
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    title = "CPU recovery review — frozen SFT512 vs replayed RL step16"
    cards = "".join([
        f'<div><b>{counts["recoverable"]}</b>recoverable</div>',
        f'<div><b>{counts["recovered_identical"]}</b>recovered identical</div>',
        f'<div><b>{counts["new_cpu_replay"]}</b>new CPU replay</div>',
        f'<div><b>{counts["recovery_failures"]}</b>recovery failures</div>',
        f'<div><b>{counts["original_no_canvas"]}</b>original no-canvas</div>',
        f'<div><b>{counts["unobserved"]}</b>unobserved</div>',
    ])
    html_text = (
        '<!doctype html><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title>'
        '<style>body{font:15px system-ui,sans-serif;max-width:1180px;margin:2em auto;color:#222}.scope{background:#fff7df;border-left:4px solid #d7a500;padding:.7em 1em}.cards{display:flex;gap:.6em;flex-wrap:wrap}.cards div{background:#f2f4f7;padding:.7em 1em;border-radius:7px;min-width:8em}.cards b{display:block;font-size:1.35em}.case{border-top:1px solid #ccc;padding:1em 0}.case h2{margin:.2em 0}.case h2 span{font-size:.75em;font-weight:normal;padding:.2em .45em;border-radius:4px;background:#e8eef8}.case.new-replay h2 span{background:#fff0d4}.case.recovery-failure h2 span{background:#fbe0e0}.family,.provenance,.muted{color:#555;font-size:.9em}.prompt{white-space:pre-wrap;background:#fafafa;border-left:3px solid #ccc;padding:.6em}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1em}.pair figure{margin:0;background:#f7f7f7;padding:.6em}.pair img{display:block;width:100%;max-width:600px;aspect-ratio:1;object-fit:contain;background:#eee}.pair figcaption{padding:.4em 0;font-weight:600}.missing{aspect-ratio:1;display:grid;place-items:center;background:#eee;color:#777;text-align:center}details{margin-top:.6em}summary{cursor:pointer;color:#145a9c}a{color:#145a9c}</style>'
        f'<h1>{html.escape(title)}</h1>'
        '<p class="scope"><b>Recovery scope:</b> this page compares only recovered step16 artifacts. A hash mismatch is labelled <b>new CPU replay</b>; missing or failed replay artifacts are <b>recovery failures</b>, never model failures.</p>'
        '<p><b>Judge labels:</b> RL preferred, SFT preferred, Tie, or Judge disagreed (uncertain). Scores are relative preference against frozen SFT; 0.5 means tie with frozen SFT; this is not absolute painting quality.</p>'
        f'<div class="cards">{cards}</div>'
        f'<p>Manifest reports {html.escape(str(summary.get("verified", "—")))} verified and {html.escape(str(summary.get("failed", "—")))} failed replay cases. Judgment receipts are optional and remain pending when absent.</p>'
        + "".join(_render_row(row) for row in rows)
        + '<p class="muted">Frozen anchors and prompts were copied only for cases listed in this manifest. The page performs no rendering or network calls.</p>'
    )
    (output_root / "index.html").write_text(html_text, encoding="utf-8")
    return {"output": str(output_root / "index.html"), "counts": counts, "cases": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original-no-canvas", type=int, default=DEFAULT_ORIGINAL_NO_CANVAS)
    parser.add_argument("--unobserved", type=int, default=DEFAULT_UNOBSERVED)
    args = parser.parse_args(argv)
    build_gallery(
        args.manifest,
        args.anchors,
        args.prompts,
        args.output,
        original_no_canvas=args.original_no_canvas,
        unobserved=args.unobserved,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
