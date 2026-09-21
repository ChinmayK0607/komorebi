"""Build a paired reference/base/SFT gallery from native eval evidence.

This is a report-only command.  It delegates trace parsing and usage/status
handling to ``painter.native_painting_report`` and adds the exact four image
slots needed by this capability check: first and final recorded canvases for
each policy.  Missing traces and artifacts stay explicit; no image is copied,
rendered, scored, or substituted from another turn.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.native_painting_report import build_report  # noqa: E402


# The node server and task configs use these actual policy labels.  The
# reviewed native report predates this run and internally calls its second
# policy bucket ``rl``; keep that implementation detail out of this gallery.
POLICY_KEYS = ("base", "sft")
POLICY_LABELS = {"base": "base", "sft": "sft"}
NATIVE_POLICY_KEYS = {"base": "rl", "sft": "sft"}
DISPLAY_NAMES = {"base": "Base unadapted", "sft": "SFT step24"}
POSTHOC_CONTROLS = (
    ("render-deadline-control", "000000000357", "Photo 357 · baseball"),
    ("render-deadline-clock", "000000000078", "Photo 078 · clock"),
)


def _existing(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return str(path.resolve()) if path.is_file() else None


def _first_canvas(episode: dict[str, Any] | None) -> str | None:
    if not episode:
        return None
    # A first canvas must be an actual recorded valid action.  In particular,
    # do not call a previous canvas the first output after an invalid turn.
    for turn in episode.get("turns", []):
        if isinstance(turn, dict) and turn.get("valid") is not False:
            path = _existing(turn.get("canvas"))
            if path:
                return path
    return None


def _final_canvas(episode: dict[str, Any] | None) -> str | None:
    if not episode:
        return None
    return _existing(episode.get("final_canvas"))


def _episode_metadata(episode: dict[str, Any] | None) -> dict[str, Any]:
    if not episode:
        return {
            "trace_found": False,
            "status": "missing",
            "errors": [],
            "flags": ["missing_trace"],
            "turn_count": 0,
            "usage": None,
            "first_canvas": None,
            "final_canvas": None,
        }
    first = _first_canvas(episode)
    final = _final_canvas(episode)
    return {
        "trace_found": True,
        "trace_id": episode.get("trace_id"),
        "trace_source": episode.get("trace_source"),
        "status": episode.get("status"),
        "errors": episode.get("errors", []),
        "flags": episode.get("flags", []),
        "turn_count": len(episode.get("turns", [])),
        "turn_cap": episode.get("turn_cap"),
        "finished": episode.get("finished"),
        "final_canvas_valid": episode.get("final_canvas_valid"),
        "tokens": episode.get("tokens"),
        "first_canvas": first,
        "final_canvas": final,
        "first_canvas_missing": bool(episode.get("turns")) and first is None,
        "final_canvas_missing": bool(episode.get("final_canvas_recorded")) and final is None,
    }


def _image(path: str | None, outdir: Path, alt: str) -> str:
    if not path:
        return f'<span class="missing">{html.escape(alt)} unavailable</span>'
    source = Path(path)
    if not source.is_file():
        return f'<span class="missing">{html.escape(alt)} artifact missing</span>'
    relative = html.escape(os.path.relpath(source, outdir), quote=True)
    return f'<a href="{relative}"><img loading="lazy" src="{relative}" alt="{html.escape(alt, quote=True)}"></a>'


def _metadata_block(metadata: dict[str, Any]) -> str:
    compact = json.dumps(
        {
            "status": metadata.get("status"),
            "trace_found": metadata.get("trace_found"),
            "turn_count": metadata.get("turn_count"),
            "finished": metadata.get("finished"),
            "final_canvas_valid": metadata.get("final_canvas_valid"),
            "tokens": metadata.get("tokens"),
            "flags": metadata.get("flags", []),
            "errors": metadata.get("errors", []),
        },
        sort_keys=True,
    )
    return f"<details><summary>recorded metadata</summary><pre>{html.escape(compact)}</pre></details>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _posthoc_control_root(evidence_root: Path, name: str) -> Path:
    # The collected bundle uses evidence/coco-capability/<control>.  The
    # second candidate keeps this helper usable when handed that subdirectory
    # directly during a local artifact smoke test.
    nested = evidence_root / "coco-capability" / name
    return nested if nested.is_dir() else evidence_root / name


def _posthoc_checks(evidence_root: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only receipt-valid saved-program replay controls.

    These controls are intentionally separate from the matched base/SFT case
    records.  A receipt must mark the render valid and its recorded hashes must
    match the local program.js and painting.png bytes before an image is shown.
    """

    included: list[dict[str, Any]] = []
    for directory, suffix, label in POSTHOC_CONTROLS:
        root = _posthoc_control_root(evidence_root, directory)
        program = root / "program.js"
        painting = root / "painting.png"
        receipt_path = root / "painting.json"
        protocol_path = root / "protocol.json"
        if not all(path.is_file() for path in (program, painting, receipt_path, protocol_path)):
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(receipt, dict) or not isinstance(protocol, dict):
            continue
        if receipt.get("valid") is not True or protocol.get("timeout_seconds") != 180:
            continue
        source_hash = receipt.get("source_sha256")
        png_hash = receipt.get("png_sha256")
        if not isinstance(source_hash, str) or not isinstance(png_hash, str):
            continue
        try:
            program_hash = _sha256(program)
            painting_hash = _sha256(painting)
        except OSError:
            continue
        if source_hash.lower() != program_hash or png_hash.lower() != painting_hash:
            continue
        case = next((row for row in cases if str(row.get("id", "")).endswith(suffix)), None)
        if case is None:
            continue
        base = (case.get("policies") or {}).get("base") or {}
        included.append(
            {
                "id": directory,
                "label": label,
                "case_id": case.get("id"),
                "reference": case.get("reference"),
                "primary_base_canvas": base.get("final_canvas"),
                "extended_canvas": str(painting.resolve()),
                "program": str(program.resolve()),
                "receipt": {
                    "valid": True,
                    "source_sha256": source_hash,
                    "png_sha256": png_hash,
                    "elapsed_seconds": receipt.get("elapsed_seconds"),
                    "program_sha256_matches": True,
                    "png_sha256_matches": True,
                },
                "protocol": {
                    "timeout_seconds": protocol.get("timeout_seconds"),
                    "source": protocol.get("source"),
                    "primary_deadline_seconds": 120,
                },
            }
        )
    return included


def _posthoc_html(checks: list[dict[str, Any]], outdir: Path) -> str:
    if not checks:
        return ""
    rows: list[str] = []
    for check in checks:
        receipt = check.get("receipt") or {}
        elapsed = receipt.get("elapsed_seconds")
        elapsed_text = f"; elapsed {elapsed}s" if isinstance(elapsed, (int, float)) else ""
        rows.append(
            "<tr>"
            f"<th><strong>{html.escape(str(check.get('label')))}</strong><br><small>{html.escape(str(check.get('case_id')))}</small>"
            f"<div class=\"reference\">{_image(check.get('reference'), outdir, 'reference')}</div></th>"
            f"<td>{_image(check.get('primary_base_canvas'), outdir, 'primary base retained canvas')}</td>"
            f"<td>{_image(check.get('extended_canvas'), outdir, '180-second saved-program replay')}"
            f"<p><small>valid receipt; saved program; 180s deadline{html.escape(elapsed_text)}</small></p></td>"
            "</tr>"
        )
    return (
        '<section class="posthoc">'
        "<h2>Post-hoc render-deadline checks — excluded from matched results</h2>"
        "<p>These optional controls rerender the same saved program with a 180-second renderer deadline (the primary run used 120 seconds). They use no new model generation and do not change the matched base/SFT counts or case pairs. Only receipt-valid images with matching program and PNG hashes are shown.</p>"
        '<table><thead><tr><th>Reference</th><th>Primary base retained canvas</th><th>Extended deadline replay</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></section>"
    )


def build_gallery(
    *,
    evidence_root: Path,
    reference_manifest: Path,
    outdir: Path,
    node_root: str = "/root/painter",
) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    reference_manifest = reference_manifest.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Parse through the reviewed report implementation in a private temporary
    # directory.  Its compatibility schema names the base bucket ``rl``;
    # exposing that HTML/JSON here would mislabel this base-vs-SFT run.
    with tempfile.TemporaryDirectory(prefix="coco-capability-native-") as native_dir:
        report = build_report(
            evidence_root,
            reference_manifest,
            Path(native_dir),
            node_root=node_root,
            sft_label=POLICY_LABELS["sft"],
            rl_label=POLICY_LABELS["base"],
        )

    metadata_cases: list[dict[str, Any]] = []
    html_rows: list[str] = []
    for case in report["cases"]:
        policy_records: dict[str, dict[str, Any]] = {}
        for policy in POLICY_KEYS:
            episodes = case.get("policies", {}).get(NATIVE_POLICY_KEYS[policy], [])
            # Preserve duplicates in the native summary.  A paired cell is
            # usable only when exactly one trace exists; otherwise it is an
            # explicit ambiguous/multiple state.
            if len(episodes) == 1:
                policy_records[policy] = _episode_metadata(episodes[0])
            elif len(episodes) > 1:
                policy_records[policy] = {
                    "trace_found": True,
                    "status": "duplicate_traces",
                    "errors": [],
                    "flags": ["duplicate_traces", f"count={len(episodes)}"],
                    "turn_count": None,
                    "usage": None,
                    "first_canvas": None,
                    "final_canvas": None,
                }
            else:
                policy_records[policy] = _episode_metadata(None)
        metadata_case = {
            "id": case["id"],
            "family": case.get("family"),
            "source_group": case.get("source_group"),
            "reference": case.get("reference"),
            "policies": policy_records,
        }
        metadata_cases.append(metadata_case)
        cells = []
        for policy in POLICY_KEYS:
            record = policy_records[policy]
            label = DISPLAY_NAMES[policy]
            cells.append(
                "<td>"
                f"<h3>{label}</h3>"
                f"<div class=\"pair\"><div><strong>first</strong>{_image(record.get('first_canvas'), outdir, 'first canvas')}</div>"
                f"<div><strong>final</strong>{_image(record.get('final_canvas'), outdir, 'final canvas')}</div></div>"
                f"{_metadata_block(record)}</td>"
            )
        html_rows.append(
            "<tr>"
            f"<th><strong>{html.escape(str(case.get('family')))}</strong><br><small>{html.escape(str(case['id']))}</small>"
            f"<div class=\"reference\">{_image(case.get('reference'), outdir, 'reference')}</div></th>"
            + "".join(cells)
            + "</tr>"
        )

    posthoc_checks = _posthoc_checks(evidence_root, metadata_cases)
    metadata = {
        "schema": "coco-capability-gallery-v1",
        "report_schema": report.get("schema_version"),
        "reference_manifest": str(reference_manifest),
        "evidence_root": str(evidence_root),
        "policy_labels": POLICY_LABELS,
        "policy_display_names": DISPLAY_NAMES,
        "expected": {
            **(report.get("expected") or {}),
            "policies": list(POLICY_KEYS),
        },
        "discovered": {
            **(report.get("discovered") or {}),
            "duplicate_cells": {
                str(key).replace("/rl", "/base"): value
                for key, value in (report.get("discovered", {}).get("duplicate_cells", {}) or {}).items()
            },
        },
        "partial": report.get("partial"),
        "cases": metadata_cases,
        "posthoc_render_deadline_checks": posthoc_checks,
        "notes": [
            "First and final links use only recorded valid canvases and native report final_canvas artifacts.",
            "Status, errors, flags, turn counts, and provider usage are copied from native traces.",
            "Missing and duplicate cells remain explicit; no quality score or comparator is fabricated.",
        ],
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    heading = f"COCO capability · {report['discovered']['matched_episodes']}/{report['expected']['episodes']} matched episodes"
    page = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(heading)}</title>"
        "<style>body{font:14px/1.4 system-ui,sans-serif;margin:20px;background:#f3f0e9;color:#292621}h1{font-size:1.4rem}table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #d5cec2;padding:9px;vertical-align:top;text-align:left}th{background:#ece7dd}.reference img{max-width:180px;max-height:180px}.pair{display:flex;gap:12px}.pair img{display:block;max-width:220px;max-height:220px;margin-top:5px}.missing{display:block;color:#9e3c31;font-style:italic;margin:7px 0}.summary{background:#fff;padding:10px;margin-bottom:16px}.summary code{white-space:pre-wrap}.status{font-weight:600}details{margin-top:8px}pre{white-space:pre-wrap;font-size:11px;background:#f0eee9;padding:6px;max-width:45em}</style></head><body>"
        f"<h1>{html.escape(heading)}</h1>"
        "<div class=\"summary\"><p>Reference, first recorded valid canvas, and final recorded canvas are linked from preserved evidence. Metadata below is copied from native traces; missing, invalid, censored, and duplicate states remain visible.</p>"
        "<p><a href=\"metadata.json\">gallery metadata JSON</a></p></div>"
        "<table><thead><tr><th>Reference</th><th>Base unadapted</th><th>SFT step24</th></tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
        + _posthoc_html(posthoc_checks, outdir)
        + "</body></html>"
    )
    (outdir / "index.html").write_text(page, encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--node-root", default="/root/painter")
    args = parser.parse_args(argv)
    metadata = build_gallery(
        evidence_root=args.evidence_root,
        reference_manifest=args.reference_manifest,
        outdir=args.outdir,
        node_root=args.node_root,
    )
    print(
        json.dumps(
            {
                "gallery": str((args.outdir / "index.html").resolve()),
                "metadata": str((args.outdir / "metadata.json").resolve()),
                "partial": metadata["partial"],
                "matched_episodes": metadata["discovered"]["matched_episodes"],
                "expected_episodes": metadata["expected"]["episodes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
