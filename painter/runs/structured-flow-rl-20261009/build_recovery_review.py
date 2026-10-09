#!/usr/bin/env python3
"""Build an offline, portable three-policy viewer for checkpoint 4.

The terminal canvas is selected from the native trace's explicit
``final_canvas`` and ``final_canvas_valid`` fields. A previous valid turn is
reported as a fallback for inspection only; it is never treated as a valid
terminal result or passed to the visual judge.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import importlib.util
from pathlib import Path
import re
import shutil
from typing import Any

RUN_NAME = "structured-flow-rl-20261009"
DEFAULT_SEED = 918210
DEFAULT_TURNS = 3
CASE_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_case_id(value: object) -> str:
    if not isinstance(value, str) or not value or not CASE_RE.fullmatch(value):
        raise ValueError(f"unsafe case id: {value!r}")
    return value


def _valid_png(receipt: Path) -> tuple[Path | None, str]:
    try:
        value = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "malformed_receipt"
    if not isinstance(value, dict) or value.get("valid") is not True:
        reason = value.get("reason") if isinstance(value, dict) else "invalid_receipt"
        return None, str(reason or "invalid_receipt")
    png = receipt.with_name("painting.png")
    if png.is_symlink() or not png.is_file():
        return None, "missing_png"
    expected = value.get("png_sha256")
    if not isinstance(expected, str) or sha256(png) != expected:
        return None, "png_hash_mismatch"
    return png, "valid"


def _resolve_manifest_file(value: object, *, manifest: Path, expected_sha256: str | None = None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(manifest.parent / raw)
    marker = "/structured-flow-rl-20261009/"
    if marker in value:
        candidates.append(manifest.parent.parent / value.split(marker, 1)[1])
    checked = [path for path in candidates if _hash_matches(path, expected_sha256)]
    if len(checked) == 1:
        return checked[0].resolve()
    # Hash-constrained basename lookup is used only for a collected bundle
    # that omitted the original absolute prefix.
    if expected_sha256:
        matches = [path for path in manifest.parent.parent.rglob(raw.name) if _hash_matches(path, expected_sha256)]
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def _brief(row: dict[str, Any], manifest: Path) -> str:
    """Read the admitted brief bytes rather than displaying a path."""
    for key, value in (("brief", row.get("brief")), ("casebrief", row.get("casebrief")), ("reference", row.get("reference"))):
        if not isinstance(value, str) or not value:
            continue
        expected = row.get("casebrief_sha256") if key == "casebrief" else row.get("reference_sha256")
        path = _resolve_manifest_file(value, manifest=manifest, expected_sha256=expected if isinstance(expected, str) else None)
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                text = ""
            if text:
                if "[USER]" in text:
                    text = text.split("[USER]", 1)[1]
                    if "[ASSISTANT]" in text:
                        text = text.split("[ASSISTANT]", 1)[0]
                return text.strip()
        if key == "brief" and ("\n" in value or not Path(value).suffix):
            return value.strip()
    return ""


def _hash_matches(path: Path, expected: str | None) -> bool:
    return path.is_file() and not path.is_symlink() and (expected is None or sha256(path) == expected)


def _resolve_canvas(
    value: object,
    *,
    root: Path,
    source: Path,
    case_id: str,
    expected_sha256: str | None,
    native_resolve: Any,
) -> Path | None:
    """Resolve Linux trace paths after collection without basename guessing."""
    direct = native_resolve(value, evidence_root=root, base_dir=source.parent, node_root="/root/painter")
    if direct is not None and _hash_matches(direct, expected_sha256):
        return direct
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates: list[Path] = []
    parts = list(raw.parts)
    # The archive contains an artifact root ending in evaluation-rollouts,
    # while the trace path includes an extra evaluation-rollouts component.
    for index, part in enumerate(parts):
        if part == "evaluation-rollouts":
            suffix = Path(*parts[index + 1:])
            if suffix.parts:
                candidates.append(root / suffix)
    marker = "/structured-flow-rl-20261009/"
    if marker in value:
        tail = value.split(marker, 1)[1]
        candidates.append(root / Path(tail))
    checked = [path for path in candidates if _hash_matches(path, expected_sha256)]
    if len(checked) == 1:
        return checked[0].resolve()
    # Last resort: constrain search to this exact case and select only a
    # unique longest suffix match, with the recorded canvas hash as a second
    # identity check.  Never pick the first matching basename.
    suffix = tuple(parts[-8:])
    matches: list[tuple[int, Path]] = []
    for path in root.rglob("painting.png"):
        if path.is_symlink() or case_id not in path.parts or not _hash_matches(path, expected_sha256):
            continue
        relative = tuple(path.relative_to(root).parts)
        common = 0
        for left, right in zip(reversed(relative), reversed(suffix)):
            if left != right:
                break
            common += 1
        matches.append((common, path))
    if not matches:
        return None
    best = max(score for score, _ in matches)
    winners = [path for score, path in matches if score == best]
    return winners[0].resolve() if len(winners) == 1 else None


def _trace_cells(root: Path, manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read native traces and retain strict terminal/fallback metadata."""
    try:
        from painter.native_painting_report import _case_identity, _trace_data, read_traces, resolve_existing
    except ImportError:
        # The node invokes this file directly with the Prime interpreter, so
        # the repository parent is not necessarily on sys.path.
        native_path = Path(__file__).resolve().parents[2] / "native_painting_report.py"
        spec = importlib.util.spec_from_file_location("structured_native_painting_report", native_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _case_identity = module._case_identity
        _trace_data = module._trace_data
        read_traces = module.read_traces
        resolve_existing = module.resolve_existing
    if not root.is_dir() or root.is_symlink():
        return {}
    by_id = {row["id"]: row for row in manifest}
    traces, _, _ = read_traces(root)
    result: dict[str, dict[str, Any]] = {}
    for trace in traces:
        case_id, source_group, _ = _case_identity(trace)
        case = by_id.get(case_id or "") or next(
            (row for row in manifest if row.get("source_group") == source_group), None
        )
        if case is None:
            continue
        case_id = case["id"]
        _, info = _trace_data(trace)
        source = Path(str(trace.get("_source_path", root)))
        final_value = info.get("final_canvas")
        final_recorded = isinstance(final_value, str) and bool(final_value)
        final_hash = None
        turns = info.get("turns") if isinstance(info.get("turns"), list) else []
        for raw_turn in turns:
            if not isinstance(raw_turn, dict) or raw_turn.get("canvas") != final_value:
                continue
            if isinstance(raw_turn.get("canvas_sha256"), str):
                final_hash = raw_turn["canvas_sha256"]
                break
        final_path = _resolve_canvas(
            final_value, root=root, source=source, case_id=case_id,
            expected_sha256=final_hash, native_resolve=resolve_existing,
        )
        fallback = None
        last_action_valid = turns[-1].get("valid") if turns and isinstance(turns[-1], dict) else None
        context = trace.get("_episode_context") if isinstance(trace.get("_episode_context"), dict) else {}
        trace_errors = bool(trace.get("errors") or context.get("errors") or info.get("service_failure"))
        # A bounded native evaluator commonly closes with user_closed and an
        # operational cutoff while retaining a valid final paint.  That is a
        # valid terminal canvas when the final action itself did not fail.
        terminal_valid = (
            bool(info.get("final_canvas_valid"))
            and final_path is not None
            and not trace_errors
            and last_action_valid is not False
        )
        for raw in reversed(turns):
            if not isinstance(raw, dict) or raw.get("valid") is not True:
                continue
            fallback_hash = raw.get("canvas_sha256") if isinstance(raw.get("canvas_sha256"), str) else None
            fallback = _resolve_canvas(
                raw.get("canvas") or raw.get("final_canvas"), root=root, source=source,
                case_id=case_id, expected_sha256=fallback_hash, native_resolve=resolve_existing,
            )
            if fallback is not None:
                break
        if terminal_valid:
            status = "valid_terminal"
            selected = final_path
        elif last_action_valid is False:
            status = "censored_invalid_terminal_fallback_prior" if fallback else "censored_invalid_terminal"
            selected = None
        elif trace_errors:
            status = "error_terminal_fallback_prior" if fallback else "error_terminal"
            selected = None
        elif final_recorded:
            prefix = "incomplete_terminal" if not info.get("finished") else "invalid_terminal"
            status = f"{prefix}_fallback_prior" if fallback else prefix
            selected = None
        else:
            status = "terminal_not_recorded_fallback_prior" if fallback else "terminal_not_recorded"
            selected = None
        if case_id in result:
            result[case_id] = {
                "path": None, "status": "censored_duplicate_trace", "terminal_recorded": True,
                "terminal_valid": False, "fallback_path": None, "trace_source": "multiple_native_traces",
            }
            continue
        result[case_id] = {
            "path": str(selected) if selected else None,
            "status": status,
            "terminal_recorded": final_recorded,
            "terminal_valid": terminal_valid,
            "fallback_path": str(fallback) if fallback else None,
            "trace_source": str(source),
            "finished": bool(info.get("finished")),
            "operational_cutoff": bool(info.get("operational_cutoff") or info.get("censored")),
            "last_action_valid": last_action_valid,
            "completed_at_cap": bool(info.get("operational_cutoff") or info.get("censored")) and terminal_valid,
            "finish_turn": info.get("finish_turn"),
            "turn_count": len(turns),
            "stop_condition": trace.get("stop_condition"),
        }
    return result


def _receipt_cells(root: Path, manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fallback for older collected evidence without traces.jsonl.

    This labels the result as receipt-scan evidence and never infers terminal
    action validity from a repeated rendering receipt.
    """
    result: dict[str, dict[str, Any]] = {}
    if not root.is_dir() or root.is_symlink():
        return result
    for case in manifest:
        case_id = case["id"]
        receipts: list[tuple[int, Path, Path | None, str]] = []
        for receipt in sorted(root.rglob("painting.json")):
            if receipt.is_symlink() or case_id not in receipt.parts:
                continue
            turn = -1
            for part in receipt.parts:
                match = re.fullmatch(r"turn-([0-9]+)", part)
                if match:
                    turn = int(match.group(1))
            png, status = _valid_png(receipt)
            receipts.append((turn, receipt, png, status))
        if not receipts:
            result[case_id] = {
                "path": None, "status": "case_missing", "terminal_valid": False,
                "terminal_recorded": False, "fallback_path": None, "trace_source": "receipt_scan_missing",
            }
            continue
        _, receipt, png, status = max(receipts, key=lambda item: (item[0], item[1].as_posix()))
        valid = png is not None and status == "valid"
        result[case_id] = {
            # Keep a hash-verified receipt image for visual inspection, but
            # never call it a valid terminal without native final trace data.
            "path": str(png) if valid else None,
            "status": "receipt_scan_valid_unverified_terminal" if valid else f"invalid_terminal_receipt_{status}",
            "terminal_valid": False, "terminal_recorded": True, "fallback_path": None,
            "receipt": str(receipt), "trace_source": "receipt_scan",
        }
    return result


def find_canvas(root: Path, case_id: str) -> tuple[Path | None, str, Path | None]:
    """Compatibility helper returning only a strict terminal canvas."""
    manifest = [{"id": case_id, "source_group": case_id}]
    cell = _trace_cells(root, manifest).get(case_id)
    if cell is None:
        cell = _receipt_cells(root, manifest).get(case_id)
    path = Path(cell["path"]) if cell and cell.get("terminal_valid") and cell.get("path") else None
    receipt = Path(cell["receipt"]) if cell and cell.get("receipt") else None
    return path, str(cell.get("status", "case_missing")) if cell else "case_missing", receipt


def _manifest_rows(manifest: Path) -> list[dict[str, Any]]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("matched evaluation manifest must contain exactly 16 cases")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("matched evaluation rows must be objects")
        result.append({**row, "id": _safe_case_id(row.get("id"))})
    if len({row["id"] for row in result}) != 16:
        raise ValueError("matched evaluation case IDs must be unique")
    return result


def collect(*, manifest: Path, baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    cases = _manifest_rows(manifest)
    baseline = _trace_cells(baseline_root, cases)
    candidate = _trace_cells(candidate_root, cases)
    baseline.update({key: value for key, value in _receipt_cells(baseline_root, cases).items() if key not in baseline})
    candidate.update({key: value for key, value in _receipt_cells(candidate_root, cases).items() if key not in candidate})
    rows = []
    for row in cases:
        case_id = row["id"]
        anchor_value = row.get("opponent")
        expected_anchor = row.get("opponent_sha256")
        anchor = _resolve_manifest_file(
            anchor_value, manifest=manifest,
            expected_sha256=expected_anchor if isinstance(expected_anchor, str) else None,
        )
        flash = baseline.get(case_id, {"path": None, "status": "case_missing", "terminal_valid": False, "terminal_recorded": False})
        cand = candidate.get(case_id, {"path": None, "status": "case_missing", "terminal_valid": False, "terminal_recorded": False})
        rows.append({
            "id": case_id,
            "family": row.get("family"),
            "source_group": row.get("source_group"),
            "brief": _brief(row, manifest),
            "flash12": {**flash, "path": flash.get("path")},
            "sft512_anchor": {"path": str(anchor) if anchor else None, "status": "valid" if anchor else "missing_or_hash_mismatch", "terminal_valid": bool(anchor)},
            "checkpoint4": {**cand, "path": cand.get("path")},
        })
    return {
        "schema": "structured-flow-evaluation-recovery-review-v2",
        "run_name": RUN_NAME,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "seed": DEFAULT_SEED,
        "max_turns": DEFAULT_TURNS,
        "policies": ["flash12", "sft512_anchor", "checkpoint4"],
        "rows": rows,
        "failed_or_missing": sum(1 for row in rows if any(not row[key].get("terminal_valid", False) for key in ("flash12", "checkpoint4"))),
    }


def _copy_assets(review: dict[str, Any], output: Path) -> None:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for row in review["rows"]:
        for policy in ("flash12", "sft512_anchor", "checkpoint4"):
            data = row[policy]
            source = Path(data["path"]) if data.get("path") else None
            if source is None or not source.is_file() or source.is_symlink():
                data["uri"] = ""
                continue
            target = assets / f"{policy}-{row['id'].replace(':', '_')}.png"
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError(f"portable asset collision: {target}")
            if not target.exists():
                shutil.copyfile(source, target)
            data["uri"] = f"assets/{target.name}"
            data["asset_sha256"] = sha256(target)


def build_html(review: dict[str, Any]) -> str:
    payload = json.dumps(review, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structured-flow checkpoint 4 recovery</title><style>
body{font:15px/1.45 system-ui,sans-serif;background:#f2efe8;color:#27251f;margin:24px}main{max-width:1500px;margin:auto}header{display:flex;justify-content:space-between;gap:12px;align-items:end;flex-wrap:wrap}button{font:inherit;padding:8px 14px;border:1px solid #777;border-radius:6px;background:#fff;cursor:pointer}.meta{color:#5a554c}.brief{background:#fff;padding:14px;border-radius:8px;white-space:pre-wrap}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.grid figure{background:#fff;padding:10px;margin:0;border-radius:8px}figure img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eee}figcaption{font-weight:600;margin-bottom:8px}.valid_terminal,.valid{color:#176d3c}.missing,.receipt_scan_valid_unverified_terminal,.invalid_terminal,.invalid_terminal_fallback_prior,.incomplete_terminal,.incomplete_terminal_fallback_prior,.censored_invalid_terminal,.censored_invalid_terminal_fallback_prior,.error_terminal,.error_terminal_fallback_prior,.terminal_not_recorded,.terminal_not_recorded_fallback_prior,.case_missing,.censored_duplicate_trace{color:#9a251c}.missing{height:240px;display:grid;place-items:center;background:#eee}.controls{display:flex;gap:8px;align-items:center}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><main><header><div><h1>Checkpoint 4 matched recovery</h1><div class="meta" id="counter"></div></div><div class="controls"><button id="prev">Previous</button><button id="next">Next</button></div></header>
<p class="meta">Flash12 is the matched comparison policy. SFT512 is shown as the fixed reward anchor. Terminal invalid and censored canvases remain visible as status; prior-turn fallbacks are inspection-only.</p><h2 id="title"></h2><p id="brief" class="brief"></p><section id="grid" class="grid"></section></main><script>
const review=__REVIEW__;let index=0;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function cell(label,data){return `<figure><figcaption>${esc(label)} · <span class="${esc(data.status)}">${esc(data.status)}</span>${data.fallback_path?' · prior fallback retained':''}</figcaption>${data.uri?`<img src="${esc(data.uri)}" alt="${esc(label)}">`:'<div class="missing">No valid terminal canvas</div>'}</figure>`}function render(){const row=review.rows[index];document.querySelector('#counter').textContent=`Case ${index+1} of ${review.rows.length} · seed ${review.seed} · ${review.max_turns} turns`;document.querySelector('#title').textContent=`${row.id}${row.family?' · '+row.family:''}`;document.querySelector('#brief').textContent=row.brief||'(brief unavailable)';document.querySelector('#grid').innerHTML=cell('Flash12',row.flash12)+cell('SFT512 anchor',row.sft512_anchor)+cell('Checkpoint4',row.checkpoint4)}document.querySelector('#prev').onclick=()=>{index=(index+review.rows.length-1)%review.rows.length;render()};document.querySelector('#next').onclick=()=>{index=(index+1)%review.rows.length;render()};document.onkeydown=e=>{if(e.key==='ArrowLeft')document.querySelector('#prev').click();if(e.key==='ArrowRight')document.querySelector('#next').click()};render();
</script>'''
    return template.replace("__REVIEW__", payload)


def build(*, manifest: Path, baseline_root: Path, candidate_root: Path, output: Path) -> dict[str, Any]:
    review = collect(manifest=manifest, baseline_root=baseline_root, candidate_root=candidate_root)
    output.mkdir(parents=True, exist_ok=True)
    _copy_assets(review, output)
    (output / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "index.html").write_text(build_html(review), encoding="utf-8")
    return review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    review = build(**vars(args))
    print(json.dumps({"schema": review["schema"], "cases": len(review["rows"]), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
