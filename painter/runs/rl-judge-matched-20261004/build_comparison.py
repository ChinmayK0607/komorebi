#!/usr/bin/env python3
"""Build a local, read-only four-arm terminal-evidence gallery.

The completed archives are expected below ``local-evidence/completed``.  The
builder only reads cases, trace records, receipts, and rendered PNGs.  It
never renders, calls a model, scores a painting, or chooses a result using a
visual or reward signal.  The output HTML keeps image links relative to the
completed input roots, so the evidence remains inspectable in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


RUN = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = RUN / "local-evidence" / "completed"
DEFAULT_OUTPUT = RUN / "local-evidence" / "comparison" / "index.html"
CASE_IDS = (
    "t600-512",
    "t600-516",
    "t600-523",
    "t600-534",
    "t600-423",
    "t600-425",
    "t600-426",
    "t600-433",
    "t600-309",
    "t600-316",
    "t600-323",
    "t600-332",
    "t600-339",
    "t600-341",
    "t600-342",
    "t600-349",
)
ARM_SPECS = (
    ("baseline", "Starting checkpoint (RL16)", 1),
    ("flash", "RL • Flash judge • 12 updates", 12),
    ("pro", "RL • Pro judge • 12 updates", 12),
    ("luna", "RL • Luna judge • 12 updates", 12),
)
ARM_IDS = tuple(spec[0] for spec in ARM_SPECS)
PAIR_OPTIONS = tuple(
    (ARM_IDS[left], ARM_IDS[right])
    for left in range(len(ARM_IDS))
    for right in range(left + 1, len(ARM_IDS))
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_case_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe case id: {value!r}")
    return value


def _relative_href(source: Path, output: Path) -> str:
    return Path(os.path.relpath(source.resolve(), output.resolve().parent)).as_posix()


def _step_root(arm_root: Path, step: int) -> Path:
    root = arm_root / "validation" / f"step-{step}-balanced16"
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing completed validation root: {root}")
    return root


def _case_list(path: Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    raw = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(raw, list):
        raise ValueError(f"cases file has no cases list: {path}")
    cases: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"case is not an object: {path}")
        case = dict(item)
        case["id"] = _safe_case_id(case.get("id") or case.get("case_id"))
        cases.append(case)
    return cases


def _find_traces(eval_root: Path) -> Path | None:
    paths = sorted(path for path in eval_root.rglob("traces.jsonl") if path.is_file() and not path.is_symlink())
    if not paths:
        return None
    if len(paths) > 1:
        raise ValueError(f"multiple trace files under {eval_root}: {paths}")
    return paths[0]


def _trace_info(eval_root: Path) -> dict[str, dict[str, Any]]:
    path = _find_traces(eval_root)
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trace JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            continue
        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
        traces = row.get("traces") if isinstance(row.get("traces"), list) else []
        trace = traces[-1] if traces and isinstance(traces[-1], dict) else {}
        info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
        case_id = task_data.get("name") or info.get("task_id") or task_data.get("task_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        turns = info.get("turns") if isinstance(info.get("turns"), list) else []
        normalized = [dict(turn) for turn in turns if isinstance(turn, dict) and isinstance(turn.get("turn"), int)]
        normalized.sort(key=lambda turn: turn["turn"])
        result[case_id] = {
            "row": row,
            "info": info,
            "turns": normalized,
            "trace_path": str(path),
        }
    return result


def _find_case_episode(eval_root: Path, case_id: str) -> Path | None:
    matches = sorted(
        path
        for path in eval_root.rglob(case_id)
        if path.is_dir() and not path.is_symlink() and path.name == case_id
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"ambiguous evaluation case root for {case_id}: {matches}")
    case_root = matches[0]
    children = sorted(path for path in case_root.iterdir() if path.is_dir() and not path.is_symlink())
    # Native evaluation stores one UUID episode below the case id.  Accept a
    # direct case root too, which keeps synthetic tests and small audits easy.
    if len(children) == 1 and any(child.name.startswith("turn-") for child in children[0].iterdir()):
        return children[0]
    return case_root


def _turn_number(path: Path) -> int | None:
    match = re.fullmatch(r"turn-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _turn_dirs(episode: Path | None) -> list[tuple[int, Path]]:
    if episode is None or not episode.is_dir():
        return []
    result = []
    for path in episode.iterdir():
        number = _turn_number(path) if path.is_dir() and not path.is_symlink() else None
        if number is not None and number > 0:
            result.append((number, path))
    return sorted(result)


def _read_receipt(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return None, "missing painting receipt"
    try:
        value = _load_json(path)
    except ValueError:
        return None, "malformed painting receipt"
    if not isinstance(value, dict):
        return None, "painting receipt is not an object"
    return value, None


def _errors(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _record_turn(turn: int, turn_root: Path, *, output: Path) -> dict[str, Any]:
    receipt_path = turn_root / "painting.json"
    png_path = turn_root / "painting.png"
    receipt, receipt_error = _read_receipt(receipt_path)
    png_present = png_path.is_file() and not png_path.is_symlink()
    actual_hash = _sha256(png_path) if png_present else None
    expected_hash = receipt.get("png_sha256") if receipt else None
    hash_verified = bool(png_present and isinstance(expected_hash, str) and actual_hash == expected_hash)
    errors = list(_errors(receipt.get("errors") if receipt else None))
    if receipt_error:
        errors.insert(0, receipt_error)
    if png_present and not hash_verified:
        errors.append("PNG hash is missing or does not match painting.json")
    return {
        "turn": turn,
        "receipt_valid": receipt.get("valid") is True if receipt else None,
        "finished": receipt.get("finished") if receipt and isinstance(receipt.get("finished"), bool) else None,
        "png": _relative_href(png_path, output) if png_present else None,
        "png_source": str(png_path) if png_present else None,
        "receipt_source": str(receipt_path) if receipt else None,
        "png_hash_verified": hash_verified,
        "errors": errors,
        "receipt": _relative_href(receipt_path, output) if receipt else None,
    }


def _trace_final(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    turns = trace.get("turns") if isinstance(trace.get("turns"), list) else []
    return dict(turns[-1]) if turns else None


def _native_trace(episode: Path | None) -> dict[str, Any] | None:
    """Read one archived episode's native trace when the collector provided it."""
    if episode is None:
        return None
    for name in ("native_trace.json", "trace.json"):
        path = episode / name
        if not path.is_file() or path.is_symlink():
            continue
        value = _load_json(path)
        if not isinstance(value, dict):
            continue
        nested = value.get("trace") if isinstance(value.get("trace"), dict) else value
        info = nested.get("info") if isinstance(nested.get("info"), dict) else nested
        if not isinstance(info, dict):
            continue
        turns = info.get("turns") if isinstance(info.get("turns"), list) else nested.get("turns", [])
        normalized = [dict(turn) for turn in turns if isinstance(turn, dict) and isinstance(turn.get("turn"), int)]
        normalized.sort(key=lambda turn: turn["turn"])
        return {"row": nested, "info": info, "turns": normalized, "trace_path": str(path)}
    return None


def _terminal_image_from_canvas(canvas: Any, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(canvas, str) or not canvas:
        return None
    # The archive may preserve the Linux absolute path.  Its final turn
    # component is enough to bind to the local copied record without trusting
    # an arbitrary path from a trace as a filesystem location.
    match = re.search(r"(?:^|/)turn-(\d+)(?:/|$)", canvas)
    if not match:
        return None
    number = int(match.group(1))
    return next((record for record in records if record["turn"] == number), None)


def _case_result(
    case: dict[str, Any],
    *,
    arm_id: str,
    step_root: Path,
    output: Path,
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = case["id"]
    eval_root = step_root / "eval" / "evaluation-rollouts"
    episode = _find_case_episode(eval_root, case_id) if eval_root.is_dir() else None
    records = [
        _record_turn(number, turn_root, output=output)
        for number, turn_root in _turn_dirs(episode)
    ]
    trace = _native_trace(episode) or traces.get(case_id)
    trace_final = _trace_final(trace)
    if trace is not None and trace_final is None:
        # A copied trace with no recorded turns is still authoritative: it
        # cannot promote the newest local painting by inference.
        terminal_turn = None
        terminal_valid = None
        selected = None
        image_record = None
        terminal_error = ["trace has no recorded terminal turn"]
        status = "CENSORED TERMINAL"
    elif trace_final is not None:
        terminal_turn = trace_final.get("turn")
        terminal_valid = trace_final.get("valid") if isinstance(trace_final.get("valid"), bool) else None
        terminal_error = _errors(trace_final.get("error")) + _errors(trace_final.get("renderer_errors"))
        selected = next((record for record in records if record["turn"] == terminal_turn), None)
        retained = _terminal_image_from_canvas(trace_final.get("canvas"), records)
        # A valid no-change terminal explicitly retained the canvas.  This is
        # terminal evidence, while an invalid/failed terminal never promotes a
        # prior turn.
        image_record = selected if selected and selected["png_hash_verified"] and selected["receipt_valid"] else None
        if image_record is None and terminal_valid is True:
            if retained and retained["png_hash_verified"] and retained["receipt_valid"]:
                image_record = retained
        if selected:
            terminal_error.extend(selected["errors"])
        if terminal_valid is False:
            image_record = None
            status = "INVALID TERMINAL"
        elif terminal_valid is True and image_record:
            status = "VALID TERMINAL"
        elif terminal_valid is True:
            image_record = None
            status = "CENSORED TERMINAL"
            terminal_error.append("terminal action was valid but no verified terminal PNG is available")
        else:
            image_record = None
            status = "CENSORED TERMINAL"
            terminal_error.append("terminal validity is unavailable in the final trace")
    else:
        # Render receipts alone cannot establish the episode's authoritative
        # terminal action.  A later invalid action may have retained an
        # earlier valid canvas, so fail closed when the trace is absent.
        terminal_turn = records[-1]["turn"] if records else None
        selected = records[-1] if records else None
        terminal_valid = None
        image_record = None
        terminal_error = list(selected["errors"]) if selected else []
        if selected is None:
            status = "MISSING TERMINAL"
            terminal_error.append("no terminal turn evidence is available")
        else:
            status = "CENSORED TERMINAL"
            terminal_error.append("authoritative terminal trace is unavailable")

    # Preserve the trace's final turn identity even when a valid no-change
    # action points at a retained image from an earlier rendered turn.
    image_href = image_record["png"] if image_record else None
    return {
        "case_id": case_id,
        "arm_id": arm_id,
        "status": status,
        "terminal_turn": terminal_turn,
        "terminal_valid": terminal_valid,
        "image": image_href,
        "image_turn": image_record["turn"] if image_record else None,
        "receipt": selected["receipt"] if selected else None,
        "error": "; ".join(dict.fromkeys(error for error in terminal_error if error)) or None,
        "turns": records,
    }


def _prompt_from_trace(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    value = info.get("prompt")
    row = trace.get("row") if isinstance(trace.get("row"), dict) else {}
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    if value is None:
        value = task_data.get("prompt")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        texts: list[str] = []
        for message in value:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
            elif isinstance(content, list):
                texts.extend(
                    item.get("text", "").strip()
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
                )
        for text in texts:
            if "PAINTING BRIEF" in text:
                return text
        return "\n\n".join(texts) if texts else None
    return None


def _prompt(case: dict[str, Any], *, cases_path: Path, trace: dict[str, Any] | None) -> str | None:
    for key in ("prompt_text", "brief_text", "prompt", "brief"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    reference = case.get("reference") or case.get("reference_path") or case.get("brief_path")
    if isinstance(reference, str) and reference:
        candidate = Path(reference)
        if not candidate.is_absolute():
            candidate = cases_path.parent / candidate
        if candidate.is_file() and not candidate.is_symlink():
            try:
                text = candidate.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                text = ""
            if text:
                return text
    return _prompt_from_trace(trace)


def _page(data: dict[str, Any]) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three judges, same RL setup</title>
<style>
:root{{--ink:#1c2731;--muted:#63717d;--line:#d6dee5;--paper:#f4f6f8;--card:#fff;--good:#176b4c;--bad:#a62828;--warn:#8b5a15}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1700px;margin:auto;padding:28px 22px 55px}}h1{{font-size:clamp(1.8rem,3.5vw,2.7rem);line-height:1.05;letter-spacing:-.035em;margin:0 0 8px}}h2,p{{margin-top:0}}.lede{{max-width:1050px;color:#3c4b57}}.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0}}label{{display:grid;gap:4px;color:var(--muted);font-size:.78rem;font-weight:700}}select{{min-width:260px;padding:8px;border:1px solid #bbc7d1;border-radius:6px;background:#fff;color:var(--ink);font:inherit}}.prompt{{white-space:pre-wrap;background:#f8fafb;border-left:4px solid #8da7bf;padding:11px 13px;margin:10px 0 16px;color:#33434f}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(300px,1fr));gap:16px}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;min-width:0}}.panel h3{{font-size:1rem;margin:0;padding:12px 14px;background:#edf2f6;display:flex;justify-content:space-between;gap:12px;align-items:baseline;min-height:72px}}.panel h3 small{{font-size:.75rem;font-weight:600;color:var(--muted);white-space:nowrap}}.image-wrap{{background:#e9eef2;display:grid;place-items:center;min-height:300px}}.image-wrap a{{display:block;width:100%;min-width:0}}.image-wrap img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#fff;cursor:zoom-in}}.missing{{min-height:300px;padding:24px;display:grid;place-content:center;text-align:center;background:#f7f1ec;color:#6f453b}}.missing b{{font-size:1.05rem}}.meta{{padding:11px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:.84rem}}.status{{font-weight:800;letter-spacing:.03em}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.warn{{color:var(--warn)}}.error{{margin-top:5px;overflow-wrap:anywhere}}.foot{{color:var(--muted);font-size:.82rem;margin-top:20px}}code{{font-size:.8em;overflow-wrap:anywhere}}@media(max-width:1320px){{.grid{{grid-template-columns:repeat(2,minmax(300px,1fr))}}}}@media(max-width:680px){{main{{padding:20px 11px 42px}}.grid{{grid-template-columns:1fr}}.image-wrap,.missing{{min-height:260px}}select{{min-width:0;width:100%}}.controls label{{width:100%}}}}
</style></head><body><main><h1>Three judges, same RL setup</h1>
<p class="lede">The same 16 prompts, starting checkpoint, and 12 updates per judge. View all four results at full size in a fixed order: RL16, Flash, Pro, Luna.</p>
<div class="controls"><label>Case<select id="case"></select></label></div>
<div id="case-content"></div><p class="foot">Saved results from the completed experiment. Checkpoint labels stay fixed.</p></main>
<script>const DATA={blob};
const caseSelect=document.getElementById('case'),content=document.getElementById('case-content');
const esc=(v)=>String(v??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const arm=(id)=>DATA.arms.find(item=>item.id===id); const result=(id,caseId)=>DATA.results.find(item=>item.case_id===caseId&&item.arm_id===id);
function statusClass(status){{return status.startsWith('VALID')?'good':status.startsWith('INVALID')?'bad':'warn'}}
function panel(row,position){{const image=row&&row.image?`<div class="image-wrap"><a href="${{esc(row.image)}}" target="_blank" rel="noopener"><img src="${{esc(row.image)}}" alt="${{esc(arm(row?.arm_id||'')?.label||position)}} terminal painting" loading="lazy"></a></div>`:`<div class="missing"><b>${{esc(row?.status||'MISSING TERMINAL')}}</b><span>Missing, invalid, or censored final actions remain unshown.</span></div>`;return `<section class="panel"><h3>${{esc(arm(row?.arm_id||'')?.label||position)}}<small>${{esc(position)}} · terminal turn ${{esc(row?.terminal_turn)}}</small></h3>${{image}}<div class="meta"><span class="status ${{statusClass(row?.status||'MISSING TERMINAL')}}">${{esc(row?.status||'MISSING TERMINAL')}}</span>${{row?.image_turn&&row.image_turn!==row.terminal_turn?`<div>image retained from rendered turn ${{esc(row.image_turn)}} by valid terminal trace</div>`:''}}${{row?.error?`<div class="error">${{esc(row.error)}}</div>`:''}}</div></section>`}}
function render(){{const c=DATA.cases.find(item=>item.id===caseSelect.value);const rows=DATA.arms.map(item=>result(item.id,c.id));content.innerHTML=`<h2>${{esc(c.id)}} · ${{esc(c.index)}} / ${{esc(DATA.cases.length)}}</h2><div class="prompt">${{esc(c.prompt||'Prompt unavailable in archived cases.')}}</div><div class="grid">${{rows.map((row,index)=>panel(row,index+1)).join('')}}</div>`}}
DATA.cases.forEach((item,index)=>caseSelect.add(new Option(`${{item.id}} · ${{index+1}}/${{DATA.cases.length}}`,item.id)));
caseSelect.value=DATA.cases[0].id;caseSelect.addEventListener('change',render);render();</script></body></html>'''


def build(*, input_root: Path = DEFAULT_INPUT_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arm_meta: list[dict[str, Any]] = []
    cases_by_arm: dict[str, list[dict[str, Any]]] = {}
    traces_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    roots_by_arm: dict[str, Path] = {}
    for arm_id, label, step in ARM_SPECS:
        arm_root = input_root / arm_id
        step_root = _step_root(arm_root, step)
        cases_path = step_root / "cases.json"
        cases = _case_list(cases_path)
        if tuple(case["id"] for case in cases) != CASE_IDS:
            raise ValueError(f"{arm_id} cases must preserve the exact 16-case order")
        cases_by_arm[arm_id] = cases
        traces_by_arm[arm_id] = _trace_info(step_root / "eval") if (step_root / "eval").is_dir() else {}
        roots_by_arm[arm_id] = step_root
        arm_meta.append({"id": arm_id, "label": label, "step": step})
    if any(cases_by_arm[arm_id] != cases_by_arm[ARM_IDS[0]] for arm_id in ARM_IDS[1:]):
        raise ValueError("arm cases do not share the exact ordered matched panel")

    cases: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    baseline_cases = cases_by_arm[ARM_IDS[0]]
    for index, case in enumerate(baseline_cases, 1):
        case_id = case["id"]
        prompt = _prompt(case, cases_path=roots_by_arm[ARM_IDS[0]] / "cases.json", trace=traces_by_arm[ARM_IDS[0]].get(case_id))
        cases.append({"id": case_id, "index": index, "prompt": prompt})
        for arm_id, _, _ in ARM_SPECS:
            results.append(
                _case_result(
                    cases_by_arm[arm_id][index - 1],
                    arm_id=arm_id,
                    step_root=roots_by_arm[arm_id],
                    output=output,
                    traces=traces_by_arm[arm_id],
                )
            )
    data = {"arms": arm_meta, "cases": cases, "pairs": [list(pair) for pair in PAIR_OPTIONS], "results": results}
    output.write_text(_page(data), encoding="utf-8")
    return {"output": str(output), "cases": cases, "results": results, "arms": arm_meta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(input_root=args.input_root, output=args.output)
    counts: dict[str, int] = {}
    for row in result["results"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"output": result["output"], "case_count": len(result["cases"]), "status_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
