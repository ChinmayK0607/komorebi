"""Build a reference/SFT/RL gallery from preserved native painting evidence.

The evaluator and renderer are deliberately outside this module.  This command
only reads a reference manifest, native verifier traces, and existing PNG/JS
artifacts.  It writes a small JSON summary and an HTML page whose image tags
point at files that already exist.

The command accepts both the bare ``vf eval`` ``traces.jsonl`` format (one
trace per line) and Prime's episode stream (an episode with ``traces``).  Paths
written by a Linux node commonly start with ``/root/painter``; those paths are
resolved against ``--evidence-root`` when the node workspace was collected.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


REPORT_SCHEMA = "native-painting-report-v1"
DEFAULT_NODE_ROOT = "/root/painter"
POLICIES = ("sft", "rl")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _json_rows(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _json_rows(path)
    if isinstance(raw, list):
        cases = raw
        metadata: dict[str, Any] = {}
    elif isinstance(raw, dict):
        metadata = raw
        cases = raw.get("cases") or raw.get("eval") or raw.get("tasks") or []
    else:
        raise ValueError("reference manifest must be a JSON list or object with cases")
    if not isinstance(cases, list) or not all(isinstance(row, dict) for row in cases):
        raise ValueError("reference manifest cases must be objects")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        case = dict(row)
        case_id = _text(case.get("id") or case.get("task_id") or case.get("case_id"))
        family = _text(case.get("family")) or case_id or f"case-{index + 1}"
        source_group = _text(case.get("source_group") or case.get("group")) or case_id
        if case_id is None:
            case_id = source_group or f"case-{index + 1}"
        case["id"] = case_id
        case["family"] = family
        case["source_group"] = source_group or case_id
        normalized.append(case)
    return metadata, normalized


def _candidate_paths(
    value: Any,
    *,
    evidence_root: Path,
    base_dir: Path | None = None,
    node_root: str = DEFAULT_NODE_ROOT,
) -> Iterator[Path]:
    """Yield local candidates for a path from a controller or Linux node.

    The first existing file is selected by ``resolve_existing``.  Keeping the
    candidates separate makes it possible to report a missing artifact without
    fabricating a link in the gallery.
    """

    if not isinstance(value, (str, os.PathLike)):
        return
    raw = str(value)
    if not raw or raw.startswith("data:"):
        return
    path = Path(raw)
    seen: set[Path] = set()

    def emit(candidate: Path) -> Iterator[Path]:
        candidate = candidate.expanduser()
        if candidate in seen:
            return
        seen.add(candidate)
        yield candidate

    if path.is_absolute():
        yield from emit(path)
        node = Path(node_root)
        try:
            relative = path.relative_to(node)
        except ValueError:
            relative = None
        if relative is not None:
            yield from emit(evidence_root / relative)
        # A collected bundle sometimes retains one extra evidence/ directory.
        if relative is not None:
            yield from emit(evidence_root / "evidence" / relative)
    else:
        if base_dir is not None:
            yield from emit(base_dir / path)
        yield from emit(evidence_root / path)
        yield from emit(evidence_root / "evidence" / path)

def resolve_existing(
    value: Any,
    *,
    evidence_root: Path,
    base_dir: Path | None = None,
    node_root: str = DEFAULT_NODE_ROOT,
) -> Path | None:
    for candidate in _candidate_paths(
        value,
        evidence_root=evidence_root,
        base_dir=base_dir,
        node_root=node_root,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _reference_path(
    case: dict[str, Any], *, manifest_path: Path, evidence_root: Path, node_root: str
) -> Path | None:
    value = case.get("reference_png") or case.get("reference") or case.get("reference_path")
    # Prefer the collected eval-assets copy when available.  The controller's
    # source manifest often resolves to a sibling run that is not present when
    # the report is moved with the collected evidence bundle.
    case_id = case.get("id")
    if case_id:
        for candidate in (
            evidence_root / "eval-assets" / f"{case_id}.png",
            evidence_root / "evidence" / "eval-assets" / f"{case_id}.png",
        ):
            if candidate.is_file():
                return candidate.resolve()
    return resolve_existing(
        value,
        evidence_root=evidence_root,
        base_dir=manifest_path.parent,
        node_root=node_root,
    )


def _trace_files(root: Path) -> list[Path]:
    """Find native trace streams without treating every evidence JSONL as one."""

    patterns = ("traces.jsonl", "traces.jsonl.zst", "*.traces.jsonl", "*.traces.jsonl.zst")
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                found.add(path.resolve())
    # A monitor stream has chunk names rather than traces.jsonl.  Include it
    # only below a directory literally named traces/stream to avoid manifests.
    for path in root.rglob("*.jsonl"):
        if path.is_file() and "traces" in path.parent.parts and "stream" in path.parent.parts:
            found.add(path.resolve())
    for path in root.rglob("*.jsonl.zst"):
        if path.is_file() and "traces" in path.parent.parts and "stream" in path.parent.parts:
            found.add(path.resolve())
    return sorted(found)


def _open_jsonl(path: Path) -> Iterable[str]:
    if path.suffix != ".zst":
        return path.open(encoding="utf-8")
    try:
        import zstandard as zstd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"cannot read compressed trace stream without zstandard: {path}"
        ) from exc
    stream = path.open("rb")
    decompressor = zstd.ZstdDecompressor()
    reader = decompressor.stream_reader(stream)
    import io

    return io.TextIOWrapper(reader, encoding="utf-8")


def _iter_trace_objects(record: Any) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(trace, episode_context)`` from bare and episode-wrapped records."""

    if isinstance(record, list):
        for item in record:
            yield from _iter_trace_objects(item)
        return
    if not isinstance(record, dict):
        return
    nested = record.get("traces")
    if isinstance(nested, list):
        context = {
            key: record.get(key)
            for key in ("id", "ok", "errors", "env", "run", "group", "task")
            if key in record
        }
        if not nested:
            # Prime can emit an episode with no trace when setup/provider work
            # fails.  Retain that failure as a synthetic trace-shaped record so
            # the expected case/policy cell says ``error`` rather than looking
            # like an unexplained omission.
            pseudo: dict[str, Any] = {}
            if isinstance(context.get("task"), dict):
                pseudo["task"] = context["task"]
            if context.get("id"):
                pseudo["id"] = context["id"]
            if context.get("errors"):
                pseudo["errors"] = context["errors"]
            if pseudo:
                yield pseudo, context
            return
        for item in nested:
            if isinstance(item, dict):
                yield item, context
        return
    if isinstance(record.get("trace"), dict):
        yield record["trace"], {key: record.get(key) for key in ("id", "ok", "errors") if key in record}
        return
    # A native bare trace has task/info/calls, but accepting any object with
    # task+info makes this parser tolerant of future verifier wire additions.
    if isinstance(record.get("task"), dict) or isinstance(record.get("info"), dict):
        yield record, {}


def read_traces(root: Path) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    traces: list[dict[str, Any]] = []
    errors: list[str] = []
    files = _trace_files(root)
    for path in files:
        try:
            stream = _open_jsonl(path)
            with stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{path}:{line_no}: invalid JSON ({exc.msg})")
                        continue
                    for trace, context in _iter_trace_objects(record):
                        item = dict(trace)
                        item["_source_path"] = str(path)
                        item["_source_line"] = line_no
                        item["_episode_context"] = context
                        traces.append(item)
        except (OSError, RuntimeError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return traces, errors, files


def _trace_data(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    return data, info


def _case_identity(trace: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    data, info = _trace_data(trace)
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    candidates = (info, data, state, task)

    def first(*keys: str) -> str | None:
        for source in candidates:
            for key in keys:
                value = _text(source.get(key)) if isinstance(source, dict) else None
                if value:
                    return value
        return None

    case_id = first("task_id", "case_id", "source_group", "id", "name")
    source_group = first("source_group", "group", "task_id", "case_id") or case_id
    family = first("family")
    if case_id and ":" in case_id and source_group:
        # Some older painting tasks name correction states ``state:anchor``;
        # leave source_group as the stable identity when it is available.
        case_id = source_group
    return case_id, source_group, family


def _policy(trace: dict[str, Any], source_path: str, aliases: dict[str, str]) -> tuple[str, str]:
    data, info = _trace_data(trace)
    agent = trace.get("agent") if isinstance(trace.get("agent"), dict) else {}
    values = [
        info.get("policy_label"),
        info.get("policy"),
        data.get("policy_label"),
        data.get("policy"),
        agent.get("name"),
        source_path,
    ]
    raw = next((str(value) for value in values if isinstance(value, str) and value), "unknown")
    lowered = raw.lower()
    for alias, canonical in aliases.items():
        if alias.lower() in lowered:
            return canonical, raw
    return "unknown", raw


def _usage_value(usage: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _tokens(trace: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Use only provider usage embedded in native trace calls.

    ``token_ids`` and generated program lengths are intentionally ignored: they
    are not equivalent to the provider's actual prompt/completion accounting.
    """

    calls = _as_list(trace.get("calls"))
    per_turn: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    present = {key: False for key in totals}
    usage_calls = 0
    for index, call in enumerate(calls):
        call = call if isinstance(call, dict) else {}
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        prompt = _usage_value(usage, "prompt_tokens", "input_tokens")
        completion = _usage_value(usage, "completion_tokens", "output_tokens", "generated_tokens")
        total = _usage_value(usage, "total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        available = any(value is not None for value in (prompt, completion, total))
        if available:
            usage_calls += 1
        for key, value in (("prompt_tokens", prompt), ("completion_tokens", completion), ("total_tokens", total)):
            if value is not None:
                totals[key] += value
                present[key] = True
        per_turn.append(
            {
                "turn": index + 1,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "usage_available": available,
            }
        )
    # Keep turn rows visible when the trace has environment turns but no call
    # record (for example a provider error before a response was received).
    while len(per_turn) < len(turns):
        index = len(per_turn)
        per_turn.append(
            {
                "turn": index + 1,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "usage_available": False,
            }
        )
    if not calls and not turns:
        per_turn = []
    for key in totals:
        if not present[key]:
            totals[key] = None
    return {
        **totals,
        "reported_calls": len(calls),
        "calls_with_usage": usage_calls,
        "missing_usage_calls": max(0, len(calls) - usage_calls),
        "complete": bool(calls) and usage_calls == len(calls),
        "source": "trace.calls[].usage",
        "per_turn": per_turn,
    }


def _status(trace: dict[str, Any], info: dict[str, Any], turns: list[dict[str, Any]]) -> tuple[str, list[str]]:
    errors = _as_list(trace.get("errors"))
    context = trace.get("_episode_context")
    if isinstance(context, dict):
        errors = errors + _as_list(context.get("errors"))
        if context.get("ok") is False:
            errors.append("episode ok=false")
    if trace.get("ok") is False:
        errors.append("trace ok=false")
    service_failure = info.get("service_failure")
    operational_cutoff = bool(info.get("operational_cutoff") or info.get("censored"))
    stop = str(trace.get("stop_condition") or "")
    if stop in {"error", "timeout", "cancelled", "failed"} or errors or service_failure:
        status = "error"
    elif operational_cutoff or trace.get("is_truncated"):
        status = "censored"
    elif bool(info.get("finished")) and bool(info.get("final_canvas_valid")):
        status = "complete"
    elif bool(info.get("final_canvas_valid")):
        status = "incomplete"
    elif any(turn.get("failure_kind") == "service" for turn in turns):
        status = "error"
    else:
        status = "invalid"
    flags: list[str] = []
    if errors:
        flags.append("trace_errors")
    if service_failure:
        flags.append("service_failure")
    if operational_cutoff or trace.get("is_truncated"):
        flags.append("operational_cutoff")
    if info.get("invalid_finish_turns"):
        flags.append("invalid_finish")
    if not info.get("final_canvas_valid"):
        flags.append("no_valid_final_canvas")
    if info.get("finished"):
        flags.append("finished")
    if info.get("final_canvas_valid"):
        flags.append("final_canvas_valid")
    return status, flags


def _turn_artifacts(
    info: dict[str, Any], *, evidence_root: Path, trace_path: Path, node_root: str
) -> list[dict[str, Any]]:
    raw_turns = info.get("turns")
    if not isinstance(raw_turns, list):
        return []
    resolved_canvases: list[Path | None] = []
    episode_root: Path | None = None
    for raw in raw_turns:
        row = dict(raw) if isinstance(raw, dict) else {"reply": str(raw)}
        canvas = resolve_existing(
            row.get("canvas") or row.get("final_canvas"),
            evidence_root=evidence_root,
            base_dir=trace_path.parent,
            node_root=node_root,
        )
        resolved_canvases.append(canvas)
        if canvas is not None and canvas.parent.name.startswith("turn-"):
            episode_root = canvas.parent.parent
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_turns, 1):
        row = dict(raw) if isinstance(raw, dict) else {"reply": str(raw)}
        canvas_value = row.get("canvas") or row.get("final_canvas")
        canvas = resolved_canvases[index - 1]
        code_value = row.get("program_path") or row.get("code_path")
        code = resolve_existing(
            code_value,
            evidence_root=evidence_root,
            base_dir=trace_path.parent,
            node_root=node_root,
        )
        if code is None and canvas is not None and row.get("valid") is not False:
            sibling = canvas.parent / "program.js"
            if sibling.is_file():
                code = sibling.resolve()
        if code is None and episode_root is not None:
            # Invalid renderer turns preserve the previous canvas path.  Walk
            # to the episode directory and use this turn's own program.js so
            # a bad action never links to the previous turn's source.
            turn_number = row.get("turn", index)
            own_turn = episode_root / f"turn-{turn_number}" / "program.js"
            if own_turn.is_file():
                code = own_turn.resolve()
        turns.append(
            {
                "turn": row.get("turn", index),
                "action": row.get("action"),
                "valid": row.get("valid"),
                "failure_kind": row.get("failure_kind"),
                "error": row.get("error") or row.get("renderer_errors"),
                "canvas": str(canvas) if canvas else None,
                "code": str(code) if code else None,
                "reply": row.get("reply") if isinstance(row.get("reply"), str) else None,
                "no_change": row.get("no_change"),
                "render_seconds": row.get("render_seconds"),
            }
        )
    return turns


def _episode(
    trace: dict[str, Any],
    *,
    evidence_root: Path,
    aliases: dict[str, str],
    node_root: str,
    manifest_by_id: dict[str, dict[str, Any]],
    manifest_by_group: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data, info = _trace_data(trace)
    case_id, source_group, family = _case_identity(trace)
    case = manifest_by_id.get(case_id or "") or manifest_by_group.get(source_group or "")
    if case:
        case_id = case["id"]
        source_group = case.get("source_group") or source_group
        family = case.get("family") or family
    policy, policy_raw = _policy(trace, str(trace.get("_source_path", "")), aliases)
    source_path = Path(str(trace.get("_source_path", "")))
    turns = _turn_artifacts(info, evidence_root=evidence_root, trace_path=source_path, node_root=node_root)
    final_value = info.get("final_canvas")
    final_recorded = isinstance(final_value, str) and bool(final_value)
    final_canvas = resolve_existing(
        final_value,
        evidence_root=evidence_root,
        base_dir=source_path.parent,
        node_root=node_root,
    )
    if final_canvas is None and not final_recorded:
        for turn in reversed(turns):
            if turn.get("canvas") and turn.get("valid"):
                final_canvas = Path(turn["canvas"])
                break
    status, flags = _status(trace, info, turns)
    if final_recorded and final_canvas is None:
        flags.append("final_canvas_artifact_missing")
        if status == "complete":
            status = "artifact_missing"
    errors: list[str] = []
    for value in _as_list(trace.get("errors")):
        errors.append(str(value.get("message") if isinstance(value, dict) else value))
    context = trace.get("_episode_context")
    if isinstance(context, dict):
        for value in _as_list(context.get("errors")):
            message = value.get("message") if isinstance(value, dict) else value
            if str(message) not in errors:
                errors.append(str(message))
        if context.get("ok") is False and "episode ok=false" not in errors:
            errors.append("episode ok=false")
    if trace.get("ok") is False and "trace ok=false" not in errors:
        errors.append("trace ok=false")
    if info.get("service_failure"):
        errors.append(str(info["service_failure"]))
    for turn in turns:
        if turn.get("error"):
            errors.append(str(turn["error"]))
    return {
        "trace_id": trace.get("id"),
        "case_id": case_id,
        "source_group": source_group,
        "family": family,
        "policy": policy,
        "policy_raw": policy_raw,
        "status": status,
        "flags": flags,
        "mode": info.get("mode") or data.get("mode"),
        "finished": bool(info.get("finished")),
        "final_canvas_valid": bool(info.get("final_canvas_valid")),
        "final_canvas_recorded": final_recorded,
        "final_canvas_artifact_missing": final_recorded and final_canvas is None,
        "final_canvas": str(final_canvas) if final_canvas else None,
        "finish_turn": info.get("finish_turn"),
        "turn_cap": info.get("turn_cap") or data.get("max_turns"),
        "turns": turns,
        "tokens": _tokens(trace, turns),
        "errors": errors,
        "trace_source": str(source_path),
        "trace_line": trace.get("_source_line"),
        "reference_trace_path": info.get("reference") or data.get("reference"),
        "raw_stop_condition": trace.get("stop_condition"),
        "raw_ok": trace.get("ok"),
    }


def _aliases(sft_label: str, rl_label: str) -> dict[str, str]:
    return {
        sft_label: "sft",
        rl_label: "rl",
        "sft": "sft",
        "contract": "sft",
        "step_24": "sft",
        "step24": "sft",
        "rl": "rl",
        "aesthetic": "rl",
        "step_4": "rl",
        "step4": "rl",
    }


def _link(path_value: str | None, *, outdir: Path, label: str) -> str:
    if not path_value:
        return f'<span class="missing">{html.escape(label)} missing</span>'
    path = Path(path_value)
    if not path.is_file():
        return f'<span class="missing">{html.escape(label)} missing</span>'
    relative = os.path.relpath(path, outdir)
    return f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>'


def _image(path_value: str | None, *, outdir: Path, alt: str) -> str:
    if not path_value:
        return f'<span class="missing">{html.escape(alt)} missing</span>'
    path = Path(path_value)
    if not path.is_file() or path.suffix.lower() != ".png":
        return f'<span class="missing">{html.escape(alt)} missing</span>'
    relative = os.path.relpath(path, outdir)
    src = html.escape(relative, quote=True)
    return f'<a href="{src}"><img loading="lazy" src="{src}" alt="{html.escape(alt, quote=True)}"></a>'


def _token_label(tokens: dict[str, Any]) -> str:
    values = [tokens.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    if not any(value is not None for value in values):
        return "tokens unavailable"
    return "tokens " + ", ".join(
        f"{name}={value}" for name, value in (("in", values[0]), ("out", values[1]), ("total", values[2])) if value is not None
    )


def _episode_html(episode: dict[str, Any], *, outdir: Path) -> str:
    if not episode:
        return '<div class="missing">No native trace for this policy/case.</div>'
    bits = [
        f'<div class="episode status-{html.escape(str(episode["status"]))}">',
        f'<div><strong>{html.escape(str(episode["status"]).upper())}</strong> · {_token_label(episode["tokens"])} · {len(episode["turns"])} turns</div>',
    ]
    if episode.get("flags"):
        bits.append(f'<div class="flags">{html.escape(", ".join(episode["flags"]))}</div>')
    bits.append(_image(episode.get("final_canvas"), outdir=outdir, alt="final canvas"))
    for turn in episode["turns"]:
        turn_no = html.escape(str(turn.get("turn")))
        turn_bits = [f'<details><summary>turn {turn_no} · {html.escape(str(turn.get("action") or ""))}']
        token_row = next((row for row in episode["tokens"]["per_turn"] if row.get("turn") == turn.get("turn")), None)
        if token_row:
            turn_bits.append(f' · {_token_label(token_row)}')
        turn_bits.append("</summary>")
        turn_bits.append(_image(turn.get("canvas"), outdir=outdir, alt=f"turn {turn_no} canvas"))
        turn_bits.append(" ")
        turn_bits.append(_link(turn.get("code"), outdir=outdir, label="program.js"))
        if turn.get("reply"):
            turn_bits.append(
                f'<details><summary>raw reply</summary><pre>{html.escape(str(turn["reply"]))}</pre></details>'
            )
        if turn.get("failure_kind") or turn.get("error"):
            details = turn.get("error") or turn.get("failure_kind")
            turn_bits.append(f'<pre class="error">{html.escape(str(details))}</pre>')
        turn_bits.append("</details>")
        bits.append("".join(turn_bits))
    if episode.get("errors"):
        bits.append(f'<pre class="error">{html.escape("\n".join(episode["errors"]))}</pre>')
    bits.append(_link(episode.get("trace_source"), outdir=outdir, label="native trace"))
    bits.append("</div>")
    return "".join(bits)


def _summary_stats(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(episodes)
    counts = Counter(item.get("status", "unknown") for item in values)
    sums: dict[str, int | float | None] = {}
    complete_tokens = 0
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        present = [item["tokens"].get(key) for item in values if item["tokens"].get(key) is not None]
        sums[key] = sum(present) if present else None
        complete_tokens += sum(1 for item in values if item["tokens"].get(key) is not None)
    return {
        "episodes": len(values),
        "statuses": dict(sorted(counts.items())),
        "prompt_tokens": sums["prompt_tokens"],
        "completion_tokens": sums["completion_tokens"],
        "total_tokens": sums["total_tokens"],
        "episodes_with_reported_usage": sum(1 for item in values if item["tokens"].get("calls_with_usage")),
        "token_fields_reported": complete_tokens,
        "token_source": "native trace calls[].usage only",
    }


def build_report(
    evidence_root: Path,
    reference_manifest: Path,
    outdir: Path,
    *,
    node_root: str = DEFAULT_NODE_ROOT,
    sft_label: str = "sft",
    rl_label: str = "rl",
) -> dict[str, Any]:
    """Read evidence and write ``summary.json`` plus ``index.html``."""

    evidence_root = Path(evidence_root).resolve()
    reference_manifest = Path(reference_manifest).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metadata, cases = _manifest_cases(reference_manifest)
    manifest_by_id = {case["id"]: case for case in cases}
    manifest_by_group = {case["source_group"]: case for case in cases if case.get("source_group")}
    references = {
        case["id"]: _reference_path(case, manifest_path=reference_manifest, evidence_root=evidence_root, node_root=node_root)
        for case in cases
    }
    traces, parse_errors, trace_files = read_traces(evidence_root)
    aliases = _aliases(sft_label, rl_label)
    episodes = [
        _episode(
            trace,
            evidence_root=evidence_root,
            aliases=aliases,
            node_root=node_root,
            manifest_by_id=manifest_by_id,
            manifest_by_group=manifest_by_group,
        )
        for trace in traces
    ]
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []
    for episode in episodes:
        if episode["case_id"] in manifest_by_id and episode["policy"] in POLICIES:
            by_cell[(episode["case_id"], episode["policy"])].append(episode)
        else:
            unmatched.append(episode)

    expected_cells = {(case["id"], policy) for case in cases for policy in POLICIES}
    missing_cells = sorted(expected_cells - set(by_cell))
    duplicate_cells = {
        f"{case_id}/{policy}": len(values)
        for (case_id, policy), values in by_cell.items()
        if len(values) > 1
    }
    # Preserve every episode in JSON.  The HTML displays all duplicates inside
    # one cell instead of silently selecting a winner.
    summary_cases: list[dict[str, Any]] = []
    for case in cases:
        row = {
            "id": case["id"],
            "family": case.get("family"),
            "source_group": case.get("source_group"),
            "reference": str(references[case["id"]]) if references[case["id"]] else None,
            "policies": {policy: by_cell.get((case["id"], policy), []) for policy in POLICIES},
        }
        summary_cases.append(row)

    policy_stats = {
        policy: _summary_stats(episode for (case_id, found_policy), values in by_cell.items() if found_policy == policy for episode in values)
        for policy in POLICIES
    }
    expected_count = len(cases) * len(POLICIES)
    report = {
        "schema_version": REPORT_SCHEMA,
        "reference_manifest": str(reference_manifest),
        "reference_manifest_metadata": {
            key: metadata[key]
            for key in ("schema_version", "run_id", "status", "provenance")
            if key in metadata
        },
        "evidence_root": str(evidence_root),
        "node_root_mapping": {"from": node_root, "to": str(evidence_root)},
        "trace_files": [str(path) for path in trace_files],
        "expected": {"cases": len(cases), "policies": list(POLICIES), "episodes": expected_count},
        "discovered": {
            "trace_records": len(episodes),
            "matched_episodes": sum(len(values) for values in by_cell.values()),
            "missing_cells": len(missing_cells),
            "unmatched_episodes": len(unmatched),
            "duplicate_cells": duplicate_cells,
        },
        "partial": bool(parse_errors or missing_cells or unmatched or len(episodes) < expected_count),
        "parse_errors": parse_errors,
        "missing_cells": [{"case_id": case_id, "policy": policy} for case_id, policy in missing_cells],
        "unmatched_episodes": unmatched,
        "policies": policy_stats,
        "cases": summary_cases,
        "notes": [
            "Status, invalid, censored, and error fields are preserved from native traces; missing cells are explicit.",
            "Token totals use only provider usage in native trace calls[].usage. Missing usage is reported as unavailable.",
            "No reward or visual-quality values are computed by this report.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "index.html").write_text(_html(report, outdir=outdir), encoding="utf-8")
    return report


def _html(report: dict[str, Any], *, outdir: Path) -> str:
    expected = report["expected"]
    discovered = report["discovered"]
    heading = (
        f"Native painter controlled comparison · {discovered['matched_episodes']}/{expected['episodes']} matched episodes"
    )
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>{html.escape(heading)}</title>",
        "<style>body{font:15px/1.45 system-ui,sans-serif;margin:24px;background:#f4f1ea;color:#25231f}h1{font-size:1.5rem}table{border-collapse:collapse;width:100%;background:#fff}th,td{vertical-align:top;border:1px solid #d8d2c7;padding:10px;text-align:left}th{background:#ebe6dc}.ref img{max-width:180px}.episode{background:#faf9f6;padding:8px;margin:0 0 8px;border-left:4px solid #999}.status-complete{border-color:#378a55}.status-censored{border-color:#c28b23}.status-error,.status-invalid{border-color:#ad4747}.episode img{max-width:220px;max-height:220px;margin:5px 5px 5px 0}.episode details{margin-top:6px}.missing{color:#9a3e32;font-style:italic}.flags{font-size:.85em;color:#6f5d3d}.error{white-space:pre-wrap;background:#2b2521;color:#f7eee1;padding:8px;max-width:42em}.summary{background:#fff;padding:12px;margin:12px 0 18px}.unmatched{background:#fff;padding:12px;margin-top:18px}</style></head><body>",
        f"<h1>{html.escape(heading)}</h1>",
        f"<div class=\"summary\"><p>Expected {expected['cases']} families × {len(expected['policies'])} policies. This page links to preserved PNG and program files only; it does not render, score, or infer quality.</p>",
        "<table><thead><tr><th>Policy</th><th>Episodes</th><th>Status counts</th><th>Prompt tokens</th><th>Completion tokens</th><th>Total tokens</th></tr></thead><tbody>",
    ]
    for policy in POLICIES:
        stats = report["policies"][policy]
        parts.append(
            "<tr>"
            f"<td>{html.escape(policy)}</td><td>{stats['episodes']}</td>"
            f"<td>{html.escape(json.dumps(stats['statuses'], sort_keys=True))}</td>"
            f"<td>{stats['prompt_tokens'] if stats['prompt_tokens'] is not None else 'unavailable'}</td>"
            f"<td>{stats['completion_tokens'] if stats['completion_tokens'] is not None else 'unavailable'}</td>"
            f"<td>{stats['total_tokens'] if stats['total_tokens'] is not None else 'unavailable'}</td></tr>"
        )
    parts.append("</tbody></table></div><table><thead><tr><th>Family / reference</th><th>SFT</th><th>RL</th></tr></thead><tbody>")
    for case in report["cases"]:
        ref = _image(case.get("reference"), outdir=outdir, alt=f"{case['family']} reference")
        parts.append(
            f"<tr><th class=\"ref\">{html.escape(str(case['family']))}<br><small>{html.escape(str(case['id']))}</small><br>{ref}</th>"
        )
        for policy in POLICIES:
            cell = case["policies"].get(policy, [])
            if not cell:
                rendered = '<div class="missing">No native trace for this policy/case.</div>'
            else:
                rendered = "".join(_episode_html(item, outdir=outdir) for item in cell)
            parts.append(f"<td><h3>{policy.upper()}</h3>{rendered}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if report.get("unmatched_episodes"):
        parts.append("<section class=\"unmatched\"><h2>Unmatched native traces</h2><p>These records were retained in summary.json but could not be assigned to one of the twelve manifest cases and two policy cells.</p><ul>")
        for item in report["unmatched_episodes"]:
            parts.append(
                f"<li>{html.escape(str(item.get('case_id') or '?'))} · {html.escape(str(item.get('policy_raw') or item.get('policy')))} · {html.escape(str(item.get('status')))} · {_link(item.get('trace_source'), outdir=outdir, label='trace')}</li>"
            )
        parts.append("</ul></section>")
    if report.get("parse_errors"):
        parts.append(f"<section class=\"unmatched\"><h2>Evidence parse errors</h2><pre class=\"error\">{html.escape(chr(10).join(report['parse_errors']))}</pre></section>")
    parts.append("</body></html>")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", "--collected-root", dest="evidence_root", type=Path, required=True)
    parser.add_argument("--reference-manifest", "--source-manifest", dest="reference_manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--node-root", default=DEFAULT_NODE_ROOT, help="absolute path used by the Linux evaluator")
    parser.add_argument("--sft-label", default="sft", help="label fragment used to identify the SFT policy")
    parser.add_argument("--rl-label", default="rl", help="label fragment used to identify the RL policy")
    args = parser.parse_args(argv)
    report = build_report(
        args.evidence_root,
        args.reference_manifest,
        args.outdir,
        node_root=args.node_root,
        sft_label=args.sft_label,
        rl_label=args.rl_label,
    )
    print(
        json.dumps(
            {
                "html": str(args.outdir.resolve() / "index.html"),
                "summary": str(args.outdir.resolve() / "summary.json"),
                "partial": report["partial"],
                "matched_episodes": report["discovered"]["matched_episodes"],
                "expected_episodes": report["expected"]["episodes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
