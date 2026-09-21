"""Summarize preserved COCO native-painting evidence without inference.

The input is a collected evidence directory containing native ``traces.jsonl``
streams under ``eval/base`` and ``eval/sft`` (episode wrappers are accepted).
The output is an audit summary of recorded episode/turn metadata and provider
usage.  Validity and canvas retention are operational observations; this
command does not make an aesthetic or reference-fidelity judgment.

GPU telemetry is optional.  Every parseable row in the bounded GPU CSV found
under the evidence root is summarized, including setup, rendering, and idle
tail samples within that recorded run window.  Trace timing is retained as
context but is not used to trim telemetry.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from painter.native_painting_report import read_traces
except ImportError:  # pragma: no cover - useful when loaded from painter/
    from native_painting_report import read_traces


SCHEMA = "coco-capability-evidence-summary-v1"
POLICIES = ("base", "sft")
_MISSING = "missing"
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _trace_parts(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    context = trace.get("_episode_context") if isinstance(trace.get("_episode_context"), dict) else {}
    return data, info, context


def _first_text(*sources: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for source in sources:
        for key in keys:
            value = _text(source.get(key))
            if value:
                return value
    return None


def _identity(trace: dict[str, Any]) -> tuple[str, str | None, str]:
    data, info, context = _trace_parts(trace)
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    case_id = _first_text(
        info,
        data,
        context.get("task") if isinstance(context.get("task"), dict) else {},
        task,
        keys=("task_id", "case_id", "source_group", "name", "id"),
    ) or "unknown"
    source_group = _first_text(info, data, keys=("source_group", "group", "task_id", "case_id"))
    family = _first_text(info, data, keys=("family",))
    return case_id, family, source_group


def _policy(trace: dict[str, Any]) -> str:
    data, info, context = _trace_parts(trace)
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    raw = _first_text(
        info,
        data,
        context,
        context.get("task") if isinstance(context.get("task"), dict) else {},
        task,
        keys=("policy_label", "policy", "model"),
    )
    if raw:
        lowered = raw.lower()
        if lowered == "base" or "base" in lowered:
            return "base"
        if lowered == "sft" or "sft" in lowered:
            return "sft"
    # A collected stream can omit policy metadata while retaining the reviewed
    # eval/base or eval/sft path.  Path inference is deliberately last.
    source = str(trace.get("_source_path", "")).replace(os.sep, "/").lower()
    for candidate in POLICIES:
        if f"/{candidate}/" in source or f"/{candidate}-" in source:
            return candidate
    return "unknown"


def _turns(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in _as_list(info.get("turns")) if isinstance(row, dict)]


def _action(turn: dict[str, Any]) -> str:
    action = turn.get("action")
    return str(action).lower() if action is not None else ""


def _has_code(turn: dict[str, Any]) -> bool:
    for key in ("code", "program", "javascript", "js", "program_path", "code_path"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return True
    reply = turn.get("reply")
    if not isinstance(reply, str):
        return False
    # Native painting responses conventionally carry a complete fenced JS
    # program.  Match a complete block so a closing fence, or a fenced HTML
    # diagnostic response, is never mistaken for accepted JavaScript.
    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", reply, flags=re.IGNORECASE | re.DOTALL):
        language = match.group(1).strip().lower()
        if language in {"", "javascript", "js"} and match.group(2).strip():
            return True
    return False


def _canvas_recorded(turn: dict[str, Any]) -> bool:
    for key in ("canvas", "final_canvas", "canvas_sha256", "canvas_hash"):
        value = turn.get(key)
        if isinstance(value, str) and value:
            return True
    return False


def _flatten_messages(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for key in ("message", "error", "detail", "reason", "type"):
            if key in value:
                yield from _flatten_messages(value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_messages(item)
        return
    text = str(value).strip()
    if text:
        yield text


def _errors(trace: dict[str, Any], turns: list[dict[str, Any]], context: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for value in _as_list(trace.get("errors")) + _as_list(context.get("errors")):
        result.update(_flatten_messages(value))
    for turn in turns:
        # Native traces may retain the same renderer failure in both
        # ``error`` and ``renderer_errors``.  Count one canonical turn error;
        # the detailed renderer list remains in the raw trace evidence.
        if turn.get("error") is not None:
            result.update(_flatten_messages(turn.get("error")))
        elif turn.get("failure") is not None:
            result.update(_flatten_messages(turn.get("failure")))
        else:
            result.update(_flatten_messages(turn.get("renderer_errors")))
    return result


def _cutoff_reasons(trace: dict[str, Any], info: dict[str, Any], turns: list[dict[str, Any]]) -> Counter[str]:
    result: Counter[str] = Counter()
    if info.get("operational_cutoff"):
        reason = _first_text(
            info,
            keys=("cutoff_reason", "operational_cutoff_reason", "censor_reason", "failure_kind"),
        ) or "operational_cutoff"
        result[reason] += 1
    if info.get("censored"):
        result["censored"] += 1
    stop = _text(trace.get("stop_condition"))
    if stop and stop.lower() in {"timeout", "timed_out", "cancelled", "canceled", "context", "max_turns", "turn_cap"}:
        result[f"stop_condition:{stop}"] += 1
    for turn in turns:
        failure = str(turn.get("failure_kind") or "").lower()
        if failure in {"timeout", "timed_out", "context", "cancelled", "canceled", "cutoff"}:
            result[f"turn_failure:{failure}"] += 1
    for call in _as_list(trace.get("calls")):
        if not isinstance(call, dict):
            continue
        reason = str(call.get("finish_reason") or "").strip().lower()
        if reason in {"length", "timeout", "timed_out", "cancelled", "canceled"}:
            result[f"finish_reason:{reason}"] += 1
    return result


def _usage(trace: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    present = {key: False for key in totals}
    calls = 0
    calls_with_usage = 0
    finish_reasons: Counter[str] = Counter()
    for call in _as_list(trace.get("calls")):
        if not isinstance(call, dict):
            continue
        calls += 1
        reason = _text(call.get("finish_reason")) or _MISSING
        finish_reasons[reason] += 1
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        values: dict[str, int | None] = {}
        for key, aliases in {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens", "generated_tokens"),
            "total_tokens": ("total_tokens",),
        }.items():
            value = next((usage.get(alias) for alias in aliases if isinstance(usage.get(alias), (int, float)) and not isinstance(usage.get(alias), bool)), None)
            values[key] = int(value) if value is not None else None
        if values["total_tokens"] is None and values["prompt_tokens"] is not None and values["completion_tokens"] is not None:
            values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]
        if any(value is not None for value in values.values()):
            calls_with_usage += 1
        for key, value in values.items():
            if value is not None:
                totals[key] += value
                present[key] = True
    for key in totals:
        if not present[key]:
            totals[key] = None
    return {
        **totals,
        "calls": calls,
        "calls_with_usage": calls_with_usage,
        "missing_usage_calls": calls - calls_with_usage,
        "finish_reason_counts": _counter(finish_reasons),
    }


def _trace_window(traces: Iterable[dict[str, Any]]) -> tuple[float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []

    def add(value: Any, output: list[float]) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            output.append(float(value))

    for trace in traces:
        timing = trace.get("timing") if isinstance(trace.get("timing"), dict) else {}
        add(timing.get("start"), starts)
        for key in ("end", "finish", "finalize", "scoring"):
            section = timing.get(key)
            if isinstance(section, dict):
                add(section.get("end"), ends)
        for call in _as_list(trace.get("calls")):
            if not isinstance(call, dict):
                continue
            call_time = call.get("time") if isinstance(call.get("time"), dict) else {}
            add(call_time.get("start"), starts)
            add(call_time.get("end"), ends)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _episode_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    data, info, context = _trace_parts(trace)
    turns = _turns(info)
    paint_turns = [turn for turn in turns if _action(turn) == "paint"]
    valid_paint = [turn for turn in paint_turns if turn.get("valid") is True]
    valid_retained = [turn for turn in valid_paint if _canvas_recorded(turn)]
    no_code = [turn for turn in paint_turns if not _has_code(turn)]
    unchanged = [turn for turn in valid_paint if turn.get("no_change") is True]
    errors = _errors(trace, turns, context)
    errors.update(_flatten_messages(info.get("service_failure")))
    if context.get("ok") is False:
        errors["episode ok=false"] += 1
    if trace.get("ok") is False:
        errors["trace ok=false"] += 1
    cutoffs = _cutoff_reasons(trace, info, turns)
    usage = _usage(trace)
    final_valid = bool(info.get("final_canvas_valid"))
    final_recorded = bool(_text(info.get("final_canvas")))
    had_turn_errors = any(
        turn.get("error") is not None
        or turn.get("renderer_errors")
        or turn.get("failure") is not None
        or turn.get("failure_kind") in {"program", "service", "timeout", "context"}
        for turn in turns
    )
    case_id, family, source_group = _identity(trace)
    policy = _policy(trace)
    status = "complete" if bool(info.get("finished")) and final_valid else "incomplete"
    if info.get("operational_cutoff") or info.get("censored"):
        status = "censored"
    fatal_error = bool(
        _as_list(trace.get("errors"))
        or _as_list(context.get("errors"))
        or info.get("service_failure")
        or context.get("ok") is False
        or trace.get("ok") is False
        or any(str(turn.get("failure_kind") or "").lower() == "service" for turn in turns)
    )
    if fatal_error:
        status = "error"
    return {
        "trace_id": _text(trace.get("id")),
        "case_id": case_id,
        "source_group": source_group,
        "family": family,
        "policy": policy,
        "status": status,
        "had_turn_errors": had_turn_errors,
        "turns": len(turns),
        "paint_attempts": len(paint_turns),
        "valid_attempts": len(valid_paint),
        # This is the episode's final valid canvas retained by the trace.  A
        # separate turn count keeps intermediate valid renders auditable.
        "retained_valid_canvases": int(final_valid and final_recorded),
        "valid_canvas_turns_count": len(valid_retained),
        "valid_final_canvas": final_valid,
        "final_canvas_recorded": final_recorded,
        "no_code_count": len(no_code),
        "unchanged_valid_count": len(unchanged),
        "errors": _counter(errors),
        "cutoff_reasons": _counter(cutoffs),
        "tokens": usage,
        "operational_cutoff": bool(info.get("operational_cutoff") or info.get("censored")),
        "finish_turn": info.get("finish_turn"),
        "trace_source": trace.get("_source_path"),
        "trace_line": trace.get("_source_line"),
        # Kept to make an episode's counted canvas rows auditable without
        # preserving large replies or model-generated programs in the report.
        "valid_canvas_turns": [turn.get("turn") for turn in valid_retained],
        "no_code_turns": [turn.get("turn") for turn in no_code],
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "episodes": 0,
        "valid_final_canvases": 0,
        "final_canvas_records": 0,
        "paint_attempts": 0,
        "valid_attempts": 0,
        "retained_valid_canvases": 0,
        "valid_canvas_turns": 0,
        "no_code_count": 0,
        "no_code_episodes": 0,
        "episodes_with_turn_errors": 0,
        "unchanged_valid_count": 0,
        "turns": 0,
        "statuses": {},
        "tokens": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "calls": 0,
            "calls_with_usage": 0,
            "missing_usage_calls": 0,
        },
        "finish_reason_counts": {},
        "errors": {},
        "cutoff_reasons": {},
    }


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = _empty_metrics()
    status_counts: Counter[str] = Counter()
    finish: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    cutoffs: Counter[str] = Counter()
    sums = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    present = {key: False for key in sums}
    for episode in episodes:
        result["episodes"] += 1
        result["turns"] += int(episode.get("turns") or 0)
        result["paint_attempts"] += int(episode.get("paint_attempts") or 0)
        result["valid_attempts"] += int(episode.get("valid_attempts") or 0)
        result["retained_valid_canvases"] += int(episode.get("retained_valid_canvases") or 0)
        result["valid_canvas_turns"] += int(episode.get("valid_canvas_turns_count") or 0)
        result["no_code_count"] += int(episode.get("no_code_count") or 0)
        result["no_code_episodes"] += int(bool(episode.get("no_code_count")))
        result["episodes_with_turn_errors"] += int(bool(episode.get("had_turn_errors")))
        result["unchanged_valid_count"] += int(episode.get("unchanged_valid_count") or 0)
        result["valid_final_canvases"] += int(bool(episode.get("valid_final_canvas")))
        result["final_canvas_records"] += int(bool(episode.get("final_canvas_recorded")))
        status_counts[str(episode.get("status") or "unknown")] += 1
        finish.update(episode.get("tokens", {}).get("finish_reason_counts", {}))
        errors.update(episode.get("errors", {}))
        cutoffs.update(episode.get("cutoff_reasons", {}))
        usage = episode.get("tokens", {})
        for key in sums:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sums[key] += int(value)
                present[key] = True
        result["tokens"]["calls"] += int(usage.get("calls") or 0)
        result["tokens"]["calls_with_usage"] += int(usage.get("calls_with_usage") or 0)
        result["tokens"]["missing_usage_calls"] += int(usage.get("missing_usage_calls") or 0)
    for key in sums:
        result["tokens"][key] = sums[key] if present[key] else None
    result["statuses"] = _counter(status_counts)
    result["finish_reason_counts"] = _counter(finish)
    result["tokens"]["finish_reason_counts"] = result["finish_reason_counts"]
    result["errors"] = _counter(errors)
    result["cutoff_reasons"] = _counter(cutoffs)
    return result


def _parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    parsed = float(match.group())
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    numeric = _parse_number(raw) if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", raw) else None
    if numeric is not None and numeric > 1_000_000_000:
        return numeric
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _gpu_csv_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.csv"):
        if not path.is_file():
            continue
        if "gpu" in path.name.lower() or "telemetry" in path.name.lower():
            paths.append(path)
    exact = [path for path in paths if path.name.lower() == "gpu.csv"]
    return sorted(exact or paths)


def _gpu_summary(root: Path, trace_window: tuple[float | None, float | None]) -> dict[str, Any]:
    start, end = trace_window
    files = _gpu_csv_paths(root)
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                raw_rows = list(csv.reader(handle, skipinitialspace=True))
            if not raw_rows:
                continue
            header = [cell.strip().lower() for cell in raw_rows[0]]
            has_header = any(
                "utilization.gpu" in cell or "gpu utilization" in cell or "gpu_utilization" in cell or cell in {"timestamp", "time", "datetime"}
                for cell in header
            )
            if has_header:
                fields = {cell: index for index, cell in enumerate(header) if cell}
                util_index = next(
                    (index for cell, index in fields.items() if "utilization.gpu" in cell or "gpu utilization" in cell or "gpu_utilization" in cell),
                    None,
                )
                time_index = next((index for cell, index in fields.items() if cell in {"timestamp", "time", "datetime", "date"}), None)
                index_index = next((index for cell, index in fields.items() if cell in {"index", "gpu", "gpu index"}), None)
                data_rows = raw_rows[1:]
                line_offset = 2
            else:
                # nvidia-smi is also commonly preserved with --format=csv,
                # but without a header.  The usual forms are
                # timestamp,index,util,memory,power and timestamp,util,memory,power.
                util_index = None
                time_index = 0 if raw_rows and _parse_timestamp(raw_rows[0][0] if raw_rows[0] else None) is not None else None
                index_index = 1 if time_index is not None and len(raw_rows[0]) > 1 else None
                data_rows = raw_rows
                line_offset = 1
                if time_index is not None and len(raw_rows[0]) > 2 and (_parse_number(raw_rows[0][1]) == 0 and (_parse_number(raw_rows[0][2]) or -1) <= 100):
                    util_index = 2
                elif time_index is not None and len(raw_rows[0]) > 1:
                    util_index = 1
                else:
                    # Without timestamps, prefer a value in the utilization
                    # range after a possible GPU index column.
                    for candidate in range(1, len(raw_rows[0])):
                        value = _parse_number(raw_rows[0][candidate])
                        if value is not None and 0 <= value <= 100:
                            util_index = candidate
                            break
            if util_index is None:
                continue
            for line_no, row in enumerate(data_rows, line_offset):
                if util_index >= len(row):
                    continue
                util = _parse_number(row[util_index])
                if util is None:
                    continue
                timestamp = _parse_timestamp(row[time_index]) if time_index is not None and time_index < len(row) else None
                index = row[index_index] if index_index is not None and index_index < len(row) else None
                rows.append({"path": str(path.resolve()), "line": line_no, "utilization": util, "timestamp": timestamp, "index": index})
        except (OSError, csv.Error, UnicodeError) as exc:
            parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if not rows:
        return {
            "available": False,
            "files": [str(path.resolve()) for path in files],
            "sample_count": 0,
            "mean_utilization_percent": None,
            "at_least_90_count": 0,
            "fraction_at_least_90": None,
            "sample_window": None,
            "selection": "no_parseable_samples",
            "parse_errors": parse_errors,
        }
    # The collector's gpu.csv is bounded by its own eval start/end markers;
    # filtering to model-call timings would drop rendering and idle-tail
    # samples and inflate the reported run-level busy fraction.
    selected = rows
    selection = "all_parseable_samples"
    values = [float(row["utilization"]) for row in selected]
    timestamps = [float(row["timestamp"]) for row in selected if row["timestamp"] is not None]
    sample_window = None
    if timestamps:
        sample_window = {
            "start": dt.datetime.fromtimestamp(min(timestamps), tz=dt.timezone.utc).isoformat(),
            "end": dt.datetime.fromtimestamp(max(timestamps), tz=dt.timezone.utc).isoformat(),
        }
    denominator = len(values)
    at_least = sum(value >= 90 for value in values)
    return {
        "available": True,
        "files": sorted({row["path"] for row in selected}),
        "sample_count": denominator,
        "mean_utilization_percent": sum(values) / denominator if denominator else None,
        "at_least_90_count": at_least,
        "fraction_at_least_90": at_least / denominator if denominator else None,
        "sample_window": sample_window,
        "trace_window": {
            "start_epoch": start,
            "end_epoch": end,
        },
        "selection": selection,
        "parse_errors": parse_errors,
    }


def summarize(evidence_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    traces, read_errors, trace_files = read_traces(evidence_root)
    trace_window = _trace_window(traces)
    episodes = [_episode_metrics(trace) for trace in traces]
    known = [episode for episode in episodes if episode["policy"] in POLICIES]
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    by_case: dict[str, dict[str, Any]] = {}
    unknown: list[dict[str, Any]] = []
    for episode in known:
        by_policy[episode["policy"]].append(episode)
        case = by_case.setdefault(episode["case_id"], {"case_id": episode["case_id"], "family": episode.get("family"), "source_group": episode.get("source_group"), "policies": {policy: [] for policy in POLICIES}})
        case["policies"][episode["policy"]].append(episode)
    for episode in episodes:
        if episode["policy"] not in POLICIES:
            unknown.append(episode)
    case_summaries: list[dict[str, Any]] = []
    for case in sorted(by_case.values(), key=lambda row: row["case_id"]):
        case_summaries.append({
            "case_id": case["case_id"],
            "family": case.get("family"),
            "source_group": case.get("source_group"),
            "policies": {
                policy: {
                    **_aggregate(case["policies"][policy]),
                    "episodes_detail": case["policies"][policy],
                }
                for policy in POLICIES
            },
        })
    policy_summaries = {
        policy: {
            **_aggregate(by_policy[policy]),
            "episodes_detail": by_policy[policy],
        }
        for policy in POLICIES
    }
    return {
        "schema": SCHEMA,
        "evidence_root": str(evidence_root),
        "trace_files": [str(path) for path in trace_files],
        "parse_errors": read_errors,
        "trace_window": {"start_epoch": trace_window[0], "end_epoch": trace_window[1]},
        "episodes": len(known),
        "policies": policy_summaries,
        "cases": case_summaries,
        "unknown_policy_episodes": unknown,
        "gpu": _gpu_summary(evidence_root, trace_window),
        "notes": [
            "Counts are taken from preserved native trace fields and recorded canvas/hash fields.",
            "Validity and retained canvases are operational evidence only; no aesthetic or reference-fidelity score is inferred.",
            "Missing finish_reason is counted explicitly as 'missing' and is not interpreted as a quality signal.",
        ],
    }


# Keep a report-style entry point for callers that use the neighboring
# build_gallery.py naming convention.
build_summary = summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.evidence_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.out.resolve()), "episodes": summary["episodes"], "gpu_samples": summary["gpu"]["sample_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
