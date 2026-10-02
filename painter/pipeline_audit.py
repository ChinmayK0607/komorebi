"""Offline, evidence-preserving audit for a finite painter pipeline.

The audit intentionally reads extracted metadata only.  It emits aggregate
facts, relative evidence paths, and hashes; it does not emit programs,
provider payloads, model responses, addresses, or credentials.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "painter-pipeline-audit-v1"
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    with handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    try:
        result = float(match.group(0))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _quantile(values: Iterable[float], probability: float) -> float | None:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return None
    # The pipeline's saved summaries use the nearest-rank (higher) method:
    # p90 of 225 values is element ceil(0.9 * 225) rather than an interpolated
    # value.  Keeping that convention makes this audit directly comparable.
    index = max(0, min(len(values) - 1, int(math.ceil(len(values) * probability)) - 1))
    return values[index]


def _stats(values: Iterable[float], *, unit: str | None = None) -> dict[str, Any]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    if unit:
        result["unit"] = unit
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def _evidence_index(arm: Path) -> dict[str, str]:
    manifest = _read_json(arm / "evidence-manifest.json")
    index: dict[str, str] = {}
    if isinstance(manifest, dict):
        for item in manifest.get("files", []):
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                index[item["path"]] = item["sha256"]
    return index


def _evidence(root: Path, arm: Path, paths: Iterable[Path], hashes: Mapping[str, str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not path.is_file():
            continue
        relative_arm = _relative(arm, path)
        relative_root = _relative(root, arm / relative_arm)
        if relative_root in seen:
            continue
        seen.add(relative_root)
        result.append({"path": relative_root, "sha256": hashes.get(relative_arm) or _file_hash(path) or "unknown"})
    return result


def _timestamp(value: str) -> _datetime.datetime | None:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _datetime.datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def _parse_gpu_csv(root: Path, arm: Path) -> dict[str, Any]:
    csv_path = _first(arm / "logs", "*.csv")
    if not csv_path:
        return {"status": "missing"}
    rows: list[tuple[int, str, float]] = []
    try:
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                index_value = row.get("index") or row.get(" index") or ""
                index = _number(index_value)
                util_value = None
                for key, value in row.items():
                    if key and "utilization.gpu" in key.lower():
                        util_value = _number(value)
                        break
                timestamp = str(row.get("timestamp", "")).strip()
                if index is None or util_value is None or not timestamp:
                    continue
                rows.append((int(index), timestamp, util_value))
    except (OSError, csv.Error):
        return {"status": "unreadable", "evidence": _relative(root, csv_path)}
    if not rows:
        return {"status": "empty", "evidence": _relative(root, csv_path)}

    timestamps = [_timestamp(timestamp) for _, timestamp, _ in rows]
    valid_timestamps = [value for value in timestamps if value is not None]
    window_start = min(valid_timestamps).isoformat() if valid_timestamps else None
    window_end = max(valid_timestamps).isoformat() if valid_timestamps else None
    # nvidia-smi emits one row per physical device.  Estimate cadence within
    # each device stream so timestamp jitter between devices cannot become a
    # false 3ms sample interval.
    per_gpu_cadence: dict[int, float | None] = {}
    for index in sorted({index for index, _, _ in rows}):
        device_times = sorted({_timestamp(timestamp) for row_index, timestamp, _ in rows if row_index == index} - {None})
        deltas = [
            (right - left).total_seconds()
            for left, right in zip(device_times, device_times[1:])
            if (right - left).total_seconds() > 0
        ]
        per_gpu_cadence[index] = statistics.median(deltas) if deltas else None
    roles: dict[str, Any] = {}
    for role, indices in (("inference", set(range(7))), ("trainer", {7})):
        selected = [util for index, _, util in rows if index in indices]
        role_times = [_timestamp(timestamp) for index, timestamp, _ in rows if index in indices]
        role_times = [value for value in role_times if value is not None]
        roles[role] = {
            "physical_indices": sorted(indices),
            "samples": len(selected),
            "denominator_samples": len(selected),
            "gpu_util_mean_percent": statistics.fmean(selected) if selected else None,
            "busy_fraction_nonzero": (sum(value > 0 for value in selected) / len(selected)) if selected else None,
            "window_start": min(role_times).isoformat() if role_times else None,
            "window_end": max(role_times).isoformat() if role_times else None,
            "window_seconds": ((max(role_times) - min(role_times)).total_seconds() if len(role_times) > 1 else 0.0) if role_times else None,
            "per_gpu_cadence_seconds": {str(index): per_gpu_cadence.get(index) for index in sorted(indices)},
        }
    role_cadences = [value for value in per_gpu_cadence.values() if value is not None]
    return {
        "status": "ok",
        "evidence": _relative(root, csv_path),
        "window_start": window_start,
        "window_end": window_end,
        "sample_cadence_seconds": statistics.median(role_cadences) if role_cadences else None,
        "roles": roles,
    }


def _parse_trainer_metrics(root: Path, arm: Path) -> tuple[dict[str, Any], Path | None]:
    logs = sorted(arm.glob("async-output/*/logs/*/trainer.log"))
    if not logs:
        return {"status": "missing"}, None
    path = logs[0]
    grad_norms: list[float] = []
    peak_memory: list[float] = []
    step_count = 0
    step_re = re.compile(r"\bSUCCESS\b.*?\bStep\s+(\d+).*?Grad\.?\s*Norm\s+([-+]?\d+(?:\.\d+)?).*?Peak\s+Mem\.?\s+([-+]?\d+(?:\.\d+)?)\s*(GiB|MiB|GB|MB)", re.I)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"status": "unreadable"}, path
    for line in lines:
        match = step_re.search(_strip_ansi(line))
        if not match:
            continue
        step_count += 1
        grad_norms.append(float(match.group(2)))
        memory = float(match.group(3))
        if match.group(4).lower() in {"mib", "mb"}:
            memory /= 1024.0
        peak_memory.append(memory)
    return {
        "status": "ok",
        "completed_step_lines": step_count,
        "grad_norm": _stats(grad_norms),
        "peak_memory": _stats(peak_memory, unit="GiB"),
    }, path


def _parse_orchestrator_steps(root: Path, arm: Path) -> tuple[dict[str, Any], Path | None]:
    paths = sorted(arm.glob("async-output/*/monitors/file/traces/annotations/orch.index.jsonl"))
    if paths:
        path = paths[0]
        steps: set[int] = set()
        for row in _read_jsonl(path):
            ship = row.get("info", {}).get("ship") if isinstance(row.get("info"), dict) else None
            value = _number(ship.get("step")) if isinstance(ship, dict) else None
            if value is not None:
                steps.add(int(value))
        return {"status": "ok", "steps": sorted(steps), "completed_batch_steps": max(steps) if steps else None}, path
    logs = sorted(arm.glob("async-output/*/logs/*/orchestrator.log"))
    if logs:
        path = logs[0]
        steps = [int(value) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() for value in re.findall(r"Train batch\s+(\d+)", _strip_ansi(line), re.I)]
        return {"status": "log_fallback", "steps": sorted(set(steps)), "completed_batch_steps": max(steps) if steps else None}, path
    return {"status": "missing"}, None


def _parse_admission(root: Path, arm: Path) -> tuple[dict[str, Any], Path | None]:
    path = arm / "consistent-group-admission.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return {"status": "missing"}, path if path.exists() else None
    final = rows[-1]
    decisions: dict[str, int] = {}
    for row in rows:
        key = str(row.get("decision", "unknown"))
        decisions[key] = decisions.get(key, 0) + 1
    observed = _number(final.get("groups_observed"))
    admitted = _number(final.get("groups_admitted"))
    rejected = _number(final.get("groups_rejected"))
    return {
        "status": "ok",
        "groups_observed": int(observed) if observed is not None else None,
        "groups_admitted": int(admitted) if admitted is not None else None,
        "groups_censored": int(rejected) if rejected is not None else None,
        "decision_counts": decisions,
        "evidence_rows": len(rows),
    }, path


def _timing_duration(phase: Any) -> float | None:
    if not isinstance(phase, dict):
        return None
    duration = _number(phase.get("duration"))
    if duration is not None:
        return duration
    start = _number(phase.get("start"))
    end = _number(phase.get("end"))
    if start is None or end is None or end < start:
        return None
    return end - start


def _trace_summary(episodes: list[dict[str, Any]], *, source: str, evidence: Path | None, root: Path) -> dict[str, Any]:
    reward_status: dict[str, int] = {}
    invalid_final_action = 0
    retained_final_canvas_valid = 0
    absent_final_canvas = 0
    invalid_final_canvas = 0
    model_times: list[float] = []
    harness_times: list[float] = []
    scoring_times: list[float] = []
    episode_ok_false = 0
    trace_ok_false = 0
    for episode in episodes:
        status = episode.get("reward_status")
        status_key = status if isinstance(status, str) and status else "unknown"
        reward_status[status_key] = reward_status.get(status_key, 0) + 1
        if episode.get("episode_ok") is False:
            episode_ok_false += 1
        if episode.get("trace_ok") is False:
            trace_ok_false += 1
        # This is intentionally strict: only the recorded final-action
        # reward status is counted as a model-invalid final action.  A missing
        # validation record is censored rather than inferred.
        if status == "model_invalid_final_action":
            invalid_final_action += 1
            if episode.get("final_canvas_valid") is True:
                retained_final_canvas_valid += 1
            elif episode.get("final_canvas_valid") is False:
                invalid_final_canvas += 1
            else:
                absent_final_canvas += 1
        timing = episode.get("timing") if isinstance(episode.get("timing"), dict) else {}
        agent = timing.get("agent") if isinstance(timing.get("agent"), dict) else {}
        for destination, phase in ((model_times, agent.get("model")), (harness_times, agent.get("harness"))):
            duration = _timing_duration(phase)
            if duration is not None:
                destination.append(duration)
        scoring = _timing_duration(timing.get("scoring"))
        if scoring is not None:
            scoring_times.append(scoring)
    result: dict[str, Any] = {
        "status": "ok",
        "source": source,
        "completed_trace_count": len(episodes),
        "reward_status_counts": reward_status,
        "final_action_invalid_count": invalid_final_action,
        "final_action_invalid_split": {
            "retained_final_canvas_valid": retained_final_canvas_valid,
            "final_canvas_invalid": invalid_final_canvas,
            "final_canvas_absent": absent_final_canvas,
        },
        "timing_seconds": {
            "model": _stats(model_times, unit="seconds"),
            "harness": _stats(harness_times, unit="seconds"),
            "scoring": _stats(scoring_times, unit="seconds"),
        },
        "denominator": {
            "kind": "completed_trace_records",
            "count": len(episodes),
            "episode_ok_false_count": episode_ok_false,
            "trace_ok_false_count": trace_ok_false,
        },
        "censor_caveat": "Counts and timing use completed trace records present in the extracted evidence; missing, unfinished, or operationally censored traces are excluded and are not inferred.",
    }
    if evidence is not None:
        result["evidence"] = _relative(root, evidence)
    return result


def _trace_episodes_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract episode-shaped records without copying raw streamed payloads."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    scalar_keys = ("reward_status", "final_canvas_valid", "episode_ok", "trace_ok")

    def add_record(record: dict[str, Any], identity: Any) -> None:
        if not record:
            return
        # A repeated trace id is an update to one trace.  Distinct ids must
        # remain distinct even when their aggregate values are equal.
        key = f"id:{identity}" if identity is not None else _canonical_hash(record)
        if key in seen:
            return
        seen.add(key)
        records.append(record)

    def aggregate(candidate: Mapping[str, Any], *, info: Mapping[str, Any] | None = None, trace_ok: Any = None) -> dict[str, Any]:
        record: dict[str, Any] = {}
        info = info if isinstance(info, dict) else {}
        for key in scalar_keys:
            if key in candidate:
                record[key] = candidate[key]
            elif key in info:
                record[key] = info[key]
        if trace_ok is not None and "trace_ok" not in record:
            record["trace_ok"] = trace_ok
        timing = candidate.get("timing")
        if not isinstance(timing, dict):
            timing = info.get("timing")
        if isinstance(timing, dict):
            record["timing"] = timing
        return record

    for row in rows:
        # Actual streamed records wrap one or more traces in a `traces` list
        # and put the reward fields under each trace's `info` object.
        raw_traces = row.get("traces")
        if isinstance(raw_traces, dict):
            raw_traces = [raw_traces]
        if isinstance(raw_traces, list):
            for index, trace in enumerate(raw_traces):
                if not isinstance(trace, dict):
                    continue
                info = trace.get("info") if isinstance(trace.get("info"), dict) else None
                record = aggregate(trace, info=info, trace_ok=trace.get("ok"))
                trace_id = trace.get("trace_id") or trace.get("id")
                if trace_id is None:
                    parent_id = row.get("trace_id") or row.get("id")
                    trace_id = f"{parent_id}:{index}" if parent_id is not None else None
                add_record(record, trace_id)
            # The wrapper's `ok`/`id` describe the stream item, not another
            # completed episode, so do not count the wrapper itself.
            continue

        candidates: list[Mapping[str, Any]] = [row]
        for key in ("episode", "trace", "result", "summary", "metrics"):
            nested = row.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        candidate = next((value for value in candidates if any(key in value for key in scalar_keys) or isinstance(value.get("timing"), dict)), None)
        if candidate is None:
            continue
        info = candidate.get("info") if isinstance(candidate.get("info"), dict) else None
        record = aggregate(candidate, info=info)
        identity = row.get("trace_id") or row.get("episode_id") or row.get("id") or candidate.get("trace_id") or candidate.get("episode_id") or candidate.get("id")
        add_record(record, identity)
    return records


def _unknown_trace_summary(*, source: str, evidence: Path | None, root: Path) -> dict[str, Any]:
    missing = ["completed_trace_aggregate", "reward_status", "final_canvas_valid", "timing"]
    result: dict[str, Any] = {
        "status": "unknown",
        "source": source,
        "completed_trace_count": None,
        "reward_status_counts": {},
        "final_action_invalid_count": None,
        "final_action_invalid_split": {
            "retained_final_canvas_valid": None,
            "final_canvas_invalid": None,
            "final_canvas_absent": None,
        },
        "timing_seconds": {
            "model": _stats([], unit="seconds"),
            "harness": _stats([], unit="seconds"),
            "scoring": _stats([], unit="seconds"),
        },
        "denominator": {
            "kind": "completed_trace_records",
            "count": None,
            "status": "unknown",
            "missing_fields": missing,
        },
        "missing_fields": missing,
        "censor_caveat": "A stream index was present, but it did not contain explicit completed trace aggregate records; index rows were not treated as completed traces, so completion-dependent counts remain unknown.",
    }
    if evidence is not None:
        result["evidence"] = _relative(root, evidence)
    return result


def _parse_trace_metrics(root: Path, arm: Path) -> tuple[dict[str, Any], Path | None]:
    compact = arm / "compact-traces-metrics.json"
    data = _read_json(compact)
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        episodes = [episode for episode in data["episodes"] if isinstance(episode, dict)]
        return _trace_summary(episodes, source="compact-traces-metrics", evidence=compact, root=root), compact

    # Prefer streamed raw trace records when the compact extraction is absent.
    # Only episode-shaped aggregate fields are retained; raw programs and
    # replies are never copied into the audit result.
    stream_paths = sorted(arm.glob("async-output/*/monitors/file/traces/stream/*.jsonl"))
    if stream_paths:
        # Read all shards before extraction so duplicate trace IDs are removed
        # globally, including when an update lands in a later shard.
        rows: list[dict[str, Any]] = []
        for stream_path in stream_paths:
            rows.extend(_read_jsonl(stream_path))
        episodes = _trace_episodes_from_rows(rows)
        if episodes:
            return _trace_summary(episodes, source="streamed-traces", evidence=stream_paths[0], root=root), stream_paths[0]

    # Older extracted archives may only retain the streamed trace index.  It
    # can provide a completed denominator and any aggregate fields present in
    # the index, but cannot manufacture final-action or timing fields.
    stream_paths = sorted(arm.glob("async-output/*/monitors/file/traces/stream.index.jsonl"))
    if stream_paths:
        rows = _read_jsonl(stream_paths[0])
        episodes = _trace_episodes_from_rows(rows)
        if episodes:
            return _trace_summary(episodes, source="streamed-trace-index", evidence=stream_paths[0], root=root), stream_paths[0]
        return _unknown_trace_summary(source="streamed-trace-index", evidence=stream_paths[0], root=root), stream_paths[0]
    return {"status": "missing", "censor_caveat": "No compact or streamed trace evidence was present; no completed denominator was inferred."}, None


def _event_count(paths: Iterable[Path], kind: str) -> tuple[int | None, list[str]]:
    """Count explicit log events, returning None when no event evidence exists."""
    count = 0
    found = False
    evidence: list[str] = []
    if kind == "stale_queued_trace_drops":
        patterns = (
            re.compile(r"\bDropped\s+(?P<n>\d+)\s+queued\s+traces?\b", re.I),
            re.compile(r"(?P<n>\d+)\s+stale\s+queued\s+traces?\s+(?:were\s+)?(?:drop|discard)", re.I),
            re.compile(r"stale\s+queued\s+traces?.{0,48}?(?P<n>\d+).{0,18}(?:drop|discard)", re.I),
        )
        predicate = lambda text: "queue" in text.lower() and any(word in text.lower() for word in ("drop", "discard"))
    else:
        patterns = (
            re.compile(r"\bCancelled\s+(?P<n>\d+)(?:\s+\w+){0,3}\s+episodes?\b", re.I),
            re.compile(r"(?P<n>\d+)\s+active\s+episodes?\s+(?:were\s+)?(?:cancel|abort)", re.I),
            re.compile(r"active\s+episodes?.{0,48}?(?P<n>\d+).{0,18}(?:cancel|abort)", re.I),
        )
        predicate = lambda text: "episode" in text.lower() and any(word in text.lower() for word in ("cancel", "abort"))
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            text = _strip_ansi(line)
            if not predicate(text):
                continue
            found = True
            match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
            count += int(match.group("n")) if match and match.group("n") else 1
            rel = str(path)
            if rel not in evidence:
                evidence.append(rel)
    return (count if found else None), evidence


def _parse_queue_and_cancel(root: Path, arm: Path) -> dict[str, Any]:
    paths = sorted(arm.glob("async-output/*/logs/**/*.log"))
    stale, stale_paths = _event_count(paths, "stale_queued_trace_drops")
    cancels, cancel_paths = _event_count(paths, "active_episode_cancels")
    result: dict[str, Any] = {
        "stale_queued_trace_drops": {"count": stale, "status": "observed" if stale is not None else "not_observed"},
        "active_episode_cancels": {"count": cancels, "status": "observed" if cancels is not None else "not_observed"},
    }
    if stale_paths:
        result["stale_queued_trace_drops"]["evidence"] = [_relative(root, Path(path)) for path in stale_paths]
    if cancel_paths:
        result["active_episode_cancels"]["evidence"] = [_relative(root, Path(path)) for path in cancel_paths]
    return result


def _usage_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    prompt: list[float] = []
    completion: list[float] = []
    total: list[float] = []
    reasoning: list[float] = []
    costs: list[float] = []
    unknown_cost = 0
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        for destination, key in ((prompt, "prompt_tokens"), (completion, "completion_tokens"), (total, "total_tokens")):
            value = _number(usage.get(key))
            if value is not None:
                destination.append(value)
        nested = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        value = _number(nested.get("reasoning_tokens"))
        if value is None:
            value = _number(call.get("observed_reasoning_tokens"))
        if value is not None:
            reasoning.append(value)
        cost = _number(usage.get("cost"))
        if cost is None:
            unknown_cost += 1
        else:
            costs.append(cost)
    return {
        "prompt_tokens": int(sum(prompt)) if prompt else None,
        "completion_tokens": int(sum(completion)) if completion else None,
        "total_tokens": int(sum(total)) if total else None,
        "reasoning_tokens": int(sum(reasoning)) if reasoning else None,
        "recorded_cost_usd": sum(costs) if costs else 0.0,
        "unknown_cost_count": unknown_cost,
        "cost_lower_bound_usd": sum(costs),
    }


def _parse_judges(root: Path, arm: Path) -> dict[str, Any]:
    receipts = sorted((arm / "rl-rollouts").rglob("text-judge.json"))
    sections: dict[str, list[dict[str, Any]]] = {"views": [], "recheck_views": []}
    raw_calls: dict[str, list[dict[str, Any]]] = {"views": [], "recheck_views": []}
    malformed = 0
    for path in receipts:
        value = _read_json(path)
        if not isinstance(value, dict):
            malformed += 1
            continue
        for section in sections:
            entries = value.get(section)
            if not isinstance(entries, list):
                continue
            sections[section].extend(entry for entry in entries if isinstance(entry, dict))
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("calls"), list):
                    raw_calls[section].extend(call for call in entry["calls"] if isinstance(call, dict))

    output: dict[str, Any] = {"receipts": len(receipts), "malformed_receipts": malformed, "views": {}, "recheck_views": {}}
    all_unique: list[dict[str, Any]] = []
    global_seen: set[str] = set()
    for section in ("views", "recheck_views"):
        calls = raw_calls[section]
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        reused = 0
        for call in calls:
            # Hash the full call privately; the raw response is never copied to output.
            key = _canonical_hash(call)
            if key in seen or key in global_seen:
                reused += 1
                continue
            seen.add(key)
            global_seen.add(key)
            unique.append(call)
            all_unique.append(call)
        latencies = [_number(call.get("elapsed_seconds")) for call in unique]
        latencies = [value for value in latencies if value is not None]
        failed = sum(1 for call in unique if call.get("finish_reason") not in (None, "stop"))
        output[section] = {
            "view_entries": len(sections[section]),
            "raw_call_count": len(calls),
            "deduplicated_call_count": len(unique),
            "exact_record_reuse_count": reused,
            "failed_call_count": failed,
            "latency_seconds": _stats(latencies, unit="seconds"),
            "usage": _usage_totals(unique),
        }
    output["total_deduplicated_call_count"] = len(all_unique)
    output["total_failed_call_count"] = sum(output[name]["failed_call_count"] for name in ("views", "recheck_views"))
    output["total_latency_seconds"] = _stats(
        [value for value in (_number(call.get("elapsed_seconds")) for call in all_unique) if value is not None],
        unit="seconds",
    )
    output["total_usage"] = _usage_totals(all_unique)
    return output


def _parse_render(root: Path, arm: Path) -> dict[str, Any]:
    receipts = sorted((arm / "rl-rollouts").rglob("painting.json"))
    valid: list[float] = []
    invalid: list[float] = []
    timed_out = 0
    unknown = 0
    timing_unknown = 0
    valid_count = 0
    invalid_count = 0
    for path in receipts:
        value = _read_json(path)
        if not isinstance(value, dict):
            unknown += 1
            continue
        duration = _number(value.get("elapsed_seconds"))
        is_valid = value.get("valid")
        if is_valid is True:
            valid_count += 1
            if duration is not None:
                valid.append(duration)
            else:
                timing_unknown += 1
        elif is_valid is False:
            invalid_count += 1
            if duration is not None:
                invalid.append(duration)
            else:
                timing_unknown += 1
        else:
            unknown += 1
            if duration is None:
                timing_unknown += 1
        if value.get("timed_out") is True or value.get("error_code") == "render_timeout":
            timed_out += 1
    durations = valid + invalid
    return {
        "receipt_count": len(receipts),
        "valid_receipts": valid_count,
        "invalid_receipts": invalid_count,
        "unknown_receipts": unknown,
        "timing_unknown_receipts": timing_unknown,
        "timed_out_receipts": timed_out,
        "duration_seconds": _stats(durations, unit="seconds"),
    }


def _initializer(root: Path, arms: Mapping[str, Path], sources: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "base_model_revision",
        "initializer_revision",
        "initializer_prefix",
        "initializer_sha256",
        "frozen_sft_model_revision",
        "frozen_sft_initializer",
    )
    records: dict[str, dict[str, Any]] = {}
    for arm, source in sources.items():
        if not isinstance(source, dict):
            records[arm] = {}
            continue
        frozen = source.get("frozen_sft_initializer") if isinstance(source.get("frozen_sft_initializer"), dict) else {}
        records[arm] = {
            "base_model_revision": source.get("base_model_revision"),
            "initializer_revision": source.get("initializer_revision"),
            "initializer_prefix": source.get("initializer_prefix"),
            "initializer_sha256": source.get("initializer_sha256"),
            "frozen_sft_model_revision": source.get("frozen_sft_model_revision"),
            "frozen_sft_initializer": {key: frozen.get(key) for key in ("revision", "prefix", "adapter_sha256")},
        }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    match_fields = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records.values() if record]
    missing: dict[str, list[str]] = {}
    for arm, record in records.items():
        missing_fields: list[str] = []
        for key in ("base_model_revision", "initializer_revision", "initializer_prefix", "initializer_sha256", "frozen_sft_model_revision"):
            if not record.get(key):
                missing_fields.append(key)
        frozen = record.get("frozen_sft_initializer") if isinstance(record.get("frozen_sft_initializer"), dict) else {}
        for key in ("revision", "prefix", "adapter_sha256"):
            if not frozen.get(key):
                missing_fields.append(f"frozen_sft_initializer.{key}")
        if missing_fields:
            missing[arm] = missing_fields
    return {
        "exact_match_across_arms": bool(match_fields) and not missing and len(set(match_fields)) == 1 and len(match_fields) == len(arms),
        "matching_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "arms": records,
        "missing_fields": missing,
    }


def audit(root: str | Path, *, arm_names: Iterable[str] | None = None) -> dict[str, Any]:
    root = Path(root)
    if root.is_file():
        root = root.parent
    if root.is_dir() and (root / "source-identity.json").is_file():
        discovered = {root.name: root}
    else:
        discovered = {path.name: path for path in sorted(root.iterdir()) if path.is_dir() and (path / "source-identity.json").is_file()} if root.exists() else {}
    if arm_names:
        discovered = {name: discovered[name] for name in arm_names if name in discovered}
    arms: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for name, arm in discovered.items():
        source = _read_json(arm / "source-identity.json")
        sources[name] = source
        supervisor = _read_json(arm / "supervisor-state.json") or {}
        analysis = _read_json(arm / "analysis-summary.json") or {}
        trainer, trainer_path = _parse_trainer_metrics(root, arm)
        orchestrator, orchestrator_path = _parse_orchestrator_steps(root, arm)
        admission, admission_path = _parse_admission(root, arm)
        traces, traces_path = _parse_trace_metrics(root, arm)
        queue_cancel = _parse_queue_and_cancel(root, arm)
        evidence_paths = [
            arm / "source-identity.json",
            arm / "evidence-manifest.json",
            arm / "supervisor-state.json",
            arm / "analysis-summary.json",
            arm / "gpu-summary.json",
            admission_path,
            trainer_path,
            orchestrator_path,
            traces_path,
            _first(arm / "logs", "*.csv"),
        ]
        hashes = _evidence_index(arm)
        completed = supervisor.get("completed_optimizer_steps")
        if completed is None:
            completed = trainer.get("completed_step_lines")
        budget = supervisor.get("budget", {}) if isinstance(supervisor.get("budget"), dict) else {}
        started = _number(supervisor.get("started_epoch"))
        finished = _number(supervisor.get("finished_epoch"))
        price = _number(budget.get("price_usd_per_hour"))
        elapsed = (finished - started) if started is not None and finished is not None and finished >= started else None
        estimated_cost = (elapsed / 3600.0 * price) if elapsed is not None and price is not None else None
        cap = _number(budget.get("gpu_rental_cap_usd", budget.get("max_gpu_usd")))
        reserve = _number(budget.get("reserve_seconds"))
        arms[name] = {
            "completion": {
                "completed_optimizer_steps": int(completed) if isinstance(completed, (int, float)) else None,
                "supervisor_last_optimizer_step": supervisor.get("last_optimizer_step"),
                "admitted_batch_counter": supervisor.get("admitted_batch_counter", supervisor.get("last_admitted_batch")),
                "orchestrator_batch_steps": orchestrator.get("completed_batch_steps"),
                "orchestrator_steps_observed": orchestrator.get("steps", []),
                "evidence": _relative(root, arm / "supervisor-state.json"),
            },
            "budget": {
                "gpu_cap_usd": cap,
                "price_usd_per_hour": price,
                "reserve_seconds": reserve,
                "elapsed_seconds": elapsed,
                "estimated_elapsed_cost_usd": estimated_cost,
                "estimated_cap_headroom_usd": (cap - estimated_cost) if cap is not None and estimated_cost is not None else None,
                "started_epoch": supervisor.get("started_epoch"),
                "finished_epoch": supervisor.get("finished_epoch"),
            },
            "training": trainer,
            "traces": traces,
            "admission": admission,
            "queue_and_cancellation": queue_cancel,
            "render": _parse_render(root, arm),
            "judge": _parse_judges(root, arm),
            "gpu": _parse_gpu_csv(root, arm),
            "evidence": _evidence(root, arm, evidence_paths, hashes),
            "quality_comparison": {"status": "not_computed", "reason": "audit does not rank quality at finite early steps"},
        }
        # Keep the saved high-level aggregate available without copying any raw response/provider data.
        if isinstance(analysis, dict):
            arms[name]["saved_aggregate"] = {
                key: analysis[key]
                for key in ("render_elapsed", "valid_render_elapsed", "invalid_render_elapsed", "judge_status", "judge_consensus", "group_decisions")
                if key in analysis
            }
    try:
        evidence_root = root.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        evidence_root = root.name
    return {
        "schema": SCHEMA,
        "evidence_root": evidence_root,
        "arms": arms,
        "initializer": _initializer(root, discovered, sources),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="offline-analysis directory or one extracted arm")
    parser.add_argument("--arm", action="append", dest="arms", help="restrict audit to an arm directory name")
    parser.add_argument("--output", type=Path, help="write compact JSON to this path as well as stdout")
    args = parser.parse_args(argv)
    result = audit(args.root, arm_names=args.arms)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
