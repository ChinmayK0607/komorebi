"""Read-only throughput and failure profile for a native painter RL run.

The profiler consumes the small receipts that are normally kept with a run:
Prime metrics JSONL, trainer/orchestrator logs, renderer ``painting.json``
receipts, judge ``text-judge.json`` receipts, and JSONL rollout records.  It
never imports the trainer, executes a renderer, reads model weights, or copies
prompts/programs into the report.  Tar archives are streamed in place so a
collected evidence archive can be profiled without extraction.

Counts are intentionally split into ``window`` (records observed in the
selected evidence) and ``cumulative`` (a run-level counter only when an
explicit counter is present).  The profiler cannot reconstruct Python logging
guard counters when they were not emitted; those fields remain
``unavailable``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
from typing import Any, Iterable, Iterator, TextIO

try:
    import zstandard as _zstandard
except ImportError:  # Optional on collection hosts.
    _zstandard = None
_ZSTD_ERROR = getattr(_zstandard, "ZstdError", RuntimeError)


SCHEMA = "painter.rl-throughput-profile.v1"
DEFAULT_RENDER_TIMEOUT_SECONDS = 180.0
DEFAULT_JUDGE_TIMEOUT_SECONDS = 240.0

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    "renderer-env", "checkpoints", "weights", "cache", "caches", "blobs",
    "private", "credentials", "provider-state", ".aws", "teacher-candidates",
    "validation", "anchors-frozen",
})
_SAFE_LOG_NAMES = frozenset({"trainer.log", "orchestrator.log", "painter-aesthetic.log", "painter.log"})
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STEP = re.compile(r"\bSUCCESS\s+Step\s+(\d+)\s*\|", re.I)
_DISCARDED = re.compile(
    r"Discarded\s+(?P<discarded>\d+)(?:/(?P<total>\d+))?[^\n]*?"
    r"stale\s*=\s*(?P<stale>\d+)[^\n]*?"
    r"(?:errored|error)\s*=\s*(?P<error>\d+)[^\n]*?"
    r"no[_ -]?signal\s*=\s*(?P<no_signal>\d+)", re.I,
)
_WAIT_WARNING = re.compile(
    r"Trainer waited\s+(?P<wait>[^ ]+(?:\s+[^ ]+)*)\s+for a batch,"
    r"\s*at least as long as its\s+(?P<active>[^ ]+(?:\s+[^ ]+)*)\s+active step",
    re.I,
)
_DURATION_PART = re.compile(r"(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?$", re.I)
_TURN_FILE = re.compile(r"(?:^|/)turn-(\d+)\.json$", re.I)
_KNOWN_SYNTAX = (
    (re.compile(r"unexpected end of input", re.I), "unexpected_end_of_input"),
    (re.compile(r"unexpected token", re.I), "unexpected_token"),
    (re.compile(r"syntaxerror", re.I), "syntax_error"),
    (re.compile(r"referenceerror", re.I), "reference_error"),
    (re.compile(r"p5\.brush could not resolve an active webgl sketch", re.I), "brush_no_active_webgl"),
    (re.compile(r"no canvas produced", re.I), "no_canvas_produced"),
    (re.compile(r"invalid color string", re.I), "invalid_color"),
    (re.compile(r"brush\.[a-z0-9_]+ is not a function", re.I), "missing_brush_method"),
    (re.compile(r"object is not iterable", re.I), "object_not_iterable"),
    (re.compile(r"typeerror", re.I), "type_error"),
    (re.compile(r"no available workers|providererror|connection error", re.I), "provider_error"),
    (re.compile(r"timeout|timed out|time limit", re.I), "timeout"),
)
_OPERATIONAL_BROWSER = re.compile(
    r"(?:browser[_ .-]*type\.launch|playwright[^\n]{0,120}(?:launch|connect|error|timeout)|"
    r"(?:chromium|chrome|firefox|webkit)[^\n]{0,120}(?:launch|executable|connect|error)|"
    r"browser[^\n]{0,120}(?:launch|executable|context|connect|closed|crash)|"
    r"(?:launch|start|spawn|connect)[^\n]{0,120}(?:browser|playwright|chromium|chrome|firefox|webkit))",
    re.I,
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _first_present(mapping: Any, *keys: str) -> Any:
    """Return the first non-None field, preserving legitimate numeric zero."""

    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _first_present_from(*mappings: Any, keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        value = _first_present(mapping, *keys)
        if value is not None:
            return value
    return None


def _duration(value: Any) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    value = value.strip()
    match = _DURATION_PART.fullmatch(value)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(item or 0) for item in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield ordinary files without following symlinks into model/provider state."""

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part.lower() in _SKIP_DIRS for part in relative.parts):
            continue
        yield path


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.name.lower().endswith(".zst"):
        if _zstandard is None:
            raise RuntimeError("zstandard support is unavailable")
        raw = path.open("rb")
        return io.TextIOWrapper(_zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        with _open_text(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, EOFError, gzip.BadGzipFile, RuntimeError, _ZSTD_ERROR):
        malformed += 1
    return rows, malformed


def _archive_member_text(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Iterator[str]:
    stream = archive.extractfile(member)
    if stream is None:
        return
    raw: io.BufferedIOBase | Any = stream
    if member.name.lower().endswith(".gz"):
        raw = gzip.GzipFile(fileobj=stream)
    elif member.name.lower().endswith(".zst"):
        if _zstandard is None:
            stream.close()
            raise RuntimeError("zstandard support is unavailable")
        raw = _zstandard.ZstdDecompressor().stream_reader(stream)
    wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    try:
        yield from wrapper
    finally:
        wrapper.close()


def _jsonl_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        for line in _archive_member_text(archive, member):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError, RuntimeError, _ZSTD_ERROR):
        malformed += 1
    return rows, malformed


def _latency_summary(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {"count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None,
                "min_seconds": None, "max_seconds": None}
    p95_index = min(len(data) - 1, max(0, math.ceil(0.95 * len(data)) - 1))
    return {
        "count": len(data),
        "mean_seconds": round(statistics.mean(data), 4),
        "median_seconds": round(statistics.median(data), 4),
        "p95_seconds": round(data[p95_index], 4),
        "min_seconds": round(data[0], 4),
        "max_seconds": round(data[-1], 4),
    }


def _count(window: int, *, cumulative: int | None = None, basis: str) -> dict[str, Any]:
    return {
        "window": int(window),
        "cumulative": int(cumulative) if cumulative is not None else None,
        "cumulative_status": "reported" if cumulative is not None else "unavailable",
        "basis": basis,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "window": None, "cumulative": None,
            "status": "unavailable", "reason": reason}


def _error_category(value: Any) -> str | None:
    if value is None or value is False or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        categories = [_error_category(item) for item in value]
        return next((category for category in categories if category), None)
    if isinstance(value, dict):
        value = value.get("message") or value.get("error") or value.get("type")
    if value is None or value is False or value == "":
        return None
    text = str(value)
    if _OPERATIONAL_BROWSER.search(text):
        return "browser_launch_operational"
    for pattern, category in _KNOWN_SYNTAX:
        if pattern.search(text):
            return category
    if text.strip():
        return "other_error"
    return None


_JUDGE_CALL_KEYS = frozenset({
    "elapsed_seconds", "latency_seconds", "usage", "model", "finish_reason",
    "provider_error", "parse_error",
})


def _judge_call_leaves(value: Any) -> Iterator[dict[str, Any]]:
    """Yield verified provider-call leaves beneath one recorded view.

    A view wrapper can contain one or more retry/attempt records.  The view
    count is tracked separately; only leaves with provider-call metadata are
    counted as calls or used for call latency.
    """

    if not isinstance(value, dict):
        return
    nested = []
    for key in ("call", "calls", "attempt", "attempts", "response", "responses", "result"):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            nested.append(candidate)
        elif isinstance(candidate, list):
            nested.extend(candidate)
    if nested:
        for item in nested:
            yield from _judge_call_leaves(item)
        return
    if _JUDGE_CALL_KEYS.intersection(value):
        yield value


class _ProfileState:
    def __init__(self, render_timeout: float, judge_timeout: float) -> None:
        self.render_timeout = render_timeout
        self.judge_timeout = judge_timeout
        self.sources = Counter()
        self.malformed_json_lines = 0
        self.metrics: list[dict[str, Any]] = []
        self.metric_fingerprints: set[str] = set()
        self.paintings = 0
        self.painting_valid = 0
        self.painting_invalid = 0
        self.painting_unknown = 0
        self.painting_timeouts = 0
        self.render_latencies: list[float] = []
        self.judge_records = 0
        self.judge_statuses = Counter()
        self.judge_consensus = Counter()
        self.judge_views = 0
        self.judge_calls = 0
        self.judge_timeouts = 0
        self.judge_latencies: list[float] = []
        self.error_categories = Counter()
        self.operational_error_categories = Counter()
        self.fallback_error_categories = Counter()
        self.failed_trajectories = 0
        self.trace_records = 0
        self.native_turns = 0
        self.native_completion_tokens = 0
        self.native_prompt_tokens = 0
        self.native_total_tokens = 0
        self.native_usage_calls = 0
        self.native_calls = 0
        self.fallback_turns = 0
        self.fallback_completion_tokens = 0
        self.fallback_prompt_tokens = 0
        self.fallback_total_tokens = 0
        self.fallback_usage_rows = 0
        self.log_turns = 0
        self.log_rollouts = 0
        self.steps: set[int] = set()
        self.metric_steps: set[int] = set()
        self.trainer_log_steps: set[int] = set()
        self.audit_max_update: int | None = None
        self.admitted_batches: list[int] = []
        self.admitted_metric_steps: set[int] = set()
        self.effective_reward_steps: set[int] = set()
        self.discarded = Counter()
        self.discard_events = 0
        self.discard_warning_windows = 0
        self.guard_counters: dict[str, int] = {}
        self.wait_seconds: list[float] = []
        self.active_seconds: list[float] = []
        self.metric_wait_seconds: list[float] = []
        self.metric_active_seconds: list[float] = []
        self.metric_matched_wait_seconds: list[float] = []
        self.metric_matched_active_seconds: list[float] = []
        self.metric_wait_rows = 0
        self.metric_active_rows = 0
        self.timing_scope = "trainer wait warning logs"
        self.inference_running: list[float] = []
        self.inference_waiting: list[float] = []
        self.metric_output_tokens_first: float | None = None
        self.metric_output_tokens_last: float | None = None

    def add_error(self, value: Any) -> None:
        category = _error_category(value)
        if category:
            if category == "browser_launch_operational":
                self.operational_error_categories[category] += 1
            else:
                self.error_categories[category] += 1

    def painting(self, value: dict[str, Any]) -> None:
        self.paintings += 1
        valid = value.get("valid")
        if valid is True:
            self.painting_valid += 1
        elif valid is False:
            self.painting_invalid += 1
        else:
            self.painting_unknown += 1
        errors = value.get("errors")
        if isinstance(errors, list):
            for error in errors:
                self.add_error(error)
        for error in (value.get("error"), value.get("failure"), value.get("failure_reason")):
            self.add_error(error)
        elapsed = _number(value.get("elapsed_seconds"))
        painting_seconds = _number(value.get("painting_seconds"))
        latency = elapsed if elapsed is not None else painting_seconds
        if latency is not None:
            self.render_latencies.append(latency)
        explicit_timeout = bool(value.get("timeout") or value.get("timed_out")) or str(value.get("error_code") or "").lower() in {"render_timeout", "timeout"} or any(
            _error_category(error) == "timeout" for error in (errors if isinstance(errors, list) else [])
        ) or any(_error_category(value.get(name)) == "timeout" for name in ("error", "failure", "failure_reason"))
        if explicit_timeout or (latency is not None and latency >= self.render_timeout):
            self.painting_timeouts += 1

    def judge(self, value: dict[str, Any]) -> None:
        self.judge_records += 1
        status = str(value.get("status") or "unknown").strip().lower()
        if "no_signal" in status or status in {"no-signal", "nosignal"} or str(value.get("consensus") or "").lower() == "no_signal":
            status = "no_signal"
        elif status in {"judge_uncertain", "uncertain", "censored", "missing"}:
            status = "uncertain"
        elif status not in {"scored", "error", "unknown"}:
            status = "other"
        self.judge_statuses[status] += 1
        consensus = value.get("consensus")
        if isinstance(consensus, str) and consensus:
            self.judge_consensus[consensus] += 1
        view_rows: list[dict[str, Any]] = []
        for key in ("views", "initial_views", "recheck_views"):
            rows = value.get(key)
            if isinstance(rows, list):
                view_rows.extend(row for row in rows if isinstance(row, dict))
        # Some collected provider receipts place call metadata under evidence.
        evidence = value.get("score_evidence")
        if isinstance(evidence, dict):
            nested = evidence.get("views")
            if isinstance(nested, list):
                view_rows.extend(row for row in nested if isinstance(row, dict))
        self.judge_views += len(view_rows)
        calls: list[dict[str, Any]] = []
        for view in view_rows:
            calls.extend(_judge_call_leaves(view))
        for call in calls:
            self.judge_calls += 1
            elapsed = _number(_first_present(call, "elapsed_seconds", "latency_seconds"))
            if elapsed is not None:
                self.judge_latencies.append(elapsed)
            provider_error = call.get("provider_error") or call.get("parse_error") or call.get("error")
            if _error_category(provider_error) == "timeout" or (elapsed is not None and elapsed >= self.judge_timeout):
                self.judge_timeouts += 1
            self.add_error(provider_error)

    def trace(self, value: dict[str, Any]) -> None:
        traces = value.get("traces")
        if isinstance(traces, list):
            trace_rows = [row for row in traces if isinstance(row, dict)]
        elif isinstance(value.get("calls"), list) or isinstance(value.get("info"), dict):
            trace_rows = [value]
        else:
            return
        for trace in trace_rows:
            self.trace_records += 1
            info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
            turns = info.get("turns") if isinstance(info.get("turns"), list) else trace.get("turns")
            turns = turns if isinstance(turns, list) else []
            calls = trace.get("calls") if isinstance(trace.get("calls"), list) else []
            self.native_turns += max(len(turns), len(calls))
            self.native_calls += len(calls)
            for call in calls:
                usage = call.get("usage") if isinstance(call, dict) and isinstance(call.get("usage"), dict) else {}
                prompt = _number(_first_present(usage, "prompt_tokens", "input_tokens"))
                completion = _number(_first_present(usage, "completion_tokens", "output_tokens", "generated_tokens"))
                total = _number(usage.get("total_tokens"))
                if total is None and prompt is not None and completion is not None:
                    total = prompt + completion
                if prompt is not None:
                    self.native_prompt_tokens += int(prompt)
                if completion is not None:
                    self.native_completion_tokens += int(completion)
                if total is not None:
                    self.native_total_tokens += int(total)
                if prompt is not None or completion is not None or total is not None:
                    self.native_usage_calls += 1
            trace_error = trace.get("error") or trace.get("errors")
            if trace.get("ok") is False or trace.get("service_failure") or trace.get("stop_condition") in {"error", "timeout", "cancelled", "failed"}:
                self.failed_trajectories += 1
            if isinstance(trace_error, list):
                for error in trace_error:
                    self.add_error(error)
            else:
                self.add_error(trace_error)
            for turn in turns:
                if isinstance(turn, dict):
                    self.add_error(turn.get("error") or turn.get("renderer_errors") or turn.get("failure_kind"))

    def fallback_turn(self, value: dict[str, Any]) -> None:
        self.fallback_turns += 1
        token = _number(_first_present(value, "tokens", "completion_tokens", "output_tokens"))
        if token is not None:
            self.fallback_completion_tokens += int(token)
            self.fallback_usage_rows += 1
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        prompt = _number(_first_present_from(usage, value, keys=("prompt_tokens", "input_tokens")))
        total = _number(_first_present_from(usage, value, keys=("total_tokens",)))
        if prompt is not None:
            self.fallback_prompt_tokens += int(prompt)
        if total is not None:
            self.fallback_total_tokens += int(total)
        category = _error_category(value.get("error") or value.get("failure_kind"))
        if category:
            self.fallback_error_categories[category] += 1

    def response(self, value: dict[str, Any]) -> None:
        usage = value.get("usage")
        if not isinstance(usage, dict):
            response = value.get("response")
            if isinstance(response, dict):
                choices = response.get("choices")
                if isinstance(choices, list) and choices:
                    usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        self.fallback_turns += 1
        self.fallback_usage_rows += 1
        for source, attr in (("prompt_tokens", "fallback_prompt_tokens"), ("input_tokens", "fallback_prompt_tokens"),
                             ("completion_tokens", "fallback_completion_tokens"), ("output_tokens", "fallback_completion_tokens"),
                             ("total_tokens", "fallback_total_tokens")):
            number = _number(usage.get(source))
            if number is not None:
                setattr(self, attr, getattr(self, attr) + int(number))

    def metric(self, value: dict[str, Any]) -> None:
        fingerprint = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if fingerprint in self.metric_fingerprints:
            return
        self.metric_fingerprints.add(fingerprint)
        self.metrics.append(value)
        producer = str(value.get("producer") or "").lower()
        step = _integer(value.get("step"))
        if producer == "trainer" and step is not None and step > 0:
            if any(key in value for key in ("time/step", "loss/mean", "loss")):
                self.steps.add(step)
                self.metric_steps.add(step)
            wait = _number(_first_present(value, "time/wait_for_batch", "time/wait"))
            step_time = _number(value.get("time/step"))
            active = _number(value.get("time/active"))
            if wait is not None:
                self.wait_seconds.append(wait)
                self.metric_wait_seconds.append(wait)
                self.metric_wait_rows += 1
            # Prime's time/step includes time/wait_for_batch. Prefer an
            # explicit active metric; otherwise derive active time only when
            # the matching wait value is present.
            if active is None and step_time is not None and wait is not None:
                active = max(0.0, step_time - wait)
            if active is not None:
                self.active_seconds.append(active)
                self.metric_active_seconds.append(active)
                self.metric_active_rows += 1
                if wait is not None:
                    self.metric_matched_wait_seconds.append(wait)
                    self.metric_matched_active_seconds.append(active)
        if step is not None and step > 0 and producer in {"orch", "orchestrator", "trainer"}:
            for key, raw in value.items():
                lowered = str(key).lower()
                number = _number(raw)
                if lowered == "train/agg/effective/agent/is_admitted/mean":
                    if number is not None and number > 0:
                        self.admitted_metric_steps.add(step)
                if lowered == "train/agg/effective/agent/reward/mean":
                    # The exact effective reward aggregate identifies an
                    # admitted reward batch even when its numeric reward is
                    # zero.  Presence is the signal; do not discard zero
                    # reward batches.
                    if number is not None:
                        self.effective_reward_steps.add(step)
        for key, raw in value.items():
            lowered = str(key).lower()
            number = _number(raw)
            if number is None:
                continue
            if "admitted" in lowered and ("batch" in lowered or "payload" in lowered or "group" in lowered):
                self.admitted_batches.append(int(number))
            if producer == "orch" and ("num_requests_running" in lowered or lowered.endswith("/running")):
                self.inference_running.append(number)
            if producer == "orch" and "num_requests_waiting" in lowered and "reason" not in lowered:
                self.inference_waiting.append(number)
            if producer == "orch" and ("generation_tokens_total" in lowered or "output_tokens_total" in lowered):
                # Prime emits the same cumulative counter with min/max/mean/sum
                # suffixes.  Prefer the sum and keep the final observed value.
                if lowered.endswith("/sum") or lowered.endswith("_total"):
                    if self.metric_output_tokens_first is None:
                        self.metric_output_tokens_first = number
                    self.metric_output_tokens_last = number

    def audit(self, value: dict[str, Any]) -> None:
        # Optimizer audit receipts are allowed to bind the update count, but
        # their tensor names and contents are intentionally discarded.
        completed = _integer(_first_present(value, "completed_updates", "completed_optimizer_updates"))
        if completed is not None and completed >= 0:
            self.audit_max_update = max(self.audit_max_update or 0, completed)
        for key in ("admitted_reward_batches", "admitted_batches"):
            number = _integer(value.get(key))
            if number is not None:
                self.admitted_batches.append(number)

    def log(self, line: str, *, role: str | None = None) -> None:
        clean = _ANSI.sub("", line)
        steps = _STEP.search(clean)
        if steps and role == "trainer":
            step = int(steps.group(1))
            if step > 0:
                self.steps.add(step)
                self.trainer_log_steps.add(step)
        wait_warning = _WAIT_WARNING.search(clean)
        if wait_warning:
            wait = _duration(wait_warning.group("wait"))
            active = _duration(wait_warning.group("active"))
            if wait is not None:
                self.wait_seconds.append(wait)
            if active is not None:
                self.active_seconds.append(active)
        discarded = _DISCARDED.search(clean)
        if discarded:
            self.discard_events += 1
            discarded_count = int(discarded.group("discarded"))
            total_text = discarded.group("total")
            if total_text is not None and discarded_count / int(total_text) > 0.5:
                self.discard_warning_windows += 1
            for name in ("stale", "error", "no_signal"):
                self.discarded[name] += int(discarded.group(name))
        rollout = re.search(r"rollout done:.*?\bturns=(\d+)", clean, re.I)
        if rollout:
            self.log_rollouts += 1
            self.log_turns += int(rollout.group(1))
        # Keep this narrow: generic ERROR lines include setup stack traces and
        # warnings.  Structured trace receipts remain the primary trajectory
        # error source.
        if re.search(r"\b(?:SyntaxError|Unexpected end of input|Unexpected token)\b", clean, re.I):
            self.add_error(clean)
        elif _OPERATIONAL_BROWSER.search(clean):
            self.add_error(clean)

        admitted = re.search(r"(?:admitted[^\n]{0,80}?(?:batch|payload|group)|(?:batch|payload|group)[^\n]{0,80}?admitted)\s*[:= ]\s*(\d+)", clean, re.I)
        if admitted and "no admitted" not in clean.lower():
            self.admitted_batches.append(int(admitted.group(1)))

        # The Python logging guard, when present, emits explicit cumulative
        # counters.  Never infer these from step count or discarded lines.
        for name, pattern in (
            ("groups_observed", r"groups[_ ]observed\s*[:=]\s*(\d+)"),
            ("groups_admitted", r"groups[_ ]admitted\s*[:=]\s*(\d+)"),
            ("groups_rejected", r"groups[_ ]rejected\s*[:=]\s*(\d+)"),
        ):
            match = re.search(pattern, clean, re.I)
            if match:
                self.guard_counters[name] = int(match.group(1))


def _is_metrics(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered == "metrics.jsonl" or lowered.startswith("metrics.jsonl.") or (
        "metrics" in lowered and lowered.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst"))
    )


def _is_zstd_jsonl_name(name: str) -> bool:
    return name.lower().endswith(".jsonl.zst")


def _is_rollout_jsonl(path: Path) -> bool:
    if not path.name.lower().endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
        return False
    lowered = str(path).lower()
    return any(token in lowered for token in ("trace", "rollout", "trajectory", "episode", "/stream/"))


def _is_audit_jsonl(path: Path) -> bool:
    if not path.name.lower().endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
        return False
    lowered = path.name.lower()
    return any(token in lowered for token in ("optimizer", "admission", "guard", "audit"))


def _member_kind(name: str) -> str | None:
    lowered = name.lower()
    base = Path(lowered).name
    if base == "painting.json":
        return "painting"
    if base == "text-judge.json":
        return "judge"
    if base in _SAFE_LOG_NAMES or base.endswith(".log"):
        return "log"
    if base == "metrics.jsonl" or base.startswith("metrics.jsonl."):
        return "metrics"
    if base.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
        if any(token in base for token in ("optimizer", "admission", "guard", "audit")):
            return "audit"
    if base.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
        if any(token in lowered for token in ("trace", "rollout", "trajectory", "episode", "/stream/")):
            return "rollout_jsonl"
    return None


def _consume_jsonl_rows(state: _ProfileState, kind: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if kind == "metrics":
            state.metric(row)
        elif kind == "rollout_jsonl":
            state.trace(row)
        else:
            state.audit(row)


def _consume_small_json(state: _ProfileState, path: Path, value: dict[str, Any]) -> None:
    name = path.name.lower()
    if name == "painting.json":
        state.painting(value)
    elif name == "text-judge.json":
        state.judge(value)
    elif _TURN_FILE.search(str(path).replace("\\", "/")):
        state.fallback_turn(value)
    elif name in {"response.json", "result.json"}:
        # result.json is a per-turn receipt in the native harness.  A response
        # receipt contributes only as a fallback when no native trace exists.
        if name == "result.json":
            state.fallback_turn(value)
        else:
            state.response(value)
    elif name in {"summary.json", "progress.json"} and isinstance(value.get("turns"), list):
        # Summary rows are a last-resort token/turn source.  They are not used
        # when per-turn/native evidence is available.
        for turn in value["turns"]:
            if isinstance(turn, dict):
                state.fallback_turn(turn)
    # Small receipts may contain explicit guard/update counters.  Copy only
    # numeric counters and never persist arbitrary receipt content.
    for key in ("groups_observed", "groups_admitted", "groups_rejected", "admitted_reward_batches", "admitted_batches"):
        number = _integer(value.get(key))
        if number is not None:
            state.guard_counters[key] = number
            if "admitted" in key:
                state.admitted_batches.append(number)


def _log_paths(root: Path) -> list[Path]:
    paths = []
    for path in _iter_files(root):
        if path.suffix.lower() not in {".log", ".txt"}:
            continue
        lowered = path.name.lower()
        relative_parts = {part.lower() for part in path.relative_to(root).parts}
        if (path.name in _SAFE_LOG_NAMES
                or any(token in lowered for token in ("trainer", "orchestrator", "painter-aesthetic"))
                or ("trainer" in relative_parts and path.name.lower() in {"stdout.log", "stderr.log"})):
            paths.append(path)
    return paths


def _log_role(name: str | Path) -> str | None:
    """Assign optimizer-step provenance from a log path, conservatively."""

    path = str(name).replace("\\", "/").lower()
    parts = set(Path(path).parts)
    basename = Path(path).name
    if "trainer" in parts or "trainer" in basename:
        return "trainer"
    if "orchestrator" in parts or "orchestrator" in basename or basename.startswith("orch"):
        return "orchestrator"
    return None


def _profile_counts(state: _ProfileState, fallback: str) -> tuple[int, int, str]:
    if state.trace_records:
        return state.native_turns, state.native_completion_tokens, "native trace calls[].usage and info.turns"
    if state.fallback_turns:
        return state.fallback_turns, state.fallback_completion_tokens, fallback
    if state.log_turns:
        return state.log_turns, 0, "rollout done log lines; completion token usage unavailable"
    return 0, 0, "no usable rollout or turn evidence"


def _apply_metric_timing(state: _ProfileState) -> None:
    """Select a matched trainer wait/active metric scope, if available.

    A partial wait series must not be paired with an unrelated active series.
    When either side is incomplete, use only rows where both were observed and
    state that scope in the report.
    """

    if not state.metric_wait_seconds and not state.metric_active_seconds:
        state.timing_scope = "trainer wait warning logs"
        return
    if (state.metric_matched_wait_seconds
            and len(state.metric_matched_wait_seconds) == len(state.metric_wait_seconds)
            and len(state.metric_matched_active_seconds) == len(state.metric_active_seconds)):
        state.wait_seconds = list(state.metric_wait_seconds)
        state.active_seconds = list(state.metric_active_seconds)
        state.timing_scope = "trainer metric rows with matched wait and active values"
        return
    if state.metric_matched_wait_seconds:
        state.wait_seconds = list(state.metric_matched_wait_seconds)
        state.active_seconds = list(state.metric_matched_active_seconds)
        state.timing_scope = "matched trainer metric rows only; wait/active series were partial"
        return
    # One-sided metrics remain visible, but do not manufacture a denominator
    # by combining them with log warnings or another metric series.
    state.wait_seconds = list(state.metric_wait_seconds)
    state.active_seconds = list(state.metric_active_seconds)
    if state.metric_wait_seconds:
        state.timing_scope = "trainer wait metric rows only; active metric unavailable"
    else:
        state.timing_scope = "trainer active metric rows only; wait metric unavailable"


def profile_run(run_dir: Path, *, render_timeout_seconds: float = DEFAULT_RENDER_TIMEOUT_SECONDS,
                judge_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Profile ``run_dir`` without changing it.

    ``run_dir`` may be a live Linux run root or a collected evidence root.
    The returned dictionary contains no absolute paths and no prompts/programs.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    state = _ProfileState(float(render_timeout_seconds), float(judge_timeout_seconds))
    files = list(_iter_files(root))
    direct_relpaths = {
        str(path.relative_to(root)).replace("\\", "/").lower()
        for path in files
    }
    archives: list[Path] = []
    has_turn_receipts = any(_TURN_FILE.search(str(path).replace("\\", "/")) for path in files)
    has_main_metrics = any(_is_metrics(path) and "attempts" not in {part.lower() for part in path.relative_to(root).parts} for path in files)
    log_candidates = _log_paths(root)
    has_main_logs = any("attempts" not in {part.lower() for part in path.relative_to(root).parts} for path in log_candidates)

    for path in files:
        name = path.name.lower()
        in_attempt = "attempts" in {part.lower() for part in path.relative_to(root).parts}
        if path.suffix.lower() in {".tar", ".gz", ".tgz"} and (path.name.endswith(".tar.gz") or path.suffix.lower() in {".tar", ".tgz"}):
            archives.append(path)
            continue
        if name == "painting.json" or name == "text-judge.json":
            value = _json_object(path)
            state.sources["painting" if name == "painting.json" else "judge"] += 1
            if value is None:
                state.malformed_json_lines += 1
            else:
                _consume_small_json(state, path, value)
            continue
        if _is_metrics(path) and in_attempt and has_main_metrics:
            continue
        if _is_metrics(path) or _is_rollout_jsonl(path) or _is_audit_jsonl(path):
            if _is_zstd_jsonl_name(path.name) and _zstandard is None:
                state.sources["unsupported_zstd_jsonl"] += 1
                continue
            rows, malformed = _jsonl(path)
            state.malformed_json_lines += malformed
            kind = "metrics" if _is_metrics(path) else ("rollout_jsonl" if _is_rollout_jsonl(path) else "audit")
            state.sources[kind] += 1
            _consume_jsonl_rows(state, kind, rows)
            continue
        if _TURN_FILE.search(str(path).replace("\\", "/")) or name in {"response.json", "result.json", "summary.json", "progress.json"}:
            # Avoid model/eval metadata summaries outside rollout directories.
            lower_path = str(path).lower()
            if "rollout" in lower_path or "/turn-" in lower_path or name in {"response.json", "result.json"}:
                if has_turn_receipts and name in {"response.json", "result.json", "summary.json", "progress.json"}:
                    continue
                value = _json_object(path)
                if value is not None:
                    state.sources["rollout_json"] += 1
                    _consume_small_json(state, path, value)
                else:
                    state.malformed_json_lines += 1

    for path in log_candidates:
        if "attempts" in {part.lower() for part in path.relative_to(root).parts} and has_main_logs:
            continue
        state.sources["log"] += 1
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    state.log(line, role=_log_role(path))
        except OSError:
            continue

    # Trainer metrics carry the authoritative wait/active measurements when
    # present.  Log warnings are a fallback and must not be summed on top of
    # the same step timings.
    _apply_metric_timing(state)

    # Archives are read only when present.  If a direct basename already
    # exists, prefer it to avoid counting a copied evidence member twice.
    for archive_path in archives:
        state.sources["archives_scanned"] += 1
        try:
            with tarfile.open(archive_path, mode="r:*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    kind = _member_kind(member.name)
                    if kind is None:
                        continue
                    member_name = member.name.replace("\\", "/").lstrip("./").lower()
                    duplicate_direct = any(
                        member_name == relative or member_name.endswith("/" + relative)
                        for relative in direct_relpaths
                    )
                    if any(part.lower() in _SKIP_DIRS for part in Path(member_name).parts):
                        continue
                    if _is_zstd_jsonl_name(member_name) and _zstandard is None:
                        state.sources["unsupported_zstd_jsonl"] += 1
                        continue
                    if duplicate_direct and kind in {"metrics", "painting", "judge", "log", "audit", "rollout_jsonl"}:
                        continue
                    if kind == "log":
                        state.sources["archive_logs"] += 1
                        try:
                            for line in _archive_member_text(archive, member):
                                state.log(line, role=_log_role(member.name))
                        except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError, RuntimeError, _ZSTD_ERROR):
                            pass
                    elif kind == "painting" or kind == "judge":
                        stream = archive.extractfile(member)
                        if stream is None:
                            continue
                        try:
                            value = json.load(stream)
                        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                            state.malformed_json_lines += 1
                            continue
                        if isinstance(value, dict):
                            state.sources["archive_painting" if kind == "painting" else "archive_judge"] += 1
                            if kind == "painting":
                                state.painting(value)
                            else:
                                state.judge(value)
                    else:
                        rows, malformed = _jsonl_member(archive, member)
                        source_name = "archive_metrics" if kind == "metrics" else ("archive_rollout_jsonl" if kind == "rollout_jsonl" else "archive_audit")
                        state.sources[source_name] += 1
                        state.malformed_json_lines += malformed
                        _consume_jsonl_rows(state, kind, rows)
        except (OSError, tarfile.TarError):
            state.sources["archive_open_errors"] += 1

    _apply_metric_timing(state)

    turns, completion_tokens, token_source = _profile_counts(state, "per-turn receipt tokens field")
    if state.trace_records:
        prompt_tokens = state.native_prompt_tokens
        total_tokens = state.native_total_tokens
        usage_calls = state.native_usage_calls
        calls = state.native_calls
    elif state.fallback_turns:
        prompt_tokens = state.fallback_prompt_tokens
        total_tokens = state.fallback_total_tokens
        usage_calls = state.fallback_usage_rows
        calls = state.fallback_turns
    else:
        prompt_tokens = total_tokens = usage_calls = calls = 0

    trainer_update_candidates = []
    if state.metric_steps:
        trainer_update_candidates.append(len(state.metric_steps))
    if state.trainer_log_steps:
        # A trainer log's highest completed step is authoritative even when
        # the collected window contains only its final SUCCESS line.
        trainer_update_candidates.append(max(state.trainer_log_steps))
    if trainer_update_candidates:
        update_count = max(trainer_update_candidates)
        update_cumulative = state.audit_max_update if state.audit_max_update == update_count else None
        update_basis = "trainer metrics and/or trainer-provenance SUCCESS Step records"
        if state.audit_max_update is not None and state.audit_max_update != update_count:
            update_basis += "; disagreeing audit count was not used"
    elif state.audit_max_update is not None:
        update_count = state.audit_max_update
        update_cumulative = state.audit_max_update
        update_basis = "explicit optimizer audit; trainer completion evidence unavailable"
    else:
        update_count = None
        update_cumulative = None
        update_basis = "no completed optimizer Step or audit record"
    completed_updates = (
        {
            "value": update_count,
            "window": update_count,
            "cumulative": update_cumulative,
            "cumulative_status": "reported" if update_cumulative is not None else "unavailable",
            "status": "reported",
            "basis": update_basis,
        }
        if update_count is not None else
        _unavailable(update_basis)
    )
    if state.effective_reward_steps:
        admitted_value = len(state.effective_reward_steps)
        admitted = {"value": admitted_value, "window": admitted_value, "cumulative": None,
                    "cumulative_status": "unavailable",
                    "status": "reported", "basis": "distinct keyed steps with exact train/agg/effective/agent/reward/mean metric"}
    elif state.admitted_metric_steps:
        admitted_value = len(state.admitted_metric_steps)
        admitted = {"value": admitted_value, "window": admitted_value, "cumulative": None,
                    "cumulative_status": "unavailable",
                    "status": "reported", "basis": "distinct keyed steps with train/agg/effective/agent/is_admitted/mean > 0; fallback admission metric"}
    elif state.admitted_batches:
        admitted_explicit = state.admitted_batches[-1]
        admitted = {"value": admitted_explicit, "window": admitted_explicit, "cumulative": None,
                    "cumulative_status": "unavailable",
                    "status": "reported", "basis": "explicit admitted batch/payload counter"}
    else:
        admitted = _unavailable("no explicit admitted reward-batch counter; optimizer updates are not used as a proxy")

    wait_total = round(sum(state.wait_seconds), 4) if state.wait_seconds else None
    active_total = round(sum(state.active_seconds), 4) if state.active_seconds else None
    denominator = (wait_total or 0.0) + (active_total or 0.0)
    utilization = None if not denominator else round((active_total or 0.0) / denominator, 6)

    if state.guard_counters:
        guard = {key: {"value": value, "status": "reported", "basis": "explicit receipt/log counter"}
                 for key, value in sorted(state.guard_counters.items())}
    else:
        guard = _unavailable("Python-logging guard counters were not emitted; not inferred from logs or metrics")

    failed_trajectories = (
        _count(state.failed_trajectories,
               basis="structured trace error/stop_condition only; invalid renders excluded")
        if state.trace_records
        else _unavailable("no readable native trace rows; failed trajectories were not inferred from render failures")
    )

    observed_discarded = {name: _count(state.discarded[name], basis="explicit Discarded log summaries in observed evidence")
                          for name in ("stale", "error", "no_signal")}
    error_categories = Counter(state.error_categories)
    if not state.trace_records:
        error_categories.update(state.fallback_error_categories)
    syntax = {name: count for name, count in sorted(error_categories.items())}
    operational_errors = {
        name: count for name, count in sorted(state.operational_error_categories.items())
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run": {"label": root.name, "evidence_window": "all readable records under the supplied run directory"},
        "evidence": {
            "sources": dict(sorted(state.sources.items())),
            "malformed_json_lines": state.malformed_json_lines,
            "read_only": True,
            "raw_prompts_or_programs_in_report": False,
        },
        "training": {
            "completed_optimizer_updates": completed_updates,
            "admitted_reward_batches": admitted,
            "python_logging_guard_counters": guard,
            "trainer_wait_vs_active": {
                "wait_seconds": wait_total,
                "active_seconds": active_total,
                "wait_fraction": round((wait_total or 0.0) / denominator, 6) if denominator else None,
                "active_fraction": utilization,
                "denominator_seconds": round(denominator, 4) if denominator else None,
                "measurement_window": state.timing_scope,
                "metric_wait_rows": state.metric_wait_rows,
                "metric_active_rows": state.metric_active_rows,
                "matched_metric_rows": len(state.metric_matched_wait_seconds),
                "status": "reported" if denominator else "unavailable",
            },
        },
        "output": {
            "turns": _count(turns, basis=token_source),
            "completion_tokens": _count(completion_tokens, basis=token_source),
            "prompt_tokens": _count(prompt_tokens, basis=token_source) if prompt_tokens else _unavailable("provider prompt usage was not reported"),
            "total_tokens": _count(total_tokens, basis=token_source) if total_tokens else _unavailable("provider total usage was not reported"),
            "calls": _count(calls, basis=token_source),
            "calls_with_usage": _count(usage_calls, basis=token_source),
            "metric_generation_token_delta": _unavailable(
                "generation_tokens_total metric schema is unverified; use provider usage or native trace tokens"
            ),
        },
        "render": {
            "observations": _count(state.paintings, basis="painting.json receipts"),
            "valid": _count(state.painting_valid, basis="painting.json valid=true"),
            "invalid": _count(state.painting_invalid, basis="painting.json valid=false; intermediate invalid renders are not failed trajectories"),
            "unknown": _count(state.painting_unknown, basis="painting.json without boolean valid field"),
            "timeouts": _count(state.painting_timeouts, basis=f"explicit timeout or latency >= {state.render_timeout:g}s"),
            "latency_seconds": _latency_summary(state.render_latencies),
            "failed_trajectories": failed_trajectories,
        },
        "judge": {
            "records": _count(state.judge_records, basis="text-judge.json receipts"),
            "views": _count(state.judge_views, basis="recorded judge views; views are not assumed to be HTTP calls"),
            "statuses": {name: _count(value, basis="text-judge.json status") for name, value in sorted(state.judge_statuses.items())},
            "consensus": {name: _count(value, basis="text-judge.json consensus") for name, value in sorted(state.judge_consensus.items())},
            "calls": _count(state.judge_calls, basis="verified provider-call leaves under judge views"),
            "timeouts": _count(state.judge_timeouts, basis=f"judge call timeout or latency >= {state.judge_timeout:g}s"),
            "call_latency_seconds": _latency_summary(state.judge_latencies),
            "no_signal_is_separate_from_uncertain": True,
        },
        "discarded": {
            "stale": observed_discarded["stale"],
            "error": observed_discarded["error"],
            "no_signal": observed_discarded["no_signal"],
            "discard_events": _count(state.discard_events, basis="explicit Discarded log summaries"),
            "warning_windows": _count(
                state.discard_warning_windows,
                basis="explicit Discarded summaries with total and >50% discarded; warning-window count, not run total",
            ),
            "scope": "logged Discarded summary counters only; not a run-total; warning threshold is >50% when a total is present",
        },
        "failures": {
            "error_categories": syntax,
            "error_category_scope": "overlapping error mentions; counts are not unique failure instances",
            "operational_error_categories": operational_errors,
            "operational_error_scope": "operational browser/service failures are separated from model failure categories",
            "repeated_deterministic_errors": [
                {"signature": name, "count": count}
                for name, count in sorted(syntax.items()) if count > 1
            ],
        },
        "inference": {
            "mean_requests_running": round(statistics.mean(state.inference_running), 4) if state.inference_running else None,
            "mean_requests_waiting": round(statistics.mean(state.inference_waiting), 4) if state.inference_waiting else None,
            "gpu_utilization": _unavailable("no run-level GPU utilization interval was present in the supplied evidence"),
        },
        "limitations": [
            "Counts are limited to readable evidence under the supplied directory; missing archives or censored records are not reconstructed.",
            "Python-logging guard counters are unavailable unless explicit counters were emitted.",
            "Prime no_signal discard labels are retained as emitted; they may include whole-group guard rejections and are not interpreted as judge uncertainty or absent reward without distinguishing evidence.",
            "No GPU utilization is inferred from instantaneous snapshots, MFU, or trainer busy time.",
        ] + ([
            "Compressed .jsonl.zst evidence was present but zstandard support was unavailable; related counts remain unavailable.",
        ] if state.sources.get("unsupported_zstd_jsonl") else []),
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact archive-safe Markdown summary."""

    training = report["training"]
    output = report["output"]
    render = report["render"]
    judge = report["judge"]
    discarded = report["discarded"]
    failures = report["failures"]

    def value(section: dict[str, Any], key: str) -> str:
        item = section.get(key)
        if not isinstance(item, dict):
            return str(item)
        return str(item.get("value") if item.get("value") is not None else item.get("window", "unavailable"))

    lines = [
        f"# RL throughput and failure profile: {report['run']['label']}",
        "",
        "Read-only summary of the supplied evidence window. Raw prompts, programs, private paths, and credentials are excluded.",
        "",
        "## Training",
        "",
        f"- Completed optimizer updates: {value(training, 'completed_optimizer_updates')}",
        f"- Admitted reward batches: {value(training, 'admitted_reward_batches')}",
        f"- Trainer wait/active seconds: {training['trainer_wait_vs_active']['wait_seconds']} / {training['trainer_wait_vs_active']['active_seconds']}",
        f"- Python logging guard counters: {training['python_logging_guard_counters'].get('status', 'unavailable')}",
        "",
        "## Output and rendering",
        "",
        f"- Turns / completion tokens: {value(output, 'turns')} / {value(output, 'completion_tokens')}",
        f"- Render valid / invalid / timeouts: {value(render, 'valid')} / {value(render, 'invalid')} / {value(render, 'timeouts')}",
        f"- Failed trajectories: {value(render, 'failed_trajectories')} (intermediate invalid renders excluded)",
        f"- Render latency median/p95 seconds: {render['latency_seconds']['median_seconds']} / {render['latency_seconds']['p95_seconds']}",
        "",
        "## Judge and discarded work",
        "",
        f"- Judge statuses: {', '.join(f'{k}={value(judge.get('statuses', {}), k)}' for k in sorted(judge.get('statuses', {}))) or 'unavailable'}",
        f"- Judge views / verified provider calls: {value(judge, 'views')} / {value(judge, 'calls')}",
        f"- Judge call latency median/p95 seconds: {judge['call_latency_seconds']['median_seconds']} / {judge['call_latency_seconds']['p95_seconds']}",
        f"- Discarded stale / error / no_signal: {value(discarded, 'stale')} / {value(discarded, 'error')} / {value(discarded, 'no_signal')}",
        f"- Discard scope: logged summary counters only, not run totals; >50% warning windows: {value(discarded, 'warning_windows')}",
        "- `no_signal` is reported separately from judge uncertainty.",
        "",
        "## Repeated error categories",
        "",
        f"- Error-category counts are overlapping mentions, not unique failures.",
        f"- Operational browser/service errors are reported separately: {', '.join(f'{k}={v}' for k, v in sorted(failures.get('operational_error_categories', {}).items())) or 'none observed'}.",
    ]
    repeated = failures.get("repeated_deterministic_errors") or []
    lines.extend([f"- `{item['signature']}`: {item['count']}" for item in repeated] or ["- None observed in structured error evidence."])
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report.get("limitations", [])], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json-output", type=Path, help="write JSON report; omitted means stdout")
    parser.add_argument("--markdown-output", type=Path, help="write Markdown report")
    parser.add_argument("--render-timeout-seconds", type=float, default=DEFAULT_RENDER_TIMEOUT_SECONDS)
    parser.add_argument("--judge-timeout-seconds", type=float, default=DEFAULT_JUDGE_TIMEOUT_SECONDS)
    args = parser.parse_args()
    report = profile_run(args.run_dir, render_timeout_seconds=args.render_timeout_seconds,
                         judge_timeout_seconds=args.judge_timeout_seconds)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded)
    else:
        print(encoded, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report))


if __name__ == "__main__":
    main()
