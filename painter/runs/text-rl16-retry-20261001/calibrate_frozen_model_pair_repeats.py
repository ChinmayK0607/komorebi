"""Compare two explicitly selected Gateway models on frozen painting pairs.

The input manifest freezes the exact brief/image bytes, model IDs, v2 rubric
source hashes, temperature and response schema.  Each model receives the
same balanced side-order protocol at temperature zero, repeated twice by
default.  This is a consistency calibration with no gold labels; it does not
estimate judge accuracy or change the live reward judge.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable

try:
    from painter import text_pairwise_consistent_side_judge_v2_model as model_judge
except ImportError:  # pragma: no cover - direct execution from a checkout
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_consistent_side_judge_v2_model as model_judge


# Reuse only the frozen-pair input/path/identity helpers.  The prior runner's
# manifests and receipts have a different schema and are never read by this
# runner's output paths.
_OLD_RUNNER_PATH = Path(__file__).with_name("calibrate_frozen_pair_repeats.py")
_OLD_SPEC = importlib.util.spec_from_file_location("frozen_pair_repeat_helpers", _OLD_RUNNER_PATH)
assert _OLD_SPEC and _OLD_SPEC.loader
_OLD = importlib.util.module_from_spec(_OLD_SPEC)
_OLD_SPEC.loader.exec_module(_OLD)


SCHEMA = "text-rl16-frozen-pair-model-repeat-input-v1"
RECEIPT_SCHEMA = "text-rl16-frozen-pair-model-repeat-receipt-v1"
SUMMARY_SCHEMA = "text-rl16-frozen-pair-model-repeat-summary-v1"
METHODS = ("baseline", "candidate")
SPLITS = ("calibration", "confirmation")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPLICATES = 10_000


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _model_config_digest(models: dict[str, str]) -> str:
    value = {
        "models": models,
        "protocol": model_judge.PROTOCOL,
        "format_version": model_judge.FORMAT_VERSION,
        "temperature": model_judge.TEMPERATURE,
        "max_tokens": model_judge.MAX_TOKENS,
        "response_format": model_judge._response_format(),
        "reasoning": {"mode": "provider_default", "request_field_omitted": True},
    }
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str:
    value = str(value)
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe pair identifier: {value!r}")
    return value


def _load_pair_rows(source: Path) -> tuple[list[dict[str, Any]], str]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows, note = value, "The selected panel is descriptive; it is not a random accuracy sample."
    elif isinstance(value, dict) and isinstance(value.get("pairs"), list):
        rows = value["pairs"]
        note_value = value.get("selection_note")
        if not isinstance(note_value, str) or not note_value.strip():
            notes = value.get("notes")
            if isinstance(notes, list) and all(isinstance(item, str) and item.strip() for item in notes):
                note_value = " ".join(item.strip() for item in notes)
        note = note_value if isinstance(note_value, str) else ""
    else:
        raise ValueError("pairs input must be a list or an object with pairs")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("pairs input requires a nonempty selection note")
    return rows, note.strip()


def _freeze_pair_rows(pairs_file: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("each pair must be an object")
        pair_id = _safe_id(raw.get("pair_id", raw.get("trace_id", "")))
        if pair_id in seen:
            raise ValueError(f"duplicate pair identifier: {pair_id}")
        seen.add(pair_id)
        split = raw.get("split")
        if split not in SPLITS:
            raise ValueError(f"pair {pair_id} must use split calibration or confirmation")
        anchor = _OLD._path(raw.get("anchor_path"), base=pairs_file.parent)
        candidate = _OLD._path(raw.get("candidate_path"), base=pairs_file.parent)
        brief_path_value = raw.get("brief_path")
        brief_text = raw.get("brief")
        if brief_path_value is not None:
            brief_path = _OLD._path(brief_path_value, base=pairs_file.parent)
            brief = brief_path.read_text(encoding="utf-8").strip()
            stored_brief_path: str | None = str(brief_path)
        elif isinstance(brief_text, str):
            brief = brief_text.strip()
            stored_brief_path = None
        else:
            raise ValueError(f"pair {pair_id} needs brief_path or brief")
        if not brief or len(brief) > 4000:
            raise ValueError(f"pair {pair_id} has an invalid brief")
        identity = _OLD._source_identity(
            brief_sha256=_OLD.sha256_text(brief), anchor=anchor, candidate=candidate,
        )
        for field in ("brief_sha256", "image_ids", "image_sha256"):
            if field in raw and raw[field] != identity[field]:
                raise ValueError(f"selected pair {pair_id} has stale {field}")
        frozen.append({
            "pair_id": pair_id,
            "split": split,
            "brief": brief if stored_brief_path is None else None,
            "brief_path": stored_brief_path,
            "brief_sha256": identity["brief_sha256"],
            "anchor_path": str(anchor),
            "candidate_path": str(candidate),
            "image_ids": identity["image_ids"],
            "image_sha256": identity["image_sha256"],
            "source_note": str(raw.get("source_note", "")).strip() or None,
        })
    return frozen


def prepare_manifest(*, pairs_file: Path, output: Path, repeat_count: int = 2,
                     baseline_model: str = model_judge.DEFAULT_MODEL,
                     candidate_model: str = model_judge.DEFAULT_MODEL) -> dict[str, Any]:
    """Freeze exactly the selected pair rows without making provider calls."""

    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    models = {
        "baseline": model_judge.validate_model(baseline_model),
        "candidate": model_judge.validate_model(candidate_model),
    }
    rows, selection_note = _load_pair_rows(pairs_file)
    if not rows:
        raise ValueError("pairs input is empty")
    frozen = _freeze_pair_rows(pairs_file, rows)
    if {row["split"] for row in frozen} != set(SPLITS):
        raise ValueError("manifest must contain both calibration and confirmation pairs")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": model_judge.PROTOCOL,
        "format_version": model_judge.FORMAT_VERSION,
        "source_format_sha256": model_judge.SOURCE_FORMAT_SHA256,
        "rubric_source_format_sha256": model_judge.RUBRIC_SOURCE_FORMAT_SHA256,
        "prompt_sha256": model_judge.PROMPT_SHA256,
        "response_schema_sha256": hashlib.sha256(_canonical(model_judge._response_schema()).encode()).hexdigest(),
        "models": models,
        "model_config_sha256": _model_config_digest(models),
        "temperature": model_judge.TEMPERATURE,
        "max_tokens": model_judge.MAX_TOKENS,
        "reasoning": {"mode": "provider_default", "request_field_omitted": True},
        "repeat_count": repeat_count,
        "selection_note": selection_note,
        "gold_labels": False,
        "scope_note": (
            "This is a selected frozen-pair consistency calibration with no gold labels. "
            "Calibration and confirmation are descriptive subsets, not a judge accuracy estimate "
            "or evidence of aggregate reward improvement."
        ),
        "pairs": frozen,
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def read_manifest(path: Path, *, expected_models: dict[str, str] | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected model calibration manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("model calibration manifest integrity mismatch")
    if manifest.get("protocol") != model_judge.PROTOCOL:
        raise ValueError("model calibration protocol changed since preparation")
    for field, expected in (
        ("format_version", model_judge.FORMAT_VERSION),
        ("source_format_sha256", model_judge.SOURCE_FORMAT_SHA256),
        ("rubric_source_format_sha256", model_judge.RUBRIC_SOURCE_FORMAT_SHA256),
        ("prompt_sha256", model_judge.PROMPT_SHA256),
        ("temperature", model_judge.TEMPERATURE),
        ("max_tokens", model_judge.MAX_TOKENS),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"{field} changed since preparation")
    if manifest.get("reasoning") != {"mode": "provider_default", "request_field_omitted": True}:
        raise ValueError("reasoning configuration changed since preparation")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(METHODS):
        raise ValueError("manifest must bind baseline and candidate models")
    models = {method: model_judge.validate_model(models[method]) for method in METHODS}
    if manifest.get("model_config_sha256") != _model_config_digest(models):
        raise ValueError("model configuration digest mismatch")
    if expected_models is not None:
        expected = {method: model_judge.validate_model(expected_models[method]) for method in METHODS}
        if models != expected:
            raise ValueError(f"manifest model configuration differs: {models!r} != {expected!r}")
    if not isinstance(manifest.get("repeat_count"), int) or manifest["repeat_count"] < 1:
        raise ValueError("manifest repeat_count is invalid")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest has no selected pairs")
    if {row.get("split") for row in pairs} != set(SPLITS):
        raise ValueError("manifest must contain both calibration and confirmation pairs")
    return manifest


def _identity(manifest: dict[str, Any], pair: dict[str, Any], method: str, repeat: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"], "split": pair["split"], "method": method,
        "repeat": repeat, "model": manifest["models"][method],
        "model_config_sha256": manifest["model_config_sha256"],
        "protocol": manifest["protocol"], "format_version": manifest["format_version"],
        "source_format_sha256": manifest["source_format_sha256"],
        "rubric_source_format_sha256": manifest["rubric_source_format_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "brief_sha256": pair["brief_sha256"], "image_ids": pair["image_ids"],
        "image_sha256": pair["image_sha256"],
    }


def _pair_paths(pair: dict[str, Any]) -> tuple[str, Path, Path, str]:
    return _OLD._pair_paths(pair)


def _brief_temp(brief: str, output: Path, pair_id: str) -> Path:
    path = output / "briefs" / f"{pair_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"inline brief path is a symlink for {pair_id}")
    if path.is_file() and path.read_text(encoding="utf-8") != brief:
        raise ValueError(f"inline brief changed for {pair_id}")
    if not path.is_file():
        path.write_text(brief, encoding="utf-8")
    return path


def _safe_failure(exc: BaseException) -> str:
    name = type(exc).__name__
    return "judge_failure_" + (name if name.replace("_", "").isalnum() else "ProviderError")


def _exception_status(exc: BaseException) -> str:
    if isinstance(exc, (AttributeError, AssertionError, FileNotFoundError,
                        ImportError, KeyError, PermissionError, TypeError, ValueError)):
        return "implementation_error"
    return "censored"


def _validate_result_identity(result: dict[str, Any], *, manifest: dict[str, Any],
                              pair: dict[str, Any], method: str) -> None:
    expected_model = manifest["models"][method]
    expected = _identity(manifest, pair, method, 0)
    for field in ("protocol", "format_version", "source_format_sha256",
                  "rubric_source_format_sha256", "prompt_sha256"):
        result_field = field
        if result.get(result_field) != expected[result_field]:
            raise ValueError(f"judge result {field} identity mismatch")
    if result.get("requested_model") != expected_model:
        raise ValueError("judge result requested model identity mismatch")
    if result.get("model_config_sha256") != model_judge._model_config_sha256(expected_model):
        raise ValueError("judge result request configuration mismatch")
    if result.get("brief_sha256") != pair["brief_sha256"]:
        raise ValueError("judge result brief identity mismatch")
    if result.get("image_ids") != pair["image_ids"] or result.get("image_sha256") != pair["image_sha256"]:
        raise ValueError("judge result image identity mismatch")
    for view in list(result.get("views") or []) + list(result.get("recheck_views") or []):
        for call in view.get("calls") or []:
            if call.get("requested_model") != expected_model:
                raise ValueError("judge call requested model identity mismatch")
            if call.get("request_config_sha256") != model_judge._model_config_sha256(expected_model):
                raise ValueError("judge call request configuration mismatch")


async def _one(*, manifest: dict[str, Any], pair: dict[str, Any], method: str,
               repeat: int, output: Path, key_file: Path,
               sem: asyncio.Semaphore) -> dict[str, Any]:
    brief, anchor, candidate, pair_id = _pair_paths(pair)
    expected = _identity(manifest, pair, method, repeat)
    path = output / "receipts" / method / f"{pair_id}--repeat-{repeat}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("schema") != RECEIPT_SCHEMA or cached.get("identity") != expected:
            raise ValueError(f"stale receipt: {method}/{pair_id}/repeat-{repeat}")
        return cached
    model = manifest["models"][method]
    async with sem:
        started = time.monotonic()
        try:
            brief_path = _brief_temp(brief, output, pair_id)
            result = await model_judge.judge_pair(
                brief=brief_path, anchor=anchor, candidate=candidate,
                key_file=key_file, model=model,
            )
            _validate_result_identity(result, manifest=manifest, pair=pair, method=method)
            accepted = result.get("status") in {"scored", "identical_canvases"}
            receipt = {
                "schema": RECEIPT_SCHEMA, "identity": expected,
                "status": "scored" if accepted else "censored",
                "reason": None if accepted else result.get("consensus", "incomplete_evidence"),
                "elapsed_seconds_runner": time.monotonic() - started,
                "judge": result,
            }
        except Exception as exc:
            receipt = {
                "schema": RECEIPT_SCHEMA, "identity": expected,
                "status": _exception_status(exc), "reason": _safe_failure(exc),
                "elapsed_seconds_runner": time.monotonic() - started,
            }
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
    return receipt


def _read_receipts(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        for method in METHODS:
            for repeat in range(1, manifest["repeat_count"] + 1):
                path = output / "receipts" / method / f"{pair['pair_id']}--repeat-{repeat}.json"
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(path)
                row = json.loads(path.read_text(encoding="utf-8"))
                expected = _identity(manifest, pair, method, repeat)
                if row.get("schema") != RECEIPT_SCHEMA or row.get("identity") != expected:
                    raise ValueError(f"receipt identity mismatch: {method}/{pair['pair_id']}/repeat-{repeat}")
                row["_pair"] = pair
                rows.append(row)
    return rows


def _decision(row: dict[str, Any]) -> tuple[str, str] | None:
    result = row.get("judge") or {}
    if row.get("status") != "scored" or result.get("status") not in {"scored", "identical_canvases"}:
        return None
    winner, preference = result.get("winner_id"), result.get("preference")
    if not isinstance(winner, str) or not isinstance(preference, str):
        return None
    return winner, preference


def _calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for view in list(result.get("views") or []) + list(result.get("recheck_views") or []):
        calls.extend(call for call in view.get("calls") or [] if isinstance(call, dict))
    return calls


def _usage(result: dict[str, Any]) -> tuple[int, float, dict[str, int], list[float], Counter[str]]:
    calls = _calls(result)
    totals: dict[str, int] = {}
    cost = 0.0
    latencies: list[float] = []
    failures: Counter[str] = Counter()
    for call in calls:
        if isinstance(call.get("elapsed_seconds"), (int, float)):
            latencies.append(float(call["elapsed_seconds"]))
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(name), (int, float)) and not isinstance(usage.get(name), bool):
                totals[name] = totals.get(name, 0) + int(usage[name])
        reasoning = usage.get("completion_tokens_details")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("reasoning_tokens"), (int, float)):
            totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + int(reasoning["reasoning_tokens"])
        if isinstance(usage.get("reasoning_tokens"), (int, float)):
            totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + int(usage["reasoning_tokens"])
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            cost += float(usage["cost"])
        for field in ("provider_error", "parse_error"):
            if isinstance(call.get(field), str):
                failures[call[field]] += 1
    return len(calls), cost, totals, latencies, failures


def _order(row: dict[str, Any]) -> tuple[bool | None, list[str]]:
    result = row.get("judge") or {}
    votes = result.get("initial_votes")
    if not isinstance(votes, list) or len(votes) != 2 or not all(isinstance(v, str) for v in votes):
        return None, []
    return votes[0] == votes[1], votes


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "receipt_count": len(rows), "scored_count": 0, "censored_count": 0,
        "implementation_error_count": 0, "uncertain_count": 0,
        "winner_counts": {}, "preference_counts": {}, "status_counts": {},
        "failure_reasons": {}, "initial_order_valid_count": 0,
        "initial_order_agreement_count": 0, "initial_order_conflict_count": 0,
        "api_call_count": 0, "reported_cost_usd": 0.0, "usage": {},
        "latency_seconds": {"call_count": 0, "total": 0.0, "mean": None, "max": None},
    }
    all_latencies: list[float] = []
    failures: Counter[str] = Counter()
    for row in rows:
        status = row.get("status", "missing")
        out["status_counts"][status] = out["status_counts"].get(status, 0) + 1
        result = row.get("judge") or {}
        decision = _decision(row)
        if decision is None:
            if status == "implementation_error":
                out["implementation_error_count"] += 1
            else:
                out["censored_count"] += 1
                if result.get("status") == "uncertain":
                    out["uncertain_count"] += 1
            if isinstance(row.get("reason"), str):
                failures[row["reason"]] += 1
        else:
            out["scored_count"] += 1
            winner, preference = decision
            out["winner_counts"][winner] = out["winner_counts"].get(winner, 0) + 1
            out["preference_counts"][preference] = out["preference_counts"].get(preference, 0) + 1
        agreement, _ = _order(row)
        if agreement is not None:
            out["initial_order_valid_count"] += 1
            key = "initial_order_agreement_count" if agreement else "initial_order_conflict_count"
            out[key] += 1
        calls, cost, usage, latencies, call_failures = _usage(result)
        out["api_call_count"] += calls
        out["reported_cost_usd"] += cost
        all_latencies.extend(latencies)
        failures.update(call_failures)
        for name, value in usage.items():
            out["usage"][name] = out["usage"].get(name, 0) + value
    out["failure_reasons"] = dict(failures)
    out["latency_seconds"] = {
        "call_count": len(all_latencies),
        "total": sum(all_latencies),
        "mean": sum(all_latencies) / len(all_latencies) if all_latencies else None,
        "max": max(all_latencies) if all_latencies else None,
    }
    return out


def _repeat_metrics(rows: list[dict[str, Any]], repeat_count: int) -> tuple[dict[str, Any], list[bool]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["_pair"]["pair_id"]].append(row)
    eligible = winner_consistent = exact_consistent = magnitude_only = unresolved = implementation_errors = 0
    flags: list[bool] = []
    for pair_rows in by_pair.values():
        pair_rows.sort(key=lambda row: row["identity"]["repeat"])
        decisions = [_decision(row) for row in pair_rows]
        if len(pair_rows) != repeat_count or any(decision is None for decision in decisions):
            unresolved += 1
            implementation_errors += int(any(row.get("status") == "implementation_error" for row in pair_rows))
            continue
        eligible += 1
        assert all(decision is not None for decision in decisions)
        winners = {decision[0] for decision in decisions}
        winner_ok = len(winners) == 1
        flags.append(winner_ok)
        winner_consistent += int(winner_ok)
        exact_consistent += int(len(set(decisions)) == 1)
        magnitude_only += int(winner_ok and len(set(decisions)) != 1)
    return ({
        "pair_count": len(by_pair), "repeat_count": repeat_count,
        "all_repeats_scored_pairs": eligible,
        "unresolved_or_censored_pairs": unresolved,
        "implementation_error_pairs": implementation_errors,
        "winner_consistent_pairs": winner_consistent,
        "exact_decision_consistent_pairs": exact_consistent,
        "magnitude_only_change_pairs": magnitude_only,
        "winner_consistency_rate": winner_consistent / eligible if eligible else None,
        "exact_decision_consistency_rate": exact_consistent / eligible if eligible else None,
    }, flags)


def _paired_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[bool]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        ident = row["identity"]
        by_key[(ident["pair_id"], ident["repeat"])][ident["method"]] = row
    paired = winner_agreement = exact_agreement = winner_flip = unresolved = implementation_errors = 0
    flags: list[bool] = []
    for methods in by_key.values():
        baseline, candidate = methods.get("baseline"), methods.get("candidate")
        if baseline is None or candidate is None:
            continue
        first, second = _decision(baseline), _decision(candidate)
        if first is None or second is None:
            unresolved += 1
            implementation_errors += int(any(row.get("status") == "implementation_error" for row in (baseline, candidate)))
            continue
        paired += 1
        same = first[0] == second[0]
        flags.append(same)
        winner_agreement += int(same)
        winner_flip += int(not same)
        exact_agreement += int(first == second)
    return ({
        "paired_scored_repeats": paired,
        "unresolved_or_censored_repeats": unresolved,
        "implementation_error_repeats": implementation_errors,
        "winner_agreement": winner_agreement, "winner_flip": winner_flip,
        "exact_decision_agreement": exact_agreement,
        "winner_agreement_rate": winner_agreement / paired if paired else None,
        "exact_decision_agreement_rate": exact_agreement / paired if paired else None,
    }, flags)


def _paired_pair_flags(rows: list[dict[str, Any]], repeat_count: int) -> list[bool]:
    """Return one cross-model agreement flag per fully eligible pair."""

    by_pair: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        ident = row["identity"]
        by_pair[ident["pair_id"]][ident["repeat"]][ident["method"]] = row
    flags: list[bool] = []
    for repeats in by_pair.values():
        if set(repeats) != set(range(1, repeat_count + 1)):
            continue
        comparisons: list[bool] = []
        for repeat in range(1, repeat_count + 1):
            pair_rows = repeats[repeat]
            baseline = _decision(pair_rows.get("baseline", {})) if "baseline" in pair_rows else None
            candidate = _decision(pair_rows.get("candidate", {})) if "candidate" in pair_rows else None
            if baseline is None or candidate is None:
                break
            comparisons.append(baseline[0] == candidate[0])
        if len(comparisons) == repeat_count:
            flags.append(all(comparisons))
    return flags


def _bootstrap_interval(values: list[bool], *, seed: int = BOOTSTRAP_SEED,
                        replicates: int = BOOTSTRAP_REPLICATES) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    return [draws[int(0.025 * (replicates - 1))], draws[int(0.975 * (replicates - 1))]]


def summarize(*, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    repeats: dict[str, Any] = {}
    repeat_flags: dict[str, list[bool]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["identity"]["method"] == method]
        metrics[method] = {
            split: _metrics([row for row in method_rows if row["_pair"]["split"] == split])
            for split in SPLITS
        }
        metrics[method]["total"] = _metrics(method_rows)
        repeats[method] = {}
        for split in SPLITS:
            value, flags = _repeat_metrics(
                [row for row in method_rows if row["_pair"]["split"] == split],
                manifest["repeat_count"],
            )
            repeats[method][split] = value
        repeats[method]["total"], repeat_flags[method] = _repeat_metrics(method_rows, manifest["repeat_count"])
    paired: dict[str, Any] = {}
    for split in SPLITS:
        paired[split], _ = _paired_metrics([row for row in rows if row["_pair"]["split"] == split])
    paired["total"], _ = _paired_metrics(rows)
    paired_flags = _paired_pair_flags(rows, manifest["repeat_count"])
    compact_rows = []
    for row in rows:
        result = row.get("judge") or {}
        decision = _decision(row)
        order_agreement, order_votes = _order(row)
        calls = _calls(result)
        compact_rows.append({
            "pair_id": row["identity"]["pair_id"], "split": row["identity"]["split"],
            "method": row["identity"]["method"], "model": row["identity"]["model"],
            "repeat": row["identity"]["repeat"], "status": row["status"],
            "reason": row.get("reason"), "judge_status": result.get("status"),
            "winner_id": decision[0] if decision else None,
            "preference": decision[1] if decision else None,
            "score": result.get("score") if decision else None,
            "requested_model": result.get("requested_model"),
            "returned_models": sorted({call.get("returned_model") for call in calls if call.get("returned_model")}),
            "observed_reasoning_tokens": sum(
                _usage(result)[2].get("reasoning_tokens", 0) for _ in [0]
            ),
            "initial_order_agreement": order_agreement, "initial_votes": order_votes,
        })
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "protocol": manifest["protocol"], "format_version": manifest["format_version"],
        "source_format_sha256": manifest["source_format_sha256"],
        "rubric_source_format_sha256": manifest["rubric_source_format_sha256"],
        "prompt_sha256": manifest["prompt_sha256"], "models": manifest["models"],
        "model_config_sha256": manifest["model_config_sha256"],
        "temperature": manifest["temperature"], "reasoning": manifest["reasoning"],
        "pair_count": len(manifest["pairs"]), "repeat_count": manifest["repeat_count"],
        "selection_note": manifest["selection_note"], "gold_labels": False,
        "scope_note": manifest["scope_note"], "metrics": metrics,
        "repeat_consistency": repeats,
        "total_repeat_consistency": {method: repeats[method]["total"] for method in METHODS},
        "paired_model_comparison": paired,
        "bootstrap_diagnostics": {
            "gold_labels": False, "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "eligible selected frozen-pair diagnostics",
            "repeat_winner_consistency_95pct": {
                method: _bootstrap_interval(flags, seed=BOOTSTRAP_SEED + index)
                for index, (method, flags) in enumerate(repeat_flags.items())
            },
            "cross_model_winner_agreement_95pct": _bootstrap_interval(
                paired_flags, seed=BOOTSTRAP_SEED + 2,
            ),
            "note": "Descriptive internal consistency intervals only; no gold-label accuracy claim.",
        },
        "rows": compact_rows,
    }


async def run(*, manifest_path: Path, output: Path, key_file: Path,
              baseline_model: str, candidate_model: str,
              concurrency: int = 4) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    expected_models = {
        "baseline": model_judge.validate_model(baseline_model),
        "candidate": model_judge.validate_model(candidate_model),
    }
    manifest = read_manifest(manifest_path, expected_models=expected_models)
    output.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    tasks: list[Awaitable[dict[str, Any]]] = []
    for pair in manifest["pairs"]:
        for method in METHODS:
            for repeat in range(1, manifest["repeat_count"] + 1):
                tasks.append(_one(manifest=manifest, pair=pair, method=method,
                                  repeat=repeat, output=output, key_file=key_file, sem=sem))
    await asyncio.gather(*tasks)
    rows = _read_receipts(output, manifest)
    summary = summarize(manifest=manifest, rows=rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, help="JSON selected pair list for --prepare-only")
    parser.add_argument("--manifest", type=Path, help="prepared model-comparison manifest for --run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, help="externally supplied Gateway credential file")
    parser.add_argument("--baseline-model", default=model_judge.DEFAULT_MODEL)
    parser.add_argument("--candidate-model", default=model_judge.DEFAULT_MODEL)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.pairs is None:
            parser.error("--pairs is required with --prepare-only")
        manifest = prepare_manifest(
            pairs_file=args.pairs, output=args.output, repeat_count=args.repeat_count,
            baseline_model=args.baseline_model, candidate_model=args.candidate_model,
        )
        print(json.dumps({"manifest": str(args.output / "input-manifest.json"),
                          "pairs": len(manifest["pairs"]), "models": manifest["models"]}, sort_keys=True))
        return 0
    if args.manifest is None or args.key_file is None:
        parser.error("--manifest and --key-file are required with --run")
    summary = asyncio.run(run(
        manifest_path=args.manifest, output=args.output, key_file=args.key_file,
        baseline_model=args.baseline_model, candidate_model=args.candidate_model,
        concurrency=args.concurrency,
    ))
    print(json.dumps({"pairs": summary["pair_count"], "repeat_count": summary["repeat_count"],
                      "models": summary["models"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
