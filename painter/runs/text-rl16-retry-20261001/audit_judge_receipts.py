#!/usr/bin/env python3
"""Audit exported RL16 trace metadata and side-judge receipts offline.

The live trainer writes one ``text-judge.json`` beside each text rollout.  A
sanitized evidence bundle may instead contain only JSON/JSONL metadata, with
the receipt path replaced by a relative path.  This checker accepts either
shape, joins records by trace ID where possible, and never calls a renderer,
provider, or model.  It is intentionally standard-library only.

Usage::

    python audit_judge_receipts.py /path/to/judge-audit

The default output files are ``judge-audit-report.json`` and
``judge-audit-report.html`` below the supplied directory.  Use ``--output`` to
choose another report prefix.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


MAGNITUDES = ("tied", "slight", "moderate", "clear")
UNCERTAIN_STATUSES = frozenset({
    "judge_uncertain", "uncertain", "censored", "missing", "service_failure",
})
RECEIPT_NAMES = frozenset({"text-judge.json", "judge-receipt.json", "receipt.json"})
REPORT_NAMES = frozenset({"judge-audit-report.json", "judge-audit-report.html"})


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _nested(record: dict[str, Any], *names: str) -> dict[str, Any]:
    """Return the first nested trace/info/data object, if present."""

    for name in names:
        value = record.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _trace_id(record: dict[str, Any], *, fallback: str | None = None) -> str | None:
    value = _first(record, "trace_id", "trace", "episode_id", "rollout_id", "id")
    if isinstance(value, dict):
        value = _first(value, "id", "trace_id", "episode_id")
    if value is None:
        value = _first(_nested(record, "trace_info", "info", "trace"), "trace_id", "id")
    return str(value) if value is not None else fallback


def _looks_like_receipt(record: dict[str, Any], path: Path) -> bool:
    if path.name in RECEIPT_NAMES or path.parent.name in {"text-judge", "judges", "receipts"}:
        return True
    return (
        "views" in record
        and ("protocol" in record or "winner_id" in record)
        and ("score" in record or "status" in record)
    )


def _unwrap_records(value: Any, source: Path) -> Iterable[tuple[dict[str, Any], Path]]:
    """Yield record-like dictionaries from common export wrappers."""

    if isinstance(value, dict):
        # A receipt or a trace itself must remain one record; do not recurse
        # through its nested ``views`` or ``info`` dictionaries.
        if _looks_like_receipt(value, source) or any(
            key in value for key in ("trace_id", "reward_status", "final_canvas_score", "comparison_images")
        ):
            yield value, source
            return
        for key in ("records", "rows", "traces", "episodes", "items", "rollouts", "data"):
            child = value.get(key)
            if isinstance(child, list):
                for item in child:
                    yield from _unwrap_records(item, source)
                return
        yield value, source
        return
    if isinstance(value, list):
        for item in value:
            yield from _unwrap_records(item, source)


def _looks_like_trace(record: dict[str, Any], path: Path) -> bool:
    """Exclude raw Prime metric rows from the trace denominator."""

    if path.name in {"skinny-traces.json", "traces.json", "trace-metadata.jsonl"}:
        return True
    return any(key in record for key in (
        "trace_id", "task_id", "brief", "group", "info", "rewards", "comparison_images",
    ))


def _read_documents(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    traces: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        if (path.name in REPORT_NAMES or path.name == "visual-selection.json"
                or path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}):
            continue
        try:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                values = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {type(exc).__name__}")
            continue
        for value in values:
            for record, source in _unwrap_records(value, path):
                if _looks_like_receipt(record, source):
                    receipts.append({
                        "record": record,
                        "source": str(source),
                        "trace_id": _trace_id(record),
                        # Repeat receipts are deliberately kept separate from
                        # the shipped receipt denominator.  They are selected
                        # frozen-pair diagnostics, not additional rollouts.
                        "is_repeat": "repeat" in source.parts,
                    })
                elif _looks_like_trace(record, source):
                    traces.append({"record": record, "source": str(source), "trace_id": _trace_id(record)})
                elif isinstance(record, dict) and (
                    "train/agg/effective/agent/reward/mean" in record
                    or "train/agg/all/agent/reward/mean" in record
                ):
                    metrics.append({"record": record, "source": str(source)})
    return traces, receipts, metrics, errors


def _receipt_key(item: dict[str, Any]) -> str | None:
    value = item.get("trace_id")
    if value:
        return str(value)
    path = Path(str(item.get("source", "")))
    if path.name in RECEIPT_NAMES:
        return path.parent.name or None
    stem = path.stem
    return stem if stem and stem not in {"judges", "receipts", "text-judge"} else None


def _receipt_from_trace(trace: dict[str, Any], receipts_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    record = trace["record"]
    embedded = _first(record, "text_judge", "judge_receipt", "receipt", "judge")
    if isinstance(embedded, dict) and ("score" in embedded or "views" in embedded):
        return {"record": embedded, "source": trace["source"], "trace_id": trace.get("trace_id")}
    paths: list[str] = []
    for container in (record, _nested(record, "info", "trace_info")):
        value = _first(container, "text_judge_path", "judge_receipt_path", "receipt_path", "judge_path")
        if isinstance(value, str):
            paths.append(value)
    trace_id = trace.get("trace_id")
    if trace_id and str(trace_id) in receipts_by_id:
        return receipts_by_id[str(trace_id)]
    for path in paths:
        candidate = Path(path)
        for key, item in receipts_by_id.items():
            if key == path or Path(str(item.get("source", ""))).name == candidate.name:
                return item
    return None


def _view_assessment(view: Any) -> dict[str, Any] | None:
    if not isinstance(view, dict):
        return None
    value = view.get("assessment")
    if isinstance(value, dict):
        return value
    if "winner_id" in view or "preference" in view:
        return view
    return None


def _votes(receipt: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {"initial": [], "recheck": [], "fresh": []}
    for bucket, names in (
        ("initial", ("views",)),
        ("recheck", ("recheck_views",)),
        ("fresh", ("fresh_view",)),
    ):
        for name in names:
            value = receipt.get(name)
            if isinstance(value, list):
                values = value
            elif value is None:
                values = []
            else:
                values = [value]
            for view in values:
                assessment = _view_assessment(view)
                if not assessment:
                    continue
                winner = assessment.get("winner_id")
                preference = assessment.get("preference")
                if isinstance(winner, str) and isinstance(preference, str):
                    result[bucket].append({"winner_id": winner, "preference": preference})
    return result


def _expected_score(receipt: dict[str, Any]) -> float | None:
    winner = receipt.get("winner_id")
    preference = receipt.get("preference")
    image_ids = receipt.get("image_ids")
    candidate = image_ids.get("candidate") if isinstance(image_ids, dict) else None
    if winner == "tie" and preference == "tied":
        return 0.5
    if not isinstance(winner, str) or not isinstance(preference, str) or not isinstance(candidate, str):
        return None
    if preference not in MAGNITUDES or winner == "tie":
        return None
    delta = MAGNITUDES.index(preference) / 6.0
    return 0.5 + delta if winner == candidate else 0.5 - delta


def _consensus_choice(receipt: dict[str, Any]) -> tuple[str, str] | None:
    """Recompute the side judge's selected winner and weakest magnitude."""

    if receipt.get("status") == "identical_canvases":
        return ("tie", "tied")
    votes = _votes(receipt)
    initial = votes["initial"]
    if len(initial) != 2:
        return None
    if initial[0]["winner_id"] == initial[1]["winner_id"]:
        winner = initial[0]["winner_id"]
        if winner == "tie":
            return ("tie", "tied")
        return (winner, min((vote["preference"] for vote in initial), key=MAGNITUDES.index))
    recheck = votes["recheck"]
    if len(recheck) != 2:
        return None
    all_votes = initial + recheck
    counts = collections.Counter(vote["winner_id"] for vote in all_votes)
    majority = next((winner for winner, count in counts.items() if count >= 3), None)
    if majority is None:
        return None
    if majority == "tie":
        return ("tie", "tied")
    magnitudes = [vote["preference"] for vote in all_votes if vote["winner_id"] == majority]
    return (majority, min(magnitudes, key=MAGNITUDES.index)) if magnitudes else None


def _hash_file(path_value: Any) -> str | None:
    if not isinstance(path_value, str):
        return None
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _metadata_image_hash_issues(receipt: dict[str, Any], trace: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    canonical_ids = receipt.get("image_ids")
    canonical_hashes = receipt.get("image_sha256")
    if not isinstance(canonical_ids, dict) or not isinstance(canonical_hashes, dict):
        return ["missing_canonical_image_identity"]
    if not all(isinstance(canonical_ids.get(k), str) and isinstance(canonical_hashes.get(k), str)
               for k in ("anchor", "candidate")):
        issues.append("incomplete_canonical_image_identity")
    for bucket in ("views", "recheck_views", "fresh_view"):
        value = receipt.get(bucket)
        values = value if isinstance(value, list) else ([] if value is None else [value])
        for view_index, view in enumerate(values):
            calls = view.get("calls", []) if isinstance(view, dict) else []
            calls = calls if isinstance(calls, list) else []
            call = next((item for item in reversed(calls) if isinstance(item, dict) and "assessment" in item), None)
            if call is None:
                continue
            ids = call.get("image_ids")
            hashes = call.get("image_sha256")
            if not isinstance(ids, dict) or not isinstance(hashes, dict) or set(ids) != {"A", "B"}:
                issues.append(f"{bucket}[{view_index}].missing_side_identity")
                continue
            # A/B are deliberately swapped between the two canonical views.
            for side in ("A", "B"):
                role = "anchor" if ids.get(side) == canonical_ids.get("anchor") else (
                    "candidate" if ids.get(side) == canonical_ids.get("candidate") else None
                )
                if role is None or hashes.get(side) != canonical_hashes.get(role):
                    issues.append(f"{bucket}[{view_index}].{side}_identity_mismatch")
    # If paths survived sanitization, verify downloaded image bytes as well.
    for container in (receipt, _nested(receipt, "comparison_images"), _nested(trace or {}, "comparison_images")):
        paths = container.get("paths") if isinstance(container, dict) else None
        if isinstance(paths, dict):
            candidates = paths
        else:
            candidates = container if isinstance(container, dict) else {}
        for role in ("anchor", "candidate", "reference"):
            digest = _hash_file(candidates.get(role))
            expected = canonical_hashes.get(role) if role in canonical_hashes else None
            if digest is not None and expected is not None and digest != expected:
                issues.append(f"downloaded_{role}_hash_mismatch")
    return sorted(set(issues))


def _trace_fields(trace: dict[str, Any]) -> dict[str, Any]:
    record = trace["record"]
    info = _nested(record, "info", "trace_info")
    data = _nested(record, "data", "task", "task_data")
    group = record.get("group") if isinstance(record.get("group"), dict) else {}
    trace_id = trace.get("trace_id") or _trace_id(record)
    batch = _first(record, "batch", "batch_id", "optimizer_step", "step", "train_step", "update")
    if batch is None:
        batch = _first(info, "batch", "batch_id", "optimizer_step", "step", "train_step", "update")
    dispatch = info.get("dispatch") if isinstance(info.get("dispatch"), dict) else {}
    arrival = info.get("arrival") if isinstance(info.get("arrival"), dict) else {}
    if batch is None:
        batch = dispatch.get("step")
    family = _first(record, "family", "tier", "source_family") or _first(data, "family", "tier") or "unknown"
    prompt = _first(record, "prompt_id", "task_id", "case_id", "source_group", "group", "name")
    prompt = prompt or _first(data, "task_id", "source_group", "group", "id") or "unknown"
    source_group = _first(record, "source_group", "group") or _first(data, "source_group", "group")
    status = _first(record, "reward_status", "status") or _first(info, "reward_status", "status")
    score = _finite(_first(record, "final_canvas_score", "score"))
    if score is None:
        score = _finite(_first(info, "final_canvas_score", "score"))
    if score is None:
        rewards = record.get("rewards")
        if isinstance(rewards, dict):
            reward = rewards.get("final_canvas_aesthetic")
            if isinstance(reward, dict):
                score = _finite(reward.get("score"))
    candidate_sha = _first(record, "candidate_sha256", "image_sha256_candidate")
    images = _nested(record, "comparison_images") or _nested(info, "comparison_images")
    image_hashes = images.get("image_sha256") if isinstance(images, dict) else None
    if candidate_sha is None and isinstance(image_hashes, dict):
        candidate_sha = image_hashes.get("candidate")
    final_validation = _nested(info, "final_action_validation")
    turns = info.get("turns") if isinstance(info.get("turns"), list) else []
    turn_failures: list[str] = []
    turn_error_classes: collections.Counter[str] = collections.Counter()
    no_change_count = 0
    finish_count = 0
    valid_turn_count = 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if turn.get("no_change"):
            no_change_count += 1
        if str(turn.get("action", "")).lower() == "finish":
            finish_count += 1
        if turn.get("valid"):
            valid_turn_count += 1
        failure = turn.get("failure_kind")
        if isinstance(failure, str) and failure:
            turn_failures.append(failure)
        text = " ".join(str(turn.get(key, "")) for key in ("error", "renderer_errors"))
        lowered = text.lower()
        if re.search(r"syntaxerror|unexpected token|missing initializer|check_syntax", lowered):
            turn_error_classes["syntax_error"] += 1
        elif re.search(r"timeout|timed out|time limit|deadline", lowered):
            turn_error_classes["timeout"] += 1
        elif re.search(r"brush|p5\.|referenceerror|not defined|undefined", lowered):
            turn_error_classes["unsupported_or_runtime_api"] += 1
        elif text.strip():
            turn_error_classes["other_program_error"] += 1
    if not turn_error_classes and isinstance(final_validation.get("failure_kind"), str):
        turn_error_classes[str(final_validation["failure_kind"])] += 1
    retained = bool(
        final_validation.get("retained_final_canvas")
        and final_validation.get("retained_final_canvas_valid")
    )
    if not retained and not info.get("final_canvas_valid") and valid_turn_count:
        retained = any(isinstance(turn, dict) and turn.get("canvas") and turn.get("valid") for turn in turns)
    final_action = _first(final_validation, "action") or (turns[-1].get("action") if turns and isinstance(turns[-1], dict) else None)
    return {
        "trace_id": str(trace_id) if trace_id is not None else None,
        "batch": str(batch) if batch is not None else "unknown",
        "dispatch_step": str(dispatch.get("step")) if dispatch.get("step") is not None else None,
        "arrival_step": str(arrival.get("step")) if arrival.get("step") is not None else None,
        "group_id": str(_first(group, "id", "group_id")) if _first(group, "id", "group_id") is not None else None,
        "family": str(family),
        "prompt": str(prompt),
        "source_group": str(source_group) if source_group is not None else None,
        "status": str(status) if status is not None else None,
        "score": score,
        "candidate_sha256": str(candidate_sha) if candidate_sha else None,
        "final_canvas_valid": bool(info.get("final_canvas_valid")),
        "retained_valid_earlier": retained,
        "all_turns_invalid": bool(turns) and valid_turn_count == 0,
        "failure_kind": _first(info, "failure_kind") or _first(final_validation, "failure_kind"),
        "turn_failures": dict(sorted(collections.Counter(turn_failures).items())),
        "turn_error_classes": dict(sorted(turn_error_classes.items())),
        "final_action": str(final_action) if final_action is not None else None,
        "finish_count": finish_count,
        "no_change_count": no_change_count,
        "valid_turn_count": valid_turn_count,
        "source": trace.get("source"),
    }


def _category(fields: dict[str, Any], receipt: dict[str, Any] | None) -> str:
    status = (fields.get("status") or "").lower()
    if status.startswith("model_invalid"):
        return "model_invalid_final"
    if status == "anchor_invalid_model_output":
        return "anchor_invalid_model_output"
    if status in UNCERTAIN_STATUSES or "failure" in status or fields.get("score") is None:
        return "uncertain_or_missing"
    if status == "scored" or (receipt and receipt.get("status") in {"scored", "identical_canvases"}):
        return "scored"
    return "other"


def _audit_receipt(item: dict[str, Any], trace: dict[str, Any] | None) -> dict[str, Any]:
    receipt = item["record"]
    votes = _votes(receipt)
    initial = votes["initial"]
    recheck = votes["recheck"]
    winners = [vote["winner_id"] for vote in initial]
    magnitudes = [vote["preference"] for vote in initial]
    expected = _expected_score(receipt)
    expected_consensus = _consensus_choice(receipt)
    reported = _finite(receipt.get("score"))
    issues: list[str] = []
    if expected is not None and (reported is None or abs(expected - reported) > 1e-9):
        issues.append("score_recomputation_mismatch")
    if (receipt.get("winner_id") == "tie") != (receipt.get("preference") == "tied"):
        issues.append("winner_preference_inconsistency")
    if receipt.get("status") in {"scored", "identical_canvases"}:
        selected = (receipt.get("winner_id"), receipt.get("preference"))
        if expected_consensus is None:
            issues.append("scored_without_consensus_evidence")
        elif selected != expected_consensus:
            issues.append("consensus_selection_mismatch")
    if len(initial) == 2 and len(set(winners)) > 1:
        issues.append("initial_swapped_winner_conflict")
    if len(initial) == 2 and winners[0] == winners[1] and len(set(magnitudes)) > 1:
        issues.append("initial_magnitude_conflict")
    if len(recheck) == 2 and len({vote["winner_id"] for vote in recheck}) > 1:
        issues.append("recheck_swapped_winner_conflict")
    if len(recheck) == 2 and len({vote["preference"] for vote in recheck}) > 1:
        issues.append("recheck_magnitude_conflict")
    issues.extend(_metadata_image_hash_issues(receipt, trace))
    trace_fields = _trace_fields(trace) if trace else None
    if trace_fields and trace_fields.get("score") is not None and reported is not None:
        if abs(trace_fields["score"] - reported) > 1e-9:
            issues.append("trace_receipt_score_mismatch")
    return {
        "trace_id": item.get("trace_id") or _trace_id(receipt),
        "source": item.get("source"),
        "status": receipt.get("status"),
        "score": reported,
        "expected_score": expected,
        "expected_consensus": list(expected_consensus) if expected_consensus else None,
        "winner_id": receipt.get("winner_id"),
        "preference": receipt.get("preference"),
        "image_ids": receipt.get("image_ids"),
        "image_sha256": receipt.get("image_sha256"),
        "votes": votes,
        "issues": sorted(set(issues)),
        "trace": trace_fields,
    }


def _batch_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get("batch", "unknown"))].append(row)
    result: list[dict[str, Any]] = []
    for batch, members in grouped.items():
        counts = collections.Counter(member["category"] for member in members)
        scored = [member["score"] for member in members if member["category"] == "scored" and member["score"] is not None]
        result.append({
            "batch": batch,
            "count": len(members),
            "categories": dict(sorted(counts.items())),
            "scored_count": len(scored),
            "visual_conditional_mean": sum(scored) / len(scored) if scored else None,
            "families": dict(sorted(collections.Counter(member["family"] for member in members).items())),
            "prompts": dict(sorted(collections.Counter(member["prompt"] for member in members).items())),
        })
    def order(item: dict[str, Any]) -> tuple[int, str]:
        match = re.fullmatch(r"-?\d+", item["batch"])
        return (int(item["batch"]) if match else 10**12, item["batch"])
    return sorted(result, key=order)


def _conditional_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = row.get(field) or "unknown"
        grouped[str(key)].append(row)
    result: list[dict[str, Any]] = []
    for key, members in grouped.items():
        scored = [member["score"] for member in members if member["category"] == "scored" and member["score"] is not None]
        result.append({
            field: key,
            "count": len(members),
            "categories": dict(sorted(collections.Counter(member["category"] for member in members).items())),
            "scored_count": len(scored),
            "visual_conditional_mean": sum(scored) / len(scored) if scored else None,
            "families": dict(sorted(collections.Counter(member["family"] for member in members).items())),
            "prompts": dict(sorted(collections.Counter(member["prompt"] for member in members).items())),
        })
    def order(item: dict[str, Any]) -> tuple[int, str]:
        raw = str(item[field])
        return (int(raw) if raw.lstrip("-").isdigit() else 10**12, raw)
    return sorted(result, key=order)


def _repeated_prompt_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        prompt = row.get("prompt")
        if prompt and prompt != "unknown":
            grouped[str(prompt)].append(row)
    result: list[dict[str, Any]] = []
    for prompt, members in grouped.items():
        if len(members) < 2:
            continue
        scores = [member["score"] for member in members if member.get("score") is not None]
        signatures = sorted(
            {(member.get("score"), member.get("winner_id"), member.get("preference")) for member in members},
            key=lambda signature: (signature[0] is None, signature[0] if signature[0] is not None else 0,
                                   str(signature[1]), str(signature[2])),
        )
        result.append({
            "prompt": prompt,
            "count": len(members),
            "scored_count": len(scores),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_range": max(scores) - min(scores) if scores else None,
            "distinct_decisions": len(signatures),
            "decisions": [list(signature) for signature in signatures],
            "trace_ids": sorted(str(member.get("trace_id")) for member in members if member.get("trace_id")),
        })
    return sorted(result, key=lambda item: (-item["distinct_decisions"], item["prompt"]))


def _metric_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the trainer's effective/all means separate from rollout means."""

    by_step: dict[str, dict[str, Any]] = {}
    for item in items:
        record = item["record"]
        step = record.get("step")
        if step is None:
            continue
        key = str(step)
        effective = _finite(record.get("train/agg/effective/agent/reward/mean"))
        all_mean = _finite(record.get("train/agg/all/agent/reward/mean"))
        if effective is None and all_mean is None:
            continue
        row = by_step.setdefault(key, {"step": step, "effective_mean": None, "all_mean": None})
        if effective is not None:
            row["effective_mean"] = effective
        if all_mean is not None:
            row["all_mean"] = all_mean
    rows = sorted(by_step.values(), key=lambda row: (int(row["step"]) if str(row["step"]).lstrip("-").isdigit() else 10**12))
    effective = [row["effective_mean"] for row in rows if row["effective_mean"] is not None]
    all_values = [row["all_mean"] for row in rows if row["all_mean"] is not None]
    return {
        "rows": rows,
        "step_count": len(rows),
        "effective_mean_over_steps": sum(effective) / len(effective) if effective else None,
        "all_mean_over_steps": sum(all_values) / len(all_values) if all_values else None,
        "denominator_note": "These are trainer metric means by optimizer step; they are separate from the exported rollout trace denominator.",
    }


def _invalid_detail(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [row for row in rows if row["category"] == "model_invalid_final"]
    return {
        "count": len(invalid),
        "failure_kinds": dict(sorted(collections.Counter(row.get("failure_kind") or "unknown" for row in invalid).items())),
        "error_classes": dict(sorted(collections.Counter(
            error_class
            for row in invalid
            for error_class, count in row.get("turn_error_classes", {}).items()
            for _ in range(int(count))
        ).items())),
        "retained_valid_earlier": sum(bool(row.get("retained_valid_earlier")) for row in invalid),
        "all_turns_invalid": sum(bool(row.get("all_turns_invalid")) for row in invalid),
        "finish_action_count": sum(int(row.get("finish_count") or 0) for row in invalid),
        "no_change_turn_count": sum(int(row.get("no_change_count") or 0) for row in invalid),
        "turn_failure_kinds": dict(sorted(collections.Counter(
            failure
            for row in invalid
            for failure, count in row.get("turn_failures", {}).items()
            for _ in range(int(count))
        ).items())),
    }


def _visual_cards(root: Path, receipt_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the fixed visual sample, root notes, and original judge reasons."""

    selection_path = root / "visual-selection.json"
    review_path = root / "root-visual-review.json"
    if not selection_path.is_file():
        selection_path = root / "visuals" / "visual-selection.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        selection = []
    try:
        reviews = json.loads(review_path.read_text(encoding="utf-8")).get("reviews", []) if review_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        reviews = []
    by_review = {str(item.get("trace_id")): item for item in reviews if isinstance(item, dict)}
    by_receipt = {str(item.get("trace_id")): item["record"] for item in receipt_items if item.get("trace_id")}
    cards: list[dict[str, Any]] = []
    for item in selection if isinstance(selection, list) else []:
        if not isinstance(item, dict) or not item.get("trace_id"):
            continue
        trace_id = str(item["trace_id"])
        receipt = by_receipt.get(trace_id, {})
        original: list[dict[str, Any]] = []
        for bucket in ("views", "recheck_views", "fresh_view"):
            values = receipt.get(bucket)
            values = values if isinstance(values, list) else ([] if values is None else [values])
            for index, view in enumerate(values):
                assessment = _view_assessment(view)
                if isinstance(assessment, dict):
                    original.append({
                        "bucket": bucket,
                        "index": index,
                        "winner_id": assessment.get("winner_id"),
                        "preference": assessment.get("preference"),
                        "reason": assessment.get("reason"),
                    })
        image = root / "visuals" / f"{trace_id}.png"
        cards.append({
            "trace_id": trace_id,
            "task_id": item.get("task_id"),
            "family": item.get("family"),
            "score": item.get("score"),
            "status": item.get("status"),
            "sheet": str(image.relative_to(root)) if image.is_file() else None,
            "root_review": by_review.get(trace_id),
            "original_judge_views": original,
        })
    return cards


def _root_reviews(root: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load the reviewer scope and notes without treating them as labels."""

    review_path = root / "root-visual-review.json"
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    reviews = payload.get("reviews", []) if isinstance(payload, dict) else []
    by_id = {
        str(item.get("trace_id")): item
        for item in reviews
        if isinstance(item, dict) and item.get("trace_id")
    }
    scope = str(payload.get("scope", "")) if isinstance(payload, dict) else ""
    return scope, by_id


def _repeat_summary(
    root: Path,
    primary_receipts: list[dict[str, Any]],
    repeat_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare selected frozen-pair rechecks with their original receipts.

    These receipts intentionally do not enter the main score/receipt counts:
    each is a second judge run for an existing image pair, selected by the
    reviewer to diagnose judge stability.
    """

    scope, reviews = _root_reviews(root)
    originals = {}
    for item in primary_receipts:
        key = item.get("trace_id") or _receipt_key(item)
        if key and key not in originals:
            originals[str(key)] = item
    rows: list[dict[str, Any]] = []
    for repeat in sorted(repeat_receipts, key=lambda item: str(item.get("source", ""))):
        key = str(repeat.get("trace_id") or _receipt_key(repeat) or "unknown")
        original = originals.get(key)
        repeat_audit = _audit_receipt(repeat, None)
        original_audit = _audit_receipt(original, None) if original else None
        original_record = original["record"] if original else {}
        repeat_record = repeat["record"]
        original_ids = original_record.get("image_ids")
        repeat_ids = repeat_record.get("image_ids")
        original_hashes = original_record.get("image_sha256")
        repeat_hashes = repeat_record.get("image_sha256")
        same_identity = (
            isinstance(original_ids, dict)
            and isinstance(repeat_ids, dict)
            and isinstance(original_hashes, dict)
            and isinstance(repeat_hashes, dict)
            and original_ids == repeat_ids
            and original_hashes == repeat_hashes
        )
        original_score = original_audit.get("score") if original_audit else None
        repeat_score = repeat_audit.get("score")
        review = reviews.get(key)
        rows.append({
            "trace_id": key,
            "original_source": original.get("source") if original else None,
            "repeat_source": repeat.get("source"),
            "matched_original": original is not None,
            "same_image_identity": same_identity,
            "original_status": original_audit.get("status") if original_audit else None,
            "repeat_status": repeat_audit.get("status"),
            "original_score": original_score,
            "repeat_score": repeat_score,
            "score_delta": (repeat_score - original_score
                            if repeat_score is not None and original_score is not None else None),
            "original_winner_id": original_audit.get("winner_id") if original_audit else None,
            "repeat_winner_id": repeat_audit.get("winner_id"),
            "winner_changed": bool(original_audit and original_audit.get("winner_id") != repeat_audit.get("winner_id")),
            "original_preference": original_audit.get("preference") if original_audit else None,
            "repeat_preference": repeat_audit.get("preference"),
            "preference_changed": bool(original_audit and original_audit.get("preference") != repeat_audit.get("preference")),
            "original_issues": original_audit.get("issues", []) if original_audit else [],
            "repeat_issues": repeat_audit.get("issues", []),
            "original_votes": original_audit.get("votes", {}) if original_audit else {},
            "repeat_votes": repeat_audit.get("votes", {}),
            "root_review": review,
        })
    both_scored = [
        row for row in rows
        if row["original_status"] in {"scored", "identical_canvases"}
        and row["repeat_status"] in {"scored", "identical_canvases"}
    ]
    availability_transitions = sum(
        (row["original_status"] in {"scored", "identical_canvases"}) !=
        (row["repeat_status"] in {"scored", "identical_canvases"})
        for row in rows
    )
    changed_winner = sum(row["winner_changed"] for row in rows)
    changed_preference = sum(row["preference_changed"] for row in rows)
    direction_flips = sum(
        row["original_winner_id"] != row["repeat_winner_id"]
        for row in both_scored
    )
    magnitude_only_changes = sum(
        row["original_winner_id"] == row["repeat_winner_id"]
        and row["original_preference"] != row["repeat_preference"]
        for row in both_scored
    )
    return {
        "scope": scope,
        "selection_note": (
            "Selected frozen image-pair repeats; this is a judge-stability diagnostic, "
            "not a random sample or human-labelled accuracy estimate."
        ),
        "count": len(rows),
        "matched_original_count": sum(row["matched_original"] for row in rows),
        "same_image_identity_count": sum(row["same_image_identity"] for row in rows),
        "original_scored_count": sum(row["original_status"] in {"scored", "identical_canvases"} for row in rows),
        "repeat_scored_count": sum(row["repeat_status"] in {"scored", "identical_canvases"} for row in rows),
        "both_scored_count": len(both_scored),
        "availability_transition_count": availability_transitions,
        "direction_flip_count": direction_flips,
        "magnitude_only_change_count": magnitude_only_changes,
        "winner_changed_count": changed_winner,
        "preference_changed_count": changed_preference,
        "rows": rows,
    }


def audit(root: Path) -> dict[str, Any]:
    traces, receipt_items, metric_items, read_errors = _read_documents(root)
    primary_receipts = [item for item in receipt_items if not item.get("is_repeat")]
    repeat_receipts = [item for item in receipt_items if item.get("is_repeat")]
    receipts_by_id: dict[str, dict[str, Any]] = {}
    duplicate_receipt_ids: list[str] = []
    for item in primary_receipts:
        key = _receipt_key(item)
        if item.get("trace_id") is None and key:
            item["trace_id"] = key
        if key:
            if key in receipts_by_id:
                duplicate_receipt_ids.append(key)
            else:
                receipts_by_id[key] = item
    trace_by_id = {str(item.get("trace_id")): item for item in traces if item.get("trace_id")}
    joined: list[dict[str, Any]] = []
    receipt_audits: list[dict[str, Any]] = []
    for trace in traces:
        fields = _trace_fields(trace)
        receipt = _receipt_from_trace(trace, receipts_by_id)
        receipt_audit: dict[str, Any] | None = None
        if receipt:
            receipt_audit = _audit_receipt(receipt, trace)
            receipt_audits.append(receipt_audit)
        category = _category(fields, receipt["record"] if receipt else None)
        joined.append({
            **fields,
            "winner_id": receipt_audit.get("winner_id") if receipt_audit else None,
            "preference": receipt_audit.get("preference") if receipt_audit else None,
            "category": category,
            "receipt_source": receipt.get("source") if receipt else None,
        })
    orphan_receipts = [item for item in primary_receipts if str(item.get("trace_id") or _receipt_key(item)) not in trace_by_id]
    # A bundle may contain receipts only; retain them in repeated-decision and
    # integrity reports without inventing batch/family metadata.
    for item in orphan_receipts:
        receipt_audits.append(_audit_receipt(item, None))
    report = {
        "schema": "painter-text-rl16-judge-audit-v1",
        "input_root": str(root),
        "input_files": len({item["source"] for item in traces + receipt_items}),
        "trace_records": len(traces),
        "receipt_records": len(primary_receipts),
        "repeat_receipt_records": len(repeat_receipts),
        "metric_records": len(metric_items),
        "unmatched_receipts": len(orphan_receipts),
        "duplicate_receipt_ids": sorted(set(duplicate_receipt_ids)),
        "read_errors": read_errors,
        "totals": dict(sorted(collections.Counter(row["category"] for row in joined).items())),
        "batch_summary": _batch_summary(joined),
        "dispatch_step_summary": _conditional_summary(joined, "dispatch_step"),
        "arrival_step_summary": _conditional_summary(joined, "arrival_step"),
        "group_summary": _conditional_summary(joined, "group_id"),
        "family_summary": _conditional_summary(joined, "family"),
        "invalid_detail": _invalid_detail(joined),
        "trainer_metrics": _metric_summary(metric_items),
        "family_counts": dict(sorted(collections.Counter(row["family"] for row in joined).items())),
        "prompt_counts": dict(sorted(collections.Counter(row["prompt"] for row in joined).items())),
        "visual_cards": _visual_cards(root, primary_receipts),
        "receipt_issues": [row for row in receipt_audits if row["issues"]],
        "receipt_issue_counts": dict(sorted(collections.Counter(issue for row in receipt_audits for issue in row["issues"]).items())),
        "repeated_prompt_note": (
            "Repeated prompt rows contain different rollout canvases and may also differ in policy/family; "
            "they are not same-image judge repeat tests."
        ),
        "repeated_prompt_variation": _repeated_prompt_summary(joined),
    }
    report["frozen_repeat_diagnostics"] = _repeat_summary(root, primary_receipts, repeat_receipts)
    repeat_by_id = {
        str(row["trace_id"]): row
        for row in report["frozen_repeat_diagnostics"]["rows"]
    }
    for card in report["visual_cards"]:
        repeat = repeat_by_id.get(str(card.get("trace_id")))
        if repeat:
            card["frozen_repeat_diagnostic"] = repeat
    return report


def _html_report(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return html.escape(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value))

    rows = []
    for batch in report["batch_summary"]:
        rows.append("<tr>" + "".join(f"<td>{cell(batch.get(key))}</td>" for key in (
            "batch", "count", "scored_count", "visual_conditional_mean", "categories", "families")) + "</tr>")
    issue_rows = []
    for item in report["receipt_issues"]:
        issue_rows.append("<tr>" + "".join(f"<td>{cell(item.get(key))}</td>" for key in (
            "trace_id", "score", "expected_score", "issues", "source")) + "</tr>")
    variation_rows = []
    for item in report["repeated_prompt_variation"]:
        variation_rows.append("<tr>" + "".join(f"<td>{cell(item.get(key))}</td>" for key in (
            "prompt", "count", "distinct_decisions", "score_min", "score_max", "decisions")) + "</tr>")
    metric_rows = []
    for item in report["trainer_metrics"]["rows"]:
        metric_rows.append("<tr>" + "".join(f"<td>{cell(item.get(key))}</td>" for key in (
            "step", "effective_mean", "all_mean")) + "</tr>")
    repeat_rows = []
    for item in report.get("frozen_repeat_diagnostics", {}).get("rows", []):
        review = item.get("root_review") or {}
        repeat_rows.append("<tr>" + "".join(f"<td>{cell(value)}</td>" for value in (
            item.get("trace_id"), item.get("original_score"), item.get("repeat_score"),
            item.get("score_delta"), item.get("original_winner_id"), item.get("repeat_winner_id"),
            item.get("original_preference"), item.get("repeat_preference"),
            item.get("same_image_identity"), review.get("winner"), review.get("preference"),
            review.get("reason", ""))) + "</tr>")
    visual_cards = []
    for item in report.get("visual_cards", []):
        review = item.get("root_review") or {}
        image = (
            f'<img loading="lazy" src="{html.escape(str(item["sheet"]))}" '
            f'alt="{html.escape(str(item.get("task_id") or item.get("trace_id")))}" style="max-width:760px;width:100%;border:1px solid #ccd">'
            if item.get("sheet") else "<p>sheet not present in export</p>"
        )
        reasons = "".join(
            f'<li><b>{html.escape(str(view.get("bucket")))} {view.get("index", 0)}</b> '
            f'{html.escape(str(view.get("winner_id")))} / {html.escape(str(view.get("preference")))}: '
            f'{html.escape(str(view.get("reason") or "(no reason)"))}</li>'
            for view in item.get("original_judge_views", [])
        )
        visual_cards.append(
            "<article><h3>" + html.escape(str(item.get("task_id") or item.get("trace_id"))) +
            " · score " + html.escape(str(item.get("score"))) + "</h3>" + image +
            "<p><b>Root review:</b> " + html.escape(str(review.get("winner", ""))) +
            " / " + html.escape(str(review.get("preference", ""))) +
            " (" + html.escape(str(review.get("confidence", ""))) + "): " +
            html.escape(str(review.get("reason", ""))) + "</p><details><summary>Original judge reasons</summary><ul>" +
            reasons + "</ul></details></article>"
        )
    return """<!doctype html>
<meta charset="utf-8"><title>RL16 side-judge audit</title>
<style>body{font:14px system-ui,sans-serif;margin:2rem;color:#202124}table{border-collapse:collapse;margin:1rem 0 2rem}th,td{border:1px solid #ccd;padding:.35rem .5rem;text-align:left;vertical-align:top}th{background:#eef}code{white-space:pre-wrap}</style>
<h1>RL16 side-judge audit</h1>
<p>Offline report from sanitized trace metadata and exported receipts. No renderer or provider was called.</p>
<h2>Totals</h2><pre>""" + html.escape(json.dumps({
        key: report[key] for key in ("trace_records", "receipt_records", "metric_records", "unmatched_receipts", "totals", "invalid_detail", "receipt_issue_counts")
    }, indent=2, sort_keys=True)) + """</pre>
<h2>Per shipped batch</h2><table><thead><tr><th>batch</th><th>count</th><th>scored</th><th>visual mean</th><th>categories</th><th>families</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table>
<h2>Receipt integrity issues</h2><table><thead><tr><th>trace</th><th>reported</th><th>recomputed</th><th>issues</th><th>source</th></tr></thead><tbody>""" + "".join(issue_rows) + """</tbody></table>
<h2>Trainer metrics</h2><table><thead><tr><th>step</th><th>effective mean</th><th>all mean</th></tr></thead><tbody>""" + "".join(metric_rows) + """</tbody></table>
<h2>Repeated prompt decisions</h2><p>""" + html.escape(str(report.get("repeated_prompt_note", ""))) + "</p><table><thead><tr><th>prompt</th><th>count</th><th>distinct decisions</th><th>min</th><th>max</th><th>decisions</th></tr></thead><tbody>""" + "".join(variation_rows) + """</tbody></table>
<h2>Selected frozen-pair repeats</h2><p>""" + html.escape(str(report.get("frozen_repeat_diagnostics", {}).get("selection_note", ""))) + " " + html.escape(str(report.get("frozen_repeat_diagnostics", {}).get("scope", ""))) + "</p><pre>""" + html.escape(json.dumps({
        key: report.get("frozen_repeat_diagnostics", {}).get(key)
        for key in (
            "count", "matched_original_count", "same_image_identity_count",
            "original_scored_count", "repeat_scored_count", "both_scored_count",
            "availability_transition_count", "direction_flip_count",
            "magnitude_only_change_count", "winner_changed_count", "preference_changed_count",
        )
    }, indent=2, sort_keys=True)) + """</pre><table><thead><tr><th>trace</th><th>original score</th><th>repeat score</th><th>delta</th><th>original winner</th><th>repeat winner</th><th>original magnitude</th><th>repeat magnitude</th><th>same image IDs/hashes</th><th>root winner</th><th>root magnitude</th><th>root note</th></tr></thead><tbody>""" + "".join(repeat_rows) + """</tbody></table>
<h2>Fixed visual review sample</h2>""" + "".join(visual_cards) + """
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="sanitized audit directory or one metadata file")
    parser.add_argument("--output", type=Path, help="report prefix or .json path")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.exists():
        parser.error(f"audit path does not exist: {root}")
    report = audit(root)
    output = args.output or (root / "judge-audit-report.json" if root.is_dir() else root.with_name("judge-audit-report.json"))
    if output.suffix.lower() == ".html":
        json_output = output.with_suffix(".json")
        html_output = output
    else:
        json_output = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        html_output = output.with_suffix(".html")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_output.write_text(_html_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_output), "html": str(html_output), "totals": report["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
