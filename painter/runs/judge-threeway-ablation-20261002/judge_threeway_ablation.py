"""Offline finite three-model frozen-image side-judge ablation.

This runner is intentionally separate from the live reward path.  It uses the
current v2 prompt, strict response schema, side parser, and joint-sheet image
transport, but makes the model and request configuration explicit per call.
The primary assay has exactly two image orders per model/pair.  A fixed set of
pairs can receive a predeclared number of additional two-order rounds.  There
are no adaptive retries or rechecks: an invalid response or order conflict is
censored and never converted into a tie.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_consistent_side_judge_v2 as rubric
    from painter import text_pairwise_identity_judge as sheet_transport
    from painter.text_pairwise_judge import ENDPOINT, digest, load_key
except ImportError:  # pragma: no cover - direct checkout execution
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_consistent_side_judge_v2 as rubric
    from painter import text_pairwise_identity_judge as sheet_transport
    from painter.text_pairwise_judge import ENDPOINT, digest, load_key


SCHEMA = "judge-threeway-ablation-manifest-v1"
CALL_SCHEMA = "judge-threeway-ablation-call-v1"
ROUND_SCHEMA = "judge-threeway-ablation-round-v1"
SUMMARY_SCHEMA = "judge-threeway-ablation-summary-v1"
PROMPT = rubric.PROMPT
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
RUBRIC_SOURCE_SHA256 = rubric.SOURCE_FORMAT_SHA256
RUBRIC_FORMAT_VERSION = rubric.FORMAT_VERSION
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RESPONSE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(rubric._response_schema(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
MODELS = (
    "xiaomi/mimo-v2.6-flash",
    "xiaomi/mimo-v2.6-pro",
    "openai/gpt-6-luna",
)
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
MAX_TOKENS = 4096
# Explicit reasoning requests omit sampling controls on the Gateway payload.
# Keep this as None so receipts cannot falsely claim that temperature=0 was
# sent when the provider contract requires omission.
TEMPERATURE = None
REASONING = {"effort": "medium"}
PRIMARY_ORDER_COUNT = 2
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_digest(value: dict[str, Any]) -> str:
    copy = dict(value)
    copy.pop("manifest_content_sha256", None)
    return sha256_bytes(canonical(copy).encode("utf-8"))


def validate_model(model: str) -> str:
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
        raise ValueError(f"invalid model ID: {model!r}")
    return model


def _image_id(path: Path) -> str:
    return "img_" + digest(path)[:20]


def _request_config(model: str) -> dict[str, Any]:
    """Canonical request identity.

    Gateway's explicit reasoning mode requires sampling parameters to be
    omitted.  No fallback to provider default is allowed.
    """

    return {
        "model": validate_model(model),
        "temperature": TEMPERATURE,
        "temperature_payload": "omitted_for_explicit_reasoning",
        "max_tokens": MAX_TOKENS,
        "reasoning": REASONING,
        "response_format": rubric._response_format(),
        "stream": False,
    }


def config_digest(model: str) -> str:
    return sha256_bytes(canonical(_request_config(model)).encode("utf-8"))


def _safe_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name.replace("_", "").isalnum() else "ProviderError"


def _normalize_usage(value: Any) -> dict[str, int | float | None] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int | float | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = int(item)
        else:
            result[key] = None
    reasoning: Any = None
    details = value.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
    if not isinstance(reasoning, (int, float)) or isinstance(reasoning, bool):
        reasoning = value.get("reasoning_tokens")
    result["reasoning_tokens"] = int(reasoning) if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool) else None
    for key in ("cost", "prompt_cost", "completion_cost", "reasoning_cost"):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = float(item)
    return result


def _view(*, key: str, model: str, brief: str, left: Path, right: Path,
          left_id: str, right_id: str) -> dict[str, Any]:
    """Make exactly one API call; all failures are safe, censored metadata."""

    model = validate_model(model)
    sheet, sheet_sha = sheet_transport._sheet(left, right, "LEFT", "RIGHT")
    config = _request_config(model)
    # Deliberately omit temperature: explicit reasoning medium is required for
    # all three models and must never silently fall back to provider defaults.
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "reasoning": dict(REASONING),
        "response_format": rubric._response_format(),
        "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PAINTING BRIEF:\n{brief}\nAssess the LEFT and RIGHT panels before comparing them."},
                {"type": "image_url", "image_url": {"url": sheet}},
            ]},
        ],
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    call: dict[str, Any] = {
        "schema": CALL_SCHEMA,
        "status": "censored",
        "raw_response": None,
        "image_sha256": {"LEFT": digest(left), "RIGHT": digest(right)},
        "image_ids": {"LEFT": left_id, "RIGHT": right_id},
        "sheet_sha256": sheet_sha,
        "prompt_sha256": PROMPT_SHA256,
        "rubric_source_sha256": RUBRIC_SOURCE_SHA256,
        "rubric_format_version": RUBRIC_FORMAT_VERSION,
        "source_sha256": SOURCE_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "request_config_sha256": config_digest(model),
        "requested_model": model,
        "returned_model": None,
        "reasoning": dict(REASONING),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "elapsed_seconds": None,
        "usage": None,
        "finish_reason": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        call.update({"provider_error": "HTTPError", "http_status": int(exc.code)})
        call["elapsed_seconds"] = time.monotonic() - started
        return call
    except Exception as exc:
        call.update({"provider_error": _safe_exception(exc)})
        call["elapsed_seconds"] = time.monotonic() - started
        return call
    call["elapsed_seconds"] = time.monotonic() - started
    try:
        body = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        call["parse_error"] = "malformed_json"
        return call
    if not isinstance(body, dict):
        call["parse_error"] = "malformed_provider_response"
        return call
    call["usage"] = _normalize_usage(body.get("usage"))
    returned_model = body.get("model")
    call["returned_model"] = returned_model if isinstance(returned_model, str) and returned_model else None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        call["parse_error"] = "malformed_provider_response"
        return call
    choice = choices[0]
    call["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message")
    raw = message.get("content") if isinstance(message, dict) else None
    if not isinstance(raw, str):
        call["parse_error"] = "missing_message_content"
        return call
    # Raw valid labels are retained for audit; provider failures and malformed
    # response bodies are intentionally never copied into durable receipts.
    call["raw_response"] = raw
    if call["returned_model"] != model:
        call["parse_error"] = "model_identity_mismatch"
        return call
    if call["finish_reason"] == "length":
        call["parse_error"] = "truncated"
        return call
    try:
        call["assessment"] = rubric._parse(raw, (left_id, right_id))
        call["status"] = "valid"
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        call["parse_error"] = type(exc).__name__
    return call


def _source_row(raw: dict[str, Any], side: str) -> dict[str, Any]:
    value = raw.get(side)
    if not isinstance(value, dict):
        raise ValueError(f"pair missing {side} image")
    path = Path(value.get("path", ""))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = digest(path)
    if value.get("sha256") != actual_sha:
        raise ValueError(f"{side} image hash changed for {raw.get('pair_id')}")
    actual_id = _image_id(path)
    if value.get("image_id") != actual_id:
        raise ValueError(f"{side} image identity changed for {raw.get('pair_id')}")
    return {**value, "path": str(path), "sha256": actual_sha, "image_id": actual_id}


def _load_panel(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "judge-threeway-ablation-panel-v1":
        raise ValueError("unexpected panel schema")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise ValueError("panel must contain exactly 24 pairs")
    seen: set[str] = set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not SAFE_ID.fullmatch(pair_id) or pair_id in seen:
            raise ValueError(f"invalid or duplicate pair ID: {pair_id!r}")
        seen.add(pair_id)
        brief = pair.get("brief")
        if not isinstance(brief, str) or not brief.strip() or len(brief) > 4000:
            raise ValueError(f"invalid brief for {pair_id}")
        if pair.get("brief_sha256") != sha256_bytes(brief.encode("utf-8")):
            raise ValueError(f"brief hash changed for {pair_id}")
        _source_row(pair, "left")
        _source_row(pair, "right")
    return value


def prepare_manifest(*, panel_path: Path, output: Path, fixed_repeat_rounds: int = 1,
                     fixed_repeat_pair_ids: list[str] | None = None) -> dict[str, Any]:
    if fixed_repeat_rounds < 0:
        raise ValueError("fixed_repeat_rounds must be nonnegative")
    panel = _load_panel(panel_path)
    pairs = panel["pairs"]
    default_repeat_ids = [
        "sft-oldrl16-02", "sft-oldrl16-05", "sft-oldrl16-06",
        "sft-oldrl16-07", "sft-oldrl16-08", "oldrl16-groundedrl24-02",
    ]
    repeat_ids = fixed_repeat_pair_ids if fixed_repeat_pair_ids is not None else default_repeat_ids
    if len(repeat_ids) != 6 or len(set(repeat_ids)) != 6:
        raise ValueError("fixed repeat panel must contain exactly six unique pair IDs")
    pair_ids = {pair["pair_id"] for pair in pairs}
    if not set(repeat_ids) <= pair_ids:
        raise ValueError("fixed repeat pair is not in panel")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "panel_schema": panel["schema"],
        "panel_source_evidence": panel["source_evidence"],
        "protocol": "text-paint-sft-pairwise-consistent-side-v2-threeway-ablation",
        "prompt_sha256": PROMPT_SHA256,
        "rubric_source_sha256": RUBRIC_SOURCE_SHA256,
        "rubric_format_version": RUBRIC_FORMAT_VERSION,
        "runner_source_sha256": SOURCE_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "models": list(MODELS),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"field": "reasoning", "effort": "medium", "same_requested_effort": True},
        "reasoning_caveat": (
            "All models receive reasoning.effort=medium. Provider internal reasoning "
            "compute is not assumed equal across model families; observed reasoning_tokens "
            "are recorded when returned."
        ),
        "sampling_payload_note": "Gateway explicit reasoning requires temperature/top_p omitted; no temperature value was sent.",
        "primary_order_count": PRIMARY_ORDER_COUNT,
        "fixed_repeat_rounds": fixed_repeat_rounds,
        "fixed_repeat_pair_ids": repeat_ids,
        "planned_call_count": len(MODELS) * (len(pairs) * 2 + len(repeat_ids) * fixed_repeat_rounds * 2),
        "human_labels": {
            "scope": "Explicit historical user preferences in selected panel only; no inferred gold or accuracy claim.",
            "available": True,
        },
        "selection_note": panel["selection_note"],
        "pairs": pairs,
    }
    manifest["manifest_content_sha256"] = manifest_digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("manifest_content_sha256") != manifest_digest(value):
        raise ValueError("manifest identity or content hash mismatch")
    for key, expected in (
        ("protocol", "text-paint-sft-pairwise-consistent-side-v2-threeway-ablation"),
        ("prompt_sha256", PROMPT_SHA256),
        ("rubric_source_sha256", RUBRIC_SOURCE_SHA256),
        ("rubric_format_version", RUBRIC_FORMAT_VERSION),
        ("runner_source_sha256", SOURCE_SHA256),
        ("response_schema_sha256", RESPONSE_SCHEMA_SHA256),
        ("temperature", TEMPERATURE),
        ("max_tokens", MAX_TOKENS),
        ("primary_order_count", PRIMARY_ORDER_COUNT),
    ):
        if value.get(key) != expected:
            raise ValueError(f"manifest {key} differs from current runner")
    if value.get("reasoning") != {"field": "reasoning", "effort": "medium", "same_requested_effort": True}:
        raise ValueError("manifest reasoning configuration differs")
    models = value.get("models")
    if models != list(MODELS):
        raise ValueError(f"manifest models differ: {models!r}")
    if not isinstance(value.get("fixed_repeat_rounds"), int) or value["fixed_repeat_rounds"] < 0:
        raise ValueError("manifest fixed repeat rounds invalid")
    repeat_ids = value.get("fixed_repeat_pair_ids")
    if not isinstance(repeat_ids, list) or len(repeat_ids) != 6 or len(set(repeat_ids)) != 6:
        raise ValueError("manifest fixed repeat pair selection invalid")
    _load_panel_from_manifest(value)
    return value


def _load_panel_from_manifest(manifest: dict[str, Any]) -> None:
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise ValueError("manifest pair count changed")
    seen: set[str] = set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not SAFE_ID.fullmatch(pair_id) or pair_id in seen:
            raise ValueError(f"invalid or duplicate manifest pair ID: {pair_id!r}")
        seen.add(pair_id)
        brief = pair.get("brief")
        if not isinstance(brief, str) or pair.get("brief_sha256") != sha256_bytes(brief.encode()):
            raise ValueError(f"manifest brief identity changed: {pair_id}")
        _source_row(pair, "left")
        _source_row(pair, "right")
    if not set(manifest["fixed_repeat_pair_ids"]) <= seen:
        raise ValueError("manifest fixed repeat pair is absent")


def _identity(manifest: dict[str, Any], pair: dict[str, Any], model: str,
              round_index: int, order: str) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"],
        "case_id": pair.get("case_id"),
        "model": validate_model(model),
        "round": round_index,
        "order": order,
        "prompt_sha256": manifest["prompt_sha256"],
        "rubric_source_sha256": manifest["rubric_source_sha256"],
        "rubric_format_version": manifest["rubric_format_version"],
        "runner_source_sha256": manifest["runner_source_sha256"],
        "response_schema_sha256": manifest["response_schema_sha256"],
        "request_config_sha256": config_digest(model),
        "brief_sha256": pair["brief_sha256"],
        "image_ids": {"LEFT": pair["left"]["image_id"], "RIGHT": pair["right"]["image_id"]},
        "image_sha256": {"LEFT": pair["left"]["sha256"], "RIGHT": pair["right"]["sha256"]},
        "reasoning": manifest["reasoning"],
        "temperature": manifest["temperature"],
        "max_tokens": manifest["max_tokens"],
    }


def _round_identity(manifest: dict[str, Any], pair: dict[str, Any], model: str,
                    round_index: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"], "model": validate_model(model),
        "round": round_index,
        "prompt_sha256": manifest["prompt_sha256"],
        "rubric_source_sha256": manifest["rubric_source_sha256"],
        "rubric_format_version": manifest["rubric_format_version"],
        "runner_source_sha256": manifest["runner_source_sha256"],
        "response_schema_sha256": manifest["response_schema_sha256"],
        "request_config_sha256": config_digest(model),
        "brief_sha256": pair["brief_sha256"],
        "image_sha256": {"LEFT": pair["left"]["sha256"], "RIGHT": pair["right"]["sha256"]},
        "reasoning": manifest["reasoning"],
        "temperature": manifest["temperature"],
        "max_tokens": manifest["max_tokens"],
    }


def _call_paths(output: Path, model: str, pair_id: str, round_index: int, order: str) -> Path:
    model_slug = model.replace("/", "__")
    return output / "receipts" / "calls" / model_slug / pair_id / f"round-{round_index:02d}-{order}.json"


def _round_path(output: Path, model: str, pair_id: str, round_index: int) -> Path:
    model_slug = model.replace("/", "__")
    return output / "receipts" / "rounds" / model_slug / pair_id / f"round-{round_index:02d}.json"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_existing(path: Path, identity: dict[str, Any], schema: str) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or value.get("identity") != identity:
        raise ValueError(f"stale receipt identity: {path}")
    return value


def _progress_line(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    usage = row.get("call", {}).get("usage") or {}
    event = {
        "model": row["identity"]["model"], "pair": row["identity"]["pair_id"],
        "round": row["identity"]["round"], "order": row["identity"]["order"],
        "status": row.get("status"), "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "elapsed_seconds": row.get("call", {}).get("elapsed_seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
        print(json.dumps(event, sort_keys=True), flush=True)


def _pair_paths(pair: dict[str, Any]) -> tuple[str, Path, Path, str, str]:
    return (
        pair["brief"], Path(pair["left"]["path"]), Path(pair["right"]["path"]),
        pair["left"]["image_id"], pair["right"]["image_id"],
    )


async def _one_call(*, manifest: dict[str, Any], pair: dict[str, Any], model: str,
                    round_index: int, order: str, output: Path, key: str,
                    progress_lock: threading.Lock, sem: asyncio.Semaphore) -> dict[str, Any]:
    identity = _identity(manifest, pair, model, round_index, order)
    path = _call_paths(output, model, pair["pair_id"], round_index, order)
    existing = _load_existing(path, identity, CALL_SCHEMA)
    if existing is not None:
        return existing
    brief, left, right, left_id, right_id = _pair_paths(pair)
    if order == "reverse":
        left, right, left_id, right_id = right, left, right_id, left_id
    started = time.monotonic()
    async with sem:
        try:
            call = await asyncio.to_thread(
                _view, key=key, model=model, brief=brief, left=left, right=right,
                left_id=left_id, right_id=right_id,
            )
            status = "valid" if call.get("status") == "valid" else "censored"
            row = {
                "schema": CALL_SCHEMA, "identity": identity, "status": status,
                "call": call, "runner_elapsed_seconds": time.monotonic() - started,
            }
        except Exception as exc:
            row = {
                "schema": CALL_SCHEMA, "identity": identity, "status": "implementation_error",
                "reason": _safe_exception(exc),
                "runner_elapsed_seconds": time.monotonic() - started,
            }
    _atomic_json(path, row)
    _progress_line(output / "progress.jsonl", row, progress_lock)
    return row


def _round_result(*, manifest: dict[str, Any], pair: dict[str, Any], model: str,
                  round_index: int, calls: list[dict[str, Any]]) -> dict[str, Any]:
    identity = _round_identity(manifest, pair, model, round_index)
    valid = [row for row in calls if row.get("status") == "valid" and row.get("call", {}).get("assessment")]
    votes: list[str] = []
    if len(valid) == 2:
        votes = [str(row["call"]["assessment"]["winner_id"]) for row in valid]
    agreement = len(votes) == 2 and votes[0] == votes[1]
    status = "scored" if agreement else "censored"
    reason = None
    if len(valid) != 2:
        reason = "strict_invalid_call"
    elif not agreement:
        reason = "order_conflict"
    winner_id = votes[0] if agreement else None
    preference = valid[0]["call"]["assessment"].get("preference") if agreement else None
    return {
        "schema": ROUND_SCHEMA, "identity": identity, "status": status,
        "reason": reason, "winner_id": winner_id, "preference": preference,
        "order_agreement": agreement if len(valid) == 2 else None,
        "calls": [row["identity"] for row in calls],
        "call_paths": [str(_call_paths(Path("."), model, pair["pair_id"], round_index, row["identity"]["order"])) for row in calls],
    }


async def _one_round(*, manifest: dict[str, Any], pair: dict[str, Any], model: str,
                     round_index: int, output: Path, key: str,
                     progress_lock: threading.Lock, sem: asyncio.Semaphore) -> dict[str, Any]:
    order_names = ("canonical", "reverse")
    calls = await asyncio.gather(*(
        _one_call(manifest=manifest, pair=pair, model=model, round_index=round_index,
                  order=order, output=output, key=key, progress_lock=progress_lock, sem=sem)
        for order in order_names
    ))
    round_path = _round_path(output, model, pair["pair_id"], round_index)
    identity = _round_identity(manifest, pair, model, round_index)
    existing = _load_existing(round_path, identity, ROUND_SCHEMA)
    if existing is not None:
        return existing
    result = _round_result(manifest=manifest, pair=pair, model=model,
                           round_index=round_index, calls=calls)
    _atomic_json(round_path, result)
    return result


def _work_items(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], str, int]]:
    repeat_ids = set(manifest["fixed_repeat_pair_ids"])
    rounds = manifest["fixed_repeat_rounds"]
    items: list[tuple[dict[str, Any], str, int]] = []
    for pair in manifest["pairs"]:
        for model in manifest["models"]:
            for round_index in range(0, 1 + (rounds if pair["pair_id"] in repeat_ids else 0)):
                items.append((pair, model, round_index))
    return items


def _missing_call_count(manifest: dict[str, Any], output: Path) -> int:
    missing = 0
    for pair, model, round_index in _work_items(manifest):
        for order in ("canonical", "reverse"):
            path = _call_paths(output, model, pair["pair_id"], round_index, order)
            identity = _identity(manifest, pair, model, round_index, order)
            if _load_existing(path, identity, CALL_SCHEMA) is None:
                missing += 1
    return missing


def _numbers(values: list[int | float | None]) -> dict[str, Any]:
    observed = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    observed.sort()
    if not observed:
        return {"mean": None, "median": None, "p90": None, "observed": 0, "missing": len(values)}
    index = max(0, min(len(observed) - 1, math.ceil(0.90 * len(observed)) - 1))
    return {
        "mean": sum(observed) / len(observed), "median": statistics.median(observed),
        "p90": observed[index], "observed": len(observed), "missing": len(values) - len(observed),
    }


def _read_call_rows(manifest: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, model, round_index in _work_items(manifest):
        for order in ("canonical", "reverse"):
            path = _call_paths(output, model, pair["pair_id"], round_index, order)
            identity = _identity(manifest, pair, model, round_index, order)
            row = _load_existing(path, identity, CALL_SCHEMA)
            if row is None:
                raise FileNotFoundError(path)
            row["_pair"] = pair
            rows.append(row)
    return rows


def _stats_for_rows(rows: list[dict[str, Any]], pair_count: int) -> dict[str, Any]:
    calls = [row.get("call", {}) for row in rows if isinstance(row.get("call"), dict)]
    completion = [call.get("usage", {}).get("completion_tokens") if isinstance(call.get("usage"), dict) else None for call in calls]
    reasoning = [call.get("usage", {}).get("reasoning_tokens") if isinstance(call.get("usage"), dict) else None for call in calls]
    prompt = [call.get("usage", {}).get("prompt_tokens") if isinstance(call.get("usage"), dict) else None for call in calls]
    costs = [call.get("usage", {}).get("cost") if isinstance(call.get("usage"), dict) else None for call in calls]
    elapsed = [call.get("elapsed_seconds") for call in calls]
    pair_completion = _pair_token_stats(rows, "completion_tokens")
    pair_reasoning = _pair_token_stats(rows, "reasoning_tokens")
    pair_service, pair_wall = _pair_latency_stats(rows)
    order_valid = 0
    order_agree = 0
    conflicts = 0
    invalid = 0
    for round_rows in _rounds_from_rows(rows).values():
        valid = [row for row in round_rows if row.get("status") == "valid" and row.get("call", {}).get("assessment")]
        if len(valid) == 2:
            order_valid += 1
            votes = [row["call"]["assessment"]["winner_id"] for row in valid]
            if votes[0] == votes[1]:
                order_agree += 1
            else:
                conflicts += 1
        else:
            invalid += 1
    usage_cost = sum(float(value) for value in costs if isinstance(value, (int, float)))
    return {
        "pair_count": pair_count,
        "call_count": len(rows),
        "valid_call_count": sum(row.get("status") == "valid" for row in rows),
        "censored_call_count": sum(row.get("status") == "censored" for row in rows),
        "implementation_error_count": sum(row.get("status") == "implementation_error" for row in rows),
        "order_consistency": {
            "rounds_with_two_valid_orders": order_valid,
            "agreeing_rounds": order_agree,
            "conflicting_rounds": conflicts,
            "invalid_rounds": invalid,
            "rate": order_agree / order_valid if order_valid else None,
        },
        "latency_seconds": {
            "per_call": _numbers(elapsed),
            "per_pair_service_seconds": pair_service,
            "per_pair_wall_clock_seconds": pair_wall,
        },
        "tokens": {
            "prompt": _numbers(prompt),
            "completion": {"per_call": _numbers(completion), "per_pair": pair_completion},
            "reasoning": {"per_call": _numbers(reasoning), "per_pair": pair_reasoning},
        },
        "reported_cost": {"sum": usage_cost, "observed_calls": sum(isinstance(value, (int, float)) for value in costs), "missing_calls": sum(not isinstance(value, (int, float)) for value in costs)},
    }


def _work_items_from_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, int]]:
    return [(row["_pair"], row["identity"]["model"], row["identity"]["round"]) for row in rows]


def _rounds_from_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ident = row["identity"]
        groups[(ident["pair_id"], ident["model"], ident["round"])].append(row)
    return groups


def _calls_by_pair(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ident = row["identity"]
        groups[ident["pair_id"]].append(row.get("call", {}))
    return groups


def _pair_token_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        usage = (row.get("call") or {}).get("usage") or {}
        grouped[row["identity"]["pair_id"]].append(usage.get(field))
    values: list[float | None] = []
    partial = 0
    for pair_values in grouped.values():
        numeric = [float(value) for value in pair_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if len(numeric) == len(pair_values):
            values.append(sum(numeric))
        else:
            values.append(None)
            if numeric:
                partial += 1
    stats = _numbers(values)
    stats["partial_pairs"] = partial
    return stats


def _pair_latency_stats(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rounds: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in rows:
        ident = row["identity"]
        rounds[(ident["pair_id"], ident["round"])].append((row.get("call") or {}).get("elapsed_seconds"))
    service: dict[str, list[float | None]] = defaultdict(list)
    wall: dict[str, list[float | None]] = defaultdict(list)
    for (pair_id, _round), elapsed_values in rounds.items():
        numeric = [float(value) for value in elapsed_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        service[pair_id].append(sum(numeric) if len(numeric) == len(elapsed_values) else None)
        wall[pair_id].append(max(numeric) if numeric else None)
    service_values = [sum(values) if all(value is not None for value in values) else None for values in service.values()]
    wall_values = [sum(value for value in values if value is not None) if all(value is not None for value in values) else None for values in wall.values()]
    return _numbers(service_values), _numbers(wall_values)


def _cross_model_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ident = row["identity"]
        groups[(ident["pair_id"], ident["round"])][ident["model"]].append(row)
    complete = unanimous = disagreements = 0
    for by_model in groups.values():
        decisions: list[str] = []
        for model_rows in by_model.values():
            valid = [row for row in model_rows if row.get("status") == "valid" and row.get("call", {}).get("assessment")]
            if len(valid) != 2 or valid[0]["call"]["assessment"]["winner_id"] != valid[1]["call"]["assessment"]["winner_id"]:
                decisions = []
                break
            decisions.append(valid[0]["call"]["assessment"]["winner_id"])
        if len(decisions) != len(MODELS):
            continue
        complete += 1
        if len(set(decisions)) == 1:
            unanimous += 1
        else:
            disagreements += 1
    return {
        "complete_pair_rounds": complete,
        "all_models_agree": unanimous,
        "model_disagreement": disagreements,
        "all_models_agree_rate": unanimous / complete if complete else None,
    }


def summarize(*, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for model in manifest["models"]:
        model_rows = [row for row in rows if row["identity"]["model"] == model]
        summaries[model] = _stats_for_rows(model_rows, len(manifest["pairs"]))
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "protocol": manifest["protocol"], "prompt_sha256": manifest["prompt_sha256"],
        "rubric_source_sha256": manifest["rubric_source_sha256"],
        "rubric_format_version": manifest["rubric_format_version"],
        "runner_source_sha256": manifest["runner_source_sha256"],
        "models": manifest["models"], "reasoning": manifest["reasoning"],
        "temperature": manifest["temperature"], "max_tokens": manifest["max_tokens"],
        "reasoning_caveat": manifest["reasoning_caveat"],
        "latency_caveat": (
            "per_pair_service_seconds sums individual order-call elapsed times; "
            "per_pair_wall_clock_seconds sums the maximum order elapsed time in each "
            "parallel round and is the pair-clock estimate."
        ),
        "pair_count": len(manifest["pairs"]),
        "fixed_repeat_rounds": manifest["fixed_repeat_rounds"],
        "fixed_repeat_pair_ids": manifest["fixed_repeat_pair_ids"],
        "gold_labels": False,
        "historical_human_labels_scope": manifest["human_labels"]["scope"],
        "selection_note": manifest["selection_note"],
        "cross_model_agreement": _cross_model_agreement(rows),
        "by_model": summaries,
    }


async def run(*, manifest_path: Path, output: Path, key_file: Path,
              max_paid_calls: int, concurrency: int = 6) -> dict[str, Any]:
    if max_paid_calls < 0:
        raise ValueError("max_paid_calls must be nonnegative")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    missing = _missing_call_count(manifest, output)
    if missing > max_paid_calls:
        raise ValueError(f"planned missing calls {missing} exceed --max-paid-calls {max_paid_calls}")
    key = load_key(key_file)
    sem = asyncio.Semaphore(concurrency)
    progress_lock = threading.Lock()
    tasks = [
        _one_round(manifest=manifest, pair=pair, model=model, round_index=round_index,
                   output=output, key=key, progress_lock=progress_lock, sem=sem)
        for pair, model, round_index in _work_items(manifest)
    ]
    await asyncio.gather(*tasks)
    rows = _read_call_rows(manifest, output)
    summary = summarize(manifest=manifest, rows=rows)
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-repeat-rounds", type=int, default=1)
    parser.add_argument("--max-paid-calls", type=int)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=6)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.panel is None:
            parser.error("--panel is required with --prepare-only")
        prepare_manifest(panel_path=args.panel, output=args.output,
                         fixed_repeat_rounds=args.fixed_repeat_rounds)
        return 0
    if args.manifest is None or args.key_file is None or args.max_paid_calls is None:
        parser.error("--manifest, --key-file, and --max-paid-calls are required with --run")
    summary = asyncio.run(run(manifest_path=args.manifest, output=args.output,
                              key_file=args.key_file, max_paid_calls=args.max_paid_calls,
                              concurrency=args.concurrency))
    print(json.dumps({"pair_count": summary["pair_count"], "models": summary["models"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
