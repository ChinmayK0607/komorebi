"""Model-selectable transport for the frozen v2 side-keyed judge.

This module reuses the frozen v2 rubric and consensus code while making the
Gateway model an explicit per-call argument.  It is for offline calibration
only; it is not imported by the training reward path.  A missing or different
returned model identity is censored rather than scored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

try:
    from painter import text_pairwise_consistent_side_judge_v2 as rubric
    from painter import text_pairwise_identity_judge as sheet_transport
    from painter.text_pairwise_judge import ENDPOINT, MODEL as DEFAULT_MODEL, digest, load_key
except ImportError:  # pragma: no cover - direct node execution fallback
    import text_pairwise_consistent_side_judge_v2 as rubric
    import text_pairwise_identity_judge as sheet_transport
    from text_pairwise_judge import ENDPOINT, MODEL as DEFAULT_MODEL, digest, load_key


PROTOCOL = "text-paint-sft-pairwise-consistent-v2-model-calibration"
FORMAT_VERSION = "identity-joint-side-schema-v2-model-calibration"
MAGNITUDES = rubric.MAGNITUDES
PROMPT = rubric.PROMPT
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
RUBRIC_SOURCE_FORMAT_SHA256 = rubric.SOURCE_FORMAT_SHA256
SOURCE_FORMAT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TEMPERATURE = 0
MAX_TOKENS = 4096
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def validate_model(model: str) -> str:
    """Accept only a concrete Gateway model ID without changing global state."""

    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
        raise ValueError(f"invalid model ID: {model!r}")
    return model


def image_id(path: Path) -> str:
    return rubric.image_id(path)


def _response_schema() -> dict[str, Any]:
    return rubric._response_schema()


def _response_format() -> dict[str, Any]:
    return rubric._response_format()


def _parse(raw: str, ids: tuple[str, str]) -> dict[str, Any]:
    return rubric._parse(raw, ids)


def _normalise_reasoning(reasoning: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate an explicit Gateway reasoning object without changing defaults."""

    if reasoning is None:
        return None
    if not isinstance(reasoning, Mapping):
        raise ValueError("reasoning must be a mapping when supplied")
    value = dict(reasoning)
    if value != {"effort": "medium"}:
        raise ValueError("only the registered reasoning object {'effort': 'medium'} is supported")
    return value


def _model_config_sha256(model: str, reasoning: Mapping[str, Any] | None = None) -> str:
    reasoning_value = _normalise_reasoning(reasoning)
    value = {
        "model": validate_model(model),
        "max_tokens": MAX_TOKENS,
        "response_format": _response_format(),
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
    }
    if reasoning_value is None:
        value["temperature"] = TEMPERATURE
    else:
        value["reasoning"] = reasoning_value
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _view(*, key: str, model: str, brief: str, a: Path, b: Path,
          a_id: str, b_id: str,
          reasoning: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Send one view with explicit model and strict returned-model binding."""

    model = validate_model(model)
    reasoning_value = _normalise_reasoning(reasoning)
    sheet, sheet_sha = sheet_transport._sheet(a, b, "LEFT", "RIGHT")
    payload = {
        "model": model, "max_tokens": MAX_TOKENS,
        "response_format": _response_format(), "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PAINTING BRIEF:\n{brief}\nAssess the LEFT and RIGHT panels before comparing them."},
                {"type": "image_url", "image_url": {"url": sheet}},
            ]},
        ],
    }
    if reasoning_value is None:
        payload["temperature"] = TEMPERATURE
    else:
        payload["reasoning"] = reasoning_value
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    call: dict[str, Any] = {
        "raw_response": "", "image_sha256": {"A": digest(a), "B": digest(b)},
        "image_ids": {"A": a_id, "B": b_id}, "sheet_sha256": sheet_sha,
        "prompt_sha256": PROMPT_SHA256, "format_version": FORMAT_VERSION,
        "source_format_sha256": SOURCE_FORMAT_SHA256,
        "rubric_source_format_sha256": RUBRIC_SOURCE_FORMAT_SHA256,
        "response_schema": "painting_comparison_by_side", "elapsed_seconds": 0.0,
        "usage": None, "finish_reason": None, "requested_model": model,
        "returned_model": None, "request_config_sha256": _model_config_sha256(model, reasoning_value),
        "observed_reasoning_tokens": None,
    }
    if reasoning_value is None:
        call["reasoning"] = {"mode": "provider_default", "request_field_omitted": True}
    else:
        call["reasoning"] = {
            "mode": "explicit", "request_field": "reasoning",
            "request": reasoning_value, "sampling_parameters_omitted": True,
        }
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        call.update({"elapsed_seconds": time.monotonic() - started,
                     "provider_error": "HTTPError", "http_status": int(exc.code)})
        return call
    except Exception as exc:
        name = type(exc).__name__
        call.update({"elapsed_seconds": time.monotonic() - started,
                     "provider_error": name if name.replace("_", "").isalnum() else "ProviderError"})
        return call
    call["elapsed_seconds"] = time.monotonic() - started
    try:
        body = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        call.update({"raw_response": response_bytes.decode("utf-8", errors="replace"),
                     "parse_error": "malformed_json"})
        return call
    if isinstance(body, dict):
        call["usage"] = body.get("usage")
        returned = body.get("model")
        call["returned_model"] = returned if isinstance(returned, str) and returned else None
        usage = body.get("usage")
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), (int, float)):
                call["observed_reasoning_tokens"] = int(details["reasoning_tokens"])
            elif isinstance(usage.get("reasoning_tokens"), (int, float)):
                call["observed_reasoning_tokens"] = int(usage["reasoning_tokens"])
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
        call.update({"raw_response": json.dumps(body, sort_keys=True, ensure_ascii=False),
                     "parse_error": "malformed_provider_response"})
        return call
    choice = body["choices"][0]
    message = choice.get("message") if isinstance(choice, dict) else None
    raw = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(raw, str):
        raw = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    call["raw_response"] = raw
    call["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
    if call["returned_model"] != model:
        call["parse_error"] = "model_identity_mismatch"
        return call
    if call["finish_reason"] == "length":
        call["parse_error"] = "truncated"
        return call
    try:
        call["assessment"] = _parse(raw, (a_id, b_id))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        call["parse_error"] = type(exc).__name__
    return call


async def _call_with_retry(*, key: str, model: str, brief: str, a: Path, b: Path,
                           a_id: str, b_id: str,
                           reasoning: Mapping[str, Any] | None = None) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for attempt in range(2):
        call = await asyncio.to_thread(_view, key=key, model=model, brief=brief,
                                       a=a, b=b, a_id=a_id, b_id=b_id,
                                       reasoning=reasoning)
        call["attempt"] = attempt + 1
        calls.append(call)
        if "assessment" in call or "parse_error" not in call:
            break
    if calls and "assessment" in calls[-1]:
        return {"assessment": calls[-1]["assessment"], "calls": calls}
    return {"assessment": None, "calls": calls, "error": "incomplete_evidence"}


def _result_base(*, model: str, brief: Path, anchor: Path, candidate: Path,
                 anchor_id: str, candidate_id: str,
                 reasoning: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "format_version": FORMAT_VERSION,
        "source_format_sha256": SOURCE_FORMAT_SHA256,
        "rubric_source_format_sha256": RUBRIC_SOURCE_FORMAT_SHA256,
        "prompt_sha256": PROMPT_SHA256, "requested_model": validate_model(model),
        "model_config_sha256": _model_config_sha256(model, reasoning),
        "brief_sha256": digest(brief),
        "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
        "image_sha256": {"anchor": digest(anchor), "candidate": digest(candidate)},
    }


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path, model: str = DEFAULT_MODEL,
                     reasoning: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Judge one pair using ``model`` without mutating module or provider state."""

    model = validate_model(model)
    reasoning_value = _normalise_reasoning(reasoning)
    for path in (brief, anchor, candidate):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    text = brief.read_text(encoding="utf-8").strip()
    if not text or len(text) > 4000:
        raise ValueError("invalid painting brief")
    anchor_sha, candidate_sha = digest(anchor), digest(candidate)
    anchor_id, candidate_id = image_id(anchor), image_id(candidate)
    if anchor_id == candidate_id and anchor_sha != candidate_sha:
        raise ValueError("image identity prefix collision")
    if anchor_sha == candidate_sha:
        result = _result_base(model=model, brief=brief, anchor=anchor, candidate=candidate,
                              anchor_id=anchor_id, candidate_id=candidate_id,
                              reasoning=reasoning_value)
        result.update({"status": "identical_canvases", "winner_id": "tie",
                       "preference": "tied", "score": 0.5,
                       "consensus": "identical", "acceptance": "true_tie",
                       "initial_conflict": False, "initial_votes": [],
                       "recheck_votes": [], "views": [], "recheck_views": []})
        if reasoning_value is not None:
            result["reasoning"] = {
                "mode": "explicit", "request": reasoning_value,
                "sampling_parameters_omitted": True,
            }
        return result
    key = load_key(key_file)
    ordered = sorted(((anchor_id, anchor), (candidate_id, candidate)),
                     key=lambda item: item[0])
    first_ids = (ordered[0][0], ordered[1][0])
    second_ids = (ordered[1][0], ordered[0][0])
    first, second = await asyncio.gather(
        _call_with_retry(key=key, model=model, brief=text, a=ordered[0][1], b=ordered[1][1],
                         a_id=first_ids[0], b_id=first_ids[1], reasoning=reasoning_value),
        _call_with_retry(key=key, model=model, brief=text, a=ordered[1][1], b=ordered[0][1],
                         a_id=second_ids[0], b_id=second_ids[1], reasoning=reasoning_value),
    )
    result = _result_base(model=model, brief=brief, anchor=anchor, candidate=candidate,
                          anchor_id=anchor_id, candidate_id=candidate_id,
                          reasoning=reasoning_value)
    initial_views = [first, second]
    initial_votes = rubric._assessment_votes(initial_views)
    initial_conflict = len(initial_votes) == 2 and initial_votes[0][0] != initial_votes[1][0]
    result.update({"views": initial_views, "recheck_views": [],
                   "returned_models": sorted({
                       call.get("returned_model")
                       for view in initial_views for call in view.get("calls", [])
                       if call.get("returned_model")
                   }),
                   "initial_votes": [vote for vote, _ in initial_votes],
                   "recheck_votes": [], "initial_conflict": initial_conflict})
    if reasoning_value is not None:
        result["reasoning"] = {
            "mode": "explicit", "request": reasoning_value,
            "sampling_parameters_omitted": True,
        }
    if len(initial_votes) != 2:
        result.update({"status": "uncertain", "winner_id": None,
                       "preference": None, "score": None,
                       "consensus": "failed_initial_call", "acceptance": "censored"})
        return result
    if not initial_conflict:
        winner = initial_votes[0][0]
        preference = min((magnitude for _, magnitude in initial_votes), key=MAGNITUDES.index)
        if winner == "tie":
            preference = "tied"
        result.update({"status": "scored", "winner_id": winner,
                       "preference": preference,
                       "score": rubric._score(winner, preference, anchor_id, candidate_id),
                       "consensus": "initial_agreement", "acceptance": "two_of_two"})
        return result
    recheck_first, recheck_second = await asyncio.gather(
        _call_with_retry(key=key, model=model, brief=text, a=ordered[0][1], b=ordered[1][1],
                         a_id=first_ids[0], b_id=first_ids[1], reasoning=reasoning_value),
        _call_with_retry(key=key, model=model, brief=text, a=ordered[1][1], b=ordered[0][1],
                         a_id=second_ids[0], b_id=second_ids[1], reasoning=reasoning_value),
    )
    recheck_views = [recheck_first, recheck_second]
    recheck_votes = rubric._assessment_votes(recheck_views)
    result["recheck_views"] = recheck_views
    result["recheck_votes"] = [vote for vote, _ in recheck_votes]
    result["returned_models"] = sorted({
        call.get("returned_model")
        for view in initial_views + recheck_views for call in view.get("calls", [])
        if call.get("returned_model")
    })
    all_votes = initial_votes + recheck_votes
    if len(all_votes) != 4:
        result.update({"status": "uncertain", "winner_id": None,
                       "preference": None, "score": None,
                       "consensus": "failed_recheck_call", "acceptance": "censored"})
        return result
    counts = {winner: sum(vote == winner for vote, _ in all_votes)
              for winner, _ in all_votes}
    majority = next((winner for winner, count in counts.items() if count >= 3), None)
    if majority is None:
        result.update({"status": "uncertain", "winner_id": None,
                       "preference": None, "score": None,
                       "consensus": "no_three_of_four", "acceptance": "censored"})
        return result
    magnitudes = [magnitude for vote, magnitude in all_votes if vote == majority]
    preference = "tied" if majority == "tie" else min(magnitudes, key=MAGNITUDES.index)
    result.update({"status": "scored", "winner_id": majority,
                   "preference": preference,
                   "score": rubric._score(majority, preference, anchor_id, candidate_id),
                   "consensus": "three_of_four" if counts[majority] == 3 else "four_of_four",
                   "acceptance": "recheck_consensus"})
    return result


__all__ = [
    "DEFAULT_MODEL", "FORMAT_VERSION", "MAGNITUDES", "MAX_TOKENS", "MODEL_PATTERN",
    "PROMPT", "PROMPT_SHA256", "PROTOCOL", "RUBRIC_SOURCE_FORMAT_SHA256",
    "SOURCE_FORMAT_SHA256", "TEMPERATURE", "_model_config_sha256", "_parse",
    "_response_format", "_response_schema", "_view", "image_id", "judge_pair",
    "validate_model",
]
