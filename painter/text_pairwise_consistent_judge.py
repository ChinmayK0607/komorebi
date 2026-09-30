"""Strict, identity-keyed pairwise judge with balanced conflict rechecks.

This is an isolated transport hardening of the identity judge.  It keeps the
same joint sheet, model, and evidence-first prompt while asking the gateway
for a strict dynamic JSON Schema response.  A disagreement gets a balanced
second order pair; a 3-of-4 identity consensus is required to resolve it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_identity_judge as base
    from painter.text_pairwise_judge import ENDPOINT, MODEL, digest, load_key
except ImportError:
    import text_pairwise_identity_judge as base
    from text_pairwise_judge import ENDPOINT, MODEL, digest, load_key


PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
FORMAT_VERSION = "identity-joint-schema-v1"
PROMPT = base.PROMPT
MAGNITUDES = base.MAGNITUDES
SOURCE_FORMAT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def image_id(path: Path) -> str:
    return base.image_id(path)


def _response_schema(ids: tuple[str, str]) -> dict[str, Any]:
    assessment = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "brief_fidelity": {"type": "string", "enum": ["clear", "partial", "missing"]},
            "visible_evidence": {"type": "string", "minLength": 1},
            "weaknesses": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        },
        "required": ["brief_fidelity", "visible_evidence", "weaknesses"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assessments": {
                "type": "object", "additionalProperties": False,
                "properties": {ident: assessment for ident in ids}, "required": list(ids),
            },
            "reason": {"type": "string", "minLength": 1},
            "winner_id": {"type": "string", "enum": [*ids, "tie"]},
            "preference": {"type": "string", "enum": list(MAGNITUDES)},
        },
        "required": ["assessments", "reason", "winner_id", "preference"],
    }


def _response_format(ids: tuple[str, str]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {
        "name": "painting_comparison", "strict": True, "schema": _response_schema(ids),
    }}


def _known_key_map(value: dict[str, Any], known: set[str], *, aliases: set[str] = set(), allow_unknown: bool = False) -> tuple[dict[str, Any], list[str], list[str]]:
    """Conservatively normalize known key whitespace and one explicit alias."""

    result: dict[str, Any] = {}
    transformations: list[str] = []
    ignored: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("non-string JSON key")
        canonical = key if key in known or key in aliases else None
        if canonical is None and key.strip() in known:
            canonical = key.strip()
            transformations.append(f"trimmed_key:{key!r}->{canonical}")
        if canonical in aliases:
            canonical = "preference"
            transformations.append("alias:_preference->preference")
        if canonical is None:
            if allow_unknown:
                ignored.append(key)
                transformations.append(f"ignored_top_key:{key}")
                continue
            raise ValueError(f"unknown nested key:{key}")
        if canonical in result:
            # Some gateway responses repeat the canonical field alongside the
            # explicit ``_preference`` alias.  This is lossless only when the
            # values are exactly equal; a conflicting duplicate remains
            # invalid and must stay censored.
            if result[canonical] != item:
                raise ValueError(f"conflicting key alias:{canonical}")
            transformations.append(f"duplicate_equal:{key!r}->{canonical}")
            continue
        result[canonical] = item
    return result, transformations, ignored


def _normalize_response(raw: str, ids: tuple[str, str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    value = json.loads(raw.strip())
    if not isinstance(value, dict):
        raise ValueError("identity judge response is not an object")
    normalized, transformations, ignored = _known_key_map(value, {"assessments", "reason", "winner_id", "preference"}, aliases={"_preference"}, allow_unknown=True)
    required = {"assessments", "reason", "winner_id", "preference"}
    if set(normalized) != required:
        raise ValueError("identity judge omitted exact top-level fields")
    assessments = normalized["assessments"]
    if not isinstance(assessments, dict) or set(assessments) != set(ids):
        raise ValueError("identity judge assessment IDs do not match images")
    checked: dict[str, Any] = {}
    for ident in ids:
        assessment = assessments[ident]
        if not isinstance(assessment, dict):
            raise ValueError("identity judge assessment is not an object")
        checked_assessment, changes, _ = _known_key_map(assessment, {"brief_fidelity", "visible_evidence", "weaknesses"})
        transformations.extend(changes)
        if set(checked_assessment) != {"brief_fidelity", "visible_evidence", "weaknesses"}:
            raise ValueError("identity judge omitted exact assessment fields")
        if checked_assessment["brief_fidelity"] not in {"clear", "partial", "missing"}:
            raise ValueError("identity judge omitted valid fidelity")
        if not isinstance(checked_assessment["visible_evidence"], str) or not checked_assessment["visible_evidence"].strip():
            raise ValueError("identity judge omitted visible evidence")
        weaknesses = checked_assessment["weaknesses"]
        if not isinstance(weaknesses, list) or not weaknesses or not all(isinstance(item, str) and item.strip() for item in weaknesses):
            raise ValueError("identity judge omitted weaknesses")
        checked[ident] = checked_assessment
    if not isinstance(normalized["reason"], str) or not normalized["reason"].strip():
        raise ValueError("identity judge omitted comparison reason")
    if normalized["winner_id"] not in (*ids, "tie"):
        raise ValueError("identity judge winner ID is invalid")
    if normalized["preference"] not in MAGNITUDES:
        raise ValueError("identity judge preference magnitude is invalid")
    if (normalized["winner_id"] == "tie") != (normalized["preference"] == "tied"):
        raise ValueError("identity judge winner and preference disagree")
    return ({"assessments": checked, "reason": normalized["reason"], "winner_id": normalized["winner_id"], "preference": normalized["preference"]}, {"transformations": transformations, "ignored_top_keys": ignored})


def _parse(raw: str, ids: tuple[str, str]) -> dict[str, Any]:
    return _normalize_response(raw, ids)[0]


def _error_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name.replace("_", "").isalnum() else "ProviderError"


def _view(*, key: str, brief: str, a: Path, b: Path, a_id: str, b_id: str) -> dict[str, Any]:
    sheet, sheet_sha = base._sheet(a, b, a_id, b_id)
    ids = (a_id, b_id)
    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 4096,
        "response_format": _response_format(ids), "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PAINTING BRIEF:\n{brief}\nThe two opaque image IDs are {a_id} and {b_id}. Assess each ID before comparing."},
                {"type": "image_url", "image_url": {"url": sheet}},
            ]},
        ],
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    call: dict[str, Any] = {
        "raw_response": "", "image_sha256": {"A": digest(a), "B": digest(b)},
        "image_ids": {"A": a_id, "B": b_id}, "sheet_sha256": sheet_sha,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "format_version": FORMAT_VERSION, "source_format_sha256": SOURCE_FORMAT_SHA256,
        "response_schema": "painting_comparison", "elapsed_seconds": 0.0,
        "usage": None, "finish_reason": None, "model": MODEL,
        "normalization": {"transformations": [], "ignored_top_keys": []},
    }
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        call.update({"elapsed_seconds": time.monotonic() - started,
                     "provider_error": "HTTPError", "http_status": int(exc.code)})
        return call
    except Exception as exc:
        call.update({"elapsed_seconds": time.monotonic() - started,
                     "provider_error": _error_name(exc)})
        return call
    call["elapsed_seconds"] = time.monotonic() - started
    try:
        body = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        call.update({"raw_response": response_bytes.decode("utf-8", errors="replace"), "parse_error": "malformed_json"})
        return call
    if isinstance(body, dict):
        call["usage"] = body.get("usage")
        call["model"] = body.get("model", MODEL)
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
        call.update({"raw_response": json.dumps(body, sort_keys=True, ensure_ascii=False), "parse_error": "malformed_provider_response"})
        return call
    choice = body["choices"][0]
    if not isinstance(choice, dict):
        call.update({"raw_response": json.dumps(choice, sort_keys=True, ensure_ascii=False), "parse_error": "malformed_provider_response"})
        return call
    message = choice.get("message")
    raw = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(raw, str):
        raw = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    call["raw_response"] = raw
    call["finish_reason"] = choice.get("finish_reason")
    if call["finish_reason"] == "length":
        call["parse_error"] = "truncated"
        return call
    try:
        call["assessment"], call["normalization"] = _normalize_response(raw, ids)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        call["parse_error"] = type(exc).__name__
    return call


async def _call_with_retry(*, key: str, brief: str, a: Path, b: Path,
                           a_id: str, b_id: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    # One bounded retry is reserved for a malformed/truncated response.  A
    # schema HTTP error is recorded once and is never silently retried/fallbacked.
    for attempt in range(2):
        try:
            call = await asyncio.to_thread(_view, key=key, brief=brief, a=a, b=b, a_id=a_id, b_id=b_id)
        except Exception as exc:
            call = {"provider_error": _error_name(exc), "raw_response": "", "usage": None, "elapsed_seconds": 0.0}
        call["attempt"] = attempt + 1
        calls.append(call)
        if "assessment" in call:
            return {"assessment": call["assessment"], "calls": calls}
        if "parse_error" not in call:
            break
    return {"assessment": None, "calls": calls, "error": "incomplete_evidence"}


def _score(winner: str, preference: str, anchor_id: str, candidate_id: str) -> float:
    if winner == "tie":
        return 0.5
    delta = MAGNITUDES.index(preference) / 6
    return 0.5 + delta if winner == candidate_id else 0.5 - delta


def _assessment_votes(views: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(view["assessment"]["winner_id"], view["assessment"]["preference"])
            for view in views if view.get("assessment")]


def _result_base(*, brief: Path, anchor: Path, candidate: Path,
                 anchor_id: str, candidate_id: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "format_version": FORMAT_VERSION,
        "source_format_sha256": SOURCE_FORMAT_SHA256,
        "brief_sha256": digest(brief),
        "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
        "image_sha256": {"anchor": digest(anchor), "candidate": digest(candidate)},
    }


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path) -> dict[str, Any]:
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
        result = _result_base(brief=brief, anchor=anchor, candidate=candidate,
                              anchor_id=anchor_id, candidate_id=candidate_id)
        result.update({"status": "identical_canvases", "winner_id": "tie", "preference": "tied",
                       "score": 0.5, "consensus": "identical", "acceptance": "true_tie",
                       "initial_conflict": False, "initial_votes": [], "recheck_votes": [],
                       "views": [], "recheck_views": []})
        return result
    key = load_key(key_file)
    ordered = sorted(((anchor_id, anchor), (candidate_id, candidate)), key=lambda item: item[0])
    first_ids = (ordered[0][0], ordered[1][0])
    second_ids = (ordered[1][0], ordered[0][0])
    first, second = await asyncio.gather(
        _call_with_retry(key=key, brief=text, a=ordered[0][1], b=ordered[1][1], a_id=first_ids[0], b_id=first_ids[1]),
        _call_with_retry(key=key, brief=text, a=ordered[1][1], b=ordered[0][1], a_id=second_ids[0], b_id=second_ids[1]),
    )
    result = _result_base(brief=brief, anchor=anchor, candidate=candidate,
                          anchor_id=anchor_id, candidate_id=candidate_id)
    initial_views = [first, second]
    initial_votes = _assessment_votes(initial_views)
    initial_conflict = len(initial_votes) == 2 and initial_votes[0][0] != initial_votes[1][0]
    result.update({"views": initial_views, "recheck_views": [], "initial_votes": [vote for vote, _ in initial_votes],
                   "recheck_votes": [], "initial_conflict": initial_conflict})
    if len(initial_votes) != 2:
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None,
                       "consensus": "failed_initial_call", "acceptance": "censored"})
        return result
    if not initial_conflict:
        winner = initial_votes[0][0]
        preference = min((magnitude for _, magnitude in initial_votes), key=MAGNITUDES.index)
        if winner == "tie":
            preference = "tied"
        result.update({"status": "scored", "winner_id": winner, "preference": preference,
                       "score": _score(winner, preference, anchor_id, candidate_id),
                       "consensus": "initial_agreement", "acceptance": "two_of_two"})
        return result
    recheck_first, recheck_second = await asyncio.gather(
        _call_with_retry(key=key, brief=text, a=ordered[0][1], b=ordered[1][1], a_id=first_ids[0], b_id=first_ids[1]),
        _call_with_retry(key=key, brief=text, a=ordered[1][1], b=ordered[0][1], a_id=second_ids[0], b_id=second_ids[1]),
    )
    recheck_views = [recheck_first, recheck_second]
    recheck_votes = _assessment_votes(recheck_views)
    result["recheck_views"] = recheck_views
    result["recheck_votes"] = [vote for vote, _ in recheck_votes]
    all_votes = initial_votes + recheck_votes
    if len(all_votes) != 4:
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None,
                       "consensus": "failed_recheck_call", "acceptance": "censored"})
        return result
    counts = {winner: sum(vote == winner for vote, _ in all_votes) for winner, _ in all_votes}
    majority = next((winner for winner, count in counts.items() if count >= 3), None)
    if majority is None:
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None,
                       "consensus": "no_three_of_four", "acceptance": "censored"})
        return result
    magnitudes = [magnitude for vote, magnitude in all_votes if vote == majority]
    preference = "tied" if majority == "tie" else min(magnitudes, key=MAGNITUDES.index)
    result.update({"status": "scored", "winner_id": majority, "preference": preference,
                   "score": _score(majority, preference, anchor_id, candidate_id),
                   "consensus": "three_of_four" if counts[majority] == 3 else "four_of_four",
                   "acceptance": "recheck_consensus"})
    return result


__all__ = ["FORMAT_VERSION", "MAGNITUDES", "MODEL", "PROMPT", "PROTOCOL",
           "SOURCE_FORMAT_SHA256", "_parse", "_response_format", "_response_schema",
           "image_id", "judge_pair"]
