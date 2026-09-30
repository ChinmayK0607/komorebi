"""Consistent joint-sheet judge with side-keyed model output.

This transport variant keeps the identity judge's image ordering, balanced
rechecks, score scale, and aesthetic rubric.  The response uses LEFT/RIGHT
fields instead of asking the model to reproduce long content-derived IDs; the
caller maps those sides back to the opaque image IDs before aggregation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_consistent_judge as base
    from painter.text_pairwise_judge import ENDPOINT, MODEL, digest, load_key
except ImportError:  # pragma: no cover - direct node execution fallback
    import text_pairwise_consistent_judge as base
    from text_pairwise_judge import ENDPOINT, MODEL, digest, load_key


PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
FORMAT_VERSION = "identity-joint-side-schema-v1"
MAGNITUDES = base.MAGNITUDES
SOURCE_FORMAT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Preserve the existing aesthetic criteria and replace every transport phrase
# that asked the model to copy opaque IDs or emit ``winner_id``.
_AESTHETIC_PROMPT = base.PROMPT.split("Return exactly one JSON object", 1)[0]
_AESTHETIC_PROMPT = _AESTHETIC_PROMPT.replace(
    "for each opaque image ID first", "for each labeled panel first"
).replace(
    "choose winner_id as one of the exact opaque image IDs or tie",
    "compare the labeled panels and choose the stronger one",
).replace(
    "Winner_id tie requires preference tied; a non-tie winner requires a non-tied preference. ",
    "",
).rstrip()
PROMPT = _AESTHETIC_PROMPT + """

Return exactly one JSON object with exactly these top-level fields: left_assessment, right_assessment, reason, winner, preference. The left_assessment describes the painting shown in the LEFT panel and the right_assessment describes the RIGHT panel. Each assessment must contain exactly brief_fidelity, visible_evidence, and weaknesses; weaknesses must be a nonempty list. Use winner LEFT, RIGHT, or TIE and preference tied, slight, moderate, or clear. TIE requires tied; a non-tie winner requires a non-tied preference. Do not reproduce checkpoint names.
"""


def image_id(path: Path) -> str:
    return base.image_id(path)


def _assessment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "brief_fidelity": {"type": "string", "enum": ["clear", "partial", "missing"]},
            "visible_evidence": {"type": "string", "minLength": 1},
            "weaknesses": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        },
        "required": ["brief_fidelity", "visible_evidence", "weaknesses"],
    }


def _response_schema() -> dict[str, Any]:
    assessment = _assessment_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "left_assessment": assessment,
            "right_assessment": assessment,
            "reason": {"type": "string", "minLength": 1},
            "winner": {"type": "string", "enum": ["LEFT", "RIGHT", "TIE"]},
            "preference": {"type": "string", "enum": list(MAGNITUDES)},
        },
        "required": ["left_assessment", "right_assessment", "reason", "winner", "preference"],
    }


def _response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {
        "name": "painting_comparison_by_side", "strict": True, "schema": _response_schema(),
    }}


def _check_assessment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"brief_fidelity", "visible_evidence", "weaknesses"}:
        raise ValueError("side judge omitted exact assessment fields")
    if value["brief_fidelity"] not in {"clear", "partial", "missing"}:
        raise ValueError("side judge omitted valid fidelity")
    if not isinstance(value["visible_evidence"], str) or not value["visible_evidence"].strip():
        raise ValueError("side judge omitted visible evidence")
    weaknesses = value["weaknesses"]
    if not isinstance(weaknesses, list) or not weaknesses or not all(isinstance(item, str) and item.strip() for item in weaknesses):
        raise ValueError("side judge omitted weaknesses")
    return value


def _parse(raw: str, ids: tuple[str, str]) -> dict[str, Any]:
    """Validate side-keyed output and map LEFT/RIGHT to opaque image IDs."""

    value = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    required = {"left_assessment", "right_assessment", "reason", "winner", "preference"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("side judge omitted exact top-level fields")
    left = _check_assessment(value["left_assessment"])
    right = _check_assessment(value["right_assessment"])
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("side judge omitted comparison reason")
    if value["winner"] not in {"LEFT", "RIGHT", "TIE"}:
        raise ValueError("side judge winner is invalid")
    if value["preference"] not in MAGNITUDES:
        raise ValueError("side judge preference magnitude is invalid")
    if (value["winner"] == "TIE") != (value["preference"] == "tied"):
        raise ValueError("side judge winner and preference disagree")
    winner = {"LEFT": ids[0], "RIGHT": ids[1], "TIE": "tie"}[value["winner"]]
    return {
        "assessments": {ids[0]: left, ids[1]: right},
        "reason": value["reason"],
        "winner_id": winner,
        "preference": value["preference"],
    }


def _view(*, key: str, brief: str, a: Path, b: Path, a_id: str, b_id: str) -> dict[str, Any]:
    """Send one side-keyed joint view, retaining raw/usage metadata."""

    sheet, sheet_sha = base.base._sheet(a, b, "LEFT", "RIGHT")
    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 4096,
        "response_format": _response_format(), "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PAINTING BRIEF:\n{brief}\nAssess the LEFT and RIGHT panels before comparing them."},
                {"type": "image_url", "image_url": {"url": sheet}},
            ]},
        ],
    }
    request = __import__("urllib.request", fromlist=["Request"]).Request(
        ENDPOINT, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    import time
    import urllib.error
    import urllib.request
    started = time.monotonic()
    call: dict[str, Any] = {
        "raw_response": "", "image_sha256": {"A": digest(a), "B": digest(b)},
        "image_ids": {"A": a_id, "B": b_id}, "sheet_sha256": sheet_sha,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "format_version": FORMAT_VERSION, "source_format_sha256": SOURCE_FORMAT_SHA256,
        "response_schema": "painting_comparison_by_side", "elapsed_seconds": 0.0,
        "usage": None, "finish_reason": None, "model": MODEL,
    }
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        call.update({"elapsed_seconds": time.monotonic() - started, "provider_error": "HTTPError", "http_status": int(exc.code)})
        return call
    except Exception as exc:
        name = type(exc).__name__
        call.update({"elapsed_seconds": time.monotonic() - started, "provider_error": name if name.isalnum() else "ProviderError"})
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
    message = choice.get("message") if isinstance(choice, dict) else None
    raw = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(raw, str):
        raw = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    call["raw_response"] = raw
    call["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
    if call["finish_reason"] == "length":
        call["parse_error"] = "truncated"
        return call
    try:
        call["assessment"] = _parse(raw, (a_id, b_id))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        call["parse_error"] = type(exc).__name__
    return call


async def _call_with_retry(*, key: str, brief: str, a: Path, b: Path, a_id: str, b_id: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for attempt in range(2):
        call = await asyncio.to_thread(_view, key=key, brief=brief, a=a, b=b, a_id=a_id, b_id=b_id)
        call["attempt"] = attempt + 1
        calls.append(call)
        if "assessment" in call or "parse_error" not in call:
            break
    if calls and "assessment" in calls[-1]:
        return {"assessment": calls[-1]["assessment"], "calls": calls}
    return {"assessment": None, "calls": calls, "error": "incomplete_evidence"}


def _score(winner: str, preference: str, anchor_id: str, candidate_id: str) -> float:
    if winner == "tie":
        return 0.5
    delta = MAGNITUDES.index(preference) / 6
    return 0.5 + delta if winner == candidate_id else 0.5 - delta


def _assessment_votes(views: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(view["assessment"]["winner_id"], view["assessment"]["preference"])
            for view in views if view.get("assessment")]


def _result_base(*, brief: Path, anchor: Path, candidate: Path, anchor_id: str, candidate_id: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "format_version": FORMAT_VERSION,
        "source_format_sha256": SOURCE_FORMAT_SHA256, "brief_sha256": digest(brief),
        "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
        "image_sha256": {"anchor": digest(anchor), "candidate": digest(candidate)},
    }


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path, key_file: Path) -> dict[str, Any]:
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
        result = _result_base(brief=brief, anchor=anchor, candidate=candidate, anchor_id=anchor_id, candidate_id=candidate_id)
        result.update({"status": "identical_canvases", "winner_id": "tie", "preference": "tied", "score": 0.5,
                       "consensus": "identical", "acceptance": "true_tie", "initial_conflict": False,
                       "initial_votes": [], "recheck_votes": [], "views": [], "recheck_views": []})
        return result
    key = load_key(key_file)
    ordered = sorted(((anchor_id, anchor), (candidate_id, candidate)), key=lambda item: item[0])
    first_ids = (ordered[0][0], ordered[1][0])
    second_ids = (ordered[1][0], ordered[0][0])
    first, second = await asyncio.gather(
        _call_with_retry(key=key, brief=text, a=ordered[0][1], b=ordered[1][1], a_id=first_ids[0], b_id=first_ids[1]),
        _call_with_retry(key=key, brief=text, a=ordered[1][1], b=ordered[0][1], a_id=second_ids[0], b_id=second_ids[1]),
    )
    result = _result_base(brief=brief, anchor=anchor, candidate=candidate, anchor_id=anchor_id, candidate_id=candidate_id)
    initial_views = [first, second]
    initial_votes = _assessment_votes(initial_views)
    initial_conflict = len(initial_votes) == 2 and initial_votes[0][0] != initial_votes[1][0]
    result.update({"views": initial_views, "recheck_views": [], "initial_votes": [vote for vote, _ in initial_votes], "recheck_votes": [], "initial_conflict": initial_conflict})
    if len(initial_votes) != 2:
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None, "consensus": "failed_initial_call", "acceptance": "censored"})
        return result
    if not initial_conflict:
        winner = initial_votes[0][0]
        preference = min((magnitude for _, magnitude in initial_votes), key=MAGNITUDES.index)
        if winner == "tie":
            preference = "tied"
        result.update({"status": "scored", "winner_id": winner, "preference": preference, "score": _score(winner, preference, anchor_id, candidate_id), "consensus": "initial_agreement", "acceptance": "two_of_two"})
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
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None, "consensus": "failed_recheck_call", "acceptance": "censored"})
        return result
    counts = {winner: sum(vote == winner for vote, _ in all_votes) for winner, _ in all_votes}
    majority = next((winner for winner, count in counts.items() if count >= 3), None)
    if majority is None:
        result.update({"status": "uncertain", "winner_id": None, "preference": None, "score": None, "consensus": "no_three_of_four", "acceptance": "censored"})
        return result
    magnitudes = [magnitude for vote, magnitude in all_votes if vote == majority]
    preference = "tied" if majority == "tie" else min(magnitudes, key=MAGNITUDES.index)
    result.update({"status": "scored", "winner_id": majority, "preference": preference, "score": _score(majority, preference, anchor_id, candidate_id), "consensus": "three_of_four" if counts[majority] == 3 else "four_of_four", "acceptance": "recheck_consensus"})
    return result


__all__ = ["FORMAT_VERSION", "MAGNITUDES", "MODEL", "PROMPT", "PROTOCOL", "SOURCE_FORMAT_SHA256", "_parse", "_response_format", "_response_schema", "image_id", "judge_pair"]
