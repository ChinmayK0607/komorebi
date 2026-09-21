"""Schema and reward math for the aesthetic pairwise judge.

This module deliberately contains no model client.  A Luna-low worker receives
one blinded reference/A/B comparison, returns the four-field label described by
``JUDGE_PROMPT.md``, and the queue combines the two orientations here.

The score is the fraction of the two orientations in which the original
candidate wins: win = 1, tie = .5, loss = 0.  A contradiction is evidence of
uncertainty but is still a valid averaged score; ``uncertain`` or missing work
is masked as ``None`` so it can never become a false zero reward.
"""

from __future__ import annotations

import json
import math
from typing import Any


PROTOCOL_VERSION = "aesthetic-pairwise-v1"
JUDGE_MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"

WINNERS = frozenset({"A", "B", "tie", "uncertain"})
JUDGMENT_FIELDS = frozenset({
    "winner",
    "reason",
    "weaknesses_A",
    "weaknesses_B",
})
NORMALIZED_FIELDS = frozenset({"protocol_version", "pairwise_win_fraction"})


def validate_judgment(value: Any) -> dict[str, Any]:
    """Validate and return one strict judge response.

    The model response is intentionally closed-world: missing or unknown keys,
    non-string reasons, non-list weakness fields, and non-string list members
    are rejected.  Empty weakness lists are valid because a judge may see no
    concrete weakness for one canvas.
    """

    if not isinstance(value, dict):
        raise ValueError("judgment must be a JSON object")
    if set(value) != JUDGMENT_FIELDS:
        missing = sorted(JUDGMENT_FIELDS - set(value))
        extra = sorted(set(value) - JUDGMENT_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields: {missing}")
        if extra:
            details.append(f"unknown fields: {extra}")
        raise ValueError("; ".join(details) or "invalid judgment fields")

    winner = value["winner"]
    if not isinstance(winner, str) or winner not in WINNERS:
        raise ValueError("winner must be A, B, tie, or uncertain")

    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    result: dict[str, Any] = {"winner": winner, "reason": reason}
    for field in ("weaknesses_A", "weaknesses_B"):
        weaknesses = value[field]
        if not isinstance(weaknesses, list):
            raise ValueError(f"{field} must be a list of strings")
        if any(not isinstance(item, str) or not item.strip() for item in weaknesses):
            raise ValueError(f"{field} must contain only non-empty strings")
        # Copy lists so a caller cannot mutate a validated receipt in place.
        result[field] = list(weaknesses)
    return result


def parse_judgment(text: str) -> dict[str, Any]:
    """Parse a strict JSON object returned by a judge.

    Markdown fences, prose around JSON, and duplicate/unknown fields are not
    accepted.  ``json.loads`` rejects duplicate keys only indirectly, so a
    small object-pairs hook is used to reject them explicitly as malformed.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("judge returned no JSON text")

    def pairs(pairs_list: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs_list:
            if key in result:
                raise ValueError(f"duplicate judgment field: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("judge response must be one JSON object") from exc
    return validate_judgment(value)


def _label_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return validate_judgment(value)


def orientation_score(label: Any, candidate_side: str) -> float | None:
    """Map one blinded label to the original candidate's score.

    ``candidate_side`` identifies which displayed side contains the original
    candidate in that orientation.  ``uncertain`` is masked, while ties remain
    a usable 0.5 preference.
    """

    if candidate_side not in {"A", "B"}:
        raise ValueError("candidate_side must be A or B")
    label = _label_or_none(label)
    if label is None or label["winner"] == "uncertain":
        return None
    winner = label["winner"]
    if winner == "tie":
        return 0.5
    return 1.0 if winner == candidate_side else 0.0


def pairwise_win_fraction(
    first: Any,
    reverse: Any,
    first_candidate_side: str = "A",
    reverse_candidate_side: str = "B",
) -> float | None:
    """Average the two orientation scores, or mask unusable labels.

    The default orientation pair is the queue convention: first displays the
    original candidate as A and reverse displays it as B.  Calling code may
    pass the explicit sides when importing a receipt from another queue.
    """

    first_score = orientation_score(first, first_candidate_side)
    reverse_score = orientation_score(reverse, reverse_candidate_side)
    if first_score is None or reverse_score is None:
        return None
    return (first_score + reverse_score) / 2.0


def reconcile(
    first: Any,
    reverse: Any,
    first_candidate_side: str = "A",
    reverse_candidate_side: str = "B",
) -> dict[str, Any]:
    """Return the versioned normalized judgment consumed by training."""

    score = pairwise_win_fraction(
        first, reverse, first_candidate_side, reverse_candidate_side
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pairwise_win_fraction": score,
    }


def valid_normalized_judgment(value: Any) -> bool:
    """Return whether a normalized record is exactly this protocol's shape."""

    if not isinstance(value, dict) or set(value) != NORMALIZED_FIELDS:
        return False
    if value.get("protocol_version") != PROTOCOL_VERSION:
        return False
    score = value.get("pairwise_win_fraction")
    if score is None:
        return True
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and 0.0 <= float(score) <= 1.0
    )


def reward(judgment: Any = ..., reverse: Any = ..., **kwargs: Any) -> float | None:
    """Return only the holistic final-canvas pairwise reward.

    With two label arguments this is a convenience for
    ``pairwise_win_fraction(first, reverse)``.  With one argument it accepts
    the versioned normalized record.  No action, finish, length, or
    acceptability signal is read or added.
    """

    # Accommodate an environment that keeps the older ``reward(result,
    # judgment)`` call shape.  The result is deliberately ignored; only the
    # normalized aesthetic judgment can affect the value.
    if "judgment" in kwargs and reverse is ...:
        judgment = kwargs["judgment"]
    if "result" in kwargs:
        if reverse is not ...:
            judgment = reverse
    if reverse is not ...:
        if valid_normalized_judgment(reverse):
            return reward(reverse)
        return pairwise_win_fraction(judgment, reverse)
    if not valid_normalized_judgment(judgment):
        return None
    score = judgment["pairwise_win_fraction"]
    return None if score is None else float(score)


def reward_from_orientations(first: Any, reverse: Any) -> float | None:
    """Explicitly named convenience wrapper for environment callers."""

    return pairwise_win_fraction(first, reverse)


reward_aesthetic = reward
parse_response = parse_judgment
validate_response = validate_judgment
score_orientations = pairwise_win_fraction


__all__ = [
    "JUDGE_MODEL",
    "JUDGMENT_FIELDS",
    "NORMALIZED_FIELDS",
    "PROTOCOL_VERSION",
    "REASONING_EFFORT",
    "WINNERS",
    "orientation_score",
    "parse_response",
    "pairwise_win_fraction",
    "parse_judgment",
    "reconcile",
    "reward",
    "reward_aesthetic",
    "reward_from_orientations",
    "valid_normalized_judgment",
    "validate_judgment",
    "validate_response",
    "score_orientations",
]
