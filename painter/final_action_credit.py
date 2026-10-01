"""Opt-in terminal credit gate for native painting reward evaluation.

This module is deliberately independent of the verifier runtime.  The gate is
off unless ``PAINTER_REQUIRE_VALID_FINAL_ACTION=1`` is set, so existing traces
and pilots retain their current reward contract by default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


REQUIRE_VALID_FINAL_ACTION_ENV = "PAINTER_REQUIRE_VALID_FINAL_ACTION"


def require_valid_final_action_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the optional terminal-credit contract is enabled."""

    values = os.environ if environ is None else environ
    return values.get(REQUIRE_VALID_FINAL_ACTION_ENV) == "1"


def validate_last_turn(info: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the recorded terminal turn.

    A missing or malformed record is an operational evidence error.  It must
    never be converted into a model zero because doing so would invent a
    terminal action outcome that the trace does not establish.
    """

    turns = info.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError("final-action evidence is missing the recorded last turn")
    last = turns[-1]
    if not isinstance(last, Mapping):
        raise RuntimeError("final-action evidence has a malformed last turn")
    valid = last.get("valid")
    if type(valid) is not bool:
        raise RuntimeError("final-action evidence has a malformed last-turn valid flag")
    return {
        "enabled": True,
        "status": "valid" if valid else "invalid",
        "turn": last.get("turn"),
        "action": last.get("action"),
        "valid": valid,
        "failure_kind": last.get("failure_kind"),
    }
