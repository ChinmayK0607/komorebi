"""Opt-in anchor renderer guard for the native p5.brush contract.

The native prompt requires a finite ``draw`` loop.  This patch is installed
only in the frozen-anchor evaluator through ``PAINTER_ANCHOR_REQUIRE_NO_LOOP``;
it runs after JavaScript syntax validation and before Chromium starts.  A
``FINISHED`` response never reaches ``render_action`` and remains valid.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


NO_LOOP_PATTERN = re.compile(r"\bnoLoop\s*\(")


def no_loop_error(source: Path) -> str | None:
    """Return a model-program error when a submitted sketch can keep drawing."""
    text = source.read_text(encoding="utf-8")
    if NO_LOOP_PATTERN.search(text) is None:
        return "painting program must call noLoop() in draw()"
    return None


def install() -> dict[str, Any]:
    """Install the guard on the native renderer exactly once."""
    from painter import native_painting_renderer as renderer

    if getattr(renderer, "_structured_anchor_no_loop_guard", False):
        return {"installed": True, "already": True}
    original = renderer._run_js_syntax_preflight

    def guarded(source: Path) -> str | None:
        syntax_error = original(source)
        if syntax_error is not None:
            return syntax_error
        return no_loop_error(source)

    renderer._run_js_syntax_preflight = guarded
    renderer._structured_anchor_no_loop_guard = True
    return {"installed": True, "already": False}
