"""Small node-side renderer adapter for the shared native painting loop.

The adapter has no model or scoring code.  It extracts a complete JavaScript
sketch, invokes the pinned Watercolour renderer, and returns a serialisable
observation for the next turn.  Tests replace ``subprocess.run``; rendering is
never performed by the local test suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_RENDER_TIMEOUT_SECONDS = 180
MAX_RENDER_TIMEOUT_SECONDS = 180


def extract_sketch(reply: str) -> str:
    blocks = re.findall(
        r"```[ \t]*(?:javascript|js)?[ \t]*\r?\n(.*?)```", reply, re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        raise ValueError("No complete JavaScript code block")
    return blocks[-1].strip() + "\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_timeout_seconds() -> int:
    raw = os.environ.get("PAINTER_RENDER_TIMEOUT_SECONDS", str(DEFAULT_RENDER_TIMEOUT_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("PAINTER_RENDER_TIMEOUT_SECONDS must be an integer from 1 to 180") from exc
    if not 1 <= value <= MAX_RENDER_TIMEOUT_SECONDS:
        raise ValueError("PAINTER_RENDER_TIMEOUT_SECONDS must be an integer from 1 to 180")
    return value


def render_action(
    *,
    root: Path,
    folder: Path,
    reply: str,
    initial_canvas: Path | None,
    initial_program: str | None,
    renderer: Path | None = None,
    renderer_python: Path | None = None,
) -> dict[str, Any]:
    """Render one model action, preserving the previous canvas on bad code."""
    folder.mkdir(parents=True, exist_ok=True)
    # The pinned renderer runs as the unprivileged ``painter`` user while the
    # native environment process may be root inside the Prime box.
    os.chmod(folder, 0o777)
    row: dict[str, Any] = {
        "valid": False,
        "action": "paint",
        "canvas": str(initial_canvas) if initial_canvas is not None else None,
        "no_change": False,
    }
    try:
        program = extract_sketch(reply)
    except ValueError as exc:
        row.update(error=str(exc), failure_kind="program")
        return row

    source = folder / "program.js"
    source.write_text(program)
    os.chmod(source, 0o644)
    row["program"] = program
    if initial_program is not None and program.strip() == initial_program.strip():
        row.update(valid=True, canvas=str(initial_canvas), no_change=True)
        return row

    renderer = renderer or root / "vendor/integrations/watercolour/renderer.py"
    renderer_python = renderer_python or root / "renderer-env/bin/python"
    if not renderer.is_file() or not renderer_python.is_file():
        raise RuntimeError(f"renderer runtime is missing under {root}")
    deadline_seconds = render_timeout_seconds()
    row["render_timeout_seconds"] = deadline_seconds
    row["render_process_timeout_seconds"] = deadline_seconds + 20
    output = folder / "painting.png"
    command = [
        "/usr/sbin/runuser", "-u", "painter", "--", str(renderer_python),
        str(renderer), str(source), "--output", str(output), "--timeout", str(deadline_seconds),
        "--backend", "swiftshader",
    ]
    child_env = os.environ.copy()
    # The node bootstrap stores Playwright in the run workspace.  Explicitly
    # pass that location so eval workers launched by vf/Prime do not fall back
    # to a per-user cache under /tmp.
    child_env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(root / "browsers"))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=deadline_seconds + 20, check=False, env=child_env)
    receipt_path = output.with_suffix(".json")
    if not receipt_path.is_file():
        raise RuntimeError("renderer produced no receipt: " + completed.stderr[-1000:])
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("valid"):
        error = str(receipt.get("errors") or receipt.get("error") or completed.stderr)
        error_code = receipt.get("error_code")
        # The vendor renderer uses an error_code for worker/runtime failures
        # (missing Playwright, browser startup, deadline, or child failure).
        # Those are infrastructure outages and must reach the native loop's
        # service-failure path; only a receipt with page-level errors and no
        # runtime error code is a model-program failure.
        if error_code or receipt.get("timed_out"):
            raise RuntimeError(
                f"renderer service failure ({error_code or 'timeout'}): {error}"
            )
        row.update(
            error=error,
            renderer_errors=receipt.get("errors") or receipt.get("error"),
            failure_kind="program",
        )
        return row
    if completed.returncode != 0:
        raise RuntimeError(f"renderer exited {completed.returncode}: {completed.stderr[-1000:]}")
    if not output.is_file():
        raise RuntimeError("renderer receipt is valid but output canvas is missing")
    if receipt.get("source_sha256") != _sha(source):
        raise RuntimeError("renderer source hash does not match submitted program")
    if receipt.get("png_sha256") != _sha(output):
        raise RuntimeError("renderer canvas hash does not match output")
    row.update(
        valid=True,
        canvas=str(output),
        no_change=initial_canvas is not None and output.read_bytes() == initial_canvas.read_bytes(),
        render_seconds=receipt.get("painting_seconds"),
        canvas_sha256=_sha(output),
    )
    return row


__all__ = ["DEFAULT_RENDER_TIMEOUT_SECONDS", "extract_sketch", "render_action", "render_timeout_seconds"]
