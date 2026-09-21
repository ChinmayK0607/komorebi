"""Render one native multi-turn painter action on the Linux node.

This helper keeps program errors in-band for the next model turn.  Renderer
process failures raise, which lets the task mark the trace as a service failure
and exclude it from learning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def _sketch(reply: str) -> str:
    blocks = re.findall(
        r"```[ \t]*(?:javascript|js)?[ \t]*\r?\n(.*?)```", reply, re.DOTALL
    )
    if not blocks:
        raise ValueError("No complete JavaScript code block")
    return blocks[-1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_action(
    *,
    root: Path,
    folder: Path,
    reply: str,
    initial_canvas: Path,
    initial_program: str,
) -> dict:
    """Render a model reply and return a serializable observation record."""
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "valid": False,
        "action": "paint",
        "canvas": str(initial_canvas),
        "no_change": False,
    }
    try:
        program = _sketch(reply)
    except ValueError as exc:
        row["error"] = str(exc)
        row["failure_kind"] = "program"
        return row

    source = folder / "program.js"
    source.write_text(program)
    os.chmod(source, 0o644)
    os.chmod(folder, 0o777)
    row["program"] = program
    if program.strip() == initial_program.strip():
        row.update(valid=True, canvas=str(initial_canvas), no_change=True)
        return row

    renderer = root / "vendor/integrations/watercolour/renderer.py"
    renderer_python = root / "renderer-env/bin/python"
    if not renderer.is_file() or not renderer_python.is_file():
        raise RuntimeError(f"renderer runtime is missing under {root}")
    output = folder / "painting.png"
    command = [
        "/usr/sbin/runuser",
        "-u",
        "painter",
        "--",
        str(renderer_python),
        str(renderer),
        str(source),
        "--output",
        str(output),
        "--timeout",
        "120",
        "--backend",
        "swiftshader",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=140,
        check=False,
    )
    receipt_path = output.with_suffix(".json")
    if not receipt_path.is_file():
        raise RuntimeError(
            "renderer produced no receipt"
            + f": {completed.stderr[-1000:]}"
        )
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("valid"):
        row["error"] = str(receipt.get("errors") or receipt.get("error") or completed.stderr)
        row["renderer_errors"] = receipt.get("errors") or receipt.get("error")
        row["failure_kind"] = "program"
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
        no_change=output.read_bytes() == initial_canvas.read_bytes(),
        render_seconds=receipt.get("painting_seconds"),
    )
    return row


__all__ = ["render_action"]
