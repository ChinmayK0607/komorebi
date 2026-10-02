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
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

DEFAULT_RENDER_TIMEOUT_SECONDS = 180
MAX_RENDER_TIMEOUT_SECONDS = 180
RENDER_CANCEL_GRACE_SECONDS = 5.0
RENDER_WAIT_POLL_SECONDS = 0.1
JS_SYNTAX_PREFLIGHT_ENV = "PAINTER_JS_SYNTAX_PREFLIGHT"
JS_SYNTAX_PREFLIGHT_TIMEOUT_SECONDS = 5
MIN_CANVAS_CHANNEL_VARIANCE = 1.0


@dataclass(frozen=True)
class _OwnedProcessGroup:
    pgid: int
    sid: int


class RenderCancelled(RuntimeError):
    """The caller cancelled a render before its result could be consumed."""


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _cancel_requested(cancel_event):
        raise RenderCancelled("renderer cancelled")


def _record_owned_process_group(process: subprocess.Popen[Any]) -> _OwnedProcessGroup | None:
    """Record the fresh session identity immediately after Popen."""

    try:
        pgid = os.getpgid(process.pid)
        sid = os.getsid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    if pgid != process.pid or sid != process.pid:
        return None
    return _OwnedProcessGroup(pgid=pgid, sid=sid)


def _session_has_member(identity: _OwnedProcessGroup) -> bool:
    """Prove that the recorded session still has a live member.

    The process-group id cannot be recycled while a member remains.  When the
    leader has exited, verify both the recorded process group and session id
    before using killpg; this avoids signalling a later unrelated PID/group.
    """

    try:
        os.killpg(identity.pgid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                _, fields = stat.split(") ", 1)
                values = fields.split()
                # After comm: state=0, ppid=1, pgrp=2, session=3.  A
                # zombie has no running descendant to clean up; treating it
                # as live would unnecessarily hold the semaphore through the
                # escalation grace period in PID-reaping containers.
                if values[0] in {"Z", "X"}:
                    continue
                if int(values[2]) == identity.pgid and int(values[3]) == identity.sid:
                    return True
            except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
                continue
        return False
    try:
        listed = subprocess.run(
            ["ps", "-axo", "pid=,state=,pgid=,sid="],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Some sandboxed macOS runners deny process-table inspection.  The
        # recorded fresh session plus killpg(..., 0) above is the available
        # ownership proof there; a missing group would already have returned
        # ESRCH and been rejected.
        return True
    for line in listed.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 4:
            try:
                if columns[1] in {"Z", "X"}:
                    continue
                if int(columns[2]) == identity.pgid and int(columns[3]) == identity.sid:
                    return True
            except ValueError:
                continue
    return False


def _owned_process_group(
    process: subprocess.Popen[Any],
    identity: _OwnedProcessGroup | None,
) -> int | None:
    """Return the owned group while its session identity remains provable.

    The native caller starts the renderer in a fresh session.  Checking that
    the live child remains the session leader before signalling avoids sending
    a group signal to an unrelated process after a PID reuse or wrapper exit.
    """

    if identity is None:
        return None
    if process.poll() is None:
        try:
            if os.getpgid(process.pid) == identity.pgid and os.getsid(process.pid) == identity.sid:
                return identity.pgid
        except (ProcessLookupError, PermissionError, OSError):
            return None
        return None
    return identity.pgid if _session_has_member(identity) else None


def _signal_owned_process(
    process: subprocess.Popen[Any],
    signum: int,
    identity: _OwnedProcessGroup | None,
) -> bool:
    group = _owned_process_group(process, identity)
    if group is not None:
        try:
            os.killpg(group, signum)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False
    if process.poll() is not None:
        return False
    # A launcher that is not the session leader is still an owned child.  Send
    # only to that exact Popen pid; never guess at a broader process group.
    try:
        process.send_signal(signum)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _wait_for_group_exit(identity: _OwnedProcessGroup, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _session_has_member(identity):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(RENDER_WAIT_POLL_SECONDS, remaining))
    return True


def _close_process_pipes(process: subprocess.Popen[Any]) -> None:
    # Cancelled and timed-out output is intentionally discarded.  Close the
    # parent-side wrappers directly; communicate(timeout=0) cannot close them
    # when an inherited descendant still holds the other pipe end.
    for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def _terminate_owned_process(
    process: subprocess.Popen[Any],
    *,
    identity: _OwnedProcessGroup | None,
    grace_seconds: float = RENDER_CANCEL_GRACE_SECONDS,
) -> None:
    """Terminate the renderer and wait for its complete owned session."""

    if process.poll() is not None and not (identity and _session_has_member(identity)):
        _close_process_pipes(process)
        return
    _signal_owned_process(process, signal.SIGTERM, identity)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    if identity and _wait_for_group_exit(identity, grace_seconds):
        _close_process_pipes(process)
        return
    if process.poll() is None:
        _signal_owned_process(process, signal.SIGKILL, identity)
    elif identity:
        _signal_owned_process(process, signal.SIGKILL, identity)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # Popen.kill targets this exact child.  The final wait is deliberately
        # a blocking reap only after SIGKILL; if the OS cannot reap, this
        # worker remains blocked and its semaphore slot is retained rather than
        # reporting normal completion while an owned process is still alive.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()
    _close_process_pipes(process)


def _run_renderer_process(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    cancel_event: threading.Event | None,
) -> subprocess.CompletedProcess[str]:
    """Run the renderer with bounded escalation and fail-closed final reap.

    The action deadline and SIGTERM/SIGKILL escalation are bounded.  After
    SIGKILL, the exact owned child is reaped before returning; an OS-level
    uninterruptible reap can therefore keep this worker (and its semaphore
    slot) blocked instead of claiming normal completion with a live process.
    """

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    identity = _record_owned_process_group(process)
    started = time.monotonic()
    try:
        while True:
            _raise_if_cancelled(cancel_event)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(
                    timeout=min(RENDER_WAIT_POLL_SECONDS, remaining)
                )
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                continue
    except RenderCancelled:
        _terminate_owned_process(process, identity=identity)
        raise
    except subprocess.TimeoutExpired:
        _terminate_owned_process(process, identity=identity)
        if _cancel_requested(cancel_event):
            raise RenderCancelled("renderer cancelled")
        raise
    except BaseException:
        # Includes unexpected thread cancellation/interrupts.  The slot owner
        # must never be released while this process is still alive.
        _terminate_owned_process(process, identity=identity)
        raise


def extract_sketch(reply: str) -> str:
    blocks = re.findall(
        r"```[ \t]*(?:javascript|js)?[ \t]*\r?\n(.*?)```", reply, re.DOTALL | re.IGNORECASE
    )
    if blocks:
        return blocks[-1].strip() + "\n"
    # The model sometimes emits a complete sketch without fences. Accept only
    # an unambiguously code-only reply; prose plans still receive feedback.
    candidate = reply.strip()
    if ("```" not in candidate
            and candidate.startswith(("function ", "const ", "let ", "var ", "//", "/*"))
            and re.search(r"\bfunction\s+setup\s*\(", candidate)
            and re.search(r"\bfunction\s+draw\s*\(", candidate)
            and not re.search(r"<\s*/?\s*(?:script|html)\b", candidate, re.IGNORECASE)):
        return candidate + "\n"
    raise ValueError("No complete JavaScript code block")


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


def _run_js_syntax_preflight(source: Path) -> str | None:
    """Check JavaScript Script grammar without executing the submitted code.

    ``node --check`` parses a file and exits before evaluating it.  Keep the
    child environment deliberately small: syntax checking needs only PATH and
    locale, and must not inherit model/API credentials from the renderer
    process.  Startup and timeout failures are operational failures and are
    raised for the native loop's service-failure path.  A non-zero parser exit
    is a model-program failure and returns the parser's diagnostic.
    """

    command = ["node", "--check", str(source)]
    child_env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C"),
        "LC_ALL": os.environ.get("LC_ALL", "C"),
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=JS_SYNTAX_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
            env=child_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "JavaScript syntax preflight is enabled but Node.js (`node`) is unavailable"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "JavaScript syntax preflight timed out after "
            f"{JS_SYNTAX_PREFLIGHT_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"JavaScript syntax preflight could not start Node.js: {exc}") from exc

    if completed.returncode == 0:
        return None
    detail = (getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "").strip()
    return detail or f"node --check exited with status {completed.returncode}"


def render_action(
    *,
    root: Path,
    folder: Path,
    reply: str,
    initial_canvas: Path | None,
    initial_program: str | None,
    renderer: Path | None = None,
    renderer_python: Path | None = None,
    cancel_event: threading.Event | None = None,
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
    _raise_if_cancelled(cancel_event)

    source = folder / "program.js"
    source.write_text(program)
    os.chmod(source, 0o644)
    row["program"] = program
    if os.environ.get(JS_SYNTAX_PREFLIGHT_ENV) == "1":
        syntax_error = _run_js_syntax_preflight(source)
        _raise_if_cancelled(cancel_event)
        if syntax_error is not None:
            row.update(
                error=syntax_error,
                renderer_errors=[syntax_error],
                failure_kind="program",
            )
            return row
    if initial_program is not None and program.strip() == initial_program.strip():
        _raise_if_cancelled(cancel_event)
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
    completed = _run_renderer_process(
        command,
        env=child_env,
        timeout=deadline_seconds + 20,
        cancel_event=cancel_event,
    )
    # A process can finish just as cancellation arrives.  Do not consume a
    # receipt produced by that stale action as the next model state.
    _raise_if_cancelled(cancel_event)
    receipt_path = output.with_suffix(".json")
    if not receipt_path.is_file():
        raise RuntimeError("renderer produced no receipt: " + completed.stderr[-1000:])
    receipt = json.loads(receipt_path.read_text())
    _raise_if_cancelled(cancel_event)
    if not receipt.get("valid"):
        error = str(receipt.get("errors") or receipt.get("error") or completed.stderr)
        error_code = receipt.get("error_code")
        if error_code == "render_timeout" and receipt.get("timed_out"):
            # The browser worker launched and exhausted the action deadline.
            # A non-terminating or prohibitively expensive model program is an
            # invalid action; preserve the prior canvas and let the next turn
            # repair it. Other runtime errors remain censored service failures.
            row.update(
                error=f"Your painting program exceeded the {deadline_seconds}s render limit. Simplify it and call noLoop() in draw().",
                renderer_errors=["Painting program exceeded the render limit"],
                failure_kind="program_timeout",
            )
            return row
        # The vendor renderer uses an error_code for worker/runtime failures
        # (missing Playwright, browser startup, or child failure).
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
    with Image.open(output) as image:
        channel_variance = max(ImageStat.Stat(image.convert("RGB")).var)
    if channel_variance < MIN_CANVAS_CHANNEL_VARIANCE:
        row.update(
            error="Rendered canvas is blank or nearly uniform; paint visible reference subjects.",
            renderer_errors=["Rendered canvas is blank or nearly uniform"],
            failure_kind="blank_canvas",
            canvas_variance=channel_variance,
        )
        return row
    row.update(
        valid=True,
        canvas=str(output),
        no_change=initial_canvas is not None and output.read_bytes() == initial_canvas.read_bytes(),
        render_seconds=receipt.get("painting_seconds"),
        canvas_sha256=_sha(output),
    )
    return row


__all__ = [
    "DEFAULT_RENDER_TIMEOUT_SECONDS",
    "JS_SYNTAX_PREFLIGHT_ENV",
    "JS_SYNTAX_PREFLIGHT_TIMEOUT_SECONDS",
    "extract_sketch",
    "render_action",
    "render_timeout_seconds",
]
