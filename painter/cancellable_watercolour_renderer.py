#!/usr/bin/env python3
"""Opt-in cancellable derivative of the frozen Watercolour renderer.

The vendored renderer remains byte frozen for historical runs.  This wrapper
reuses its worker, assets, and CLI but owns the worker's fresh session so a
cancelled native render can clean up a nested browser process safely.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


_DERIVATIVE_DIR = Path(__file__).resolve().parent
_ADJACENT_BASE_PATH = _DERIVATIVE_DIR / "frozen_watercolour_renderer.py"
BASE_PATH = (
    _ADJACENT_BASE_PATH
    if _ADJACENT_BASE_PATH.is_file()
    else _DERIVATIVE_DIR / "vendor/integrations/watercolour/renderer.py"
)
_SPEC = importlib.util.spec_from_file_location("frozen_watercolour_renderer", BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen renderer: {BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

RENDERER_RUNTIME_ID = "watercolour-cancellable-v1"
WORKER_CLEANUP_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class _OwnedProcessGroup:
    pgid: int
    sid: int


def _record_owned_process_group(process: subprocess.Popen) -> _OwnedProcessGroup | None:
    try:
        pgid = os.getpgid(process.pid)
        sid = os.getsid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    if pgid != process.pid or sid != process.pid:
        return None
    return _OwnedProcessGroup(pgid=pgid, sid=sid)


def _session_has_member(identity: _OwnedProcessGroup) -> bool:
    """Prove the recorded session remains alive after its leader exits."""

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
                # After comm: state=0, ppid=1, pgrp=2, session=3.  Zombies
                # are already dead and must not keep cleanup in escalation.
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
        # Sandboxed macOS runners may deny process-table inspection.  The
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
    process: subprocess.Popen,
    identity: _OwnedProcessGroup | None,
) -> int | None:
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
    process: subprocess.Popen,
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
        time.sleep(min(0.1, remaining))
    return True


def _terminate_owned_process(
    process: subprocess.Popen,
    identity: _OwnedProcessGroup | None,
    grace: float = WORKER_CLEANUP_GRACE_SECONDS,
) -> None:
    """Escalate cleanup while retaining ownership proof and slot lifetime."""

    if process.poll() is not None and not (identity and _session_has_member(identity)):
        return
    _signal_owned_process(process, signal.SIGTERM, identity)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    if identity and _wait_for_group_exit(identity, grace):
        return
    _signal_owned_process(process, signal.SIGKILL, identity)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        # A final blocking reap is intentional: if the OS cannot reap an
        # owned child after SIGKILL, this worker stays blocked and its native
        # semaphore slot is retained instead of reporting normal completion.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def render(source, output, seed=0, timeout=90, telemetry=False, backend=None):
    """Run the frozen worker with cancellation-safe owned-session cleanup."""

    backend = _BASE.resolve_backend(backend)
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.stat().st_size > 100_000:
        raise ValueError("Sketch exceeds 100KB bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite existing render/validation")
    started = time.monotonic()
    result = {
        "source": str(source),
        "source_sha256": _BASE.sha(source),
        "seed": seed,
        "canvas_contract": [600, 600],
        "valid": False,
        "assets": {p.name: _BASE.sha(p) for p in _BASE.ASSETS.glob("*.js")},
        "renderer_sha256": _BASE.sha(Path(__file__).resolve()),
        "renderer_runtime_id": RENDERER_RUNTIME_ID,
        "telemetry_enabled": telemetry,
        "requested_angle_backend": backend,
        "telemetry_sha256": _BASE.sha(_BASE.HERE / "telemetry.py") if telemetry else None,
        "network_enabled": False,
        "browser_sandbox_enabled": True,
    }
    with tempfile.TemporaryDirectory(prefix="painting-render-") as folder:
        scratch = Path(folder)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": folder,
            "TMPDIR": folder,
            "LANG": "en_US.UTF-8",
        }
        bundled = _BASE.PROJECT / "work/playwright-browsers"
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]).resolve())
        elif bundled.exists():
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(bundled)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(source),
            "--output",
            str(scratch / "image.png"),
            "--seed",
            str(seed),
            "--timeout",
            str(timeout),
            "--backend",
            backend,
        ]
        if telemetry:
            command.append("--telemetry")
        with (scratch / "worker.log").open("wb") as log:
            process = None
            identity = None
            cleanup_started = False
            previous_sigterm = None

            def cleanup_on_sigterm(signum, _frame):
                nonlocal cleanup_started
                if cleanup_started:
                    # A second signal can arrive through a relay while the
                    # first cleanup is waiting for the owned worker tree.
                    # Let the first cleanup finish rather than interrupting
                    # it and leaking descendants.
                    return
                cleanup_started = True
                if process is not None:
                    _terminate_owned_process(process, identity)
                raise SystemExit(128 + signum)

            install_handler = threading.current_thread() is threading.main_thread()
            if install_handler:
                previous_sigterm = signal.signal(signal.SIGTERM, cleanup_on_sigterm)
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                identity = _record_owned_process_group(process)
                try:
                    code = process.wait(timeout=timeout)
                    metadata = scratch / "image.json"
                    if code == 0 and metadata.exists():
                        result.update(json.loads(metadata.read_text()))
                        if (scratch / "image.png").exists():
                            output.write_bytes((scratch / "image.png").read_bytes())
                            result["png_sha256"] = _BASE.sha(output)
                    else:
                        result["error"] = (scratch / "worker.log").read_text(errors="replace")[-4000:]
                        result["error_code"] = "worker_failure"
                except subprocess.TimeoutExpired:
                    result["error"] = f"Hard render deadline exceeded ({timeout}s)"
                    result.update(timed_out=True, error_code="render_timeout")
            finally:
                if process is not None:
                    _terminate_owned_process(process, identity)
                if install_handler:
                    signal.signal(signal.SIGTERM, previous_sigterm)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    _BASE.render = render
    _BASE.main()


if __name__ == "__main__":
    main()
