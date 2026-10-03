#!/usr/bin/env python3
"""Opt-in cancellable derivative of the frozen Watercolour renderer.

The vendored renderer remains byte frozen for historical runs.  This wrapper
reuses its worker, assets, and CLI but owns the worker's fresh session so a
cancelled native render can clean up a nested browser process safely.
"""

from __future__ import annotations

import argparse
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
    anchor_pids: tuple[int, ...] = ()


_BATCH_REGISTRY_LOCK = threading.RLock()
_BATCH_REGISTRY: dict[int, tuple[subprocess.Popen, _OwnedProcessGroup | None]] = {}
_BATCH_MODE = False
_BATCH_CANCELLED = threading.Event()
_BATCH_CLEANUP_STARTED = False


def _record_owned_process_group(process: subprocess.Popen) -> _OwnedProcessGroup | None:
    try:
        pgid = os.getpgid(process.pid)
        sid = os.getsid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    if pgid != process.pid or sid != process.pid:
        return None
    return _OwnedProcessGroup(pgid=pgid, sid=sid, anchor_pids=(process.pid,))


def _proc_snapshot() -> dict[int, tuple[int, int, int, str]]:
    """Return pid -> (ppid, process-group, session, state) for live /proc rows."""
    root = Path("/proc")
    if not root.is_dir():
        return {}
    snapshot: dict[int, tuple[int, int, int, str]] = {}
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            value = (entry / "stat").read_text()
            prefix, fields = value.rsplit(") ", 1)
            pid = int(prefix.split("(", 1)[0])
            columns = fields.split()
            # After comm: state=0, ppid=1, pgrp=2, session=3.
            state = columns[0]
            if state in {"Z", "X"}:
                continue
            snapshot[pid] = (int(columns[1]), int(columns[2]), int(columns[3]), state)
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            continue
    return snapshot


def _discover_owned_process_groups(
    process: subprocess.Popen,
    identity: _OwnedProcessGroup | None,
) -> tuple[_OwnedProcessGroup, ...]:
    """Capture descendant sessions before cleanup, including fresh browser sessions."""
    if identity is None:
        return ()
    snapshot = _proc_snapshot()
    descendants = {process.pid}
    pending = [process.pid]
    while pending:
        parent = pending.pop()
        for pid, (ppid, _pgid, _sid, _state) in snapshot.items():
            if ppid == parent and pid not in descendants:
                descendants.add(pid)
                pending.append(pid)
    groups: dict[tuple[int, int], set[int]] = {(identity.pgid, identity.sid): {process.pid}}
    for pid in descendants:
        row = snapshot.get(pid)
        if row is None:
            continue
        _ppid, pgid, sid, _state = row
        groups.setdefault((pgid, sid), set()).add(pid)
    ordered = sorted(groups.items(), key=lambda item: item[0] == (identity.pgid, identity.sid))
    return tuple(
        _OwnedProcessGroup(pgid=pgid, sid=sid, anchor_pids=tuple(sorted(anchors)))
        for (pgid, sid), anchors in ordered
    )


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
    if group is not None and identity is not None:
        return _signal_owned_group(identity, signum)
    if process.poll() is not None:
        return False
    try:
        process.send_signal(signum)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _signal_owned_group(identity: _OwnedProcessGroup, signum: int) -> bool:
    snapshot = _proc_snapshot()
    if snapshot:
        owned_anchor = any(
            pid in snapshot and snapshot[pid][1:3] == (identity.pgid, identity.sid)
            for pid in identity.anchor_pids
        )
        if not owned_anchor:
            return False
    elif not _session_has_member(identity):
        return False
    try:
        os.killpg(identity.pgid, signum)
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

    _terminate_owned_processes(((process, identity),), grace=grace)


def _terminate_owned_processes(
    processes: tuple[tuple[subprocess.Popen, _OwnedProcessGroup | None], ...],
    *,
    grace: float = WORKER_CLEANUP_GRACE_SECONDS,
) -> None:
    """Signal all observed owned sessions under one shared bounded grace."""
    groups_by_key: dict[tuple[int, int], set[int]] = {}
    for process, identity in processes:
        for group in _discover_owned_process_groups(process, identity):
            groups_by_key.setdefault((group.pgid, group.sid), set()).update(group.anchor_pids)
    groups = tuple(
        _OwnedProcessGroup(pgid, sid, tuple(sorted(anchors)))
        for (pgid, sid), anchors in groups_by_key.items()
    )
    if not groups and all(process.poll() is not None for process, _identity in processes):
        return
    deadline = time.monotonic() + grace
    for group in groups:
        _signal_owned_group(group, signal.SIGTERM)
    for process, _identity in processes:
        if not groups and process.poll() is None:
            process.send_signal(signal.SIGTERM)
    while True:
        process_done = all(process.poll() is not None for process, _identity in processes)
        groups_done = all(not _session_has_member(group) for group in groups)
        if process_done and groups_done:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    for group in groups:
        if _session_has_member(group):
            _signal_owned_group(group, signal.SIGKILL)
    for process, _identity in processes:
        if process.poll() is None and not groups:
            process.kill()
    kill_deadline = time.monotonic() + max(0.1, grace * 0.25)
    while True:
        process_done = all(process.poll() is not None for process, _identity in processes)
        groups_done = all(not _session_has_member(group) for group in groups)
        if process_done and groups_done:
            return
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    for process, _identity in processes:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait()
    if any(_session_has_member(group) for group in groups):
        raise RuntimeError("owned renderer process group survived bounded cleanup")


def _register_batch_process(process: subprocess.Popen, identity: _OwnedProcessGroup | None) -> bool:
    with _BATCH_REGISTRY_LOCK:
        if not _BATCH_MODE:
            return True
        if _BATCH_CANCELLED.is_set():
            return False
        _BATCH_REGISTRY[process.pid] = (process, identity)
        return True


def _unregister_batch_process(process: subprocess.Popen) -> None:
    with _BATCH_REGISTRY_LOCK:
        _BATCH_REGISTRY.pop(process.pid, None)


def _cleanup_batch_on_sigterm(signum: int, _frame: object) -> None:
    global _BATCH_CLEANUP_STARTED
    _BATCH_CANCELLED.set()
    with _BATCH_REGISTRY_LOCK:
        if _BATCH_CLEANUP_STARTED:
            return
        _BATCH_CLEANUP_STARTED = True
        active = tuple(_BATCH_REGISTRY.values())
    try:
        _terminate_owned_processes(active, grace=WORKER_CLEANUP_GRACE_SECONDS)
    except BaseException:
        # The main process still exits with the signal status.  Each worker's
        # finally block retries bounded cleanup and reaps it.
        pass
    raise SystemExit(128 + signum)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_BASE.__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--telemetry", action="store_true", help="Opt in to runtime diagnostics; never enables an efficiency reward")
    parser.add_argument("--backend", choices=["swiftshader", "metal"], help="Explicit ANGLE backend; defaults to BRUSH_RENDER_BACKEND or swiftshader")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _run_single_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.workers <= 16 or not 1 <= args.timeout <= 180:
        parser.error("Use 1..16 workers and a 1..180 second deadline")
    if bool(args.output) == bool(args.output_dir) or (args.output and len(args.sources) != 1):
        parser.error("Use --output for one source or --output-dir for a batch")
    result = render(args.sources[0], args.output, args.seed, args.timeout, args.telemetry, args.backend)
    print(json.dumps([result], indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def _run_batch_cli() -> None:
    global _BATCH_CLEANUP_STARTED, _BATCH_MODE
    _BATCH_CANCELLED.clear()
    with _BATCH_REGISTRY_LOCK:
        _BATCH_MODE = True
    previous_sigterm = signal.signal(signal.SIGTERM, _cleanup_batch_on_sigterm)
    try:
        _BASE.render = render
        _BASE.main()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        with _BATCH_REGISTRY_LOCK:
            _BATCH_MODE = False
            _BATCH_CLEANUP_STARTED = False
            _BATCH_REGISTRY.clear()
        _BATCH_CANCELLED.clear()


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
                registered = _register_batch_process(process, identity)
                try:
                    if not registered:
                        _terminate_owned_process(process, identity)
                        result.update(error="Batch cancellation requested", error_code="render_cancelled")
                        code = None
                    else:
                        code = process.wait(timeout=timeout)
                    metadata = scratch / "image.json"
                    if not registered:
                        pass
                    elif code == 0 and metadata.exists():
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
                    try:
                        _terminate_owned_process(process, identity)
                    finally:
                        _unregister_batch_process(process)
                if install_handler:
                    signal.signal(signal.SIGTERM, previous_sigterm)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    _BASE.render = render
    argv = sys.argv[1:]
    if "--worker" in argv or "--help" in argv or "-h" in argv:
        _BASE.main()
        return
    parser = _cli_parser()
    args = parser.parse_args(argv)
    if args.output is not None and args.output_dir is None and len(args.sources) == 1:
        _run_single_cli(args, parser)
        return
    _run_batch_cli()


if __name__ == "__main__":
    main()
