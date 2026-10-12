#!/usr/bin/env python3
"""Foreground supervisor for the staged Prime launch.

The Prime entrypoint owns torchrun children, so a shell ``timeout`` only sees
the orchestrator's eventual exit.  This supervisor watches the run-owned
failure and rank-heartbeat receipts and terminates the complete process group
as soon as one trainer rank reports an exception/OOM or stops heartbeating.
It never schedules work and always returns a non-zero status for a failed
distributed launch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


FAILURE_EXIT = 70
HEARTBEAT_STALE_SECONDS = 20.0
STARTUP_GRACE_SECONDS = 120.0
# Loading a TP4 inference server and prewarming both fixed adapters happens
# before Prime returns to its launcher and creates the trainer child. This is
# a bounded pre-trainer phase; once the trainer process receipt exists, the
# shorter rank readiness grace applies again.
PRETRAINER_STARTUP_GRACE_SECONDS = 600.0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _terminate(process: subprocess.Popen[bytes], *, grace: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _failure_reason(path: Path) -> str | None:
    receipt = _read_json(path)
    if receipt is None:
        return "malformed distributed failure receipt"
    if receipt.get("status") not in {"rank_failed", "rank_oom", "rank_exited"}:
        return "invalid distributed failure receipt status"
    reason = receipt.get("reason")
    return str(reason) if reason else str(receipt.get("status"))


def _stale_rank(
    rank_dir: Path,
    *,
    expected_world_size: int,
    started_at: float,
    phase_file: Path | None = None,
) -> str | None:
    if expected_world_size < 2 or not rank_dir.is_dir():
        return None
    started = sorted(rank_dir.glob("rank-*.started.json"))
    # Do not call a rank missing during torchrun startup dead until the
    # process has had a bounded opportunity to initialize.
    phase = _read_json(phase_file) if phase_file is not None else None
    phase_name = phase.get("phase") if isinstance(phase, dict) else None
    phase_recorded = phase.get("recorded_at_unix") if isinstance(phase, dict) else None
    try:
        phase_age = max(0.0, time.time() - float(phase_recorded))
    except (TypeError, ValueError):
        phase_age = time.monotonic() - started_at
    if phase_name == "trainer_process_started":
        grace = STARTUP_GRACE_SECONDS
        elapsed = phase_age
    elif phase_name:
        grace = PRETRAINER_STARTUP_GRACE_SECONDS
        elapsed = phase_age
    else:
        grace = STARTUP_GRACE_SECONDS
        elapsed = time.monotonic() - started_at
    if len(started) < expected_world_size and elapsed < grace:
        return None
    if len(started) < expected_world_size:
        phase_detail = f" during {phase_name}" if phase_name else ""
        return f"only {len(started)} of {expected_world_size} trainer ranks created startup markers after grace{phase_detail}"
    now = time.time()
    for marker in started:
        rank = marker.name.removeprefix("rank-").removesuffix(".started.json")
        heartbeat = rank_dir / f"rank-{rank}.heartbeat"
        try:
            age = now - heartbeat.stat().st_mtime
        except OSError:
            return f"trainer rank {rank} has no heartbeat"
        if age > HEARTBEAT_STALE_SECONDS:
            return f"trainer rank {rank} heartbeat stale for {age:.1f}s"
    return None


def supervise(
    command: Sequence[str],
    *,
    failure_file: Path,
    rank_dir: Path,
    expected_world_size: int,
    timeout_seconds: int,
    phase_file: Path | None = None,
    poll_seconds: float = 1.0,
    popen: Any = subprocess.Popen,
) -> int:
    """Run one foreground command and fail the full group on rank failure."""

    failure_file = failure_file.expanduser().resolve()
    rank_dir = rank_dir.expanduser().resolve()
    phase_file = phase_file.expanduser().resolve() if phase_file is not None else None
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)
    if phase_file is not None:
        phase_file.parent.mkdir(parents=True, exist_ok=True)
    _reject_stale_state(failure_file, rank_dir, phase_file)
    started_at = time.monotonic()
    process = popen(list(command), start_new_session=True)
    deadline = started_at + timeout_seconds
    try:
        while True:
            status = process.poll()
            if status is not None:
                return int(status)
            if failure_file.exists():
                reason = _failure_reason(failure_file) or "distributed rank failure"
                _write_supervisor_receipt(failure_file, reason=reason, status="supervisor_terminated")
                _terminate(process)
                return FAILURE_EXIT
            stale = _stale_rank(
                rank_dir,
                expected_world_size=expected_world_size,
                started_at=started_at,
                phase_file=phase_file,
            )
            if stale:
                _write_supervisor_receipt(failure_file, reason=stale, status="rank_exited")
                _terminate(process)
                return FAILURE_EXIT
            if time.monotonic() >= deadline:
                _write_supervisor_receipt(failure_file, reason="bounded launch timeout", status="launch_timeout")
                _terminate(process)
                return 124
            time.sleep(poll_seconds)
    except BaseException:
        _terminate(process)
        raise


def _write_supervisor_receipt(path: Path, *, reason: str, status: str) -> None:
    # Preserve the rank's first failure evidence.  The supervisor's launch
    # receipt records the non-zero outcome; replacing this file would erase
    # whether the rank reported an OOM or another exception.
    existing = _read_json(path)
    if existing is not None and existing.get("status") in {"rank_failed", "rank_oom", "rank_exited"}:
        return
    value = {
        "schema": "staged-family-balanced-distributed-failure-v1",
        "status": status,
        "reason": reason,
        "supervisor_pid": os.getpid(),
        "recorded_at_unix": time.time(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _reject_stale_state(failure_file: Path, rank_dir: Path, phase_file: Path | None = None) -> None:
    """Require a new node-owned receipt namespace for every foreground launch."""

    if failure_file.exists():
        raise RuntimeError(f"stale distributed failure receipt exists: {failure_file}")
    if rank_dir.exists() and any(rank_dir.iterdir()):
        raise RuntimeError(f"stale rank guard state exists: {rank_dir}")
    if phase_file is not None and phase_file.exists():
        raise RuntimeError(f"stale launch phase receipt exists: {phase_file}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-file", type=Path, required=True)
    parser.add_argument("--rank-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.world_size < 1 or args.timeout_seconds < 1:
        parser.error("a command, positive world size, and positive timeout are required")
    return supervise(
        command,
        failure_file=args.failure_file,
        rank_dir=args.rank_dir,
        expected_world_size=args.world_size,
        timeout_seconds=args.timeout_seconds,
        phase_file=args.phase_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
