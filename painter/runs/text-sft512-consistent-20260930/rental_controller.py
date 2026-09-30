#!/usr/bin/env python3
"""Run one finite local command and release exactly one Lium pod.

This is a foreground controller.  It has no scheduler, daemon, TTL watcher, or
provider-output logger.  The caller creates ``release_marker`` only after its
hash-verified preservation work is complete; otherwise the calculated $20
rental deadline (with a small release margin) stops the command.  All Lium
output is consumed in memory and only a small credential-free JSONL receipt is
appended locally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


POD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
DEFAULT_MAX_GPU_USD = 20.0
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_RELEASE_MARGIN_SECONDS = 60.0
DEFAULT_RELEASE_TIMEOUT_SECONDS = 45.0
DEFAULT_CLI_TIMEOUT_SECONDS = 30.0


class RentalControllerError(RuntimeError):
    """A finite command or exact-pod release could not complete safely."""


@dataclass(frozen=True)
class RentalBudget:
    pod_id: str
    price_usd_per_hour: float
    rental_start_epoch: float
    max_gpu_usd: float = DEFAULT_MAX_GPU_USD
    release_margin_seconds: float = DEFAULT_RELEASE_MARGIN_SECONDS

    @property
    def rental_deadline_epoch(self) -> float:
        return self.rental_start_epoch + self.max_gpu_usd / self.price_usd_per_hour * 3600.0

    @property
    def release_deadline_epoch(self) -> float:
        """Stop before the provider deadline to leave room for API/billing calls."""

        return self.rental_deadline_epoch - self.release_margin_seconds


def validate_pod_id(value: str) -> str:
    if not isinstance(value, str) or not POD_ID.fullmatch(value) or value.lower() in {"all", "*"}:
        raise RentalControllerError("pod must be one exact filename-safe provider ID")
    return value


def make_budget(
    *,
    pod_id: str,
    price_usd_per_hour: float,
    rental_start_epoch: float,
    max_gpu_usd: float = DEFAULT_MAX_GPU_USD,
    release_margin_seconds: float = DEFAULT_RELEASE_MARGIN_SECONDS,
) -> RentalBudget:
    validate_pod_id(pod_id)
    values = (price_usd_per_hour, rental_start_epoch, max_gpu_usd, release_margin_seconds)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("rental budget values must be numeric")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("rental budget values must be finite")
    if price_usd_per_hour <= 0 or rental_start_epoch <= 0 or max_gpu_usd <= 0:
        raise ValueError("rental price, start epoch, and cap must be positive")
    if release_margin_seconds < 0:
        raise ValueError("release margin must be non-negative")
    return RentalBudget(
        pod_id=validate_pod_id(pod_id),
        price_usd_per_hour=float(price_usd_per_hour),
        rental_start_epoch=float(rental_start_epoch),
        max_gpu_usd=float(max_gpu_usd),
        release_margin_seconds=float(release_margin_seconds),
    )


def parse_ps_prefix(raw: str) -> Any:
    """Parse the first JSON value, tolerating provider warnings after it."""

    try:
        value, _ = json.JSONDecoder().raw_decode(raw.lstrip())
    except (AttributeError, json.JSONDecodeError) as exc:
        raise RentalControllerError("lium ps did not return a JSON value") from exc
    return value


def pod_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("pods"), list):
        rows = payload["pods"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise RentalControllerError("lium ps JSON has no pod records")
    if not all(isinstance(row, dict) for row in rows):
        raise RentalControllerError("lium ps returned a malformed pod record")
    return rows


def exact_pod_present(payload: Any, pod_id: str) -> bool:
    validate_pod_id(pod_id)
    matches = [row for row in pod_records(payload) if str(row.get("id", "")) == pod_id]
    if len(matches) > 1:
        raise RentalControllerError("lium ps returned duplicate records for the exact pod ID")
    return bool(matches)


def _run_lium(
    lium_bin: str,
    args: Sequence[str],
    *,
    run: Callable[..., Any],
    timeout_seconds: float = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> Any:
    # Capture provider output so it cannot leak into terminal logs or receipts.
    return run(
        [lium_bin, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def pod_present(*, pod_id: str, lium_bin: str, run: Callable[..., Any]) -> bool:
    result = _run_lium(lium_bin, ("ps", "--format", "json"), run=run)
    if result.returncode != 0:
        raise RentalControllerError(f"lium ps failed with exit {result.returncode}")
    return exact_pod_present(parse_ps_prefix(result.stdout), pod_id)


def release_exact_pod(
    *,
    pod_id: str,
    lium_bin: str = "lium",
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    timeout_seconds: float = DEFAULT_RELEASE_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> str:
    """Remove only ``pod_id`` after verifying it is the exact active record."""

    validate_pod_id(pod_id)
    if timeout_seconds <= 0 or not 0 < poll_seconds <= 30:
        raise ValueError("release timeout must be positive and poll must be at most 30 seconds")
    if not pod_present(pod_id=pod_id, lium_bin=lium_bin, run=run):
        return "already_absent"
    result = _run_lium(lium_bin, ("rm", pod_id, "-y"), run=run)
    if result.returncode != 0:
        raise RentalControllerError(f"lium rm for exact pod failed with exit {result.returncode}")
    release_deadline = now() + timeout_seconds
    while True:
        if not pod_present(pod_id=pod_id, lium_bin=lium_bin, run=run):
            return "released"
        current = now()
        if current >= release_deadline:
            raise RentalControllerError("exact pod still appears active after release timeout")
        sleep(min(poll_seconds, release_deadline - current))


class OwnedChild:
    """A local command in a private process group, safe to stop."""

    def __init__(self, process: subprocess.Popen[Any]):
        self.process = process
        self.pid = int(process.pid)
        try:
            self.pgid = os.getpgid(self.pid)
        except ProcessLookupError as exc:
            raise RentalControllerError("local command exited before ownership was established") from exc
        if self.pgid != self.pid:
            raise RentalControllerError("refusing to control a child without a private process group")

    def stop(self, *, grace_seconds: float = 30.0) -> None:
        if self.process.poll() is not None:
            return
        if os.getpgid(self.pid) != self.pgid or self.pgid != self.pid:
            raise RentalControllerError("local command process-group ownership changed")
        os.killpg(self.pgid, signal.SIGINT)
        try:
            self.process.wait(timeout=max(0.0, grace_seconds))
            return
        except subprocess.TimeoutExpired:
            pass
        if self.process.poll() is None:
            if os.getpgid(self.pid) != self.pgid or self.pgid != self.pid:
                raise RentalControllerError("local command process-group ownership changed during shutdown")
            os.killpg(self.pgid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15.0)
                return
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None:
            if os.getpgid(self.pid) != self.pgid or self.pgid != self.pid:
                raise RentalControllerError("local command process-group ownership changed before kill")
            os.killpg(self.pgid, signal.SIGKILL)
            self.process.wait(timeout=15.0)


def _append_receipt(path: Path, value: dict[str, Any]) -> None:
    """Append only safe controller fields; never write provider output."""

    if path.exists() and path.is_symlink():
        raise RentalControllerError("receipt path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def supervise(
    *,
    budget: RentalBudget,
    command: Sequence[str],
    release_marker: Path,
    receipt_path: Path,
    lium_bin: str = "lium",
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    release_timeout_seconds: float = DEFAULT_RELEASE_TIMEOUT_SECONDS,
    shutdown_grace_seconds: float = 30.0,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    run: Callable[..., Any] = subprocess.run,
) -> int:
    """Run one command until marker, process exit, or finite release deadline."""

    validate_pod_id(budget.pod_id)
    if not command:
        raise ValueError("local command must not be empty")
    if not 0 < poll_seconds <= 30 or release_timeout_seconds <= 0 or shutdown_grace_seconds < 0:
        raise ValueError("finite controller intervals are invalid")

    started = now()
    common = {
        "schema": "finite-rental-receipt-v1",
        "pod_id": budget.pod_id,
        "price_usd_per_hour": budget.price_usd_per_hour,
        "max_gpu_usd": budget.max_gpu_usd,
        "rental_deadline_epoch": budget.rental_deadline_epoch,
        "release_deadline_epoch": budget.release_deadline_epoch,
    }
    _append_receipt(receipt_path, {**common, "event": "started", "started_epoch": started})

    process: Any = None
    owner: OwnedChild | None = None
    child_code: int | None = None
    reason = "process_exit"
    status = "completed"

    def release_and_finish(*, release_reason: str, final_status: str, code: int) -> int:
        release_status = release_exact_pod(
            pod_id=budget.pod_id,
            lium_bin=lium_bin,
            run=run,
            sleep=sleep,
            now=now,
            timeout_seconds=release_timeout_seconds,
            poll_seconds=poll_seconds,
        )
        _append_receipt(
            receipt_path,
            {
                **common,
                "event": "released",
                "reason": release_reason,
                "release_status": release_status,
                "child_exit_code": child_code,
                "finished_epoch": now(),
                "status": final_status,
            },
        )
        return code

    try:
        # Setup is part of the finite rental budget: do not start a child when
        # the release deadline has already arrived.
        if release_marker.is_file() and not release_marker.is_symlink():
            return release_and_finish(release_reason="release_marker", final_status="completed", code=0)
        if started >= budget.release_deadline_epoch:
            return release_and_finish(release_reason="rental_deadline", final_status="stopped", code=0)

        process = popen(list(command), start_new_session=True)
        owner = OwnedChild(process)
        while True:
            child_code = process.poll()
            if release_marker.is_file() and not release_marker.is_symlink():
                reason = "release_marker"
                if child_code is None:
                    owner.stop(grace_seconds=shutdown_grace_seconds)
                break
            current = now()
            if current >= budget.release_deadline_epoch:
                reason = "rental_deadline"
                status = "stopped"
                if child_code is None:
                    owner.stop(grace_seconds=shutdown_grace_seconds)
                break
            if child_code is not None:
                break
            sleep(min(poll_seconds, max(0.0, budget.release_deadline_epoch - current)))

        if child_code is not None and child_code != 0 and reason == "process_exit":
            return release_and_finish(release_reason="child_exit_nonzero", final_status="failed", code=child_code)
        return release_and_finish(release_reason=reason, final_status=status, code=0)
    except BaseException as exc:
        if owner is not None and process.poll() is None:
            owner.stop(grace_seconds=shutdown_grace_seconds)
        release_status = "release_failed"
        try:
            release_status = release_exact_pod(
                pod_id=budget.pod_id,
                lium_bin=lium_bin,
                run=run,
                sleep=sleep,
                now=now,
                timeout_seconds=release_timeout_seconds,
                poll_seconds=poll_seconds,
            )
        except BaseException:
            pass
        _append_receipt(
            receipt_path,
            {
                **common,
                "event": "failed",
                "reason": f"controller_error:{type(exc).__name__}",
                "release_status": release_status,
                "child_exit_code": process.poll() if process is not None else None,
                "finished_epoch": now(),
                "status": "failed",
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--price-usd-per-hour", type=float, required=True)
    parser.add_argument("--rental-start-epoch", type=float, required=True)
    parser.add_argument("--max-gpu-usd", type=float, default=DEFAULT_MAX_GPU_USD)
    parser.add_argument("--release-margin-seconds", type=float, default=DEFAULT_RELEASE_MARGIN_SECONDS)
    parser.add_argument("--release-marker", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lium-bin", default="lium")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--release-timeout-seconds", type=float, default=DEFAULT_RELEASE_TIMEOUT_SECONDS)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="local command after --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("a local command is required after --")
    budget = make_budget(
        pod_id=args.pod,
        price_usd_per_hour=args.price_usd_per_hour,
        rental_start_epoch=args.rental_start_epoch,
        max_gpu_usd=args.max_gpu_usd,
        release_margin_seconds=args.release_margin_seconds,
    )
    return supervise(
        budget=budget,
        command=command,
        release_marker=args.release_marker,
        receipt_path=args.receipt,
        lium_bin=args.lium_bin,
        poll_seconds=args.poll_seconds,
        release_timeout_seconds=args.release_timeout_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RentalControllerError, ValueError) as exc:
        print(f"rental controller: {exc}", file=sys.stderr)
        raise SystemExit(2)
