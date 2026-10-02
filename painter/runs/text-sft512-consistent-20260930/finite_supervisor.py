#!/usr/bin/env python3
"""Run one bounded Prime training process and stop it on the frozen protocol.

The supervisor owns exactly one process group, reads only admitted optimizer
batch rewards, and writes a small state/trend ledger.  It is a finite process;
it does not schedule itself, daemonize, or create a watcher.
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
import statistics
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence


REWARD_KEY = "train/agg/effective/agent/reward/mean"
CHECKPOINT_STEPS = (15, 20, 25, 30, 35, 40, 45, 50)
DEFAULT_MAX_GPU_USD = 20.0
DEFAULT_RESERVE_SECONDS = 45 * 60
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_METRIC_TIMEOUT_SECONDS = 30 * 60
MAX_INITIAL_METRIC_TIMEOUT_SECONDS = 60 * 60


class SupervisorError(RuntimeError):
    """A bounded launch or metrics failure that should stop the run."""


@dataclass(frozen=True)
class Budget:
    price_usd_per_hour: float
    rental_start_epoch: float
    max_gpu_usd: float
    reserve_seconds: float

    @property
    def rental_deadline_epoch(self) -> float:
        return self.rental_start_epoch + self.max_gpu_usd / self.price_usd_per_hour * 3600.0

    @property
    def training_deadline_epoch(self) -> float:
        return self.rental_deadline_epoch - self.reserve_seconds

    def as_dict(self) -> dict[str, float]:
        return {
            "price_usd_per_hour": self.price_usd_per_hour,
            "rental_start_epoch": self.rental_start_epoch,
            "max_gpu_usd": self.max_gpu_usd,
            "gpu_rental_cap_usd": self.max_gpu_usd,
            "reserve_seconds": self.reserve_seconds,
            "rental_deadline_epoch": self.rental_deadline_epoch,
            "training_deadline_epoch": self.training_deadline_epoch,
        }


def make_budget(
    *,
    price_usd_per_hour: float,
    rental_start_epoch: float,
    max_gpu_usd: float = DEFAULT_MAX_GPU_USD,
    reserve_seconds: float = DEFAULT_RESERVE_SECONDS,
) -> Budget:
    values = {
        "price_usd_per_hour": price_usd_per_hour,
        "rental_start_epoch": rental_start_epoch,
        "max_gpu_usd": max_gpu_usd,
        "reserve_seconds": reserve_seconds,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if price_usd_per_hour <= 0:
        raise ValueError("price_usd_per_hour must be positive")
    if rental_start_epoch <= 0:
        raise ValueError("rental_start_epoch must be positive")
    if max_gpu_usd <= 0:
        raise ValueError("max_gpu_usd must be positive")
    if reserve_seconds < 0:
        raise ValueError("reserve_seconds must be non-negative")
    budget = Budget(**values)
    if budget.training_deadline_epoch <= rental_start_epoch:
        raise ValueError("reserve exceeds the finite rental budget")
    return budget


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_optimizer_rewards(path: Path, *, max_step: int = 50) -> dict[int, float]:
    """Read the unique admitted-batch reward for each optimizer step.

    Trainer rows and speculative rollout rows are ignored.  A malformed or
    duplicate admitted reward is an error so the supervisor cannot silently
    make a stop decision from a different denominator.
    """

    if not path.is_file():
        return {}
    rewards: dict[int, float] = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        # Prime appends JSONL records while the supervisor reads them. A final
        # unterminated line is an in-progress write, not a metric.
        if line_number == len(lines) and not line.endswith(("\n", "\r")):
            continue
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupervisorError(f"malformed metrics JSON on line {line_number}") from exc
        if not isinstance(row, dict) or REWARD_KEY not in row:
            continue
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= max_step:
            raise SupervisorError(f"invalid admitted optimizer step on line {line_number}")
        reward = _finite_number(row.get(REWARD_KEY))
        if reward is None or not 0.0 <= reward <= 1.0:
            raise SupervisorError(f"invalid admitted reward for step {step}")
        if step in rewards:
            raise SupervisorError(f"duplicate admitted reward for step {step}")
        rewards[step] = reward
    return rewards


def contiguous_step(rewards: dict[int, float]) -> int:
    """Return the highest contiguous admitted optimizer step from one."""

    step = 0
    while step + 1 in rewards:
        step += 1
    return step


_SUCCESS_STEP = re.compile(r"SUCCESS[^\n]*?Step\s+(\d+)\s+\|")
_TRAINER_METRIC_KEYS = frozenset({"optim/grad_norm", "time/step", "optimizer/step"})


def read_completed_optimizer_steps(
    metrics_path: Path,
    *,
    max_step: int = 50,
    trainer_log: Path | None = None,
) -> set[int]:
    """Read trainer-completed optimizer steps with explicit provenance.

    The orchestrator reward metric uses the same ``step`` field for admitted
    batches, so it cannot establish that an optimizer update finished.  Prime
    trainer rows carry optimizer timing/gradient metrics; older runs also
    preserve ``SUCCESS ... Step N |`` lines in ``trainer.log``.  Either source
    is accepted, but an admitted batch alone is never promoted to a completed
    optimizer update.
    """

    completed: set[int] = set()
    if metrics_path.is_file():
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= max_step:
                continue
            producer = row.get("producer")
            trainer_metric = bool(_TRAINER_METRIC_KEYS.intersection(row))
            # ``time/step`` is also emitted by the orchestrator.  Bind normal
            # metrics to Prime's trainer producer; permit only an explicit
            # gradient metric for legacy rows that predate producer labels.
            if (producer == "trainer" and trainer_metric) or (
                producer is None and "optim/grad_norm" in row
            ):
                completed.add(step)
    if trainer_log is None:
        # metrics.jsonl lives at <run>/async-output/<name>/monitors/file/.
        trainer_log = metrics_path.parents[2] / "logs" / "attempt_1" / "trainer.log"
    if trainer_log.is_file():
        for raw in _SUCCESS_STEP.findall(trainer_log.read_text(encoding="utf-8", errors="replace")):
            step = int(raw)
            if 1 <= step <= max_step:
                completed.add(step)
    return completed


def reward_trend(rewards: dict[int, float], checkpoint: int) -> dict[str, object]:
    """Compare the same five-step sampler used at each frozen checkpoint."""

    if checkpoint < 10:
        raise ValueError("a five-vs-five trend needs at least ten updates")
    missing = [step for step in range(1, checkpoint + 1) if step not in rewards]
    if missing:
        raise SupervisorError(
            f"missing admitted reward step(s) before checkpoint {checkpoint}: {missing[:5]}"
        )
    prior = [rewards[step] for step in range(checkpoint - 9, checkpoint - 4)]
    latest = [rewards[step] for step in range(checkpoint - 4, checkpoint + 1)]
    prior_mean = statistics.fmean(prior)
    latest_mean = statistics.fmean(latest)
    delta = latest_mean - prior_mean
    return {
        "checkpoint": checkpoint,
        "metric": REWARD_KEY,
        "denominator": "admitted optimizer batches only",
        "prior_steps": [checkpoint - 9, checkpoint - 5],
        "latest_steps": [checkpoint - 4, checkpoint],
        "prior_five_mean": prior_mean,
        "latest_five_mean": latest_mean,
        "delta": delta,
        "rising": delta > 0.0,
        "decision": "continue" if delta > 0.0 else "stop",
    }


class OwnedProcessGroup:
    """A process and its private session/process group."""

    def __init__(self, process: subprocess.Popen[bytes | str]):
        self.process = process
        self.pid = int(process.pid)
        self.pgid = self._owned_pgid()

    def _owned_pgid(self) -> int:
        try:
            pgid = os.getpgid(self.pid)
        except ProcessLookupError as exc:
            raise SupervisorError("training process exited before ownership was established") from exc
        if pgid != self.pid:
            raise SupervisorError(
                f"refusing to control training pid {self.pid}: process group {pgid} is not private"
            )
        return pgid

    def still_owned(self) -> bool:
        if self.process.poll() is not None:
            return True
        try:
            return os.getpgid(self.pid) == self.pgid == self.pid
        except ProcessLookupError:
            return True

    def stop(self, *, grace_seconds: float = 90.0) -> None:
        if self.process.poll() is not None:
            return
        if not self.still_owned():
            raise SupervisorError("training process group ownership changed; refusing broad termination")
        os.killpg(self.pgid, signal.SIGINT)
        try:
            self.process.wait(timeout=max(0.0, grace_seconds))
            return
        except subprocess.TimeoutExpired:
            pass
        if not self.still_owned():
            raise SupervisorError("training process group ownership changed during shutdown")
        os.killpg(self.pgid, signal.SIGTERM)
        try:
            self.process.wait(timeout=15.0)
            return
        except subprocess.TimeoutExpired:
            pass
        if not self.still_owned():
            raise SupervisorError("training process group ownership changed before final shutdown")
        os.killpg(self.pgid, signal.SIGKILL)
        self.process.wait(timeout=15.0)


def _append_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _write_state(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def supervise(
    *,
    command: Sequence[str],
    metrics_path: Path,
    trainer_log: Path | None = None,
    state_path: Path,
    trend_path: Path,
    budget: Budget,
    max_updates: int = 50,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    metric_timeout_seconds: float = DEFAULT_METRIC_TIMEOUT_SECONDS,
    initial_metric_timeout_seconds: float | None = None,
    shutdown_grace_seconds: float = 90.0,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    """Launch and supervise one finite training process.

    ``now`` and ``sleep`` are injectable so elapsed-budget and missing-metric
    behavior can be tested without a real GPU or long wall-clock waits.
    """

    if not command:
        raise ValueError("training command must not be empty")
    if not 1 <= max_updates <= 50:
        raise ValueError("max_updates must be in [1, 50]")
    if initial_metric_timeout_seconds is None:
        # Preserve callers that historically supplied a custom metric timeout:
        # before this two-phase guard existed, it applied from process start.
        initial_metric_timeout_seconds = metric_timeout_seconds
    if (
        not all(
            math.isfinite(float(value))
            for value in (poll_seconds, metric_timeout_seconds, initial_metric_timeout_seconds, shutdown_grace_seconds)
        )
        or poll_seconds < 0
        or metric_timeout_seconds <= 0
        or initial_metric_timeout_seconds <= 0
        or initial_metric_timeout_seconds > MAX_INITIAL_METRIC_TIMEOUT_SECONDS
        or shutdown_grace_seconds < 0
    ):
        raise ValueError("poll and timeout values are invalid")

    started = now()
    process = popen(list(command), start_new_session=True)
    owner = OwnedProcessGroup(process)
    _write_state(
        state_path,
        {
            "schema": "finite-text-paint-supervisor-v1",
            "status": "running",
            "started_epoch": started,
            "pid": owner.pid,
            "pgid": owner.pgid,
            "metrics_path": str(metrics_path),
            "trend_path": str(trend_path),
            "max_updates": max_updates,
            "admitted_batch_counter": 0,
            "completed_optimizer_steps": None,
            "budget": budget.as_dict(),
            "metric_timeout_seconds": metric_timeout_seconds,
            "initial_metric_timeout_seconds": initial_metric_timeout_seconds,
        },
    )

    seen_checkpoints: set[int] = set()
    last_admitted_batch = 0
    last_completed_optimizer_step: int | None = None
    # Rollout/admission rows are not optimizer progress.  Keep this timestamp
    # anchored at process start until the first real trainer completion, then
    # advance it only when a newer optimizer step is observed.
    last_optimizer_progress_epoch = started
    final_status = "completed"
    final_reason = "process_exit"
    exit_code: int | None = None

    def finish(*, status: str, reason: str, code: int | None) -> int:
        _write_state(
            state_path,
            {
                "schema": "finite-text-paint-supervisor-v1",
                "status": status,
                "reason": reason,
                "started_epoch": started,
                "finished_epoch": now(),
                "pid": owner.pid,
                "pgid": owner.pgid,
                "exit_code": code,
                # Keep the old field as a compatibility alias, but bind it to
                # trainer completion rather than the orchestrator batch step.
                "last_optimizer_step": last_completed_optimizer_step,
                "last_admitted_batch": last_admitted_batch,
                "completed_optimizer_steps": last_completed_optimizer_step,
                "max_updates": max_updates,
                "budget": budget.as_dict(),
                "metrics_path": str(metrics_path),
                "trend_path": str(trend_path),
            },
        )
        return 0 if status in {"completed", "stopped"} else 1

    while True:
        exit_code = process.poll()
        try:
            rewards = read_optimizer_rewards(metrics_path, max_step=max_updates)
        except SupervisorError as exc:
            final_status, final_reason = "failed", f"metrics_error:{exc}"
            if exit_code is None:
                owner.stop(grace_seconds=shutdown_grace_seconds)
            return finish(status=final_status, reason=final_reason, code=exit_code)

        current = contiguous_step(rewards)
        if current > last_admitted_batch:
            last_admitted_batch = current

        completed_steps = read_completed_optimizer_steps(
            metrics_path, max_step=max_updates, trainer_log=trainer_log
        )
        completed_current = max(completed_steps) if completed_steps else None
        if completed_current is not None and (
            last_completed_optimizer_step is None or completed_current > last_completed_optimizer_step
        ):
            last_completed_optimizer_step = completed_current
            last_optimizer_progress_epoch = now()

        for checkpoint in CHECKPOINT_STEPS:
            if (
                checkpoint > max_updates
                or checkpoint > current
                or last_completed_optimizer_step is None
                or checkpoint > last_completed_optimizer_step
                or checkpoint in seen_checkpoints
            ):
                continue
            try:
                trend = reward_trend(rewards, checkpoint)
            except SupervisorError as exc:
                final_status, final_reason = "failed", f"trend_error:{exc}"
                if exit_code is None:
                    owner.stop(grace_seconds=shutdown_grace_seconds)
                return finish(status=final_status, reason=final_reason, code=exit_code)
            trend["observed_epoch"] = now()
            _append_json(trend_path, trend)
            seen_checkpoints.add(checkpoint)
            if not bool(trend["rising"]):
                final_status, final_reason = "stopped", f"reward_not_rising_at_{checkpoint}"
                if exit_code is None:
                    owner.stop(grace_seconds=shutdown_grace_seconds)
                return finish(status=final_status, reason=final_reason, code=exit_code)

        current_time = now()
        if current_time >= budget.training_deadline_epoch:
            final_status, final_reason = "stopped", "finite_cost_deadline"
            if exit_code is None:
                owner.stop(grace_seconds=shutdown_grace_seconds)
            return finish(status=final_status, reason=final_reason, code=exit_code)

        if last_completed_optimizer_step is not None and last_completed_optimizer_step >= max_updates:
            final_status, final_reason = "stopped", "max_updates_reached"
            if exit_code is None:
                owner.stop(grace_seconds=shutdown_grace_seconds)
            return finish(status=final_status, reason=final_reason, code=exit_code)

        if exit_code is not None:
            if exit_code == 0:
                return finish(status="completed", reason="process_exit", code=exit_code)
            return finish(status="failed", reason="process_exit_nonzero", code=exit_code)

        timeout_seconds = (
            metric_timeout_seconds
            if last_completed_optimizer_step is not None
            else initial_metric_timeout_seconds
        )
        if current_time - last_optimizer_progress_epoch >= timeout_seconds:
            final_status, final_reason = "failed", "missing_metric_timeout"
            owner.stop(grace_seconds=shutdown_grace_seconds)
            return finish(status=final_status, reason=final_reason, code=exit_code)

        sleep(min(poll_seconds, max(0.0, budget.training_deadline_epoch - current_time)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--trainer-log", type=Path,
                        help="optional trainer completion log; metrics remain the admitted-batch source")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--trend", type=Path, required=True)
    parser.add_argument("--price-usd-per-hour", type=float, required=True)
    parser.add_argument("--rental-start-epoch", "--start-epoch", dest="rental_start_epoch", type=float, required=True)
    parser.add_argument("--max-gpu-usd", type=float, default=DEFAULT_MAX_GPU_USD)
    parser.add_argument("--reserve-seconds", type=float, default=DEFAULT_RESERVE_SECONDS)
    parser.add_argument("--max-updates", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--metric-timeout-seconds", type=float, default=DEFAULT_METRIC_TIMEOUT_SECONDS)
    parser.add_argument("--initial-metric-timeout-seconds", type=float, default=None)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=90.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="training command after --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("a training command is required after --")
    budget = make_budget(
        price_usd_per_hour=args.price_usd_per_hour,
        rental_start_epoch=args.rental_start_epoch,
        max_gpu_usd=args.max_gpu_usd,
        reserve_seconds=args.reserve_seconds,
    )
    return supervise(
        command=command,
        metrics_path=args.metrics,
        trainer_log=args.trainer_log,
        state_path=args.state,
        trend_path=args.trend,
        budget=budget,
        max_updates=args.max_updates,
        poll_seconds=args.poll_seconds,
        metric_timeout_seconds=args.metric_timeout_seconds,
        initial_metric_timeout_seconds=args.initial_metric_timeout_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SupervisorError, ValueError) as exc:
        print(f"finite supervisor: {exc}", file=sys.stderr)
        raise SystemExit(2)
