#!/usr/bin/env python3
"""Finite 20-update supervisor for the Diverse-flow pilot.

The proven process-group, metric provenance, and deadline implementation is
reused from the generic text runner.  This arm changes only the registered
checkpoint schedule and makes the 4/8/12/16/20 observations non-promotional:
budget, missing metrics, crashes, and the hard 20-update boundary stop the
process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "text-sft512-consistent-20260930" / "finite_supervisor.py"
spec = importlib.util.spec_from_file_location("generic_structured_supervisor", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"generic finite supervisor is unavailable: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

# Keep the finite controller on the hard 20-update boundary.  The generic
# supervisor otherwise treats a noisy four-update trend as a stop condition;
# this arm records truthful four-update observations after the process exits and
# leaves promotion to the matched panel.
base.CHECKPOINT_STEPS = ()
base.DEFAULT_MAX_GPU_USD = 20.0
MAX_UPDATES = 20
CHECKPOINTS = (4, 8, 12, 16, 20)


def _structured_observation(rewards: dict[int, float], checkpoint: int) -> dict[str, object]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError("structured checkpoints are exactly 4, 8, 12, 16, and 20")
    missing = [step for step in range(1, checkpoint + 1) if step not in rewards]
    if missing:
        raise base.SupervisorError(f"missing admitted reward step(s) before checkpoint {checkpoint}: {missing[:5]}")
    midpoint = checkpoint // 2
    prior = [rewards[step] for step in range(1, midpoint + 1)]
    latest = [rewards[step] for step in range(midpoint + 1, checkpoint + 1)]
    prior_mean = sum(prior) / len(prior)
    latest_mean = sum(latest) / len(latest)
    return {
        "checkpoint": checkpoint,
        "metric": base.REWARD_KEY,
        "denominator": "admitted optimizer batches only",
        "prior_steps": [1, midpoint],
        "latest_steps": [midpoint + 1, checkpoint],
        "prior_mean": prior_mean,
        "latest_mean": latest_mean,
        "delta": latest_mean - prior_mean,
        # Keep the observation truthful. The generic supervisor may stop at
        # an observation boundary when admitted reward is not rising; this
        # field must never claim an improvement merely to reach step 8.
        "rising": latest_mean > prior_mean,
        "decision": "observe",
        "promotion_decision": "deferred_to_matched_16_evaluation",
    }


base.reward_trend = _structured_observation


def _record_observations(metrics_path: Path, trend_path: Path) -> None:
    try:
        rewards = base.read_optimizer_rewards(metrics_path, max_step=MAX_UPDATES)
    except base.SupervisorError:
        return
    for checkpoint in CHECKPOINTS:
        if not all(step in rewards for step in range(1, checkpoint + 1)):
            continue
        value = _structured_observation(rewards, checkpoint)
        value["observed_epoch"] = time.time()
        base._append_json(trend_path, value)


def main(argv: list[str] | None = None) -> int:
    parser = base._parser()
    args = parser.parse_args(argv)
    # The reused generic pipeline supplies its historical fifty-update value;
    # an older controller may also carry eight.  This wrapper owns the
    # registered arm-level bound and clamps either legacy invocation to 20
    # before spawning any trainer process.
    if args.max_updates not in (8, 20, 50):
        raise SystemExit("Diverse-flow supervisor requires the registered 20-update target")
    if args.max_gpu_usd > 20:
        raise SystemExit("GPU cap exceeds the hard $20 limit")
    result = base.supervise(
        command=args.command[1:] if args.command[:1] == ["--"] else args.command,
        metrics_path=args.metrics,
        trainer_log=args.trainer_log,
        state_path=args.state,
        trend_path=args.trend,
        budget=base.make_budget(
            price_usd_per_hour=args.price_usd_per_hour,
            rental_start_epoch=args.rental_start_epoch,
            max_gpu_usd=args.max_gpu_usd,
            reserve_seconds=args.reserve_seconds,
        ),
        max_updates=MAX_UPDATES,
        poll_seconds=args.poll_seconds,
        metric_timeout_seconds=args.metric_timeout_seconds,
        initial_metric_timeout_seconds=args.initial_metric_timeout_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
    )
    _record_observations(args.metrics, args.trend)
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.SupervisorError, ValueError) as exc:
        print(f"structured finite supervisor: {exc}", file=sys.stderr)
        raise SystemExit(2)
