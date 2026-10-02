"""Opt-in Prime settings for a measured stale-rollout recovery profile.

This module contains only preparation-time values and arithmetic.  It does
not launch a trainer, render, call a judge, or claim that lower occupancy is
faster.  The profile keeps the off-policy bound at one and sizes the in-flight
pool to two valid policy-version windows.  Its interval-one checkpoint cadence
is an atomic post-update boundary: it makes a completed update publishable
before a finite supervisor stops the process.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib
from typing import Any, Callable, Mapping


PROFILE_NAME = "rl-throughput-recovery-v1"
INITIALIZER_MODE = "recovered_policy"
INITIALIZER_REVISION = "ac4d3eda8e5925e2d23992cfdc1a63620ebde573"
INITIALIZER_PREFIX = "text-pairwise-scaled-20260929/step_16"
INITIALIZER_SHA256 = "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c"
REWARD_PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
CHECKPOINT_INTERVAL = 1
BATCH_SIZE = 16
GROUP_SIZE = 4
MAX_OFF_POLICY_STEPS = 1
INITIAL_INFLIGHT = 32
MAX_INFLIGHT = 32
ADAPTER_CHECKPOINT_MODE = "adapter_only_post_update"
OPTIMIZER_RESUME = False
ACTIVATION_CPU_OFFLOAD = True


@dataclass(frozen=True)
class ObservedTiming:
    """Measured seconds used for a conservative completion estimate.

    ``initial_warmup`` and ``first_batch_wait`` must come from a preserved
    receipt/log.  ``update_seconds`` is one measured update duration;
    ``publish_budget`` reserves time for the interval-one adapter export and
    its verification.  No value is inferred from GPU count or model name.
    """

    initial_warmup: float
    first_batch_wait: float
    update_seconds: float
    publish_budget: float
    refill_seconds_per_update: float = 0
    policy_cycle_seconds: float = 0

    def __post_init__(self) -> None:
        for name in (
            "initial_warmup",
            "first_batch_wait",
            "update_seconds",
            "publish_budget",
            "refill_seconds_per_update",
            "policy_cycle_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number of seconds")


def profile_settings() -> dict[str, int | str]:
    """Return the exact opt-in settings for a config/receipt."""
    return {
        "profile": PROFILE_NAME,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_keep_last": 1,
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "max_off_policy_steps": MAX_OFF_POLICY_STEPS,
        "initial_inflight": INITIAL_INFLIGHT,
        "max_inflight": MAX_INFLIGHT,
        "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
    }


def required_environment() -> dict[str, str]:
    """Return launch bindings required by this opt-in profile."""
    return {
        "PAINTER_RL_CHECKPOINT_INTERVAL": str(CHECKPOINT_INTERVAL),
        "ADAPTER_ONLY_CHECKPOINTS": "1",
        "PAINTER_ACTIVATION_CPU_OFFLOAD": "1",
    }


def preparation_bindings() -> dict[str, Any]:
    """Return the complete config-writer bindings for this profile.

    A caller should pass these bindings to the text config writer as one
    immutable bundle.  The values deliberately include the RL16 initializer
    and reward protocol so a future wrapper cannot apply only the occupancy
    change while silently retaining a different source recipe.
    """
    return {
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "initial_inflight": INITIAL_INFLIGHT,
        "max_inflight": MAX_INFLIGHT,
        "mode": INITIALIZER_MODE,
        "judge_protocol": REWARD_PROTOCOL,
        "adapter_revision": INITIALIZER_REVISION,
        "adapter_prefix": INITIALIZER_PREFIX,
        "adapter_sha256": INITIALIZER_SHA256,
        "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
    }


def validate_runtime_environment(environ: Mapping[str, str]) -> None:
    """Require the runtime binding that makes adapter-only export truthful."""
    for key, value in required_environment().items():
        if environ.get(key) != value:
            raise ValueError(f"runtime environment does not satisfy {PROFILE_NAME}: {key}")


def prepare_profile(
    prepare: Callable[..., dict[str, Any]],
    *,
    output: Path,
    model: Path,
    run_root: Path,
    manifest: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare and immediately validate one profile-bound native config.

    ``prepare`` is the existing offline config writer.  This wrapper does not
    start any process; it prevents a caller from overriding profile-critical
    bindings and then validates both generated TOML and its receipt before the
    result can be handed to a launcher.
    """
    bindings = preparation_bindings()
    conflicting = {
        key: value
        for key, value in kwargs.items()
        if key in bindings and value != bindings[key]
    }
    if conflicting:
        raise ValueError(f"profile-critical preparation override: {sorted(conflicting)}")
    args = {
        "output": output,
        "model": model,
        "run_root": run_root,
        "manifest": manifest,
        **kwargs,
        **bindings,
    }
    receipt = prepare(**args)
    receipt_path = output.with_name("rl-config-receipt.json")
    if not receipt_path.is_file():
        raise ValueError("profile preparation did not write rl-config-receipt.json")
    validate_config(tomllib.loads(output.read_text()), receipt)
    on_disk = json.loads(receipt_path.read_text())
    if on_disk != receipt:
        raise ValueError("profile receipt returned by config writer differs from on-disk receipt")
    return receipt


def validate_config(config: Mapping[str, Any], receipt: Mapping[str, Any] | None = None) -> None:
    """Reject a prepared config/receipt that silently changed the profile."""
    try:
        concurrency = config["orchestrator"]["concurrency"]
        actual = {
            "checkpoint_interval": config["ckpt"]["interval"],
            "checkpoint_keep_last": config["ckpt"]["keep_last"],
            "batch_size": config["orchestrator"]["batch_size"],
            "group_size": config["orchestrator"]["group_size"],
            "max_off_policy_steps": config["orchestrator"]["max_off_policy_steps"],
            "initial_inflight": concurrency["initial_inflight"],
            "max_inflight": concurrency["max_inflight"],
            "activation_cpu_offload": config["trainer"]["model"]["ac_offloading"]
            == {"pin_memory": True, "max_inflight_activations": 1},
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("prepared config is missing the recovery profile fields") from exc
    expected = {key: value for key, value in profile_settings().items() if key != "profile"}

    def matches(actual_value: Any, expected_value: Any) -> bool:
        # Python considers True equal to 1; profile identity must not.
        if isinstance(expected_value, bool):
            return type(actual_value) is bool and actual_value is expected_value
        if isinstance(expected_value, int):
            return type(actual_value) is int and actual_value == expected_value
        return actual_value == expected_value

    if any(not matches(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(f"prepared config does not match {PROFILE_NAME}: {actual!r}")
    if receipt is not None:
        for key, value in expected.items():
            if not matches(receipt.get(key), value):
                raise ValueError(f"prepared receipt does not match {PROFILE_NAME}: {key}")
        if receipt.get("optimizer_resume") is not OPTIMIZER_RESUME:
            raise ValueError(f"prepared receipt does not match {PROFILE_NAME}: optimizer_resume")
        if receipt.get("adapter_checkpoint_mode_required") != ADAPTER_CHECKPOINT_MODE:
            raise ValueError(
                f"prepared receipt does not match {PROFILE_NAME}: adapter_checkpoint_mode_required"
            )
        if receipt.get("activation_cpu_offload") is not ACTIVATION_CPU_OFFLOAD:
            raise ValueError(f"prepared receipt does not match {PROFILE_NAME}: activation_cpu_offload")
        expected_identity = {
            "mode": INITIALIZER_MODE,
            "adapter_revision": INITIALIZER_REVISION,
            "adapter_prefix": INITIALIZER_PREFIX,
            "adapter_sha256": INITIALIZER_SHA256,
            "reward": REWARD_PROTOCOL,
        }
        for key, value in expected_identity.items():
            if receipt.get(key) != value:
                raise ValueError(f"prepared receipt does not match {PROFILE_NAME}: {key}")


def required_runtime_seconds(observation: ObservedTiming, completed_updates: int) -> float:
    """Estimate the minimum observed runtime for ``completed_updates``.

    The first update pays warmup and first-batch wait.  Each update also pays
    the supplied policy-cycle and refill observations.  This remains an
    optimistic lower bound: it cannot prove capacity because cancellation,
    variable refill, and tail latency may be larger than the supplied values.
    """
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 1:
        raise ValueError("completed_updates must be a positive integer")
    return (
        observation.initial_warmup
        + observation.first_batch_wait
        + completed_updates * observation.update_seconds
        + completed_updates * observation.publish_budget
        + completed_updates * observation.refill_seconds_per_update
        + completed_updates * observation.policy_cycle_seconds
    )


def budget_check(
    observation: ObservedTiming,
    completed_updates: int,
    available_training_seconds: float,
) -> dict[str, Any]:
    """Return an auditable capacity check without claiming success/failure quality."""
    if (
        isinstance(available_training_seconds, bool)
        or not isinstance(available_training_seconds, (int, float))
        or not math.isfinite(available_training_seconds)
        or available_training_seconds < 0
    ):
        raise ValueError("available_training_seconds must be a non-negative number")
    required = required_runtime_seconds(observation, completed_updates)
    return {
        "profile": PROFILE_NAME,
        "completed_updates": completed_updates,
        "required_runtime_seconds": required,
        "available_training_seconds": available_training_seconds,
        "fits_observed_bound": required <= available_training_seconds,
        "basis": "measured_initial_warmup_first_batch_wait_update_and_publish_budget",
        "runtime_bound_type": "optimistic_lower_bound",
        "capacity_proven": False,
        "launch_ready": False,
        "throughput_claim": None,
    }


__all__ = [
    "BATCH_SIZE",
    "ADAPTER_CHECKPOINT_MODE",
    "ACTIVATION_CPU_OFFLOAD",
    "CHECKPOINT_INTERVAL",
    "GROUP_SIZE",
    "INITIALIZER_PREFIX",
    "INITIALIZER_REVISION",
    "INITIALIZER_SHA256",
    "INITIALIZER_MODE",
    "INITIAL_INFLIGHT",
    "MAX_INFLIGHT",
    "MAX_OFF_POLICY_STEPS",
    "ObservedTiming",
    "PROFILE_NAME",
    "REWARD_PROTOCOL",
    "budget_check",
    "profile_settings",
    "preparation_bindings",
    "prepare_profile",
    "required_environment",
    "required_runtime_seconds",
    "validate_runtime_environment",
    "validate_config",
]
