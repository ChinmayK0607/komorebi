#!/usr/bin/env python3
"""Run the reviewed recovered-policy Prime entrypoint for the staged run.

The photo-curriculum wrapper owns inference readiness, both initial LoRA
loads, fixed-opponent prewarm, and the fresh trainer launch.  This thin
run-owned adapter only makes the staged runtime contract explicit to Prime's
orchestrator child; the older wrapper predates the staged activation names
and would otherwise drop them when it builds that child environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys


RUN = Path(__file__).resolve().parent
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "staged_bootstrap_for_adapter_validation", RUN / "bootstrap.py"
)
if BOOTSTRAP_SPEC is None or BOOTSTRAP_SPEC.loader is None:
    raise RuntimeError("bootstrap.py is unavailable for adapter validation")
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)
SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "run_from_sft.py"
spec = importlib.util.spec_from_file_location("staged_photo_run_from_sft", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"reviewed photo-curriculum entrypoint is missing: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


TRAIN_WRAPPER = RUN / "train_from_sft.py"
if not TRAIN_WRAPPER.is_file():
    raise RuntimeError(f"staged trainer wrapper is missing: {TRAIN_WRAPPER}")
ORIGINAL_TRAINER_COMMAND = base.trainer_command


def _staged_trainer_command(command: list[str], wrapper: Path | None = None) -> list[str]:
    """Force Prime's trainer child through this run's 4-rank initializer."""

    selected = Path(os.environ.get("PAINTER_PRIME_TRAIN_WRAPPER", str(TRAIN_WRAPPER))).resolve(strict=True)
    if selected != TRAIN_WRAPPER.resolve():
        raise ValueError(f"staged trainer wrapper must remain run-owned: {selected}")
    return ORIGINAL_TRAINER_COMMAND(command, selected)


base.trainer_command = _staged_trainer_command


POLICY_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
FROZEN_SFT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
FROZEN_SFT_ALIAS = "sft-step512"
POLICY_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
POLICY_PREFIX = "rl-judge-matched-20261004-flash/step_12"
FROZEN_SFT_REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
FROZEN_SFT_PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
ORIGINAL_ORCHESTRATOR_ENV = base._orchestrator_env
STAGED_FORWARD_KEYS = (
    "PAINTER_REQUIRE_VALID_FINAL_ACTION",
    "PAINTER_STAGED_CURRICULUM_ENABLE",
    "PAINTER_STAGED_CURRICULUM_RUN_ID",
    "PAINTER_STAGED_CURRICULUM_PROTOCOL",
    "PAINTER_STAGED_CURRICULUM_ADMISSION",
    "PAINTER_STAGED_CURRICULUM_ROOT",
    "PAINTER_STAGED_CURRICULUM_STATE",
    "PAINTER_RL_ADAPTER_REVISION",
    "PAINTER_RL_ADAPTER_PREFIX",
    "PAINTER_RL_ADAPTER_SHA256",
    "PAINTER_FROZEN_SFT_REVISION",
    "PAINTER_FROZEN_SFT_PREFIX",
    "PAINTER_FROZEN_SFT_SHA256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_environment() -> dict[str, str]:
    """Return both adapter paths after checking trusted files and pinned bytes."""

    values: dict[str, str] = {}
    for name, expected, label in (
        ("WATERCOLOUR_INIT_ADAPTER", POLICY_SHA256, "Flash12 policy initializer"),
        ("PAINTER_FROZEN_SFT_ADAPTER", FROZEN_SFT_SHA256, "frozen SFT512 adapter"),
    ):
        raw = os.environ.get(name)
        if not raw:
            raise ValueError(f"{name} is required for the staged recovered-policy run")
        path = Path(raw).expanduser().absolute()
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} adapter directory is missing or symlinked: {path}")
        try:
            weights = BOOTSTRAP.ensure_hf_snapshot_file(
                path / "adapter_model.safetensors", f"{label} weights"
            )
            BOOTSTRAP.ensure_hf_snapshot_file(
                path / "adapter_config.json", f"{label} config"
            )
        except (BOOTSTRAP.BootstrapError, OSError, RuntimeError) as exc:
            raise ValueError(f"{label} files are not trusted Hugging Face snapshot files") from exc
        actual = _sha256(weights)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 is {actual}, expected {expected}")
        values[name] = str(path.resolve(strict=True))
    if values["WATERCOLOUR_INIT_ADAPTER"] == values["PAINTER_FROZEN_SFT_ADAPTER"]:
        raise ValueError("policy initializer and frozen SFT adapter must be distinct")
    alias = os.environ.get("PAINTER_FROZEN_SFT_ALIAS", FROZEN_SFT_ALIAS)
    if alias != FROZEN_SFT_ALIAS:
        raise ValueError(f"frozen SFT alias must remain {FROZEN_SFT_ALIAS}")
    values["PAINTER_FROZEN_SFT_ALIAS"] = alias
    return values


def _staged_orchestrator_env(environment: dict[str, str]) -> dict[str, str]:
    """Forward adapters and staged activation through Prime's child Popen."""

    values = ORIGINAL_ORCHESTRATOR_ENV(environment)
    values.update(_adapter_environment())
    expected_identity = {
        "PAINTER_RL_ADAPTER_REVISION": POLICY_REVISION,
        "PAINTER_RL_ADAPTER_PREFIX": POLICY_PREFIX,
        "PAINTER_RL_ADAPTER_SHA256": POLICY_SHA256,
        "PAINTER_FROZEN_SFT_REVISION": FROZEN_SFT_REVISION,
        "PAINTER_FROZEN_SFT_PREFIX": FROZEN_SFT_PREFIX,
        "PAINTER_FROZEN_SFT_SHA256": FROZEN_SFT_SHA256,
    }
    for key, expected in expected_identity.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"{key} is not pinned to the staged adapter identity")
    for key in STAGED_FORWARD_KEYS:
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"{key} is required for staged curriculum activation")
        values[key] = value
    values["PAINTER_PRIME_ROLE"] = "orchestrator"
    return values


base._orchestrator_env = _staged_orchestrator_env


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
