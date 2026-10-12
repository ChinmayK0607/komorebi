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
import json
import os
from pathlib import Path
import sys
import threading
import time


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
ORIGINAL_WAIT_AND_LOAD = base._wait_and_load
ORIGINAL_PREWARM_FROZEN_OPPONENT = base._prewarm_frozen_opponent


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    """Write a small run-owned launch receipt without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_launch_phase(phase: str, **details: object) -> None:
    raw = os.environ.get("PAINTER_LAUNCH_PHASE_FILE")
    if not raw:
        return
    payload: dict[str, object] = {
        "schema": "staged-family-balanced-launch-phase-v1",
        "phase": phase,
        "recorded_at_unix": time.time(),
    }
    payload.update(details)
    try:
        _atomic_json(Path(raw).expanduser().resolve(), payload)
    except OSError:
        # Phase is diagnostic only. Rank/failure receipts remain authoritative
        # and the supervisor still fails closed when they are unavailable.
        return


def _trainer_guard_paths() -> tuple[Path, Path] | None:
    rank_dir = os.environ.get("PAINTER_RANK_GUARD_DIR")
    failure_file = os.environ.get("PAINTER_DISTRIBUTED_FAILURE_FILE")
    if not rank_dir or not failure_file:
        return None
    return Path(rank_dir).expanduser().resolve(), Path(failure_file).expanduser().resolve()


def _record_trainer_process_started(command: list[str], process: object) -> tuple[Path, Path] | None:
    paths = _trainer_guard_paths()
    if paths is None:
        return None
    rank_dir, failure_file = paths
    _atomic_json(rank_dir / "trainer-process.started.json", {
        "schema": "staged-family-balanced-trainer-process-start-v1",
        "pid": int(getattr(process, "pid", 0)),
        "command": [str(value) for value in command],
        "wrapper": str(TRAIN_WRAPPER.resolve()),
        "world_size": int(os.environ.get("PAINTER_NUM_TRAIN_GPUS", "0")),
        "recorded_at_unix": time.time(),
    })
    return paths


def _watch_trainer_process(process: object, command: list[str], paths: tuple[Path, Path]) -> None:
    """Turn a pre-rank torchrun exit into the same prompt failure receipt."""

    rank_dir, failure_file = paths
    try:
        status = int(process.wait())  # type: ignore[attr-defined]
    except (AttributeError, OSError, TypeError, ValueError):
        return
    if status == 0 or failure_file.exists():
        return
    _atomic_json(failure_file, {
        "schema": "staged-family-balanced-distributed-failure-v1",
        "status": "rank_exited",
        "rank": "trainer-launcher",
        "pid": int(getattr(process, "pid", 0)),
        "world_size": os.environ.get("PAINTER_NUM_TRAIN_GPUS"),
        "reason": f"trainer torchrun exited before rank readiness with exit code {status}",
        "command": [str(value) for value in command],
        "process_start_receipt": str(rank_dir / "trainer-process.started.json"),
        "recorded_at_unix": time.time(),
    })


def _staged_trainer_command(command: list[str], wrapper: Path | None = None) -> list[str]:
    """Force Prime's trainer child through this run's 4-rank initializer."""

    target = ["-m", "prime_rl.trainer.rl.train"]
    if not any(command[index : index + 2] == target for index in range(len(command) - 1)):
        # Prime's photo runner sends env-server commands through this helper
        # before it recognizes the orchestrator/trainer roles. Preserve those
        # commands; only the pinned trainer target is rewritten and enforced.
        return list(command)
    selected = Path(os.environ.get("PAINTER_PRIME_TRAIN_WRAPPER", str(TRAIN_WRAPPER))).resolve(strict=True)
    if selected != TRAIN_WRAPPER.resolve():
        raise ValueError(f"staged trainer wrapper must remain run-owned: {selected}")
    modified = ORIGINAL_TRAINER_COMMAND(command, selected)
    if str(TRAIN_WRAPPER.resolve()) not in [str(value) for value in modified]:
        raise ValueError("Prime trainer command was not rewritten to the staged trainer wrapper")
    return modified


base.trainer_command = _staged_trainer_command


def _staged_wait_and_load(*args, **kwargs):
    _record_launch_phase("inference_readiness_wait")
    try:
        result = ORIGINAL_WAIT_AND_LOAD(*args, **kwargs)
    except BaseException as exc:
        _record_launch_phase("inference_readiness_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _record_launch_phase("inference_ready")
    return result


def _staged_prewarm_frozen_opponent(*args, **kwargs):
    _record_launch_phase("opponent_prewarm_started")
    try:
        result = ORIGINAL_PREWARM_FROZEN_OPPONENT(*args, **kwargs)
    except BaseException as exc:
        _record_launch_phase("opponent_prewarm_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _record_launch_phase("opponent_prewarm_ready")
    return result


base._wait_and_load = _staged_wait_and_load
base._prewarm_frozen_opponent = _staged_prewarm_frozen_opponent


POLICY_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
FROZEN_SFT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
FROZEN_SFT_ALIAS = "sft-step512"
POLICY_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
POLICY_PREFIX = "rl-judge-matched-20261004-flash/step_12"
FROZEN_SFT_REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
FROZEN_SFT_PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
ORIGINAL_ORCHESTRATOR_ENV = base._orchestrator_env
ORIGINAL_CACHE_ENV = base._cache_env
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
    artifact_root = os.environ.get("PAINTER_RUN_ARTIFACT_ROOT")
    if artifact_root:
        values["PAINTER_RUN_ARTIFACT_ROOT"] = artifact_root
    values["PAINTER_PRIME_ROLE"] = "orchestrator"
    return values


base._orchestrator_env = _staged_orchestrator_env


def _staged_cache_env(root: Path, role: str, environment: dict[str, str]) -> dict[str, str]:
    artifact_root = os.environ.get("PAINTER_RUN_ARTIFACT_ROOT")
    if artifact_root:
        root = Path(artifact_root).expanduser().resolve()
    return ORIGINAL_CACHE_ENV(root, role, environment)


base._cache_env = _staged_cache_env


def main() -> None:
    # The reviewed photo runner owns the local ``Popen`` closure.  Wrap the
    # entrypoint's Popen before entering it so the staged trainer command gets
    # a process-level receipt even when torchrun dies before importing the
    # rank wrapper (rank-started JSON remains the separate readiness signal).
    import prime_rl.entrypoints.rl as entrypoint

    original_popen = entrypoint.Popen

    def watched_popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        values = [str(value) for value in command]
        if str(TRAIN_WRAPPER.resolve()) in values:
            _record_launch_phase("trainer_process_started", pid=int(getattr(process, "pid", 0)))
            paths = _record_trainer_process_started(values, process)
            if paths is not None:
                threading.Thread(
                    target=_watch_trainer_process,
                    args=(process, values, paths),
                    name="painter-trainer-process-watch",
                    daemon=True,
                ).start()
        elif values and Path(values[0]).name == "inference":
            _record_launch_phase("inference_process_started", pid=int(getattr(process, "pid", 0)))
        elif values and Path(values[0]).name == "orchestrator":
            _record_launch_phase("orchestrator_process_started", pid=int(getattr(process, "pid", 0)))
        return process

    entrypoint.Popen = watched_popen
    try:
        _record_launch_phase("run_started", pid=os.getpid())
        base.main()
    finally:
        entrypoint.Popen = original_popen


if __name__ == "__main__":
    main()
