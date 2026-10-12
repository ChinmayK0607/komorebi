#!/usr/bin/env python3
"""Validate a prepared staged node and hand it to the bounded launcher.

The gateway argument is a path only.  This wrapper checks its owner-only mode
without opening it, derives every run path from the bootstrap and preparation
receipts, and then invokes ``launch.sh`` in the foreground.  It never prints
the child environment or the contents of the gateway file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("staged_bootstrap_for_launch", RUN_ROOT / "bootstrap.py")
if BOOTSTRAP_SPEC is None or BOOTSTRAP_SPEC.loader is None:
    raise RuntimeError("bootstrap.py is unavailable")
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)


RUN_ID = BOOTSTRAP.RUN_ID
PRIME_COMMIT = BOOTSTRAP.PRIME_COMMIT
MODEL_REPO = BOOTSTRAP.MODEL_REPO
MODEL_REVISION = BOOTSTRAP.MODEL_REVISION
POLICY_REPO = BOOTSTRAP.POLICY_REPO
POLICY_REVISION = BOOTSTRAP.POLICY_REVISION
POLICY_PREFIX = BOOTSTRAP.POLICY_PREFIX
POLICY_SHA256 = BOOTSTRAP.POLICY_SHA256
FROZEN_REVISION = BOOTSTRAP.FROZEN_REVISION
FROZEN_PREFIX = BOOTSTRAP.FROZEN_PREFIX
FROZEN_SHA256 = BOOTSTRAP.FROZEN_SHA256
RENDERER_SHA256 = BOOTSTRAP.RENDERER_SHA256
P5_BRUSH_SHA256 = BOOTSTRAP.P5_BRUSH_SHA256


class PreparedLaunchError(RuntimeError):
    """The prepared handoff is not admissible for launch."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparedLaunchError(f"{description} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparedLaunchError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PreparedLaunchError(f"{description} is not a JSON object")
    return value


def _path(value: Any, description: str, *, directory: bool = False, executable: bool = False) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise PreparedLaunchError(f"{description} must be an absolute path")
    path = Path(value)
    if path.is_symlink():
        raise PreparedLaunchError(f"{description} cannot be a symlink")
    if directory:
        if not path.is_dir():
            raise PreparedLaunchError(f"{description} is missing")
    elif not path.is_file():
        raise PreparedLaunchError(f"{description} is missing")
    if executable and not os.access(path, os.X_OK):
        raise PreparedLaunchError(f"{description} is not executable")
    return path.resolve(strict=True)


def _runtime_executable(value: Any, description: str, runtime_root: Path) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise PreparedLaunchError(f"{description} must be an absolute path")
    try:
        return BOOTSTRAP.ensure_runtime_executable(Path(value), description, runtime_root)
    except BOOTSTRAP.BootstrapError as exc:
        raise PreparedLaunchError(str(exc)) from exc


def _same(left: Path, right: Path, description: str) -> None:
    if left.resolve() != right.resolve():
        raise PreparedLaunchError(f"{description} does not match the verified receipt")


def _weights(path: Path, expected: str, description: str) -> None:
    try:
        weights = BOOTSTRAP.ensure_hf_snapshot_file(path / "adapter_model.safetensors", f"{description} weights")
    except BOOTSTRAP.BootstrapError as exc:
        raise PreparedLaunchError(str(exc)) from exc
    try:
        BOOTSTRAP.ensure_hf_snapshot_file(path / "adapter_config.json", f"{description} config")
    except BOOTSTRAP.BootstrapError as exc:
        raise PreparedLaunchError(str(exc)) from exc
    if sha256_file(weights) != expected:
        raise PreparedLaunchError(f"{description} bytes changed after preparation")


def validate_gateway_file(path: Path) -> Path:
    """Validate metadata only; never open or echo the credential file."""

    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PreparedLaunchError("gateway env file must be a regular file")
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        raise PreparedLaunchError("gateway env file must have owner-only mode 600")
    if path.stat().st_uid != os.geteuid():
        raise PreparedLaunchError("gateway env file must be owned by the launching user")
    if not os.access(path, os.R_OK):
        raise PreparedLaunchError("gateway env file is not readable by the launching user")
    return path.resolve(strict=True)


def _validate_topology(ready: dict[str, Any], *, visible_gpus: int) -> dict[str, int]:
    topology = ready.get("topology")
    if not isinstance(topology, dict):
        raise PreparedLaunchError("bootstrap receipt lacks GPU topology")
    try:
        result = {key: int(topology[key]) for key in ("gpus_per_node", "num_train_gpus", "num_infer_gpus")}
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparedLaunchError("bootstrap topology is malformed") from exc
    if (result["gpus_per_node"], result["num_train_gpus"], result["num_infer_gpus"]) not in BOOTSTRAP.SUPPORTED_TOPOLOGIES:
        raise PreparedLaunchError("bootstrap topology is not the explicit 1+1, 2+2, or 4+4 shape")
    if visible_gpus < result["gpus_per_node"]:
        raise PreparedLaunchError("current visible GPU count is below the verified topology")
    return result


def _validate_renderer(ready: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    renderer_record = ready.get("renderer")
    if not isinstance(renderer_record, dict):
        raise PreparedLaunchError("bootstrap receipt lacks renderer identity")
    pin = renderer_record.get("pin")
    runtime = renderer_record.get("runtime_helper")
    if not isinstance(pin, dict) or not isinstance(runtime, dict):
        raise PreparedLaunchError("bootstrap receipt lacks renderer pin/helper receipts")
    if pin.get("renderer_sha256") != RENDERER_SHA256 or pin.get("p5_brush_sha256") != P5_BRUSH_SHA256 or pin.get("playwright") != BOOTSTRAP.PLAYWRIGHT_VERSION or pin.get("pillow") != BOOTSTRAP.PILLOW_VERSION:
        raise PreparedLaunchError("renderer pin is not the reviewed runtime")
    if runtime.get("sandbox_smoke") != "passed" or runtime.get("node_check") != "passed" or runtime.get("browser_root") != pin.get("browsers"):
        raise PreparedLaunchError("renderer sandbox/Node helper did not pass")
    renderer_root = _path(ready.get("renderer_root"), "renderer root", directory=True)
    if pin.get("renderer_root") != str(renderer_root):
        raise PreparedLaunchError("renderer pin root does not match the bootstrap receipt")
    renderer = _path(pin.get("renderer"), "renderer")
    p5_brush = _path(pin.get("p5_brush"), "p5.brush asset")
    renderer_python = _runtime_executable(pin.get("renderer_python"), "renderer interpreter", Path(str(pin.get("renderer_python"))).parent.parent)
    browser_root = _path(pin.get("browsers"), "renderer browser root", directory=True)
    for path, description in ((renderer, "renderer"), (p5_brush, "p5.brush asset"), (renderer_python, "renderer interpreter"), (browser_root, "renderer browser root")):
        try:
            path.absolute().relative_to(renderer_root.absolute())
        except ValueError as exc:
            raise PreparedLaunchError(f"{description} is outside the verified renderer root") from exc
    if sha256_file(renderer) != RENDERER_SHA256 or sha256_file(p5_brush) != P5_BRUSH_SHA256 or sha256_file(renderer_python) != pin.get("renderer_python_sha256"):
        raise PreparedLaunchError("renderer or interpreter bytes changed after preparation")
    return renderer, renderer_python, browser_root, renderer_root


def validate_ready(*, ready_path: Path, node_root: Path, visible_gpus: int) -> dict[str, Any]:
    ready = _read_json(ready_path, "bootstrap-ready receipt")
    if ready.get("schema") != "staged-family-balanced-bootstrap-ready-v1" or ready.get("status") != "ready_no_launch" or ready.get("run_id") != RUN_ID:
        raise PreparedLaunchError("bootstrap receipt is not a ready staged handoff")
    node = _path(ready.get("node_root"), "bootstrap node root", directory=True)
    _same(node, node_root, "node root")
    repo = _path(ready.get("repo_root"), "repository root", directory=True)
    _same(repo, REPO_ROOT, "repository root")
    source = _path(ready.get("source_root"), "staged source root", directory=True)
    _same(source, RUN_ROOT, "staged source root")
    brush_root = _path(ready.get("brush_root"), "brush root", directory=True)
    try:
        expected_brush_root = BOOTSTRAP.ensure_brush_root(repo)
    except BOOTSTRAP.BootstrapError as exc:
        raise PreparedLaunchError(str(exc)) from exc
    _same(brush_root, expected_brush_root, "brush root")
    if ready.get("training_started") is not False or ready.get("model_generation") is not False or ready.get("judge_calls") is not False or ready.get("background_processes_left_running") is not False:
        raise PreparedLaunchError("bootstrap receipt claims work already started")
    topology = _validate_topology(ready, visible_gpus=visible_gpus)

    prime = ready.get("prime")
    if not isinstance(prime, dict) or prime.get("revision") != PRIME_COMMIT:
        raise PreparedLaunchError("Prime identity is not pinned")
    prime_root = _path(prime.get("root"), "Prime root", directory=True)
    prime_python = _runtime_executable(prime.get("python"), "Prime Python", prime_root / ".venv")
    _same(prime_python, prime_root / ".venv/bin/python", "Prime Python")

    model = ready.get("model")
    if not isinstance(model, dict) or model.get("repo") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise PreparedLaunchError("base model identity is not pinned")
    model_dir = _path(model.get("path"), "base model", directory=True)
    if model_dir.name != MODEL_REVISION or sha256_file(_path(str(model_dir / "config.json"), "base model config")) != model.get("config_sha256"):
        raise PreparedLaunchError("base model bytes do not match the verified receipt")

    adapters = ready.get("adapters")
    if not isinstance(adapters, dict) or not isinstance(adapters.get("policy"), dict) or not isinstance(adapters.get("frozen_sft"), dict):
        raise PreparedLaunchError("bootstrap receipt lacks both adapter identities")
    policy_record = adapters["policy"]
    frozen_record = adapters["frozen_sft"]
    if (policy_record.get("repo"), policy_record.get("revision"), policy_record.get("prefix"), policy_record.get("sha256")) != (POLICY_REPO, POLICY_REVISION, POLICY_PREFIX, POLICY_SHA256):
        raise PreparedLaunchError("Flash12 adapter identity is not pinned")
    if (frozen_record.get("repo"), frozen_record.get("revision"), frozen_record.get("prefix"), frozen_record.get("sha256")) != (POLICY_REPO, FROZEN_REVISION, FROZEN_PREFIX, FROZEN_SHA256):
        raise PreparedLaunchError("frozen SFT adapter identity is not pinned")
    policy = _path(policy_record.get("path"), "Flash12 adapter", directory=True)
    frozen = _path(frozen_record.get("path"), "frozen SFT adapter", directory=True)
    if policy == frozen:
        raise PreparedLaunchError("policy and frozen adapters must be distinct")
    _weights(policy, POLICY_SHA256, "Flash12 adapter")
    _weights(frozen, FROZEN_SHA256, "frozen SFT adapter")

    renderer, renderer_python, browser_root, renderer_root = _validate_renderer(ready)
    materialized = _path(ready.get("materialized_root"), "materialized root", directory=True)
    materialization = _read_json(materialized / "materialization-receipt.json", "materialization receipt")
    counts = materialization.get("counts")
    if materialization.get("status") != "materialized_with_verified_frozen_sft512_anchors" or not isinstance(counts, dict) or counts.get("train_rows") != 96 or counts.get("heldout_rows") != 32:
        raise PreparedLaunchError("materialized manifests are not the verified 96/32 handoff")
    _path(str(materialized / "train-manifest.json"), "materialized train manifest")
    _path(str(materialized / "heldout-manifest.json"), "materialized heldout manifest")

    prepared = _path(ready.get("prepared_root"), "prepared node root", directory=True)
    prepare = _read_json(prepared / "prepare-receipt.json", "node preparation receipt")
    canary = _read_json(prepared / "canary-receipt.json", "canary receipt")
    if prepare.get("run_id") != RUN_ID or prepare.get("status") != "prepared_no_process_started":
        raise PreparedLaunchError("node preparation receipt is not the staged preparation")
    if canary.get("status") != "passed_import_hook_and_renderer" or canary.get("orchestrator_setup_called") is not False or canary.get("training_started") is not False or canary.get("processes_started") is not False:
        raise PreparedLaunchError("canary did not pass without starting Prime")
    renderer_probe = canary.get("renderer_probe")
    if not isinstance(renderer_probe, dict) or renderer_probe.get("status") != "verified" or renderer_probe.get("valid") is not True or renderer_probe.get("renderer_sha256") != RENDERER_SHA256:
        raise PreparedLaunchError("verified renderer canary is required")
    trainer_validation = canary.get("trainer_dependency_validation")
    if not isinstance(trainer_validation, dict) or trainer_validation.get("status") != "passed" or trainer_validation.get("model_profile") != "qwen38_27b":
        raise PreparedLaunchError("reviewed Prime trainer dependency smoke is required")
    prepared_renderer = prepare.get("renderer")
    if not isinstance(prepared_renderer, dict) or prepared_renderer.get("root") != str(renderer_root) or prepared_renderer.get("path") != str(renderer) or prepared_renderer.get("python") != str(renderer_python) or prepared_renderer.get("sha256") != RENDERER_SHA256:
        raise PreparedLaunchError("prepared renderer receipt does not match the bootstrap pin")
    if prepare.get("prime_revision") != PRIME_COMMIT or prepare.get("prime_root") != str(prime_root):
        raise PreparedLaunchError("prepared Prime receipt does not match the bootstrap pin")
    prepared_policy = prepare.get("policy_initializer")
    prepared_frozen = prepare.get("frozen_sft")
    if not isinstance(prepared_policy, dict) or not isinstance(prepared_frozen, dict) or prepared_policy.get("path") != str(policy) or prepared_frozen.get("path") != str(frozen) or prepared_policy.get("weights_sha256") != POLICY_SHA256 or prepared_frozen.get("weights_sha256") != FROZEN_SHA256:
        raise PreparedLaunchError("prepared adapter receipt does not match the bootstrap pin")
    config_path = _path(str(prepared / "native.toml"), "prepared Prime config")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PreparedLaunchError("prepared Prime config is not valid TOML") from exc
    deployment = config.get("deployment", {})
    if not isinstance(deployment, dict) or {key: int(deployment.get(key, 0)) for key in ("gpus_per_node", "num_train_gpus", "num_infer_gpus")} != topology:
        raise PreparedLaunchError("prepared Prime topology does not match the bootstrap receipt")
    orchestrator = config.get("orchestrator", {})
    if config.get("max_steps") != 20 or not isinstance(orchestrator, dict) or orchestrator.get("batch_size") != 24 or orchestrator.get("group_size") != 4 or orchestrator.get("constant_trainer_batch_size") is not True:
        raise PreparedLaunchError("prepared Prime config is not the finite constant-batch contract")
    _path(str(RUN_ROOT / "launch.sh"), "staged launcher", executable=True)

    return {
        "ready": ready, "topology": topology, "repo_root": repo, "source_root": source,
        "brush_root": brush_root, "prime_root": prime_root, "prime_python": prime_python,
        "model_dir": model_dir, "policy": policy, "frozen": frozen,
        "renderer": renderer, "renderer_python": renderer_python, "browser_root": browser_root, "renderer_root": renderer_root,
        "materialized": materialized, "prepared": prepared,
    }


def build_launch_environment(*, verified: dict[str, Any], gateway_env: Path, timeout_seconds: int) -> dict[str, str]:
    if timeout_seconds < 1 or timeout_seconds > 604800:
        raise PreparedLaunchError("timeout seconds must be in 1..604800")
    gateway_env = validate_gateway_file(gateway_env)
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"HF_TOKEN", "HF_HUB_TOKEN", "OPENAI_API_KEY", "WANDB_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "PAINTER_PRIME_ROLE"}
    }
    env.update({
        "PAINTER_REPO_ROOT": str(verified["repo_root"]),
        "PRIME_ROOT": str(verified["prime_root"]),
        "PAINTER_PRIME_PYTHON": str(verified["prime_python"]),
        "PAINTER_MATERIALIZED_ROOT": str(verified["materialized"]),
        "PAINTER_RL_CONFIG": str(verified["prepared"] / "native.toml"),
        "PAINTER_RL_MANIFEST": str(verified["materialized"] / "train-manifest.json"),
        "PAINTER_EVAL_MANIFEST": str(verified["materialized"] / "heldout-manifest.json"),
        "MODEL_DIR": str(verified["model_dir"]),
        "WATERCOLOUR_INIT_ADAPTER": str(verified["policy"]),
        "PAINTER_FROZEN_SFT_ADAPTER": str(verified["frozen"]),
        "PAINTER_RENDERER_PATH": str(verified["renderer"]),
        "PAINTER_RENDERER_PYTHON_PATH": str(verified["renderer_python"]),
        "PAINTER_RENDERER_ROOT": str(verified["renderer_root"]),
        "PAINTER_BROWSER_ROOT": str(verified["browser_root"]),
        "BRUSH_ROOT": str(verified["brush_root"]),
        "PAINTER_GATEWAY_ENV_FILE": str(gateway_env),
        "PAINTER_RUN_TIMEOUT_SECONDS": str(timeout_seconds),
        "PAINTER_PRIME_RUN_WRAPPER": str(RUN_ROOT / "run_from_sft.py"),
        "PAINTER_PRIME_TRAIN_WRAPPER": str(RUN_ROOT / "train_from_sft.py"),
        "PAINTER_NUM_TRAIN_GPUS": "4",
    })
    return env


def invoke(*, verified: dict[str, Any], gateway_env: Path, timeout_seconds: int, runner: Any = subprocess.run) -> int:
    env = build_launch_environment(verified=verified, gateway_env=gateway_env, timeout_seconds=timeout_seconds)
    completed = runner(["bash", str(RUN_ROOT / "launch.sh"), str(verified["prepared"])], env=env, check=False)
    return int(completed.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--gateway-env-file", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if platform.system() != "Linux":
            raise PreparedLaunchError("prepared staged launch requires Linux")
        visible = BOOTSTRAP.visible_gpu_count()
        node_root = _path(str(args.node_root), "node root", directory=True)
        verified = validate_ready(ready_path=node_root / "bootstrap-ready.json", node_root=node_root, visible_gpus=visible)
        return invoke(verified=verified, gateway_env=args.gateway_env_file, timeout_seconds=args.timeout_seconds)
    except (PreparedLaunchError, ValueError) as exc:
        print(f"prepared launch failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
