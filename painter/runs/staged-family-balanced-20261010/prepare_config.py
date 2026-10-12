#!/usr/bin/env python3
"""Write the finite staged-family-balanced Prime configuration.

This generator is deliberately separate from the tracked preparation JSON:
the node-specific TOML contains only explicit local paths supplied by the
operator.  It does not download models, inspect CUDA, start Prime, or read
credentials.  Prime validation is an explicit later canary step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.prepare_scaled_native_rl import build_config as _build_base_config
from painter.prepare_scaled_native_rl import render_toml
from painter.prepare_scaled_native_rl import TARGET_MODULES


RUN_ID = "staged-family-balanced-20261010"
PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
BASE_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
POLICY = {
    "repo": "CK0607/qwen3.8-27b-brush-painting",
    "revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c",
    "prefix": "rl-judge-matched-20261004-flash/step_12",
    "sha256": "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79",
}
FROZEN_SFT = {
    "alias": "sft-step512",
    "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
}
CURRICULUM_SHA256 = "38ccfb8390730d311134b7fdddf4e4c89cd9605563d29d069e55f6b59818ea7b"
HELDOUT_PANEL_SHA256 = "94be432eaff2f26443f77c9a5d668c37c92f7e873b6c9461854d6ed2aa09ca46"
BATCH_SIZE = 24
GROUP_SIZE = 4
MAX_STEPS = 20
CHECKPOINT_INTERVAL = 4
MAX_TURNS = 3
EXPECTED_TRAIN_ROWS = 96
SUPPORTED_TOPOLOGIES = frozenset({
    (2, 1, 1),
    (4, 2, 2),
    (8, 4, 4),
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file: {path}")
    return path.resolve(strict=True)


def _runtime_executable(path: Path, description: str, runtime_root: Path) -> Path:
    """Validate a venv interpreter link while preserving its venv path."""

    path = path.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    try:
        path.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(f"{description} is outside its runtime root: {path}") from exc
    if not path.exists() and not path.is_symlink():
        raise ValueError(f"{description} is missing: {path}")
    target = path.resolve(strict=True)
    if not target.is_file() or (target.stat().st_mode & 0o111) == 0:
        raise ValueError(f"{description} does not resolve to an executable file: {path}")
    if path.is_symlink():
        allowed = [runtime_root.resolve(strict=True)]
        allowed_targets: list[Path] = []
        pyvenv = runtime_root / "pyvenv.cfg"
        if pyvenv.is_file() and not pyvenv.is_symlink():
            for line in pyvenv.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("home ="):
                    home = Path(line.split("=", 1)[1].strip()).expanduser()
                    if home.is_absolute():
                        allowed.append(home.resolve(strict=True))
                elif line.strip().lower().startswith("executable ="):
                    executable = Path(line.split("=", 1)[1].strip()).expanduser()
                    if executable.is_absolute():
                        allowed_targets.append(executable.resolve(strict=True))
        if target not in allowed_targets and not any(target == root or root in target.parents for root in allowed):
            raise ValueError(f"{description} resolves outside its trusted interpreter roots: {target}")
        current = target
        while True:
            if current.stat().st_mode & 0o022:
                raise ValueError(f"{description} target is writable by group/other: {current}")
            if current.parent == current:
                break
            current = current.parent
    return path


def _directory(path: Path, description: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be a regular directory: {path}")
    return path.resolve(strict=True)


def validate_topology(*, gpus_per_node: int, num_train_gpus: int, num_infer_gpus: int) -> None:
    if (gpus_per_node, num_train_gpus, num_infer_gpus) not in SUPPORTED_TOPOLOGIES:
        raise ValueError("staged topology must be exactly 1+1, 2+2, or 4+4 GPUs")


def build_config(
    *,
    model: Path,
    node_root: Path,
    materialized_root: Path,
    repo_root: Path,
    renderer: Path | None = None,
    renderer_python: Path | None = None,
    gpus_per_node: int = 4,
    num_train_gpus: int = 2,
    num_infer_gpus: int = 2,
) -> dict[str, Any]:
    """Build the staged config with the finite 20-update/24-trace contract."""

    validate_topology(
        gpus_per_node=gpus_per_node,
        num_train_gpus=num_train_gpus,
        num_infer_gpus=num_infer_gpus,
    )
    node_root = node_root.expanduser().resolve()
    materialized_root = _directory(materialized_root, "materialized root")
    repo_root = _directory(repo_root, "repository root")
    model = model.expanduser().resolve()
    if not model.is_dir() or model.name != BASE_MODEL_REVISION:
        raise ValueError(f"model directory must end in the pinned revision {BASE_MODEL_REVISION}")
    if renderer is not None:
        renderer = _regular_file(renderer, "renderer")
    if renderer_python is not None:
        renderer_python = _runtime_executable(renderer_python, "renderer Python", renderer_python.parent.parent)

    config = _build_base_config(
        model=str(model),
        run_root=node_root,
        gpus_per_node=gpus_per_node,
        num_train_gpus=num_train_gpus,
        num_infer_gpus=num_infer_gpus,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        run_name=RUN_ID,
    )
    config["max_steps"] = MAX_STEPS
    config["output_dir"] = str((node_root / "async-output").resolve())
    config["run"]["name"] = RUN_ID
    config["trainer"]["max_steps"] = MAX_STEPS
    config["orchestrator"].update({
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "constant_trainer_batch_size": True,
        "max_steps": MAX_STEPS,
        "max_off_policy_steps": 1,
    })
    config["orchestrator"]["concurrency"].update({
        "min_inflight": BATCH_SIZE,
        "initial_inflight": BATCH_SIZE,
        "max_inflight": BATCH_SIZE,
    })
    config["inference"]["vllm"]["max_num_seqs"] = BATCH_SIZE
    config["inference"]["vllm"]["tensor_parallel_size"] = num_infer_gpus
    config["inference"]["vllm"]["data_parallel_size"] = 1

    source = config["orchestrator"]["train"]["source"][0]
    source["name"] = "staged-family-balanced"
    source["env"]["timeout"]["episode"] = 3600
    source["env"]["agent"]["timeout"]["scoring"] = 600
    taskset = source["env"]["taskset"]
    taskset.update({
        "id": "native_painting_rl",
        "root": str(materialized_root),
        "manifest": "train-manifest.json",
        "expected_count": EXPECTED_TRAIN_ROWS,
        "max_turns": MAX_TURNS,
        "preserve_manifest_order": True,
        "lazy_opponent": False,
        "output_root": "rl-rollouts",
        "judge_module_dir": str((repo_root / "painter").resolve()),
    })
    if renderer is not None:
        taskset["renderer"] = str(renderer)
    if renderer_python is not None:
        taskset["renderer_python"] = str(renderer_python)
    return config


def validate_config(config: dict[str, Any], *, node_root: Path, materialized_root: Path) -> None:
    """Fail closed on the fields that control the finite staged contract."""

    if config.get("max_steps") != MAX_STEPS or config.get("run", {}).get("name") != RUN_ID:
        raise ValueError("config is not the finite staged run")
    if config.get("trainer", {}).get("max_steps") != MAX_STEPS:
        raise ValueError("trainer max_steps must be 20")
    orchestrator = config.get("orchestrator", {})
    if orchestrator.get("batch_size") != BATCH_SIZE or orchestrator.get("group_size") != GROUP_SIZE:
        raise ValueError("Prime batch/group shape must be 24/4")
    if orchestrator.get("constant_trainer_batch_size") is not True:
        raise ValueError("Prime must require a full constant trainer batch")
    concurrency = orchestrator.get("concurrency", {})
    if any(concurrency.get(key) != BATCH_SIZE for key in ("min_inflight", "initial_inflight", "max_inflight")):
        raise ValueError("Prime in-flight concurrency must be 24")
    source = (orchestrator.get("train", {}).get("source") or [None])[0]
    taskset = ((source or {}).get("env", {}).get("taskset", {}))
    if taskset.get("manifest") != "train-manifest.json":
        raise ValueError("taskset must use the materialized train-manifest.json")
    if Path(str(taskset.get("root", ""))).resolve() != materialized_root.resolve():
        raise ValueError("taskset root does not match the materialized root")
    if taskset.get("expected_count") != EXPECTED_TRAIN_ROWS or taskset.get("max_turns") != MAX_TURNS:
        raise ValueError("taskset count/turn scope is not staged")
    if taskset.get("preserve_manifest_order") is not True or taskset.get("lazy_opponent") is not False:
        raise ValueError("taskset must preserve staged order and use fixed anchors")
    if config.get("output_dir") and not Path(str(config["output_dir"])).resolve().is_relative_to(node_root.resolve()):
        raise ValueError("Prime output_dir must remain under the node run root")


def write_config(
    *,
    output: Path,
    receipt_output: Path,
    config: dict[str, Any],
    node_root: Path,
    materialized_root: Path,
) -> dict[str, Any]:
    output = output.expanduser().absolute()
    receipt_output = receipt_output.expanduser().absolute()
    if output.exists() or receipt_output.exists():
        raise FileExistsError("refusing to overwrite a prepared Prime config or receipt")
    validate_config(config, node_root=node_root, materialized_root=materialized_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(render_toml(config), encoding="utf-8")
    temporary.replace(output)
    receipt = {
        "schema": "staged-family-balanced-prime-config-receipt-v1",
        "run_id": RUN_ID,
        "prime_revision": PRIME_COMMIT,
        "base_model_revision": BASE_MODEL_REVISION,
        "policy_initializer": POLICY,
        "frozen_sft": FROZEN_SFT,
        "curriculum_manifest_sha256": CURRICULUM_SHA256,
        "heldout_panel_sha256": HELDOUT_PANEL_SHA256,
        "optimizer_updates": MAX_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "max_turns": MAX_TURNS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "materialized_root": str(materialized_root),
        "config_path": str(output),
        "config_sha256": sha256_file(output),
        "prime_validation": "deferred_to_canary",
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = receipt_output.with_name(receipt_output.name + ".tmp")
    receipt_tmp.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    receipt_tmp.replace(receipt_output)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-train-gpus", type=int, default=2)
    parser.add_argument("--num-infer-gpus", type=int, default=2)
    args = parser.parse_args(argv)
    config = build_config(
        model=args.model,
        node_root=args.node_root,
        materialized_root=args.materialized_root,
        repo_root=args.repo_root,
        renderer=args.renderer,
        renderer_python=args.renderer_python,
        gpus_per_node=args.gpus_per_node,
        num_train_gpus=args.num_train_gpus,
        num_infer_gpus=args.num_infer_gpus,
    )
    receipt = write_config(
        output=args.output,
        receipt_output=args.receipt_output,
        config=config,
        node_root=args.node_root,
        materialized_root=args.materialized_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
