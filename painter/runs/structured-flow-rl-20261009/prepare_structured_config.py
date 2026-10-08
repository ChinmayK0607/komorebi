#!/usr/bin/env python3
"""Build the finite structured-flow Prime config without changing shared code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "prepare_rl_config.py"
spec = importlib.util.spec_from_file_location("structured_photo_config", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load reviewed config builder: {SOURCE}")
photo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = photo
spec.loader.exec_module(photo)

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    *,
    output: Path,
    model: Path,
    run_root: Path,
    manifest: Path,
    prime_root: Path | None = None,
    expected_count: int = 32,
    max_steps: int = 8,
    checkpoint_interval: int = 4,
    run_name: str = "structured-flow-rl-20261009",
    output_root: str = "rl-rollouts",
    max_turns: int = 3,
    judge_protocol: str = "text-paint-sft-pairwise-consistent-v1",
    adapter_revision: str = POLICY["revision"],
    adapter_prefix: str = POLICY["prefix"],
    adapter_sha256: str = POLICY["sha256"],
    mode: str = "recovered_policy",
    learning_rate: float = 1e-6,
    gpus_per_node: int = 2,
    num_train_gpus: int = 1,
    num_infer_gpus: int = 1,
    batch_size: int = 16,
    group_size: int = 4,
    initial_inflight: int = 32,
    max_inflight: int = 32,
    renderer: str | None = None,
    renderer_python: str | None = None,
    activation_cpu_offload: bool = True,
    flow_policy_enabled: bool = False,
    code_supervision_enabled: bool = False,
    allow_pending_anchors: bool = False,
) -> dict[str, Any]:
    if run_name != "structured-flow-rl-20261009":
        raise ValueError("run_name is fixed for this experiment")
    if expected_count < 1 or max_steps != 8 or checkpoint_interval != 4:
        raise ValueError("structured pilot requires 8 updates, 4-update checkpoints, and a nonempty manifest")
    if max_turns != 3:
        raise ValueError("bounded structured-flow pilot uses the three-turn native scope")
    if judge_protocol != "text-paint-sft-pairwise-consistent-v1":
        raise ValueError("structured flow must use the existing consistent pairwise protocol")
    if adapter_revision != POLICY["revision"] or adapter_prefix != POLICY["prefix"] or adapter_sha256 != POLICY["sha256"]:
        raise ValueError("Flash12 policy initializer binding changed")
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("training manifest must be a regular file")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} admitted training rows")
    if len({row.get("id") for row in rows}) != expected_count:
        raise ValueError("training IDs must be unique")
    for row in rows:
        if row.get("reference_kind") != "text":
            raise ValueError("every row requires text input")
        if allow_pending_anchors:
            if not isinstance(row.get("lazy_opponent"), dict):
                raise ValueError("pending-anchor config requires explicit lazy opponent identity")
        else:
            if not row.get("opponent_sha256"):
                raise ValueError("every row requires a verified frozen opponent hash")
            if row.get("anchor_status") != "verified_file":
                raise ValueError("training rows must carry verified SFT512 anchors")
    if len(adapter_sha256) != 64:
        raise ValueError("initializer SHA-256 is malformed")
    relative = str(manifest.resolve().relative_to(run_root.resolve()))
    config = photo.build_config(
        model=str(model), run_root=run_root, manifest=relative,
        max_steps=max_steps, checkpoint_interval=checkpoint_interval,
        run_name=run_name, expected_count=expected_count,
        gpus_per_node=gpus_per_node, num_train_gpus=num_train_gpus,
        num_infer_gpus=num_infer_gpus, batch_size=batch_size,
        group_size=group_size, max_turns=max_turns, max_completion_tokens=8192,
        learning_rate=learning_rate, initial_inflight=initial_inflight,
        max_inflight=max_inflight, mode=mode, adapter_revision=adapter_revision,
        adapter_prefix=adapter_prefix, adapter_sha256=adapter_sha256,
        activation_cpu_offload=activation_cpu_offload,
    )
    source = config["orchestrator"]["train"]["source"][0]
    source["name"] = "structured-flow-fixed-sft512"
    taskset = source["env"]["taskset"]
    taskset["lazy_opponent"] = allow_pending_anchors
    taskset["preserve_manifest_order"] = True
    taskset["output_root"] = output_root
    if renderer is not None:
        taskset["renderer"] = renderer
    if renderer_python is not None:
        taskset["renderer_python"] = renderer_python
    source["env"]["agent"]["timeout"]["scoring"] = 600
    config["orchestrator"]["max_off_policy_steps"] = 1
    message = photo.write_config(output, config, validate_prime=prime_root is not None, prime_root=prime_root)
    receipt = {
        "schema": "structured-flow-rl-config-receipt-v1",
        "run_name": run_name,
        "base_model_revision": photo.BASE_MODEL_REVISION,
        "policy_initializer": {**POLICY, "revision": adapter_revision, "prefix": adapter_prefix, "sha256": adapter_sha256},
        "frozen_sft": FROZEN_SFT,
        "prime_revision": photo.PRIME_COMMIT,
        "mode": mode,
        "optimizer_resume": False,
        "updates": max_steps,
        "checkpoint_steps": [4, 8],
        "checkpoint_interval": checkpoint_interval,
        "batch_size": batch_size,
        "group_size": group_size,
        "initial_inflight": initial_inflight,
        "max_inflight": max_inflight,
        "max_off_policy_steps": 1,
        "learning_rate": learning_rate,
        "max_turns": max_turns,
        "activation_cpu_offload": activation_cpu_offload,
        "deployment": config["deployment"],
        "manifest": relative,
        "manifest_sha256": _sha256(manifest),
        "reward": judge_protocol,
        "judge": {"model": "xiaomi/mimo-v2.6-flash", "effort": "medium", "format": "identity-joint-side-schema-v2-generic-v1", "rubric": "existing-generic-rubric", "new_reward_bonus": False},
        "curriculum": {"active_stage": 1, "pilot_rows": expected_count, "full_train_rows": 240, "heldout_rows": 60, "replay_fraction": 0.25},
        "anchor_bank": {"status": "pending_runtime_generation" if allow_pending_anchors else "verified", "generator": "structured-flow.anchor_bank:_batch_native_generation" if allow_pending_anchors else None},
        "structured_flow_policy": {"enabled": flow_policy_enabled, "default_off_env": "PAINTER_STRUCTURED_FLOW_POLICY", "style_or_code_reward": False},
        "code_supervision": {"enabled": code_supervision_enabled, "separate_from_rl_reward": True},
        "prime_validation": message,
    }
    output.with_name("rl-config-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--run-name", default="structured-flow-rl-20261009")
    parser.add_argument("--output-root", default="rl-rollouts")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--judge-protocol", default="text-paint-sft-pairwise-consistent-v1")
    parser.add_argument("--adapter-revision", default=POLICY["revision"])
    parser.add_argument("--adapter-prefix", default=POLICY["prefix"])
    parser.add_argument("--adapter-sha256", default=POLICY["sha256"])
    parser.add_argument("--mode", choices=("fresh", "recovered_policy"), default="recovered_policy")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gpus-per-node", type=int, default=2)
    parser.add_argument("--num-train-gpus", type=int, default=1)
    parser.add_argument("--num-infer-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--initial-inflight", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--renderer")
    parser.add_argument("--renderer-python")
    parser.add_argument("--no-activation-cpu-offload", action="store_true")
    parser.add_argument("--enable-flow-policy", action="store_true")
    parser.add_argument("--enable-code-supervision", action="store_true")
    parser.add_argument("--allow-pending-anchors", action="store_true")
    args = parser.parse_args(argv)
    raw = vars(args).copy()
    raw.update({
        "activation_cpu_offload": not raw.pop("no_activation_cpu_offload"),
        "flow_policy_enabled": raw.pop("enable_flow_policy"),
        "code_supervision_enabled": raw.pop("enable_code_supervision"),
        "allow_pending_anchors": raw.pop("allow_pending_anchors"),
    })
    result = prepare(**raw)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
