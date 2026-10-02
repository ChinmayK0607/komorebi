"""Write the finite fixed-anchor text RL config from the pinned Prime profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PHOTO = Path(__file__).resolve().parents[1] / "photo-curriculum-rl-20260922" / "prepare_rl_config.py"
spec = importlib.util.spec_from_file_location("painter_photo_rl_config", PHOTO)
assert spec and spec.loader
photo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = photo
spec.loader.exec_module(photo)

RUN_NAME = "text-pairwise-rl-20260928"
SFT_REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
SFT_PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
SFT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
DEFAULT_GPUS_PER_NODE = 2
DEFAULT_NUM_TRAIN_GPUS = 1
DEFAULT_NUM_INFER_GPUS = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_GROUP_SIZE = 4
DEFAULT_INITIAL_INFLIGHT = 8
DEFAULT_MAX_INFLIGHT = 8
SUPPORTED_JUDGE_PROTOCOLS = {
    "text-paint-sft-pairwise-v1",
    "text-paint-sft-pairwise-margin-v1",
    "text-paint-sft-pairwise-grounded-v1",
    "text-paint-sft-pairwise-consistent-v1",
}


def _resolved_checkpoint_interval(value: int | None) -> int:
    """Resolve the opt-in environment override without changing old callers.

    An explicit function/CLI value wins.  The environment is intentionally
    consulted only when the caller omitted the value, so historical callers
    that pass ``4`` remain byte-for-byte on the old cadence.
    """
    if value is None:
        raw = os.environ.get("PAINTER_RL_CHECKPOINT_INTERVAL")
        value = 4 if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("checkpoint_interval must be a positive integer")
    try:
        resolved = int(value)
    except ValueError as exc:
        raise ValueError("checkpoint_interval must be a positive integer") from exc
    if resolved < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    return resolved


def prepare(*, output: Path, model: Path, run_root: Path, manifest: Path,
            prime_root: Path | None = None, max_steps: int = 20,
            run_name: str = RUN_NAME, output_root: str = "rl-rollouts",
            judge_protocol: str = "text-paint-sft-pairwise-v1",
            expected_count: int = 40, checkpoint_interval: int | None = None,
            adapter_revision: str = SFT_REVISION,
            adapter_prefix: str = SFT_PREFIX,
            adapter_sha256: str = SFT_SHA256,
            mode: str = "fresh", learning_rate: float = 1e-6,
            gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
            num_train_gpus: int = DEFAULT_NUM_TRAIN_GPUS,
            num_infer_gpus: int = DEFAULT_NUM_INFER_GPUS,
            batch_size: int = DEFAULT_BATCH_SIZE,
            group_size: int = DEFAULT_GROUP_SIZE,
            initial_inflight: int = DEFAULT_INITIAL_INFLIGHT,
            max_inflight: int = DEFAULT_MAX_INFLIGHT,
            renderer: str | None = None,
            renderer_python: str | None = None,
            activation_cpu_offload: bool = False) -> dict:
    if not run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a simple non-empty name")
    if not output_root or "/" in output_root or "\\" in output_root:
        raise ValueError("output_root must be a simple non-empty name")
    if judge_protocol not in SUPPORTED_JUDGE_PROTOCOLS:
        raise ValueError("unsupported text judge protocol")
    if mode not in {"fresh", "recovered_policy"}:
        raise ValueError("unsupported policy initialization mode")
    checkpoint_interval = _resolved_checkpoint_interval(checkpoint_interval)
    if len(adapter_sha256) != 64 or any(c not in "0123456789abcdef" for c in adapter_sha256):
        raise ValueError("adapter_sha256 must be lowercase SHA-256")
    rows = json.loads(manifest.read_text())["rows"]
    if len(rows) != expected_count or len({row["id"] for row in rows}) != expected_count:
        raise ValueError(f"expected {expected_count} unique fixed-anchor training rows")
    if any(row.get("reference_kind") != "text" or not row.get("opponent_sha256") for row in rows):
        raise ValueError("every row must have a text brief and hashed fixed opponent")
    relative = str(manifest.resolve().relative_to(run_root.resolve()))
    config = photo.build_config(
        model=str(model), run_root=run_root, manifest=relative,
        max_steps=max_steps, checkpoint_interval=checkpoint_interval, run_name=run_name,
        expected_count=expected_count, gpus_per_node=gpus_per_node,
        num_train_gpus=num_train_gpus, num_infer_gpus=num_infer_gpus,
        batch_size=batch_size, group_size=group_size, max_turns=3,
        max_completion_tokens=8192, learning_rate=learning_rate,
        initial_inflight=initial_inflight, max_inflight=max_inflight,
        mode=mode, adapter_revision=adapter_revision,
        adapter_prefix=adapter_prefix, adapter_sha256=adapter_sha256,
        activation_cpu_offload=activation_cpu_offload,
    )
    source = config["orchestrator"]["train"]["source"][0]
    source["name"] = "text-paint-fixed-sft-pairwise"
    source["env"]["taskset"]["lazy_opponent"] = False
    # The frozen manifest is ordered simple -> intermediate. NativePaintingRLData
    # otherwise interleaves families alphabetically, starting with intermediate.
    source["env"]["taskset"]["preserve_manifest_order"] = True
    source["env"]["taskset"]["output_root"] = output_root
    if renderer is not None:
        source["env"]["taskset"]["renderer"] = renderer
    if renderer_python is not None:
        source["env"]["taskset"]["renderer_python"] = renderer_python
    source["env"]["agent"]["timeout"]["scoring"] = 600
    config["orchestrator"]["max_off_policy_steps"] = 1
    message = photo.write_config(output, config, validate_prime=prime_root is not None,
                                 prime_root=prime_root)
    receipt = {
        "run_name": run_name, "prime_revision": photo.PRIME_COMMIT,
        "model_revision": photo.BASE_MODEL_REVISION,
        "adapter_revision": adapter_revision,
        "adapter_prefix": adapter_prefix,
        "adapter_sha256": adapter_sha256,
        "mode": mode, "optimizer_resume": False,
        "adapter_checkpoint_mode_required": "adapter_only_post_update",
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_keep_last": config["ckpt"]["keep_last"],
        "batch_size": config["orchestrator"]["batch_size"],
        "group_size": config["orchestrator"]["group_size"],
        "initial_inflight": config["orchestrator"]["concurrency"]["initial_inflight"],
        "max_inflight": config["orchestrator"]["concurrency"]["max_inflight"],
        "max_off_policy_steps": config["orchestrator"]["max_off_policy_steps"],
        "activation_cpu_offload": activation_cpu_offload,
        "activation_offloading": config["trainer"]["model"]["ac_offloading"],
        "frozen_anchor": {"alias": "sft-step512", "immutable": True,
                          "checkpoint_sha256": SFT_SHA256},
        "deployment": config["deployment"], "manifest": relative,
        "reward": judge_protocol, "prime_validation": message,
    }
    output.with_name("rl-config-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--output-root", default="rl-rollouts")
    parser.add_argument("--judge-protocol", default="text-paint-sft-pairwise-v1")
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--adapter-revision", default=SFT_REVISION)
    parser.add_argument("--adapter-prefix", default=SFT_PREFIX)
    parser.add_argument("--adapter-sha256", default=SFT_SHA256)
    parser.add_argument("--mode", choices=("fresh", "recovered_policy"), default="fresh")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--num-train-gpus", type=int, default=DEFAULT_NUM_TRAIN_GPUS)
    parser.add_argument("--num-infer-gpus", type=int, default=DEFAULT_NUM_INFER_GPUS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--initial-inflight", type=int, default=DEFAULT_INITIAL_INFLIGHT)
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT)
    parser.add_argument("--renderer")
    parser.add_argument("--renderer-python")
    parser.add_argument("--activation-cpu-offload", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
