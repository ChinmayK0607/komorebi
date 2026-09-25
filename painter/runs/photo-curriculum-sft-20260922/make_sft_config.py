"""Write a finite Prime SFT config with explicit hardware and token budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def q(value):
    return json.dumps(value) if isinstance(value, str) else str(value).lower() if isinstance(value, bool) else str(value)


def config(args: argparse.Namespace) -> tuple[dict, str]:
    if args.num_train_gpus < 1 or args.gpus_per_node < args.num_train_gpus:
        raise ValueError("num_train_gpus must be positive and no larger than gpus_per_node")
    if args.micro_batch_size < 1 or args.global_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    denominator = args.num_train_gpus * args.micro_batch_size
    if args.global_batch_size % denominator:
        raise ValueError("global batch must divide evenly across DDP ranks and microbatch")
    if args.steps < 1 or args.seq_len < 8192:
        raise ValueError("steps must be positive and seq_len must be at least 8192")
    if not 0 < args.learning_rate <= 2e-5:
        raise ValueError("learning_rate must be in (0, 2e-5]")
    dp_replicate = args.dp_replicate if args.dp_replicate is not None else args.num_train_gpus
    dp_shard = args.dp_shard if args.dp_shard is not None else args.num_train_gpus // dp_replicate
    if dp_replicate < 1 or dp_shard < 1 or dp_replicate * dp_shard != args.num_train_gpus:
        raise ValueError("dp_replicate * dp_shard must equal num_train_gpus")
    # Pinned Prime derives dp_shard from world_size / model.dp_replicate;
    # expose the resolved knob while rejecting a misleading incompatible one.
    if dp_shard != args.num_train_gpus // dp_replicate:
        raise ValueError("pinned Prime derives dp_shard=num_train_gpus/dp_replicate")
    root = args.data.resolve()
    model = args.model_path.resolve()
    run = {
        "max_steps": args.steps,
        "run": {"name": args.run_name},
        "deployment": {"gpus_per_node": args.gpus_per_node, "num_train_gpus": args.num_train_gpus},
        "model": {
            "name": str(model), "impl": "custom", "compile": "None", "attn": "auto",
            "optimization_dtype": "bfloat16", "reduce_dtype": "bfloat16",
            "dp_replicate": dp_replicate,
            "vlm": {"vision_encoder_attr": "model.visual", "language_model_attr": "model.language_model", "freeze_vision_encoder": True},
            "lora": {"rank": 16, "alpha": 32, "target_modules": TARGET_MODULES},
            "ac": {"freq": 1}, "ac_offloading": "None", "optim_cpu_offload": False,
        },
        "renderer": {"name": "qwen3.5", "enable_thinking": False},
        "data": {
            "type": "sft", "name": str(root), "batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size, "seq_len": args.seq_len,
            "num_workers": 1, "seed": args.seed, "splits": ["train"], "shuffle": args.shuffle,
            "loss_mask": {"system": False, "user": False, "assistant": True, "tool": False},
        },
        "val": {
            "interval": args.validation_interval, "eval_on_start": True,
            "data": {
                "type": "sft", "name": str(root), "batch_size": args.global_batch_size,
                "micro_batch_size": args.micro_batch_size, "seq_len": args.seq_len,
                "num_workers": 1, "seed": args.seed, "splits": ["validation"], "shuffle": False,
                "loss_mask": {"system": False, "user": False, "assistant": True, "tool": False},
            },
        },
        "optim": {"type": "adamw", "lr": args.learning_rate, "betas1": 0.9, "betas2": 0.999, "weight_decay": 0.01, "max_norm": 1.0},
        "scheduler": {"type": "constant"},
        "ckpt": {"interval": args.checkpoint_interval, "keep_last": 2},
        "monitors": {"file": {}},
    }
    if args.resume:
        resume = Path(args.resume).resolve()
        if not re.fullmatch(r"step_\d+", resume.name):
            raise ValueError("--resume must point to a Prime checkpoint directory named step_<N>")
        run["resume"] = {"dir": str(resume)}
    text = render(run)
    report = {
        "schema": 1, "run_name": args.run_name, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "data": str(root), "steps": args.steps, "seq_len": args.seq_len,
        "gpus_per_node": args.gpus_per_node, "num_train_gpus": args.num_train_gpus,
        "dp_replicate": dp_replicate, "dp_shard": dp_shard,
        "global_batch_size": args.global_batch_size, "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_per_rank": args.global_batch_size // denominator,
        "effective_global_batch_size": args.global_batch_size, "learning_rate": args.learning_rate,
        "checkpoint_interval": args.checkpoint_interval, "validation_interval": args.validation_interval,
        "assistant_only_loss": True, "long_target_policy": "fail_closed_no_truncation",
        "optimizer": "fresh" if not args.resume else "resumed_from_explicit_path",
        "resume_supported": True,
        "resume_policy": "native_full_checkpoint_only on the retained local node; public adapter exports are the durable artifact",
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return report, text


def render(value: dict) -> str:
    lines = ["# Generated by make_sft_config.py; pinned model revision is recorded in the setup receipt."]

    def table(name: str, mapping: dict) -> None:
        lines.extend(["", f"[{name}]"])
        for key, item in mapping.items():
            if isinstance(item, dict):
                continue
            if isinstance(item, list):
                lines.append(f"{key} = [" + ", ".join(q(x) for x in item) + "]")
            else:
                lines.append(f"{key} = {q(item)}")

    for key in ("max_steps",):
        lines.append(f"{key} = {q(value[key])}")
    if "resume" in value:
        resume = value["resume"]
        lines.append(f"resume = {{ dir = {q(resume['dir'])} }}")
    for name in ("run", "deployment", "model", "renderer", "data", "val", "optim", "scheduler", "ckpt"):
        table(name, value[name])
        if name == "model":
            table("model.vlm", value[name]["vlm"]); table("model.lora", value[name]["lora"]); table("model.ac", value[name]["ac"])
        if name == "data": table("data.loss_mask", value[name]["loss_mask"])
        if name == "val": table("val.data", value[name]["data"]); table("val.data.loss_mask", value[name]["data"]["loss_mask"])
    lines.extend(["", "[monitors.file]"])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", default="photo-curriculum-sft-20260922")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--gpus-per-node", type=int, default=2)
    p.add_argument("--num-train-gpus", type=int, default=1)
    p.add_argument("--dp-replicate", type=int, default=None,
                   help="Prime model weight replication degree; defaults to num_train_gpus when fitting")
    p.add_argument("--dp-shard", type=int, default=None,
                   help="Resolved Prime FSDP shard degree; must equal num_train_gpus/dp_replicate")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--checkpoint-interval", type=int, default=64)
    p.add_argument("--validation-interval", type=int, default=64)
    p.add_argument("--seed", type=int, default=92022)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--resume")
    args = p.parse_args()
    report, text = config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    args.output.with_suffix(".setup.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
