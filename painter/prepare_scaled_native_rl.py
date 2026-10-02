"""Generate the pinned Prime config for the expanded native painting RL run.

The generator has no Prime, CUDA, or model dependency.  It writes a complete
TOML document from the values used by the scaled curriculum plan.  On the
Linux training node, ``--validate-prime`` additionally asks the installed
Prime revision to parse the same in-memory config before the file is written.

This preparation step never starts Prime, a renderer, a judge, or a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
RUN_NAME = "scaled-curriculum-20260921-native-rl"
TASKSET_ID = "native_painting_rl"
MAX_STEPS = 128
BATCH_SIZE = 32
GROUP_SIZE = 8
MAX_TURNS = 6
SEQ_LEN = 32_768
MAX_COMPLETION_TOKENS = 8_192
TEMPERATURE = 1.0
LEARNING_RATE = 1e-6
LORA_RANK = 16
LORA_ALPHA = 32
DEFAULT_GPUS_PER_NODE = 2
DEFAULT_NUM_TRAIN_GPUS = 1
DEFAULT_NUM_INFER_GPUS = 1
DEFAULT_CHECKPOINT_INTERVAL = 16
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_config(
    *,
    model: str,
    run_root: Path,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    num_train_gpus: int = DEFAULT_NUM_TRAIN_GPUS,
    num_infer_gpus: int = DEFAULT_NUM_INFER_GPUS,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    run_name: str = RUN_NAME,
) -> dict[str, Any]:
    """Return the complete native Prime RL config as plain Python values.

    ``gpus_per_node`` describes the available node capacity. The two role
    counts are explicit (defaulting to one trainer and one inference GPU); no
    physical device id is placed in this config.
    """

    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty model path or identifier")
    if not isinstance(run_name, str) or not run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a simple non-empty name")
    gpus_per_node = _positive_int("gpus_per_node", gpus_per_node)
    num_train_gpus = _positive_int("num_train_gpus", num_train_gpus)
    num_infer_gpus = _positive_int("num_infer_gpus", num_infer_gpus)
    if num_train_gpus + num_infer_gpus > gpus_per_node:
        raise ValueError(
            "num_train_gpus + num_infer_gpus must not exceed gpus_per_node "
            f"({num_train_gpus} + {num_infer_gpus} > {gpus_per_node})"
        )
    checkpoint_interval = _positive_int("checkpoint_interval", checkpoint_interval)

    root = str(run_root.expanduser().resolve())
    return {
        "max_steps": MAX_STEPS,
        "seq_len": SEQ_LEN,
        "output_dir": str((run_root / "async-output").expanduser().resolve()),
        "run": {"name": run_name},
        "deployment": {
            "type": "single_node",
            "gpus_per_node": gpus_per_node,
            "num_train_gpus": num_train_gpus,
            "num_infer_gpus": num_infer_gpus,
        },
        "model": {
            "name": model,
            "vlm": {
                "vision_encoder_attr": "model.visual",
                "language_model_attr": "model.language_model",
                "freeze_vision_encoder": True,
            },
        },
        "weight_broadcast": {"type": "filesystem", "timeout": 1200},
        "ckpt": {"interval": checkpoint_interval, "keep_last": 1},
        "monitors": {"file": {}},
        "trainer": {
            "max_steps": MAX_STEPS,
            "model": {
                "seq_len": SEQ_LEN,
                "impl": "custom",
                "compile": "None",
                "attn": "auto",
                "optimization_dtype": "bfloat16",
                "reduce_dtype": "bfloat16",
                "optim_cpu_offload": False,
                "lora": {
                    "rank": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "target_modules": TARGET_MODULES,
                },
                "ac": {"freq": 1},
                "ac_offloading": "None",
            },
            "optim": {
                "type": "adamw",
                "lr": LEARNING_RATE,
                "weight_decay": 0.01,
                "max_norm": 1.0,
            },
            "scheduler": {"type": "constant"},
            "loss": {"type": "ipo", "eps": 0.1, "adv_tau": 1.0, "kl_tau": 0.001},
        },
        "orchestrator": {
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "seq_len": SEQ_LEN,
            "max_steps": MAX_STEPS,
            "max_off_policy_steps": 1,
            "renderer": {"name": "qwen3.5", "enable_thinking": False},
            "concurrency": {
                "min_inflight": BATCH_SIZE,
                "initial_inflight": BATCH_SIZE,
                "max_inflight": BATCH_SIZE,
            },
            "train": {
                "sampling": {
                    "temperature": TEMPERATURE,
                    "top_p": 1.0,
                    "max_completion_tokens": MAX_COMPLETION_TOKENS,
                },
                "source": [
                    {
                        "name": "painter-native-rl",
                        "env": {
                            "id": TASKSET_ID,
                            "taskset": {
                                "id": TASKSET_ID,
                                "root": root,
                                "manifest": "scaled-training-manifest.json",
                                "max_turns": MAX_TURNS,
                                "policy_label": "training",
                                "output_root": "rl-rollouts",
                            },
                            "agent": {
                                "harness": {"id": "null"},
                                "runtime": {"type": "subprocess"},
                                "timeout": {"rollout": 1200, "scoring": 2100},
                            },
                            "timeout": {"episode": 3600},
                        },
                    }
                ],
            },
        },
        "inference": {
            "vllm": {
                "dtype": "bfloat16",
                "max_model_len": SEQ_LEN,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": BATCH_SIZE,
                "max_num_batched_tokens": SEQ_LEN,
                "limit_mm_per_prompt": {"image": 14, "video": 0},
                "gdn_prefill_backend": "triton",
                "kernel_config": {"enable_jit_warmup": False},
                "mm_processor_kwargs": {"min_pixels": 3136, "max_pixels": 1048576},
            }
        },
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        if not all(not isinstance(item, (dict, list)) for item in value):
            raise TypeError("nested TOML arrays are not supported by this writer")
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML scalar: {type(value).__name__}")


def render_toml(config: dict[str, Any]) -> str:
    """Render this fixed-shape config without requiring a TOML writer package."""

    lines: list[str] = [
        "# Generated by painter/prepare_scaled_native_rl.py; preparation only.",
        f"# Prime revision: {PRIME_COMMIT}",
    ]

    def table(path: str, values: dict[str, Any]) -> None:
        lines.append("")
        lines.append(f"[{path}]")
        for key, value in values.items():
            if not isinstance(value, dict) and not (
                isinstance(value, list) and value and isinstance(value[0], dict)
            ):
                lines.append(f"{key} = {_toml_value(value)}")

    # This ordering mirrors the prior pinned aesthetic-rl TOML and keeps diffs
    # legible when a node regenerates the concrete path-bearing config.
    for key in ("max_steps", "seq_len", "output_dir"):
        lines.append(f"{key} = {_toml_value(config[key])}")
    table("run", config["run"])
    table("deployment", config["deployment"])
    table("model", config["model"])
    table("model.vlm", config["model"]["vlm"])
    table("weight_broadcast", config["weight_broadcast"])
    table("ckpt", config["ckpt"])
    lines.extend(["", "[monitors.file]"])
    table("trainer", {"max_steps": config["trainer"]["max_steps"]})
    table("trainer.model", config["trainer"]["model"])
    table("trainer.model.lora", config["trainer"]["model"]["lora"])
    table("trainer.model.ac", config["trainer"]["model"]["ac"])
    ac_offloading = config["trainer"]["model"].get("ac_offloading")
    if isinstance(ac_offloading, dict):
        table("trainer.model.ac_offloading", ac_offloading)
    table("trainer.optim", config["trainer"]["optim"])
    table("trainer.scheduler", config["trainer"]["scheduler"])
    table("trainer.loss", config["trainer"]["loss"])
    table("orchestrator", config["orchestrator"])
    table("orchestrator.renderer", config["orchestrator"]["renderer"])
    table("orchestrator.concurrency", config["orchestrator"]["concurrency"])
    table("orchestrator.train.sampling", config["orchestrator"]["train"]["sampling"])
    source = config["orchestrator"]["train"]["source"][0]
    lines.extend(["", "[[orchestrator.train.source]]", f"name = {_toml_value(source['name'])}"])
    env = source["env"]
    table("orchestrator.train.source.env", env)
    table("orchestrator.train.source.env.taskset", env["taskset"])
    table("orchestrator.train.source.env.agent", env["agent"])
    table("orchestrator.train.source.env.agent.harness", env["agent"]["harness"])
    table("orchestrator.train.source.env.agent.runtime", env["agent"]["runtime"])
    table("orchestrator.train.source.env.agent.timeout", env["agent"]["timeout"])
    table("orchestrator.train.source.env.timeout", env["timeout"])
    table("inference.vllm", config["inference"]["vllm"])
    table("inference.vllm.limit_mm_per_prompt", config["inference"]["vllm"]["limit_mm_per_prompt"])
    table("inference.vllm.kernel_config", config["inference"]["vllm"]["kernel_config"])
    table("inference.vllm.mm_processor_kwargs", config["inference"]["vllm"]["mm_processor_kwargs"])
    return "\n".join(lines) + "\n"


def _prime_paths(prime_root: Path) -> list[Path]:
    return [
        prime_root / "src",
        prime_root / "packages/prime-rl-configs/src",
        prime_root / "deps/pydantic-config/src",
    ]


def validate_with_prime(config: dict[str, Any], prime_root: Path | None = None) -> tuple[bool, str]:
    """Validate through pinned Prime when its Python package is importable."""

    if prime_root is not None:
        for path in reversed(_prime_paths(prime_root.expanduser().resolve())):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
    try:
        from prime_rl.configs.rl import RLConfig
    except Exception as exc:  # dependency availability is node-specific
        return False, f"Prime config import unavailable: {type(exc).__name__}: {exc}"
    try:
        RLConfig.model_validate(config)
    except Exception as exc:
        return False, f"Prime RLConfig rejected generated config: {type(exc).__name__}: {exc}"
    return True, "Prime RLConfig accepted generated config"


def write_config(
    output: Path,
    config: dict[str, Any],
    *,
    validate_prime: bool = False,
    prime_root: Path | None = None,
) -> tuple[bool, str | None]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing config: {output}")
    if validate_prime:
        valid, message = validate_with_prime(config, prime_root)
        if not valid:
            raise RuntimeError(message)
    else:
        valid, message = True, None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_toml(config))
    return valid, message


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new TOML path")
    parser.add_argument("--model", required=True, help="staged model directory or model identifier")
    parser.add_argument("--run-root", type=Path, required=True, help="node workspace containing the manifest")
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--num-train-gpus", type=int, default=DEFAULT_NUM_TRAIN_GPUS)
    parser.add_argument("--num-infer-gpus", type=int, default=DEFAULT_NUM_INFER_GPUS)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--prime-root", type=Path, help="pinned Prime checkout for optional validation")
    parser.add_argument("--validate-prime", action="store_true", help="require installed Prime to accept the config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = build_config(
        model=args.model,
        run_root=args.run_root,
        gpus_per_node=args.gpus_per_node,
        num_train_gpus=args.num_train_gpus,
        num_infer_gpus=args.num_infer_gpus,
        checkpoint_interval=args.checkpoint_interval,
        run_name=args.run_name,
    )
    _, message = write_config(
        args.output,
        config,
        validate_prime=args.validate_prime,
        prime_root=args.prime_root,
    )
    print(json.dumps({"output": str(args.output.resolve()), "prime_validation": message or "not requested"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
