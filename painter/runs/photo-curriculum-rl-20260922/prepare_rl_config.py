"""Prepare the finite photo-curriculum native Prime RL configuration.

This is a run-local specialization of the previously validated native Prime
configuration.  It performs no model, renderer, judge, or provider action.
The node launcher supplies the staged model, the reviewed RL manifest, and
the pinned Prime checkout before asking for optional Prime validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.prepare_scaled_native_rl import (  # noqa: E402
    DEFAULT_CHECKPOINT_INTERVAL,
    build_config as _build_native_config,
    render_toml,
    validate_with_prime,
)


RUN_NAME = "photo-curriculum-rl-20260922"
PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
BASE_MODEL_ID = "Qwen/Qwen3.8-27B"
BASE_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# Keep role counts explicit and operator-configurable. The historical two-GPU
# default remains conservative while a four-GPU node can be selected with
# --gpus-per-node 4 --num-train-gpus 1 --num-infer-gpus 3.
DEFAULT_GPUS_PER_NODE = 2
DEFAULT_NUM_TRAIN_GPUS = 1
DEFAULT_NUM_INFER_GPUS = 1
INITIAL_UPDATES = 32
MAX_UPDATES = 128
SEQ_LEN = 32_768
MAX_COMPLETION_TOKENS = 8_192
BATCH_SIZE = 32
GROUP_SIZE = 8
VLLM_MAX_NUM_SEQS = 16
MAX_TURNS = 6
LEARNING_RATE = 1e-6
CHECKPOINT_INTERVAL = 8
ADAPTER_NAME = "r16-a32"
DEFAULT_MANIFEST = "source-manifest.json"
EXPECTED_COUNT = 2486
ROLLOUT_TIMEOUT_SECONDS = 7_200
SCORING_TIMEOUT_SECONDS = 6_300
EPISODE_TIMEOUT_SECONDS = 13_600
FROZEN_EPISODE_TIMEOUT_SECONDS = 3_600
OPPONENT_LOCK_TIMEOUT_SECONDS = 4_200
FROZEN_ALIAS = "sft-step512"
FROZEN_CHECKPOINT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
FROZEN_PROTOCOL_VERSION = "native-painting-rl-v1"
FROZEN_RENDER_PROTOCOL = "native-painting-v1"
FROZEN_GENERATOR = "frozen_opponent:generate_frozen_opponent"
DEFAULT_MODE = "fresh"
DEFAULT_ADAPTER_REVISION = ""
DEFAULT_ADAPTER_PREFIX = ""
DEFAULT_ADAPTER_SHA256 = ""


def _sha256(value: str, *, name: str, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if not value and not required:
        return value
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _adapter_identity(*, mode: str, revision: str, prefix: str, sha256: str) -> dict[str, str]:
    if mode not in {"fresh", "recovered_policy"}:
        raise ValueError("mode must be fresh or recovered_policy")
    if not isinstance(revision, str) or not revision:
        raise ValueError("adapter_revision is required")
    if not isinstance(prefix, str) or not prefix or prefix.startswith("/"):
        raise ValueError("adapter_prefix must be a non-empty repository-relative path")
    return {
        "revision": revision,
        "prefix": prefix,
        "sha256": _sha256(sha256, name="adapter_sha256"),
    }


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_config(
    *,
    model: str,
    run_root: Path,
    manifest: str = DEFAULT_MANIFEST,
    max_steps: int = INITIAL_UPDATES,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    num_train_gpus: int = DEFAULT_NUM_TRAIN_GPUS,
    num_infer_gpus: int = DEFAULT_NUM_INFER_GPUS,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    run_name: str = RUN_NAME,
    expected_count: int = EXPECTED_COUNT,
    mode: str = DEFAULT_MODE,
    adapter_revision: str = DEFAULT_ADAPTER_REVISION,
    adapter_prefix: str = DEFAULT_ADAPTER_PREFIX,
    adapter_sha256: str = DEFAULT_ADAPTER_SHA256,
    batch_size: int = BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    max_turns: int = MAX_TURNS,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    learning_rate: float = LEARNING_RATE,
    initial_inflight: int | None = None,
    max_inflight: int | None = None,
) -> dict:
    """Return the validated-shape native RL config for this run."""

    max_steps = _positive_int("max_steps", max_steps)
    if max_steps > MAX_UPDATES:
        raise ValueError(f"max_steps must be <= {MAX_UPDATES}")
    checkpoint_interval = _positive_int("checkpoint_interval", checkpoint_interval)
    if checkpoint_interval > max_steps:
        raise ValueError("checkpoint_interval must not exceed max_steps")
    if not isinstance(manifest, str) or not manifest or Path(manifest).is_absolute():
        raise ValueError("manifest must be a non-empty relative path")
    expected_count = _positive_int("expected_count", expected_count)
    batch_size = _positive_int("batch_size", batch_size)
    group_size = _positive_int("group_size", group_size)
    max_turns = _positive_int("max_turns", max_turns)
    max_completion_tokens = _positive_int("max_completion_tokens", max_completion_tokens)
    if batch_size % group_size:
        raise ValueError("batch_size must be divisible by group_size")
    if not 0 < learning_rate <= 1e-4:
        raise ValueError("learning_rate must be in (0, 1e-4]")

    config = _build_native_config(
        model=model,
        run_root=run_root,
        gpus_per_node=gpus_per_node,
        num_train_gpus=num_train_gpus,
        num_infer_gpus=num_infer_gpus,
        checkpoint_interval=checkpoint_interval,
        run_name=run_name,
    )

    # Keep the old native Prime source shape: this is the shape validated by
    # the pinned node stack.  Only run values are changed here.
    config["max_steps"] = max_steps
    config["seq_len"] = SEQ_LEN
    config["trainer"]["max_steps"] = max_steps
    config["trainer"]["model"]["seq_len"] = SEQ_LEN
    config["trainer"]["optim"]["lr"] = learning_rate
    config["orchestrator"]["batch_size"] = batch_size
    config["orchestrator"]["group_size"] = group_size
    config["orchestrator"]["seq_len"] = SEQ_LEN
    config["orchestrator"]["max_steps"] = max_steps
    config["orchestrator"]["max_off_policy_steps"] = 1
    inference_slots = num_infer_gpus * VLLM_MAX_NUM_SEQS
    if initial_inflight is None:
        initial_inflight = max(
            batch_size,
            ((inference_slots + batch_size - 1) // batch_size) * batch_size,
        )
    else:
        initial_inflight = _positive_int("initial_inflight", initial_inflight)
    if max_inflight is None:
        max_inflight = initial_inflight + batch_size
    else:
        max_inflight = _positive_int("max_inflight", max_inflight)
    if initial_inflight < batch_size or max_inflight < initial_inflight:
        raise ValueError("inflight bounds must satisfy batch_size <= initial_inflight <= max_inflight")
    if initial_inflight % group_size or max_inflight % group_size:
        raise ValueError("inflight bounds must be divisible by group_size")
    config["orchestrator"]["concurrency"] = {
        "min_inflight": batch_size,
        # The default keeps one decode window plus one training batch active.
        # The explicit bounds allow a measured run to trade some concurrency
        # for fewer stale episodes without changing the off-policy limit.
        "initial_inflight": initial_inflight,
        "max_inflight": max_inflight,
    }
    sampling = config["orchestrator"]["train"]["sampling"]
    sampling["max_completion_tokens"] = max_completion_tokens
    source = config["orchestrator"]["train"]["source"][0]
    source["env"]["agent"]["timeout"].update(
        {
            "rollout": ROLLOUT_TIMEOUT_SECONDS,
            "scoring": SCORING_TIMEOUT_SECONDS,
        }
    )
    source["env"]["timeout"]["episode"] = EPISODE_TIMEOUT_SECONDS
    source["env"]["taskset"].update(
        {
            "manifest": manifest,
            "expected_count": expected_count,
            "max_turns": max_turns,
            "policy_label": "training",
            "output_root": "rl-rollouts",
            "preserve_manifest_order": True,
            "lazy_opponent": True,
            "opponent_cache_root": "frozen-opponents",
            "opponent_model_alias": FROZEN_ALIAS,
            "opponent_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "opponent_protocol_version": FROZEN_PROTOCOL_VERSION,
            "opponent_render_protocol": FROZEN_RENDER_PROTOCOL,
            "opponent_decoding_config": "{}",
            "opponent_seed": 20260922,
            "opponent_generator": FROZEN_GENERATOR,
            "opponent_lock_timeout_seconds": OPPONENT_LOCK_TIMEOUT_SECONDS,
            "opponent_max_completion_tokens": max_completion_tokens,
            "opponent_context_length": SEQ_LEN,
        }
    )
    config["inference"]["vllm"].update(
        {
            "max_model_len": SEQ_LEN,
            # Prime's pinned RLConfig maps each inference GPU to one vLLM DP
            # replica when tensor parallelism is one. Write the resolved
            # values explicitly so a four-GPU role split cannot silently fall
            # back to one server after config composition.
            "tensor_parallel_size": 1,
            "data_parallel_size": num_infer_gpus,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_num_batched_tokens": SEQ_LEN,
            # Keep the immutable SFT anchor resident alongside Prime's live
            # base-name alias. Prime reloads the live slot in place after each
            # update; the second slot must remain frozen at step 512.
            "max_loras": 2,
        }
    )
    config["ckpt"] = {"interval": checkpoint_interval, "keep_last": 1}
    return config


def write_config(output: Path, config: dict, *, validate_prime: bool = False, prime_root: Path | None = None) -> str:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing config: {output}")
    if validate_prime:
        # Prime's pydantic validator normalizes nested values in place on the
        # pinned node (notably trainer.model.compile: "None" -> None).  Keep
        # the render source untouched so the TOML writer never sees a Python
        # None and emits an invalid document after successful validation.
        valid, message = validate_with_prime(copy.deepcopy(config), prime_root)
        if not valid:
            raise RuntimeError(message)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_toml(config).replace(
        "# Generated by painter/prepare_scaled_native_rl.py; preparation only.",
        "# Generated by painter/runs/photo-curriculum-rl-20260922/prepare_rl_config.py; preparation only.",
    ))
    frozen_limits = output.parent / "frozen-opponent-config.json"
    frozen_limits.write_text(json.dumps({
        "schema": "photo-curriculum-rl-frozen-limits-v1",
        "operational_limits": {
            "episode_timeout_seconds": FROZEN_EPISODE_TIMEOUT_SECONDS,
            "render_timeout_seconds": 180,
        },
    }, indent=2) + "\n")
    return message if validate_prime else "Prime validation not requested"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--max-steps", type=int, default=INITIAL_UPDATES)
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--num-train-gpus", type=int, default=DEFAULT_NUM_TRAIN_GPUS)
    parser.add_argument("--num-infer-gpus", type=int, default=DEFAULT_NUM_INFER_GPUS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--initial-inflight", type=int)
    parser.add_argument("--max-inflight", type=int)
    parser.add_argument("--mode", choices=("fresh", "recovered_policy"), default=DEFAULT_MODE)
    parser.add_argument("--adapter-revision", default=DEFAULT_ADAPTER_REVISION)
    parser.add_argument("--adapter-prefix", default=DEFAULT_ADAPTER_PREFIX)
    parser.add_argument("--adapter-sha256", default=DEFAULT_ADAPTER_SHA256)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--validate-prime", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    adapter = _adapter_identity(
        mode=args.mode,
        revision=args.adapter_revision,
        prefix=args.adapter_prefix,
        sha256=args.adapter_sha256,
    )
    config = build_config(
        model=args.model,
        run_root=args.run_root,
        manifest=args.manifest,
        max_steps=args.max_steps,
        gpus_per_node=args.gpus_per_node,
        num_train_gpus=args.num_train_gpus,
        num_infer_gpus=args.num_infer_gpus,
        checkpoint_interval=args.checkpoint_interval,
        run_name=args.run_name,
        expected_count=args.expected_count,
        mode=args.mode,
        adapter_revision=args.adapter_revision,
        adapter_prefix=args.adapter_prefix,
        adapter_sha256=args.adapter_sha256,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        learning_rate=args.learning_rate,
        initial_inflight=args.initial_inflight,
        max_inflight=args.max_inflight,
    )
    message = write_config(args.output, config, validate_prime=args.validate_prime, prime_root=args.prime_root)
    receipt = {
        "run_name": args.run_name,
        "prime_revision": PRIME_COMMIT,
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "manifest": args.manifest,
        "expected_count": args.expected_count,
        "initial_updates": INITIAL_UPDATES,
        "max_updates": MAX_UPDATES,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "gpus_per_node": args.gpus_per_node,
        "num_train_gpus": args.num_train_gpus,
        "num_infer_gpus": args.num_infer_gpus,
        "inference_tensor_parallel_size": 1,
        "inference_data_parallel_size": args.num_infer_gpus,
        "initial_inflight": config["orchestrator"]["concurrency"]["initial_inflight"],
        "max_inflight": config["orchestrator"]["concurrency"]["max_inflight"],
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "max_turns": args.max_turns,
        "seq_len": SEQ_LEN,
        "max_completion_tokens": args.max_completion_tokens,
        "rollout_timeout_seconds": ROLLOUT_TIMEOUT_SECONDS,
        "scoring_timeout_seconds": SCORING_TIMEOUT_SECONDS,
        "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
        "frozen_episode_timeout_seconds": FROZEN_EPISODE_TIMEOUT_SECONDS,
        "opponent_lock_timeout_seconds": OPPONENT_LOCK_TIMEOUT_SECONDS,
        "learning_rate": args.learning_rate,
        "checkpoint_interval": args.checkpoint_interval,
        "reward": "terminal_aesthetic_only; no speed reward",
        "prime_validation": message,
        "mode": args.mode,
        "initialization": "fresh_optimizer_from_recovered_policy_adapter" if args.mode == "recovered_policy" else "fresh_optimizer_from_declared_adapter",
        "optimizer_resume": False,
        "adapter_revision": adapter["revision"],
        "adapter_prefix": adapter["prefix"],
        "adapter_sha256": adapter["sha256"],
        "frozen_anchor": {
            "alias": FROZEN_ALIAS,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "immutable": True,
        },
        "deployment": {
            "gpus_per_node": args.gpus_per_node,
            "num_train_gpus": args.num_train_gpus,
            "num_infer_gpus": args.num_infer_gpus,
        },
    }
    args.output.with_name("rl-config-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "prime_validation": message}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
