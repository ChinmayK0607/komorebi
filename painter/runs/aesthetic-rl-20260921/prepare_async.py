"""Validate the pinned native Prime configuration on the Linux node.

This writes ``rl.toml`` only after hardware and the actual pinned
``prime_rl.configs.rl.RLConfig`` accept the complete config.  It deliberately
does not launch training or the judge worker.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path


SEQ_LEN = 16_384
MAX_COMPLETION_TOKENS = 8_192


def topology(info: dict) -> dict:
    gpus = info["gpus"]
    if len(gpus) < 2:
        raise ValueError("aesthetic RL requires distinct trainer and inference GPUs")
    if any(gpu["memory_gib"] < 76 or gpu["compute_capability"] < 8 for gpu in gpus[:2]):
        raise ValueError("the unsharded 27B profile needs two >=76 GiB CUDA devices")
    return {
        "gpus_per_node": len(gpus),
        "num_train_gpus": 1,
        "num_infer_gpus": 1,
        "type": "single_node",
    }


def training_limits(experiment: dict) -> tuple[int, int]:
    max_steps = experiment.get("max_steps", 4)
    checkpoint_interval = experiment.get("checkpoint_interval", 2)
    for name, value in (("max_steps", max_steps), ("checkpoint_interval", checkpoint_interval)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return max_steps, checkpoint_interval


def build_config(root: Path, model: str, run_id: str, info: dict) -> dict:
    experiment = json.loads((root / "EXPERIMENT.json").read_text())
    max_steps, checkpoint_interval = training_limits(experiment)
    deployment = topology(info)
    return dict(
        max_steps=max_steps,
        seq_len=SEQ_LEN,
        output_dir=str(root / "async-output"),
        run=dict(name=run_id),
        deployment=deployment,
        model=dict(
            name=model,
            vlm=dict(
                vision_encoder_attr="model.visual",
                language_model_attr="model.language_model",
                freeze_vision_encoder=True,
            ),
        ),
        weight_broadcast=dict(type="filesystem", timeout=1200),
        ckpt=dict(interval=checkpoint_interval, keep_last=1),
        monitors=dict(file={}),
        trainer=dict(
            max_steps=max_steps,
            model=dict(
                seq_len=SEQ_LEN,
                impl="custom",
                compile="None",
                attn="auto",
                optimization_dtype="bfloat16",
                reduce_dtype="bfloat16",
                optim_cpu_offload=False,
                lora=dict(
                    rank=16,
                    alpha=32,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "in_proj_b",
                        "in_proj_a", "out_proj",
                    ],
                ),
                ac=dict(freq=1),
                ac_offloading="None",
            ),
            optim=dict(type="adamw", lr=1e-5, weight_decay=0.01, max_norm=1.0),
            scheduler=dict(type="constant"),
            loss=dict(type="ipo", eps=0.1, adv_tau=1.0, kl_tau=0.001),
        ),
        orchestrator=dict(
            batch_size=8,
            group_size=8,
            seq_len=SEQ_LEN,
            max_steps=max_steps,
            max_off_policy_steps=1,
            renderer=dict(name="qwen3.5", enable_thinking=False),
            concurrency=dict(min_inflight=16, initial_inflight=16, max_inflight=16),
            train=dict(
                sampling=dict(
                    temperature=1.0,
                    top_p=1.0,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
                source=[
                    dict(
                        name="painter-aesthetic",
                        env=dict(
                            id="painter-revision",
                            taskset=dict(id="painter-revision", root=str(root)),
                            agent=dict(
                                harness=dict(id="null"),
                                runtime=dict(type="subprocess"),
                                timeout=dict(rollout=1200, scoring=2100),
                            ),
                            timeout=dict(episode=3600),
                        ),
                    )
                ],
            ),
        ),
        inference=dict(
            vllm=dict(
                dtype="bfloat16",
                max_model_len=SEQ_LEN,
                gpu_memory_utilization=0.90,
                max_num_seqs=16,
                max_num_batched_tokens=SEQ_LEN,
                limit_mm_per_prompt=dict(image=6, video=0),
                gdn_prefill_backend="triton",
                kernel_config=dict(enable_jit_warmup=False),
                mm_processor_kwargs=dict(min_pixels=3136, max_pixels=1048576),
            )
        ),
    )


def main() -> None:
    if sys.platform != "linux":
        raise SystemExit("configuration validation runs on the Linux training node")
    import tomli_w
    from hardware import detect
    from prime_rl.configs.rl import RLConfig
    from training_handoff import write_static_preflight

    root = Path(__file__).resolve().parent
    experiment = json.loads((root / "EXPERIMENT.json").read_text())
    run_id = os.environ.get("PAINTER_RUN_ID") or experiment.get("restart_run_name") or experiment["run_name"]
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("PAINTER_RUN_ID must be a simple non-empty run name")
    model = (root / "model-path.txt").read_text().strip()
    info = detect()
    config = build_config(root, model, run_id, info)
    resolved = RLConfig.model_validate(copy.deepcopy(config))
    (root / "rl.toml").write_text(tomli_w.dumps(config))
    max_steps, checkpoint_interval = training_limits(experiment)
    receipt = {
        "hardware": info,
        "deployment": topology(info),
        "run_id": run_id,
        "validated": True,
        "run_dir": str(resolved.run_dir),
        "seq_len": SEQ_LEN,
        "max_completion_tokens_per_turn": MAX_COMPLETION_TOKENS,
        "max_steps": max_steps,
        "checkpoint_interval": checkpoint_interval,
        "batch_size": 8,
        "group_size": 8,
        "max_off_policy_steps": 1,
        "min_initial_max_inflight": 16,
        "learning_rate": 1e-5,
        "mm_images_per_prompt": 6,
        "fresh_optimizer": True,
        "context_note": "16k is the measured one-GPU-safe context; real painter turns are ~1k. No artificial per-turn cap below 8192 is applied.",
    }
    (root / "async-config-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_static_preflight(root)
    print("Validated native aesthetic async configuration:", resolved.run_dir)


if __name__ == "__main__":
    main()
