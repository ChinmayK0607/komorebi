#!/usr/bin/env python3
"""Write and receipt-check the one profile-bound recovery config.

This file is called only after the pinned Prime interpreter has been created by
the reviewed node bootstrap.  The plan-only path never imports this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tomllib


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config_tool(repo: Path):
    source = repo / "painter/runs/text-pairwise-rl-20260928/prepare_config.py"
    spec = importlib.util.spec_from_file_location("single_recovery_prepare_config", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the reviewed config writer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--renderer-python", required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo))
    from painter import throughput_recovery_profile as profile

    config_tool = load_config_tool(args.repo)
    profile.validate_runtime_environment(os.environ)
    receipt = profile.prepare_profile(
        config_tool.prepare,
        output=args.run_root / "native.toml",
        model=args.model,
        run_root=args.run_root,
        manifest=args.manifest,
        prime_root=args.prime_root,
        run_name=os.environ["PAINTER_RUN_NAME"],
        output_root="rl-rollouts",
        expected_count=40,
        max_steps=4,
        learning_rate=1e-6,
        renderer=args.renderer,
        renderer_python=args.renderer_python,
        gpus_per_node=int(os.environ["PAINTER_GPUS_PER_NODE"]),
        num_train_gpus=int(os.environ["PAINTER_NUM_TRAIN_GPUS"]),
        num_infer_gpus=int(os.environ["PAINTER_NUM_INFER_GPUS"]),
    )
    profile.validate_config(tomllib.loads((args.run_root / "native.toml").read_text()), receipt)

    manifest_sha = digest(args.manifest)
    config_sha = digest(args.run_root / "native.toml")
    foundation = args.repo / "painter/foundation-lock.json"
    prep_path = args.run_root / "preparation-receipt.json"
    prep = json.loads(prep_path.read_text()) if prep_path.is_file() else {}
    prep.update(
        {
            "schema": "throughput-recovery-single-arm-preparation-v1",
            "run_name": os.environ["PAINTER_RUN_NAME"],
            "experiment_name": os.environ["PAINTER_EXPERIMENT_NAME"],
            "config_sha256": config_sha,
            "train_manifest_sha256": manifest_sha,
            "profile": profile.PROFILE_NAME,
            "max_updates": 4,
            "renderer": "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py",
            "judge_model": os.environ["PAINTER_JUDGE_MODEL"],
            "judge_effort": os.environ["PAINTER_JUDGE_EFFORT"],
            "judge_format": os.environ["PAINTER_JUDGE_FORMAT"],
        }
    )
    prep_path.write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")

    identity = {
        "schema": "throughput-recovery-single-arm-source-identity-v1",
        "run_name": os.environ["PAINTER_RUN_NAME"],
        "experiment_name": os.environ["PAINTER_EXPERIMENT_NAME"],
        "source_commit": os.environ.get("PAINTER_SOURCE_COMMIT"),
        "base_model": "Qwen/Qwen3.8-27B",
        "model_revision": os.environ["PAINTER_BASE_MODEL_REVISION"],
        "initializer_revision": os.environ["PAINTER_POLICY_REVISION"],
        "initializer_prefix": os.environ["PAINTER_POLICY_PREFIX"],
        "initializer_sha256": os.environ["PAINTER_POLICY_SHA256"],
        "frozen_sft_model_revision": os.environ["PAINTER_FROZEN_SFT_MODEL_REVISION"],
        "frozen_sft_adapter_sha256": os.environ["PAINTER_FROZEN_SFT_ADAPTER_SHA256"],
        "prime_revision": "26b3131d2716a4f8210b165df584b83b4bc54f61",
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "foundation_lock_sha256": digest(foundation),
        "train_briefs": 40,
        "heldout_briefs": 47,
        "evidence_contract": {
            "required_receipts": [
                "source-identity.json",
                "preparation-receipt.json",
                "rl-config-receipt.json",
                "renderer-receipt.json",
                "telemetry-receipt.json",
                "consistent-group-admission.jsonl",
            ],
            "required_metrics": [
                "async-output/<run>/monitors/file/metrics.jsonl",
                "logs/gpu-telemetry.csv",
                "logs/cpu-telemetry.txt",
            ],
            "render_timing_fields": ["render_queue_seconds", "render_active_seconds"],
            "gradient_fields": ["optim/grad_norm", "time/step"],
            "admission_denominator": "whole admitted groups of four",
        },
        "protocol": {
            "judge_protocol": os.environ["PAINTER_JUDGE_PROTOCOL"],
            "judge_model": os.environ["PAINTER_JUDGE_MODEL"],
            "judge_effort": os.environ["PAINTER_JUDGE_EFFORT"],
            "judge_format": os.environ["PAINTER_JUDGE_FORMAT"],
            "batch_size": 16,
            "group_size": 4,
            "initial_inflight": 32,
            "max_inflight": 32,
            "max_updates": 4,
            "checkpoint_interval": 1,
            "activation_cpu_offload": True,
            "strict_final_action": True,
            "p5_brush_api_reference_v2": True,
            "gpus_per_node": int(os.environ["PAINTER_GPUS_PER_NODE"]),
            "num_train_gpus": int(os.environ["PAINTER_NUM_TRAIN_GPUS"]),
            "num_infer_gpus": int(os.environ["PAINTER_NUM_INFER_GPUS"]),
        },
        "renderer": {
            "selected": "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py",
            "overlay_manifest_sha256": digest(args.run_root / "runtime/watercolour-cancellable-centroid-v1/PATCH-MANIFEST.json"),
        },
    }
    (args.run_root / "source-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    overlay_manifest = json.loads(
        (args.run_root / "runtime/watercolour-cancellable-centroid-v1/PATCH-MANIFEST.json").read_text()
    )
    (args.run_root / "renderer-receipt.json").write_text(
        json.dumps(
            {
                "schema": "throughput-recovery-cancellable-renderer-receipt-v1",
                "runtime_id": overlay_manifest["runtime_id"],
                "selected_renderer": "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py",
                "selected_renderer_sha256": overlay_manifest["selected_renderer"]["sha256"],
                "overlay_manifest_sha256": digest(
                    args.run_root / "runtime/watercolour-cancellable-centroid-v1/PATCH-MANIFEST.json"
                ),
                "frozen_baseline_sha256": overlay_manifest["frozen_baseline"]["sha256"],
                "centroid_patch_sha256": overlay_manifest["centroid_patch"]["after_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
