#!/usr/bin/env python3
"""Write and validate controller-side launch metadata without network or ML."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


RUN = Path(__file__).resolve().parent
CURRICULUM = RUN / "prepare_curriculum.py"
spec = importlib.util.spec_from_file_location("structured_flow_curriculum_for_plan", CURRICULUM)
if spec is None or spec.loader is None:
    raise RuntimeError("curriculum preparer is unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

PINS = {
    "policy_repo": "CK0607/qwen3.8-27b-brush-painting",
    "policy_revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c",
    "policy_prefix": "rl-judge-matched-20261004-flash/step_12",
    "policy_sha256": "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79",
    "frozen_sft_revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "frozen_sft_prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "frozen_sft_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
    "judge_model": "xiaomi/mimo-v2.6-flash",
    "judge_effort": "medium",
    "judge_format": "identity-joint-side-schema-v2-generic-v1",
    "judge_protocol": "text-paint-sft-pairwise-consistent-v1",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(output: Path, *, proposal_root: Path | None = None) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty plan root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = module.generate_rows()
    counts = {"all": len(rows), "train": sum(row["split"] == "train" for row in rows), "heldout": sum(row["split"] == "heldout" for row in rows)}
    if counts != {"all": 300, "train": 240, "heldout": 60}:
        raise ValueError(f"curriculum count mismatch: {counts}")
    stages = {str(stage): {"train": sum(row["split"] == "train" and row["stage"] == stage for row in rows), "heldout": sum(row["split"] == "heldout" and row["stage"] == stage for row in rows)} for stage in range(1, 5)}
    plan = {
        "schema": "structured-flow-local-launch-plan-v1",
        "run_id": module.RUN_ID,
        "source": {"prepare_curriculum": str(CURRICULUM), "prepare_curriculum_sha256": _sha(CURRICULUM)},
        "pins": PINS,
        "curriculum": {"counts": counts, "per_stage": stages, "replay_fraction": 0.25, "active_pilot": {"train": 32, "heldout": 16, "updates": 8, "checkpoint_steps": [4, 8]}},
        "training": {"batch_size": 16, "group_size": 4, "initial_inflight": 32, "max_inflight": 32, "learning_rate": 1e-6, "trainer_gpus": 1, "inference_gpus": 1, "gpus_per_node": 2, "max_policy_age": 1, "max_turns": 3, "warmup_updates": 4, "warmup_learning_rate": 1e-6, "warmup_gpu": 1, "warmup_optimizer": "fresh", "new_reward_bonus": False},
        "budget": {"gpu_cap_usd": 20.0, "includes": ["setup", "training", "matched_pre_eval", "matched_post_eval", "archive_upload", "release"]},
        "flow_policy": {"env": "PAINTER_STRUCTURED_FLOW_POLICY", "default": "1", "hint_format": "casebrief [SYSTEM] and [USER] sections", "style_or_code_reward": False},
        "code_supervision": {"env": "PAINTER_CODE_SUPERVISION_ENABLED", "default": "1", "format": "strict-jsonl-nextassistanttokenonly", "separate_from_rl_reward": True},
        "limitations": [
            "This plan contains no rendered images, model weights, credentials, provider state, or judge calls.",
            "The 240/60 rows are proposal metadata until each frozen SFT512 anchor is rendered and hash-verified.",
            "The operational run admits only the 32/16 pilot manifests produced by materialize.py.",
        ],
    }
    (output / "launch-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "flow-policy.md").write_text(
        "# Structured-flow policy hint\n\n"
        "The node launch explicitly sets `PAINTER_STRUCTURED_FLOW_POLICY=1`; the runtime remains default-off outside the launch. It is stored in each admitted casebrief as separate `[SYSTEM]` and `[USER]` text. It teaches planning hierarchy and explicit controls; it does not add a style or code reward. The runtime default remains `0` outside the explicit launch environment.\n",
        encoding="utf-8",
    )
    return plan


def validate_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "structured-flow-local-launch-plan-v1":
        raise ValueError("unexpected launch plan schema")
    if value.get("pins") != PINS:
        raise ValueError("immutable pin set changed")
    if value.get("curriculum", {}).get("counts") != {"all": 300, "train": 240, "heldout": 60}:
        raise ValueError("curriculum count contract changed")
    active = value.get("curriculum", {}).get("active_pilot", {})
    if active != {"train": 32, "heldout": 16, "updates": 8, "checkpoint_steps": [4, 8]}:
        raise ValueError("pilot bound changed")
    if value.get("budget", {}).get("gpu_cap_usd") != 20.0:
        raise ValueError("GPU cap changed")
    if value.get("training", {}).get("max_turns") != 3:
        raise ValueError("pilot turn cap changed")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("plan")
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_plan(args.output) if args.command == "plan" else validate_plan(args.plan)
    print(json.dumps({"schema": value["schema"], "run_id": value["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
