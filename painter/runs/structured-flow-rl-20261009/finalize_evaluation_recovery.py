#!/usr/bin/env python3
"""Finalize one generated checkpoint-4 evaluation without SFT scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from evaluation_recovery import (
    BASELINE_REVISION,
    CHECKPOINT_STEP,
    EVAL_SEED,
    FLASH_PREFIX,
    FLASH_REVISION,
    FLASH_SHA256,
    JUDGE_FORMAT,
    JUDGE_MODEL,
    JUDGE_PROTOCOL,
    MAX_TOKENS,
    MAX_TURNS,
    POLICY_PREFIX,
    POLICY_REVISION,
    POLICY_SHA256,
    RUN_NAME,
    SFT_PREFIX,
    SFT_REVISION,
    SFT_SHA256,
    _object,
    sha256,
)


def _checkpoint(run: Path) -> tuple[Path, dict[str, Any]]:
    root = run / "async-output" / RUN_NAME / "artifacts" / "adapters" / "step_4"
    receipt = _object(root / "hf-upload.json")
    weight = root / "adapter_model.safetensors"
    expected = receipt.get("files", {}).get("adapter_model.safetensors", {}).get("sha256")
    if receipt.get("verified") is not True or receipt.get("private") is not False or receipt.get("provenance_type") != "public_download":
        raise ValueError("checkpoint4 is not a verified public download")
    if receipt.get("revision") != POLICY_REVISION or receipt.get("path") != POLICY_PREFIX or expected != POLICY_SHA256:
        raise ValueError("checkpoint4 public identity differs from recovery pin")
    if not weight.is_file() or sha256(weight) != expected:
        raise ValueError("checkpoint4 adapter bytes do not match public receipt")
    return root, receipt


def _baseline(run: Path, manifest: Path) -> dict[str, Any]:
    receipt = _object(run / "matched-baseline" / "pre-step12" / "baseline-receipt.json")
    policy = receipt.get("policy")
    if receipt.get("schema") != "structured-flow-matched-flash12-baseline-v1" or receipt.get("status") != "complete":
        raise ValueError("cached Flash12 baseline is incomplete")
    if not isinstance(policy, dict) or policy.get("revision") != FLASH_REVISION or policy.get("prefix") != FLASH_PREFIX or policy.get("adapter_sha256") != FLASH_SHA256:
        raise ValueError("cached baseline policy is not the pinned Flash12 starting policy")
    if receipt.get("manifest_cases") != 16 or receipt.get("seed") != EVAL_SEED or receipt.get("max_turns") != MAX_TURNS or receipt.get("judge_deferred") is not True:
        raise ValueError("cached baseline is not the exact 16-case matched panel")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected_manifest = receipt.get("manifest_sha256")
    if not isinstance(expected_manifest, str) or sha256(manifest) != expected_manifest:
        raise ValueError("matched baseline manifest bytes differ from the pinned baseline")
    return receipt


def _check_eval_config(run: Path, manifest_sha: str) -> None:
    config = run / "validation" / "step-4-all" / "eval.toml"
    if not config.is_file() or config.is_symlink():
        raise ValueError("checkpoint4 evaluation config is missing")
    text = config.read_text(encoding="utf-8")
    required = {
        "seed": EVAL_SEED,
        "max_turns": MAX_TURNS,
        "max_tokens": MAX_TOKENS,
    }
    for key, value in required.items():
        match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([0-9]+)\s*$", text)
        if match is None or int(match.group(1)) != value:
            raise ValueError(f"evaluation config {key} differs from recovery pin")
    # The wrapper materializes a matched copy; validate its bytes against the
    # baseline receipt rather than trusting only the case count.
    if manifest_sha != sha256(run / "curriculum" / "matched16-manifest.json"):
        raise ValueError("candidate manifest hash is not stable")
    cases = run / "validation" / "step-4-all" / "cases.json"
    if not cases.is_file() or cases.is_symlink():
        raise ValueError("candidate evaluation cases manifest is missing")
    value = _object(cases)
    rows = value.get("cases")
    if not isinstance(rows, list) or len(rows) != 16 or len({row.get("id") for row in rows if isinstance(row, dict)}) != 16:
        raise ValueError("candidate evaluation cases manifest is not the exact 16-case panel")


def finalize(*, run: Path, step: int = CHECKPOINT_STEP, scope: str = "all") -> dict[str, Any]:
    if step != CHECKPOINT_STEP or scope != "all":
        raise ValueError("checkpoint recovery is pinned to step 4/all")
    manifest = run / "curriculum" / "matched16-manifest.json"
    generation = _object(run / "validation" / "step-4-all" / "generation-receipt.json")
    if generation.get("step") != CHECKPOINT_STEP or generation.get("scope") != scope or generation.get("max_turns") != MAX_TURNS or generation.get("max_tokens") != MAX_TOKENS:
        raise ValueError("generation receipt does not match checkpoint4 recovery")
    if generation.get("scoring_protocol") != JUDGE_PROTOCOL or generation.get("judge_format") != JUDGE_FORMAT or generation.get("judge_model") != JUDGE_MODEL:
        raise ValueError("generation receipt changed the pinned Flash judge settings")
    adapter, upload = _checkpoint(run)
    baseline = _baseline(run, manifest)
    _check_eval_config(run, sha256(manifest))
    artifact_root = run / "validation" / "step-4-all" / "eval" / "evaluation-rollouts"
    if not artifact_root.is_dir() or not list(artifact_root.rglob("painting.json")):
        raise ValueError("checkpoint4 evaluation artifacts are missing")
    value = {
        "schema": "structured-flow-checkpoint4-finalizer-v1",
        "status": "generation_complete_unscored",
        "run_name": RUN_NAME,
        "step": CHECKPOINT_STEP,
        "scope": scope,
        "adapter": {"repo": "CK0607/qwen3.8-27b-brush-painting", "revision": POLICY_REVISION, "prefix": POLICY_PREFIX, "sha256": POLICY_SHA256, "path": str(adapter), "receipt": upload},
        "matched_baseline": {"policy": "Flash12 starting policy", "revision": FLASH_REVISION, "prefix": FLASH_PREFIX, "sha256": FLASH_SHA256, "receipt": str(run / "matched-baseline" / "pre-step12" / "baseline-receipt.json"), "archive_sha256": baseline.get("source_archive_sha256")},
        "reward_opponent": {"policy": "frozen SFT512 reward opponent", "revision": SFT_REVISION, "prefix": SFT_PREFIX, "sha256": SFT_SHA256, "used_for": "training reward/anchor only"},
        "evaluation": {"manifest": str(manifest), "manifest_sha256": sha256(manifest), "expected_cases": 16, "seed": EVAL_SEED, "max_turns": MAX_TURNS, "max_tokens": MAX_TOKENS, "judge_protocol": JUDGE_PROTOCOL, "judge_format": JUDGE_FORMAT, "judge_model": JUDGE_MODEL, "comparison": "Flash12 starting policy versus checkpoint4", "sft512_quality_comparison": False},
        "topology": {"gpus_per_node": 1, "trainer_gpus": 0, "inference_gpus": 1, "evaluation_gpu": 0},
        "score_command": "score_recovery.py (API-only after GPU release)",
        "failed_and_censored_cases": "retained in evaluation-rollouts and score rows",
    }
    path = run / "checkpoint4-finalizer-receipt.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=int)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    value = finalize(run=args.run_root.resolve(), step=args.step, scope=args.scope)
    print(json.dumps({"schema": value["schema"], "status": value["status"], "step": value["step"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
