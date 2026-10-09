#!/usr/bin/env python3
"""Small metadata boundary for the checkpoint-4 evaluation recovery.

This module deliberately does not import Prime, a model, a renderer, or a
judge.  The Linux shell entry point uses it to verify the public adapter and
to expose it at the path expected by the existing evaluator.  Keeping this
boundary separate makes it possible to exercise the recovery contract on the
controller machine without touching a credential or starting a workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


RUN_NAME = "diverse-flow-rl-20261009"
CHECKPOINT_STEP = 4
POLICY_REPO = "CK0607/qwen3.8-27b-brush-painting"
POLICY_REVISION = ""
POLICY_PREFIX = "diverse-flow-rl-20261009/step_4"
POLICY_SHA256 = ""
BASELINE_REPO = "CK0607/komorebi-painter-experiments"
FLASH_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
FLASH_PREFIX = "rl-judge-matched-20261004-flash/step_12"
FLASH_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
SFT_REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
SFT_PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
SFT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
EVAL_SEED = 918210
MAX_TURNS = 3
MAX_TOKENS = 8192
JUDGE_MODEL = "xiaomi/mimo-v2.6-flash"
JUDGE_EFFORT = "medium"
JUDGE_PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
JUDGE_FORMAT = "identity-joint-side-schema-v2-generic-v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"receipt must contain an object: {path}")
    return value


def validate_downloaded_adapter(
    receipt_path: Path,
    adapter_source: Path,
    *,
    revision: str = POLICY_REVISION,
    prefix: str = POLICY_PREFIX,
    expected_sha256: str = POLICY_SHA256,
) -> dict[str, Any]:
    """Verify the pinned public download before exposing it to vLLM."""

    if not SHA_RE.fullmatch(expected_sha256):
        raise ValueError("checkpoint SHA-256 is malformed")
    receipt = _object(receipt_path)
    if (
        receipt.get("verified") is not True
        or receipt.get("revision") != revision
        or receipt.get("prefix") != prefix
        or receipt.get("adapter_sha256") != expected_sha256
    ):
        raise ValueError("download receipt does not match the checkpoint-4 public pin")
    source = adapter_source.expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("downloaded adapter root must be a regular directory")
    weight = source / "adapter_model.safetensors"
    config = source / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise ValueError("downloaded adapter is incomplete")
    actual = sha256(weight)
    if actual != expected_sha256:
        raise ValueError(f"downloaded checkpoint hash mismatch: {actual}")
    return {
        "repo_id": POLICY_REPO,
        "revision": revision,
        "prefix": prefix,
        "adapter_path": str(source),
        "adapter_sha256": actual,
        "verified": True,
    }


def _target_dir(run_root: Path, step: int = CHECKPOINT_STEP) -> Path:
    if step != CHECKPOINT_STEP:
        raise ValueError("evaluation recovery is pinned to checkpoint step 4")
    return run_root / "async-output" / RUN_NAME / "artifacts" / "adapters" / f"step_{step}"


def _same_file_or_missing(target: Path, source: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"existing staged adapter link differs: {target}")
    elif target.exists():
        if not target.is_file() or sha256(target) != sha256(source):
            raise ValueError(f"existing staged adapter file differs: {target}")


def public_hf_upload_receipt(*, adapter: Path, commit: str, prefix: str, expected_sha256: str) -> dict[str, Any]:
    """Return the normal public-upload receipt consumed by archive/evaluate."""

    weight = adapter / "adapter_model.safetensors"
    actual = sha256(weight)
    if actual != expected_sha256:
        raise ValueError("staged adapter bytes do not match the supplied checkpoint pin")
    return {
        "schema": "hf-public-adapter-upload-v1",
        "repo_id": POLICY_REPO,
        "commit": commit,
        "revision": commit,
        "path": prefix,
        "provenance_type": "public_download",
        "public_metadata_verified": True,
        "verified": True,
        "private": False,
        "files": {"adapter_model.safetensors": {"sha256": actual}},
        "adapter_sha256": actual,
    }


def stage_checkpoint(
    *,
    run_root: Path,
    download_receipt: Path,
    adapter_source: Path,
    step: int = CHECKPOINT_STEP,
) -> dict[str, Any]:
    """Stage only links and a public receipt at the existing evaluator path."""

    source_receipt = _object(download_receipt)
    revision = source_receipt.get("revision")
    prefix = source_receipt.get("prefix")
    expected_sha256 = source_receipt.get("adapter_sha256")
    if not isinstance(revision, str) or not revision or not isinstance(prefix, str) or not prefix or not isinstance(expected_sha256, str):
        raise ValueError("download receipt must carry immutable revision, prefix, and adapter hash")
    validate_downloaded_adapter(
        download_receipt,
        adapter_source,
        revision=revision,
        prefix=prefix,
        expected_sha256=expected_sha256,
    )
    source = adapter_source.expanduser().resolve()
    target = _target_dir(run_root.expanduser().resolve(), step)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        _same_file_or_missing(target / name, source / name)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        path = target / name
        if not path.exists() and not path.is_symlink():
            path.symlink_to(source / name)
    receipt = public_hf_upload_receipt(adapter=source, commit=revision, prefix=prefix, expected_sha256=expected_sha256)
    receipt_path = target / "hf-upload.json"
    if receipt_path.is_symlink():
        raise ValueError("staged checkpoint receipt must not be a symlink")
    if receipt_path.is_file():
        if _object(receipt_path) != receipt:
            raise ValueError("existing staged checkpoint receipt differs")
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "Diverse-flow-evaluation-checkpoint-stage-v1",
        "step": step,
        "adapter_dir": str(target),
        "adapter_source": str(source),
        "receipt": str(receipt_path),
        "adapter_sha256": expected_sha256,
        "links_only": True,
        "public_receipt": receipt,
    }


def recovery_receipt(*, run_root: Path, source_commit: str, manifest: Path, renderer: Path | None = None) -> dict[str, Any]:
    """Build immutable metadata for a no-training evaluation recovery."""

    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source commit must be an immutable hexadecimal commit")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    staged_receipt = run_root / "async-output" / RUN_NAME / "artifacts" / "adapters" / "step_4" / "hf-upload.json"
    staged = _object(staged_receipt)
    policy_revision = staged.get("revision")
    policy_prefix = staged.get("path")
    policy_sha256 = staged.get("adapter_sha256")
    if not isinstance(policy_revision, str) or not isinstance(policy_prefix, str) or not SHA_RE.fullmatch(str(policy_sha256)):
        raise ValueError("staged checkpoint receipt is missing immutable policy identity")
    manifest_value = _object(manifest)
    cases = manifest_value.get("cases")
    if not isinstance(cases, list) or len(cases) != 32 or len({row.get("id") for row in cases if isinstance(row, dict)}) != 32:
        raise ValueError("recovery manifest must contain exactly 32 unique cases")
    value: dict[str, Any] = {
        "schema": "Diverse-flow-evaluation-recovery-v1",
        "status": "prepared_no_training",
        "run_name": RUN_NAME,
        "mode": "evaluation_only",
        "training": {
            "started_before_recovery": True,
            "completed_optimizer_updates": 5,
            "target_optimizer_updates": 20,
            "checkpoint_step": CHECKPOINT_STEP,
            "new_optimizer_updates": 0,
            "warmup_status": "pending",
            "stop_reason": "finite_cost_deadline",
        },
        "source_commit": source_commit,
        "policy": {
            "repo": POLICY_REPO,
            "revision": policy_revision,
            "prefix": policy_prefix,
            "adapter_sha256": policy_sha256,
        },
        "matched_baseline": {
            "policy": "Flash12 starting policy",
            "repo": POLICY_REPO,
            "revision": FLASH_REVISION,
            "prefix": FLASH_PREFIX,
            "adapter_sha256": FLASH_SHA256,
            "archive_repo": None,
            "archive_revision": None,
            "archive_sha256": None,
        },
        "reward_opponent": {
            "policy": "frozen SFT512 reward opponent",
            "repo": POLICY_REPO,
            "revision": SFT_REVISION,
            "prefix": SFT_PREFIX,
            "adapter_sha256": SFT_SHA256,
        },
        "evaluation": {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "expected_cases": 32,
            "scope": "all",
            "seed": EVAL_SEED,
            "max_turns": MAX_TURNS,
            "max_tokens": MAX_TOKENS,
            "judge_model": JUDGE_MODEL,
            "judge_effort": JUDGE_EFFORT,
            "judge_protocol": JUDGE_PROTOCOL,
            "judge_format": JUDGE_FORMAT,
            "comparison_policy": "Flash12 starting policy versus checkpoint4",
            "flow_prompt_source": "restored admitted casebrief bytes",
        },
        "topology": {"gpus_per_node": 1, "trainer_gpus": 0, "inference_gpus": 1, "evaluation_gpu": 0},
        "cache_restore": {
            "full_evidence_revision": None,
            "full_evidence_sha256": None,
            "baseline_archive_revision": None,
            "baseline_archive_sha256": None,
            "generator_calls": 0,
            "anchor_generation": "reused_hash_verified_evidence",
        },
        "renderer": {"path": str(renderer), "sha256": sha256(renderer)} if renderer else None,
    }
    return value


def write_recovery_receipt(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"recovery receipt must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-checkpoint")
    stage.add_argument("--run-root", type=Path, required=True)
    stage.add_argument("--download-receipt", type=Path, required=True)
    stage.add_argument("--adapter-source", type=Path, required=True)
    stage.add_argument("--step", type=int, default=CHECKPOINT_STEP)
    receipt = sub.add_parser("write-receipt")
    receipt.add_argument("--run-root", type=Path, required=True)
    receipt.add_argument("--source-commit", required=True)
    receipt.add_argument("--manifest", type=Path, required=True)
    receipt.add_argument("--renderer", type=Path)
    receipt.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stage-checkpoint":
        print(json.dumps(stage_checkpoint(
            run_root=args.run_root,
            download_receipt=args.download_receipt,
            adapter_source=args.adapter_source,
            step=args.step,
        ), sort_keys=True))
        return 0
    value = recovery_receipt(
        run_root=args.run_root,
        source_commit=args.source_commit,
        manifest=args.manifest,
        renderer=args.renderer,
    )
    write_recovery_receipt(args.output, value)
    print(json.dumps({"schema": value["schema"], "path": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
