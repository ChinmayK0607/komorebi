"""Bounded one-update real-backward canary receipts.

This module only validates receipts.  The launch path enables the mode through
``launch_prepared.py --real-backward-canary``; no canary is run during offline
preparation.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "staged-family-balanced-checkpoint-contract-v1"
PORTABLE_FILES = ("adapter_config.json", "adapter_model.safetensors", "validation.json")


class CheckpointContractError(ValueError):
    """A native or portable checkpoint does not satisfy the run contract."""


def _native_checkpoint_path(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"step_{step}" / "trainer"


def _adapter_checkpoint_path(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / "artifacts" / "adapters" / f"step_{step}"


def _native_evidence(output_dir: Path, step: int) -> dict[str, Any]:
    checkpoint = _native_checkpoint_path(output_dir, step)
    metadata = checkpoint / ".metadata"
    files = sorted(
        path for path in checkpoint.rglob("*")
        if path.is_file() and path != metadata
    ) if checkpoint.is_dir() else []
    if not checkpoint.is_dir() or not metadata.is_file() or not files:
        raise CheckpointContractError("native checkpoint lacks DCP metadata or state files")
    return {"path": str(checkpoint), "metadata": str(metadata), "file_count": len(files)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_upload_evidence(
    output_dir: Path,
    step: int,
    *,
    expected_repo: str | None = None,
    expected_run: str | None = None,
) -> dict[str, Any]:
    """Verify the public LoRA receipt against the local adapter bytes."""

    folder = _adapter_checkpoint_path(output_dir, step)
    receipt_path = folder / "hf-upload.json"
    contract_path = folder / "checkpoint-contract.json"
    if folder.is_symlink() or not folder.is_dir() or receipt_path.is_symlink() or not receipt_path.is_file():
        raise CheckpointContractError("portable adapter upload receipt is missing")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise CheckpointContractError("checkpoint contract receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointContractError("portable adapter or checkpoint contract receipt is unreadable") from exc
    if not isinstance(receipt, dict) or receipt.get("verified") is not True or receipt.get("private") is not False:
        raise CheckpointContractError("portable adapter upload is not a verified public receipt")
    if expected_repo is not None and receipt.get("repo_id") != expected_repo:
        raise CheckpointContractError("portable adapter receipt repository does not match the launch contract")
    if expected_run is not None and receipt.get("path") != f"{expected_run}/step_{step}":
        raise CheckpointContractError("portable adapter receipt path does not match the launch run and step")
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA or contract.get("step") != step:
        raise CheckpointContractError("checkpoint contract schema or step is unsupported")
    native = contract.get("native_dcp")
    portable = contract.get("portable_adapter")
    if not isinstance(native, dict) or native.get("path") != str(_native_checkpoint_path(output_dir, step)) or native.get("resumable") is not True:
        raise CheckpointContractError("checkpoint contract does not bind the native DCP path")
    if (
        not isinstance(portable, dict)
        or portable.get("receipt_path") != str(receipt_path)
        or portable.get("repo_id") != receipt.get("repo_id")
        or portable.get("remote_path") != receipt.get("path")
        or portable.get("verified") is not True
        or portable.get("private") is not False
    ):
        raise CheckpointContractError("checkpoint contract does not bind the public adapter receipt")
    remote_files = receipt.get("files")
    expected_files = set(PORTABLE_FILES + ("upload-manifest.json",))
    if not isinstance(remote_files, dict) or set(remote_files) != expected_files:
        raise CheckpointContractError("portable adapter receipt has an unexpected file set")
    files: dict[str, dict[str, Any]] = {}
    for name in PORTABLE_FILES:
        path = folder / name
        entry = remote_files.get(name)
        if path.is_symlink() or not path.is_file() or not isinstance(entry, dict):
            raise CheckpointContractError(f"portable adapter file is missing: {name}")
        digest = _sha256(path)
        if entry.get("sha256") != digest or entry.get("bytes") != path.stat().st_size:
            raise CheckpointContractError(f"portable adapter hash receipt mismatch: {name}")
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
    entry = remote_files["upload-manifest.json"]
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64 or type(entry.get("bytes")) is not int or entry["bytes"] < 1:
        raise CheckpointContractError("portable upload manifest receipt is malformed")
    files["upload-manifest.json"] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
    return {
        "receipt_path": str(receipt_path), "contract_path": str(contract_path),
        "repo_id": receipt.get("repo_id"), "path": receipt.get("path"),
        "commit": receipt.get("commit"), "verified": True, "private": False, "files": files,
    }


MAX_SECONDS = 2400
MAX_COST_USD = 9.0


class RealBackwardCanaryError(RuntimeError):
    pass


def validate_receipt(path: Path, *, expected_world_size: int = 4) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealBackwardCanaryError("real-backward canary receipt is unreadable") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != "staged-family-balanced-real-backward-canary-v1":
        raise RealBackwardCanaryError("real-backward canary receipt schema is unsupported")
    if receipt.get("status") != "passed":
        raise RealBackwardCanaryError("real-backward canary did not report passed")
    if receipt.get("optimizer_updates") != 1 or receipt.get("checkpoint_produced") is not True:
        raise RealBackwardCanaryError("canary must prove exactly one optimizer update and a checkpoint")
    update_receipt = receipt.get("update_receipt")
    if not isinstance(update_receipt, dict) or update_receipt.get("controller_recorded") is not True or update_receipt.get("optimizer_update") != 1:
        raise RealBackwardCanaryError("canary lacks the exact WeightWatcher/controller update receipt")
    update_path = receipt.get("update_receipt_path")
    if not isinstance(update_path, str) or not Path(update_path).is_file():
        raise RealBackwardCanaryError("canary update receipt path is missing")
    if receipt.get("world_size") != expected_world_size:
        raise RealBackwardCanaryError("canary world size does not match the four-rank trainer contract")
    time_bound = receipt.get("time_bound_seconds")
    if not isinstance(time_bound, (int, float)) or isinstance(time_bound, bool) or not 0 < time_bound <= MAX_SECONDS:
        raise RealBackwardCanaryError("canary time bound is missing or exceeds 2400 seconds")
    elapsed = receipt.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0 or elapsed > time_bound:
        raise RealBackwardCanaryError("canary exceeded its declared time bound or lacks elapsed time")
    cost_bound = receipt.get("cost_bound_usd")
    if not isinstance(cost_bound, (int, float)) or isinstance(cost_bound, bool) or not 0 < cost_bound <= MAX_COST_USD:
        raise RealBackwardCanaryError("canary cost bound is missing or exceeds $9")
    cost = receipt.get("cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0 or cost > cost_bound:
        raise RealBackwardCanaryError("canary lacks an observed cost receipt within its declared bound")
    hourly = receipt.get("node_hourly_price_usd")
    if not isinstance(hourly, (int, float)) or isinstance(hourly, bool) or hourly <= 0:
        raise RealBackwardCanaryError("canary lacks the explicit node hourly price input")
    if abs(cost - hourly * float(receipt["elapsed_seconds"]) / 3600.0) > 0.01:
        raise RealBackwardCanaryError("canary cost is not computed from node hourly price and elapsed time")
    checkpoint = receipt.get("checkpoint_path")
    metadata = receipt.get("checkpoint_metadata")
    if not isinstance(checkpoint, str) or not checkpoint or not Path(checkpoint).is_dir() or Path(checkpoint).name != "trainer":
        raise RealBackwardCanaryError("canary checkpoint path is missing or not the trainer checkpoint directory")
    if not isinstance(metadata, str) or not Path(metadata).is_file() or type(receipt.get("checkpoint_file_count")) is not int or receipt["checkpoint_file_count"] < 1:
        raise RealBackwardCanaryError("canary checkpoint layout lacks DCP metadata or state files")
    output_dir = Path(checkpoint).parents[2]
    try:
        native = _native_evidence(output_dir, 1)
        if native["path"] != checkpoint or native["metadata"] != metadata or native["file_count"] != receipt["checkpoint_file_count"]:
            raise CheckpointContractError("native DCP evidence does not match the canary receipt")
        portable = _portable_upload_evidence(
            output_dir,
            1,
            expected_repo=os.environ.get("HF_CHECKPOINT_REPO"),
            expected_run=os.environ.get("HF_CHECKPOINT_RUN"),
        )
    except CheckpointContractError as exc:
        raise RealBackwardCanaryError(f"canary portable/native checkpoint contract is invalid: {exc}") from exc
    # Keep the public adapter evidence available to callers while retaining
    # the existing receipt schema for older launch records.
    receipt["portable_adapter"] = portable
    if receipt.get("processes_left_running") is not False:
        raise RealBackwardCanaryError("canary process cleanup is not proven")
    if receipt.get("cost_source") != "node_hourly_price_input":
        raise RealBackwardCanaryError("canary cost source must identify the explicit node hourly price input")
    return receipt
