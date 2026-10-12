"""Bounded one-update real-backward canary receipts.

This module only validates receipts.  The launch path enables the mode through
``launch_prepared.py --real-backward-canary``; no canary is run during offline
preparation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_SECONDS = 1800
MAX_COST_USD = 5.0


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
    elapsed = receipt.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0 or elapsed > MAX_SECONDS:
        raise RealBackwardCanaryError("canary exceeded its 1800 second bound or lacks elapsed time")
    cost = receipt.get("cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0 or cost > MAX_COST_USD:
        raise RealBackwardCanaryError("canary lacks an observed cost receipt within the $5 bound")
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
    if receipt.get("processes_left_running") is not False:
        raise RealBackwardCanaryError("canary process cleanup is not proven")
    if receipt.get("cost_source") != "node_hourly_price_input":
        raise RealBackwardCanaryError("canary cost source must identify the explicit node hourly price input")
    return receipt
