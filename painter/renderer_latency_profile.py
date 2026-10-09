"""CPU-only latency and timeout profile for saved renderer receipts.

The profiler never launches Node, Chromium, or a renderer.  It reads
``painting.json`` receipts and their sibling ``program.js`` files, preserving
the distinction between an actual renderer deadline and a completed render.
Use it against a copied evidence directory when investigating a tail:

    python painter/renderer_latency_profile.py \
        painter/runs/diverse-flow-rl-20261009/local-evidence/recovered-v11/rl-rollouts/training
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

try:
    from painter.native_painting_renderer import _js_without_comments_and_strings
except ModuleNotFoundError:  # Direct ``python painter/renderer_latency_profile.py``.
    from native_painting_renderer import _js_without_comments_and_strings


_NO_LOOP_CALL_RE = re.compile(r"\bnoLoop\s*\(")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _has_no_loop(source: str) -> bool:
    """Return whether the saved source contains a direct noLoop call.

    This is a diagnostic classifier, not a JavaScript parser.  Reuse the
    renderer's comment/string stripping so mentions in prose or literals do
    not change the offline cause split.
    """

    return bool(_NO_LOOP_CALL_RE.search(_js_without_comments_and_strings(source)))


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def profile_receipts(root: Path, *, tail_threshold_seconds: float = 120.0) -> dict[str, Any]:
    """Summarize saved receipts without changing any evidence files."""

    if (
        isinstance(tail_threshold_seconds, bool)
        or not isinstance(tail_threshold_seconds, (int, float))
        or not math.isfinite(float(tail_threshold_seconds))
        or tail_threshold_seconds <= 0
    ):
        raise ValueError("tail_threshold_seconds must be a positive finite number")

    receipts = sorted(root.rglob("painting.json"))
    parsed_receipts: list[dict[str, Any]] = []
    elapsed: list[float] = []
    timeout_count = 0
    timeout_without_no_loop = 0
    timeout_with_no_loop = 0
    valid_slow_count = 0
    missing_source_count = 0
    invalid_receipt_count = 0
    tail_without_no_loop_count = 0
    tail_with_no_loop_count = 0

    for receipt_path in receipts:
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid_receipt_count += 1
            continue
        if not isinstance(receipt, dict):
            invalid_receipt_count += 1
            continue
        parsed_receipts.append(receipt)
        seconds = _finite_number(receipt.get("elapsed_seconds"))
        if seconds is not None:
            elapsed.append(seconds)
        source_path = receipt_path.with_name("program.js")
        source = None
        try:
            source = source_path.read_text()
        except OSError:
            missing_source_count += 1

        timed_out = bool(receipt.get("timed_out")) or receipt.get("error_code") == "render_timeout"
        if timed_out:
            timeout_count += 1
            if source is None or not _has_no_loop(source):
                timeout_without_no_loop += 1
            else:
                timeout_with_no_loop += 1
        if seconds is not None and seconds >= tail_threshold_seconds:
            if source is None or not _has_no_loop(source):
                tail_without_no_loop_count += 1
            else:
                tail_with_no_loop_count += 1
        if receipt.get("valid") is True and seconds is not None and seconds >= tail_threshold_seconds:
            valid_slow_count += 1

    # ``elapsed_seconds`` includes process startup and receipt cleanup, so a
    # receipt at or just beyond a requested cap is already affected by that
    # policy even though the renderer's inner watchdog tests ``< cap``.
    cap_tradeoffs = []
    for cap_seconds in (60.0, 90.0, 120.0):
        cap_total = cap_valid = cap_invalid = 0
        for receipt in parsed_receipts:
            seconds = _finite_number(receipt.get("elapsed_seconds"))
            if seconds is None or seconds < cap_seconds:
                continue
            cap_total += 1
            if receipt.get("valid") is True:
                cap_valid += 1
            else:
                cap_invalid += 1
        cap_tradeoffs.append(
            {
                "cap_seconds": cap_seconds,
                "receipts_at_or_above_cap": cap_total,
                "historically_valid_censored": cap_valid,
                "historically_invalid_shortened": cap_invalid,
            }
        )

    return {
        "schema": "renderer-latency-profile-v1",
        "root": str(root),
        "receipt_count": len(receipts),
        "receipts_with_elapsed_seconds": len(elapsed),
        "invalid_receipt_count": invalid_receipt_count,
        "missing_source_count": missing_source_count,
        "tail_threshold_seconds": float(tail_threshold_seconds),
        "elapsed_seconds": {
            "p50": _percentile(elapsed, 50),
            "p95": _percentile(elapsed, 95),
            "p99": _percentile(elapsed, 99),
            "max": max(elapsed) if elapsed else None,
        },
        "causes": {
            "render_timeout_count": timeout_count,
            "render_timeout_without_no_loop_count": timeout_without_no_loop,
            "render_timeout_with_no_loop_count": timeout_with_no_loop,
            "valid_over_tail_threshold_count": valid_slow_count,
        },
        "tail": {
            "receipts_at_or_above_threshold_count": tail_without_no_loop_count + tail_with_no_loop_count,
            "at_or_above_threshold_without_no_loop_count": tail_without_no_loop_count,
            "at_or_above_threshold_with_no_loop_count": tail_with_no_loop_count,
        },
        "cap_tradeoffs": cap_tradeoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Evidence directory to profile")
    parser.add_argument("--tail-threshold-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = profile_receipts(
        args.root,
        tail_threshold_seconds=args.tail_threshold_seconds,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
