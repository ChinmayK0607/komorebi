"""Paid, finite control for Gateway judgment and native reward import."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PAINTER_JUDGE_PROTOCOL", "aesthetic-pairwise-gateway-mimo-v1")
os.environ.setdefault("PAINTER_JUDGE_MODEL", "xiaomi/mimo-v2.6-flash")
os.environ.setdefault("PAINTER_JUDGE_EFFORT", "gateway-default")
sys.path.insert(0, str(HERE.parent / "photo-curriculum-rl-20260922"))
import judge_queue  # noqa: E402
from gateway_judge import load_key  # noqa: E402
from gateway_queue_worker import process_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sphere = next(args.references.glob("*sphere*.png"))
    bus = next(args.references.glob("*bus*.png"))
    request_id = judge_queue.enqueue_comparison(
        root, "bus-over-sphere-control", bus, sphere, bus,
    )
    batch = "gateway-control"
    exported = judge_queue.export(root, batch, limit=1)
    if exported["requests"] != 1:
        raise RuntimeError(f"expected one request: {exported}")
    judge_queue.stage(root, batch)
    report = process_batch(root, batch, load_key(args.key_file), "xiaomi/mimo-v2.6-flash")
    receipt = json.loads((root / "judge-queue" / request_id / "receipt.json").read_text())
    score = judge_queue.score_receipt(receipt, request_id, bus, sphere, bus)
    result = {"request_id": request_id, "score": score, "passed": score == 1.0,
              "prompt_tokens": report["prompt_tokens"],
              "completion_tokens": report["completion_tokens"],
              "judge_seconds": report["judge_seconds"]}
    (root / "control-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
