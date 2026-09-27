"""Finite, node-local MiMo visual-judge worker for the existing RL queue.

Only training's active process lifetime should run this worker. It never starts
itself or schedules another run. Failed or malformed model calls leave their
requests unscored and write an operational error record, not a fake reward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_QUEUE = HERE.parent / "photo-curriculum-rl-20260922"
os.environ.setdefault("PAINTER_JUDGE_PROTOCOL", "aesthetic-pairwise-oss-v1")
os.environ.setdefault("PAINTER_JUDGE_MODEL", "XiaomiMiMo/MiMo-VL-7B-RL")
os.environ.setdefault("PAINTER_JUDGE_EFFORT", "none")
sys.path.insert(0, str(OLD_QUEUE))
import judge_queue  # noqa: E402
from oss_judge import digest, judge_view  # noqa: E402


def process_batch(root: Path, name: str, endpoint: str, model: str) -> dict:
    folder = root / "reward-batches" / name
    rows = json.loads((folder / "review-input.json").read_text())
    result = {"batch": name, "pairs": len(rows), "labeled": 0, "failed": 0}
    for row in rows:
        pair_id = row["id"]
        label_file = folder / f"labels-{pair_id}.json"
        if label_file.exists():
            result["labeled"] += 1
            continue
        images = {role: Path(row[role]) for role in ("reference", "A", "B")}
        if {role: digest(path) for role, path in images.items()} != row["image_sha256"]:
            raise ValueError(f"image identity mismatch for {pair_id}")
        try:
            if row["image_sha256"]["A"] == row["image_sha256"]["B"]:
                judgment = {"winner": "tie", "reason": "Identical candidate images",
                            "weaknesses_A": [], "weaknesses_B": []}
                evidence = {"status": "identical_images", "elapsed_seconds": 0}
            else:
                view = judge_view(endpoint=endpoint, model=model,
                                  reference=images["reference"], a=images["A"], b=images["B"])
                judgment = view.pop("label")
                evidence = {"status": "model_response", **view}
            document = {"protocol_version": judge_queue.PROTOCOL_VERSION,
                        "judge_model": judge_queue.JUDGE_MODEL,
                        "reasoning_effort": judge_queue.REASONING_EFFORT,
                        "pair_id": pair_id, "judgment": judgment}
            # Keep raw model output in a sidecar so the strict four-field label
            # envelope consumed by the queue does not acquire extra keys.
            judge_queue.write(folder / f"evidence-{pair_id}.json", evidence)
            judge_queue.write(label_file, document)
            result["labeled"] += 1
        except Exception as exc:
            result["failed"] += 1
            judge_queue.write(folder / f"error-{pair_id}.json", {
                "type": type(exc).__name__, "message": str(exc), "time": time.time()})
    receipt = judge_queue.receipts(root, name)
    installed = judge_queue.install(root, name)
    return {**result, "receipts": receipt, "installed": installed["installed"]}


def run(root: Path, endpoint: str, model: str, seconds: float,
        interval: float = 3, batch_limit: int = 16) -> dict:
    if seconds <= 0 or interval <= 0 or batch_limit <= 0:
        raise ValueError("seconds, interval, and batch_limit must be positive")
    deadline = time.monotonic() + seconds
    totals = {"batches": 0, "pairs": 0, "failed": 0, "installed": 0}
    while time.monotonic() < deadline:
        did_work = False
        # Resume incomplete batches first; assigned queue requests otherwise
        # cannot be exported again after a provider or process failure.
        for folder in sorted((root / "reward-batches").glob("oss-*")):
            if not (folder / "review-input.json").is_file():
                continue
            rows = json.loads((folder / "review-input.json").read_text())
            if all((folder / f"labels-{row['id']}.json").is_file() for row in rows):
                continue
            report = process_batch(root, folder.name, endpoint, model)
            print(json.dumps(report), flush=True)
            totals["failed"] += report["failed"]
            did_work = True
        batch_name = f"oss-{int(time.time() * 1000)}"
        exported = judge_queue.export(root, batch_name, limit=batch_limit)
        if exported["requests"]:
            judge_queue.stage(root, batch_name)
            report = process_batch(root, batch_name, endpoint, model)
            print(json.dumps(report), flush=True)
            totals["batches"] += 1
            totals["pairs"] += report["pairs"]
            totals["failed"] += report["failed"]
            totals["installed"] += report["installed"]
            did_work = True
        if not did_work:
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8200")
    parser.add_argument("--model", default="XiaomiMiMo/MiMo-VL-7B-RL")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--batch-limit", type=int, default=16)
    args = parser.parse_args()
    if judge_queue.JUDGE_MODEL != args.model:
        raise SystemExit("export PAINTER_JUDGE_MODEL matching --model before starting the worker")
    print(json.dumps(run(args.root.resolve(), args.endpoint, args.model,
                         args.seconds, batch_limit=args.batch_limit)), flush=True)


if __name__ == "__main__":
    main()
