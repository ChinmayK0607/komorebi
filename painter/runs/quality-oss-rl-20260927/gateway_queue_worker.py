"""Finite Vercel MiMo Flash worker for the existing two-orientation RL queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PAINTER_JUDGE_PROTOCOL", "aesthetic-pairwise-gateway-mimo-v1")
os.environ.setdefault("PAINTER_JUDGE_MODEL", "xiaomi/mimo-v2.6-flash")
os.environ.setdefault("PAINTER_JUDGE_EFFORT", "gateway-default")
sys.path.insert(0, str(HERE.parent / "photo-curriculum-rl-20260922"))
import judge_queue  # noqa: E402
from gateway_judge import MODEL, digest, judge_view, load_key  # noqa: E402


def _is_installed(root: Path, folder: Path) -> bool:
    private = folder / "private-key.json"
    if not private.is_file():
        return False
    rows = json.loads(private.read_text())
    return bool(rows) and all(
        (root / "judge-queue" / row["request_id"] / "receipt.json").is_file()
        for row in rows
    )


def _label_row(folder: Path, row: dict, key: str, model: str) -> dict:
    pair_id = row["id"]
    label_file = folder / f"labels-{pair_id}.json"
    if label_file.exists():
        return {"labeled": 1, "failed": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "judge_seconds": 0.0}
    report = {"labeled": 0, "failed": 0, "prompt_tokens": 0,
              "completion_tokens": 0, "judge_seconds": 0.0}
    try:
        images = {role: Path(row[role]) for role in ("reference", "A", "B")}
        if {role: digest(path) for role, path in images.items()} != row["image_sha256"]:
            raise ValueError(f"image identity mismatch for {pair_id}")
        if row["image_sha256"]["A"] == row["image_sha256"]["B"]:
            judgment = {"winner": "tie", "reason": "Identical candidate images",
                        "weaknesses_A": [], "weaknesses_B": []}
            evidence = {"status": "identical_images", "elapsed_seconds": 0}
        else:
            view = judge_view(key=key, model=model, reference=images["reference"],
                              a=images["A"], b=images["B"])
            judgment = view.pop("label")
            evidence = {"status": "model_response", **view}
            usage = view.get("usage") or {}
            report["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
            report["completion_tokens"] = usage.get("completion_tokens", 0) or 0
            report["judge_seconds"] = view["elapsed_seconds"]
        document = {"protocol_version": judge_queue.PROTOCOL_VERSION,
                    "judge_model": judge_queue.JUDGE_MODEL,
                    "reasoning_effort": judge_queue.REASONING_EFFORT,
                    "pair_id": pair_id, "judgment": judgment}
        judge_queue.write(folder / f"evidence-{pair_id}.json", evidence)
        judge_queue.write(label_file, document)
        report["labeled"] = 1
        print(json.dumps({"pair": pair_id, "winner": judgment["winner"],
                          "elapsed_seconds": round(evidence["elapsed_seconds"], 2),
                          "usage": evidence.get("usage")}), flush=True)
    except Exception as exc:
        report["failed"] = 1
        # Gateway HTTP bodies and request headers are never logged.
        judge_queue.write(folder / f"error-{pair_id}.json", {
            "type": type(exc).__name__, "message": str(exc), "time": time.time()})
        print(json.dumps({"pair": pair_id, "error_type": type(exc).__name__,
                          "error": str(exc)}), flush=True)
    return report


def process_batch(root: Path, name: str, key: str, model: str, workers: int = 8) -> dict:
    folder = root / "reward-batches" / name
    rows = json.loads((folder / "review-input.json").read_text())
    result = {"batch": name, "pairs": len(rows), "labeled": 0,
              "failed": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "judge_seconds": 0.0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_label_row, folder, row, key, model) for row in rows]
        for future in as_completed(futures):
            row_report = future.result()
            for metric, value in row_report.items():
                result[metric] += value
    receipt = judge_queue.receipts(root, name)
    installed = judge_queue.install(root, name)
    return {**result, "receipts": receipt, "installed": installed["installed"]}


def run(root: Path, key: str, model: str, seconds: float,
        interval: float = 3, batch_limit: int = 16, workers: int = 8) -> dict:
    if seconds <= 0 or interval <= 0 or batch_limit <= 0 or workers <= 0:
        raise ValueError("seconds, interval, batch_limit, and workers must be positive")
    deadline = time.monotonic() + seconds
    totals = {"batches": 0, "pairs": 0, "failed": 0, "installed": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "judge_seconds": 0.0}
    while time.monotonic() < deadline:
        did_work = False
        # Always revisit unfinished batches, including fully labeled batches
        # interrupted before receipt installation.
        for folder in sorted((root / "reward-batches").glob("gateway-*")):
            if not (folder / "review-input.json").is_file() or _is_installed(root, folder):
                continue
            report = process_batch(root, folder.name, key, model, workers)
            print(json.dumps(report), flush=True)
            for metric in ("failed", "installed", "prompt_tokens", "completion_tokens", "judge_seconds"):
                totals[metric] += report[metric]
            did_work = True
        batch_name = f"gateway-{int(time.time() * 1000)}"
        exported = judge_queue.export(root, batch_name, limit=batch_limit)
        if exported["requests"]:
            judge_queue.stage(root, batch_name)
            report = process_batch(root, batch_name, key, model, workers)
            print(json.dumps(report), flush=True)
            totals["batches"] += 1
            totals["pairs"] += report["pairs"]
            for metric in ("failed", "installed", "prompt_tokens", "completion_tokens", "judge_seconds"):
                totals[metric] += report[metric]
            did_work = True
        if not did_work:
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--batch-limit", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if judge_queue.JUDGE_MODEL != args.model:
        raise SystemExit("PAINTER_JUDGE_MODEL must match --model")
    key = load_key(args.key_file)
    print(json.dumps(run(args.root.resolve(), key, args.model,
                         args.seconds, batch_limit=args.batch_limit,
                         workers=args.workers)), flush=True)


if __name__ == "__main__":
    main()
