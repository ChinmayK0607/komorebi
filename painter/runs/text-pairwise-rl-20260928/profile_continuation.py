#!/usr/bin/env python3
"""Profile an RL continuation from immutable metrics, traces, judge receipts and GPU samples.

Generated traces may contain stale/speculative work. Their counts never stand
in for the admitted optimizer-batch denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REWARD = "train/agg/effective/agent/reward/mean"
PHASES = {
    "model": "train/agg/effective/agent/timing/agent/model/mean",
    "harness": "train/agg/effective/agent/timing/agent/harness/mean",
    "scoring": "train/agg/effective/agent/timing/scoring/mean",
    "total_rollout": "train/agg/effective/agent/timing/total/mean",
    "optimizer_step": "time/step",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Step\s+(\d+)\s*\|\s*(?:(\d+)m\s*)?([\d.]+)s")
WAIT = re.compile(r"Trainer waited\s*(?:(\d+)m\s*)?([\d.]+)s for a batch")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def metrics_profile(path: Path) -> dict:
    batches: dict[int, dict] = {}
    trainer_updates: dict[int, dict] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("producer") == "trainer" and isinstance(row.get("step"), int):
            for key in ("optim/grad_norm", "optim/lr", "mismatch_kl/all/mean"):
                if (value := finite(row.get(key))) is not None:
                    trainer_updates[row["step"]][key] = value
        if REWARD not in row:
            continue
        step = row.get("step")
        if not isinstance(step, int) or step in batches:
            raise ValueError("duplicate or invalid optimizer-step reward")
        batches[step] = row
    ordered = [batches[key] for key in sorted(batches)]
    if not ordered or [row["step"] for row in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("missing/noncontiguous admitted optimizer steps")
    rewards = [float(row[REWARD]) for row in ordered]
    half = len(rewards) // 2
    phases = {name: mean([v for row in ordered if (v := finite(row.get(key))) is not None])
              for name, key in PHASES.items()}
    gradient_norms = [trainer_updates[step]["optim/grad_norm"] for step in sorted(batches)
                      if "optim/grad_norm" in trainer_updates[step]]
    kl = [trainer_updates[step]["mismatch_kl/all/mean"] for step in sorted(batches)
          if "mismatch_kl/all/mean" in trainer_updates[step]]
    lr = [trainer_updates[step]["optim/lr"] for step in sorted(batches)
          if "optim/lr" in trainer_updates[step]]
    return {
        "denominator": "admitted optimizer batches only",
        "steps": len(ordered), "reward_mean": mean(rewards),
        "first_half_reward_mean": mean(rewards[:half]),
        "second_half_reward_mean": mean(rewards[half:]),
        "last_five_reward_mean": mean(rewards[-5:]),
        "reward_min": min(rewards), "reward_max": max(rewards),
        "phase_seconds_mean_per_admitted_batch": phases,
        "gradient_norm_steps": len(gradient_norms),
        "nonzero_gradient_norm_steps": sum(v > 0 for v in gradient_norms),
        "gradient_norm_mean": mean(gradient_norms),
        "mismatch_kl_mean": mean(kl),
        "learning_rate_first_last": [lr[0], lr[-1]] if lr else None,
        "mean_admission_rate": mean([v for row in ordered if (v := finite(row.get(
            "curriculum/text-paint-fixed-sft-pairwise/admission_rate"))) is not None]),
        "off_policy_dropped_total": sum(v for row in ordered if (v := finite(row.get("off_policy/dropped"))) is not None),
        "all_completed_cancelled_fraction_mean": mean([v for row in ordered if (v := finite(row.get(
            "train/agg/all/agent/cancelled/mean"))) is not None]),
        "all_completed_error_fraction_mean": mean([v for row in ordered if (v := finite(row.get(
            "train/agg/all/agent/has_error/mean"))) is not None]),
        "source_sha256": sha(path),
    }


def traces_profile(path: Path, judge_root: Path) -> dict:
    families: dict[str, Counter] = defaultdict(Counter)
    failures = Counter()
    turn_failures = Counter()
    error_classes = Counter()
    dispatch_steps = Counter()
    judge_scores: dict[str, list[float]] = defaultdict(list)
    judge_order_gap: dict[str, list[float]] = defaultdict(list)
    large_order_gap_cases: list[dict] = []
    valid_render_seconds: list[float] = []
    painting_seconds: list[float] = []
    all_count = valid_count = turn_count = 0
    for line in path.read_text().splitlines():
        if not line:
            continue
        for trace in json.loads(line).get("traces", []):
            info = trace.get("info") or {}
            turns = info.get("turns") or []
            if not turns:
                continue
            all_count += 1
            valid = info.get("final_canvas_valid") is True
            valid_count += valid
            family = str(info.get("family") or "unknown")
            families[family]["generated"] += 1
            families[family]["valid_final"] += valid
            if valid:
                score = finite(info.get("final_canvas_score"))
                if score is not None:
                    judge_scores[family].append(score)
                evidence = info.get("score_evidence") or {}
                receipt = evidence.get("receipt_path") if isinstance(evidence, dict) else None
                receipt_path = Path(receipt).resolve() if isinstance(receipt, str) else None
                if (receipt_path is not None and receipt_path.is_relative_to(judge_root.resolve())
                        and receipt_path.is_file()):
                    judged = json.loads(receipt_path.read_text())
                    if (gap := finite(judged.get("order_gap"))) is not None:
                        judge_order_gap[family].append(gap)
                        if gap >= 1 / 3:
                            large_order_gap_cases.append({"task_id": str(info.get("task_id")),
                                                          "family": family, "order_gap": gap})
            if not valid:
                failures[str(info.get("failure_kind") or "unknown")] += 1
            dispatch_steps[str((info.get("dispatch") or {}).get("step") or "unknown")] += 1
            for turn in turns:
                turn_count += 1
                canvas = turn.get("canvas")
                canvas_path = Path(canvas).resolve() if isinstance(canvas, str) else None
                if (canvas_path is not None and canvas_path.is_relative_to(judge_root.resolve())
                        and (receipt_path := canvas_path.with_suffix(".json")).is_file()):
                    render = json.loads(receipt_path.read_text())
                    if (v := finite(render.get("elapsed_seconds"))) is not None:
                        valid_render_seconds.append(v)
                    if (v := finite(render.get("painting_seconds"))) is not None:
                        painting_seconds.append(v)
                if turn.get("valid") is not True:
                    turn_failures[str(turn.get("failure_kind") or "unknown")] += 1
                    error_classes[classify_turn_error(turn)] += 1
    return {
        "denominator": "saved generated traces with turns, possibly speculative or stale",
        "generated": all_count, "valid_final": valid_count,
        "valid_final_fraction": valid_count / all_count if all_count else None,
        "turns": turn_count, "family_counts": {
            k: dict(v) | {"valid_canvas_mean_judge_score": mean(judge_scores[k]),
                          "mean_judge_order_gap": mean(judge_order_gap[k]),
                          "large_judge_order_gaps": sum(g >= 1 / 3 for g in judge_order_gap[k])}
            for k, v in sorted(families.items())},
        "final_failure_kinds": dict(failures.most_common()),
        "invalid_turn_failure_kinds": dict(turn_failures.most_common()),
        "invalid_turn_error_classes": dict(error_classes.most_common()),
        "valid_render_receipt_count": len(valid_render_seconds),
        "valid_render_elapsed_seconds_mean": mean(valid_render_seconds),
        "valid_render_elapsed_seconds_p50": percentile(valid_render_seconds, .5),
        "valid_render_elapsed_seconds_p90": percentile(valid_render_seconds, .9),
        "valid_render_painting_seconds_mean": mean(painting_seconds),
        "dispatch_steps": dict(dispatch_steps), "source_sha256": sha(path),
        "large_order_gap_cases": large_order_gap_cases,
    }


def classify_turn_error(turn: dict) -> str:
    """One primary class per invalid turn; preserve broad source kind separately."""
    raw = turn.get("error") or []
    if not isinstance(raw, list):
        raw = [str(raw)]
    other = turn.get("renderer_errors") or []
    if not isinstance(other, list):
        other = [str(other)]
    message = " ".join(str(item) for item in raw + other).lower()
    kind = str(turn.get("failure_kind") or "")
    if kind == "program_timeout" or "render limit" in message or "timed out" in message:
        return "render_timeout"
    if "syntaxerror" in message or "unexpected end" in message:
        return "syntax_error"
    if 'brush "' in message and "not found" in message:
        return "unknown_brush_name"
    if "brush." in message and "not a function" in message:
        return "unsupported_brush_api"
    if "referenceerror" in message or "is not defined" in message:
        return "undefined_symbol"
    if "invalid color" in message:
        return "invalid_color"
    if "typeerror" in message or "not iterable" in message:
        return "type_error"
    if kind == "finish_without_canvas":
        return "finish_without_canvas"
    if kind == "blank_canvas" or "no canvas" in message:
        return "blank_or_no_canvas"
    return "other_program_error"


def judge_profile(root: Path) -> dict:
    statuses = Counter()
    gaps: list[float] = []
    scores: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    cost_missing = 0
    paths = sorted(root.glob("*/text-judge.json"))
    for path in paths:
        row = json.loads(path.read_text())
        statuses[str(row.get("status") or "unknown")] += 1
        if (v := finite(row.get("order_gap"))) is not None:
            gaps.append(v)
        if (v := finite(row.get("score"))) is not None:
            scores.append(v)
        for view in row.get("views") or []:
            if (v := finite(view.get("elapsed_seconds"))) is not None:
                latencies.append(v)
            usage = view.get("usage") or {}
            cost = finite(usage.get("cost")) if isinstance(usage, dict) else None
            if cost is None:
                cost_missing += 1
            else:
                costs.append(cost)
    return {
        "denominator": "saved two-order judge receipts, including speculative work",
        "receipts": len(paths), "statuses": dict(statuses),
        "mean_score": mean(scores), "mean_order_gap": mean(gaps),
        "order_gap_at_least_one_third": sum(v >= 1 / 3 for v in gaps),
        "order_gap_denominator": len(gaps),
        "view_latency_seconds_mean": mean(latencies), "view_latency_denominator": len(latencies),
        "views_missing_cost": cost_missing,
        "reported_judge_cost_usd": sum(costs),
        "reported_judge_cost_view_denominator": len(costs),
    }


def gpu_profile(paths: list[Path]) -> dict:
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        with path.open(newline="") as source:
            reader = csv.DictReader(source, skipinitialspace=True)
            for row in reader:
                try:
                    index = int(row["index"])
                    time = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                    util = float(row["utilization.gpu [%]"].strip().rstrip(" %"))
                    power = float(row["power.draw [W]"].strip().rstrip(" W"))
                except (KeyError, ValueError):
                    continue
                by_gpu[index].append({"time": time, "util": util, "power": power})
    result = {}
    for index, rows in sorted(by_gpu.items()):
        rows.sort(key=lambda row: row["time"])
        unique = list({row["time"]: row for row in rows}.values())
        unique.sort(key=lambda row: row["time"])
        result[str(index)] = {
            "samples": len(unique), "window_start": unique[0]["time"].isoformat(),
            "window_end": unique[-1]["time"].isoformat(),
            "window_seconds": (unique[-1]["time"] - unique[0]["time"]).total_seconds(),
            "utilization_mean_percent": mean([row["util"] for row in unique]),
            "busy_at_least_90_percent_fraction": sum(row["util"] >= 90 for row in unique) / len(unique),
            "power_mean_watts": mean([row["power"] for row in unique]),
            "denominator": "all valid time samples for this GPU in the listed CSV files",
        }
    return result


def trainer_log_profile(path: Path) -> dict:
    waits: list[float] = []
    steps: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        plain = ANSI.sub("", line)
        if match := WAIT.search(plain):
            waits.append(60 * float(match.group(1) or 0) + float(match.group(2)))
        if match := STEP.search(plain):
            steps.append(60 * float(match.group(2) or 0) + float(match.group(3)))
    return {"completed_step_lines": len(steps), "trainer_wait_warnings": len(waits),
            "mean_completed_step_seconds": mean(steps),
            "mean_warned_wait_for_batch_seconds": mean(waits),
            "denominator": "parsed trainer SUCCESS step lines and wait warnings, respectively",
            "source_sha256": sha(path)}


def build(args: argparse.Namespace) -> dict:
    return {"schema": "painter.continuation-profile.v1",
            "metrics": metrics_profile(args.metrics),
            "generated_traces": traces_profile(args.traces, args.judges),
            "judges": judge_profile(args.judges),
            "gpu": gpu_profile(args.gpu_csv),
            "trainer_log": trainer_log_profile(args.trainer_log),
            "caveat": "Online reward and judge receipts are not held-out visual preference; phase means may overlap in parallel execution."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metrics", "traces", "judges", "trainer-log"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "steps": result["metrics"]["steps"],
                      "generated": result["generated_traces"]["generated"],
                      "judges": result["judges"]["receipts"]}))


if __name__ == "__main__":
    main()
