"""Make an auditable, dependency-free curve from native Prime metrics.

The optimizer's effective-batch reward is the primary series. Saved generated
traces are a separate denominator: they may include speculative/stale work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REWARD = "train/agg/effective/agent/reward/mean"
ALL_REWARD = "train/agg/all/agent/reward/mean"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def load_metrics(path: Path) -> list[dict]:
    batches: dict[int, dict] = {}
    trainer: dict[int, dict] = defaultdict(dict)
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            step = entry.get("step")
            if not isinstance(step, int) or step < 1:
                continue
            if REWARD in entry:
                if step in batches:
                    raise ValueError(f"duplicate effective reward for step {step}")
                value = finite(entry[REWARD])
                if value is None or not 0 <= value <= 1:
                    raise ValueError(f"invalid effective reward on line {line_number}")
                batches[step] = {
                    "step": step, "utc": datetime.fromtimestamp(
                        float(entry["time"]), timezone.utc).isoformat(),
                    "effective_reward": value,
                    "all_completed_reward": finite(entry.get(ALL_REWARD)),
                    "all_completed_cancelled_fraction": finite(entry.get(
                        "train/agg/all/agent/cancelled/mean")),
                    "all_completed_error_fraction": finite(entry.get(
                        "train/agg/all/agent/has_error/mean")),
                    "all_completed_truncated_fraction": finite(entry.get(
                        "train/agg/all/agent/is_truncated/mean")),
                    "admitted_rollouts": entry.get("progress/rollouts"),
                    "orchestrator_step_seconds": finite(entry.get("time/step")),
                }
            elif entry.get("producer") == "trainer":
                for source, target in (
                    ("optim/grad_norm", "gradient_norm"),
                    ("optim/lr", "learning_rate"),
                    ("perf/throughput", "trainer_tokens_per_second"),
                    ("time/wait_for_batch", "trainer_wait_for_batch_seconds"),
                    ("mismatch_kl/all/mean", "mismatch_kl"),
                ):
                    value = finite(entry.get(source))
                    if value is not None:
                        trainer[step][target] = value
    rows = [batches[step] | trainer.get(step, {}) for step in sorted(batches)]
    if not rows or [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("missing or noncontiguous optimizer reward steps")
    for index, row in enumerate(rows):
        row["trailing_5_reward"] = statistics.mean(
            item["effective_reward"] for item in rows[max(0, index - 4):index + 1])
    return rows


def load_generated_traces(path: Path, max_step: int) -> tuple[dict[int, dict], dict]:
    groups: dict[int, dict] = defaultdict(lambda: {
        "count": 0, "valid": 0, "scored": 0, "score_sum": 0.0,
        "families": defaultdict(int),
    })
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            for trace in record.get("traces", []):
                info = trace.get("info") or {}
                if not info.get("turns"):
                    continue
                step = (info.get("dispatch") or {}).get("step")
                if not isinstance(step, int) or not 1 <= step <= max_step:
                    continue
                group = groups[step]
                group["count"] += 1
                group["valid"] += info.get("final_canvas_valid") is True
                group["families"][str(info.get("family") or "unknown")] += 1
                score = finite(info.get("final_canvas_score"))
                if score is not None:
                    group["scored"] += 1
                    group["score_sum"] += score
    result = {
        step: {"generated_traces": group["count"],
               "generated_valid": group["valid"],
               "generated_valid_fraction": group["valid"] / group["count"],
               "generated_mean_reward": (group["score_sum"] / group["scored"]
                                         if group["scored"] else None),
               "generated_family_counts": dict(group["families"])}
        for step, group in sorted(groups.items())
    }
    total = sum(group["generated_traces"] for group in result.values())
    valid = sum(group["generated_valid"] for group in result.values())
    return result, {"generated_traces": total, "valid_final_canvases": valid,
                    "valid_fraction": valid / total if total else None,
                    "through_dispatch_step": max_step,
                    "scope": "saved generated traces, including possible speculative/stale work"}


def _chart(rows: list[dict]) -> str:
    width, height = 1200, 690
    left, right, top = 80, 1160, 80
    reward_bottom, valid_top, valid_bottom = 370, 465, 625
    max_step = max(2, rows[-1]["step"])

    def x(step: int) -> float:
        return left + (step - 1) * (right - left) / (max_step - 1)

    def y(value: float, low: float, high: float) -> float:
        return high - value * (high - low)

    def poly(field: str, color: str, low: float, high: float,
             *, dash: str = "", width_px: int = 3) -> str:
        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for row in rows:
            value = row.get(field)
            if value is None:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append((x(row["step"]), y(value, low, high)))
        if current:
            segments.append(current)
        marks = []
        for segment in segments:
            if len(segment) == 1:
                xx, yy = segment[0]
                marks.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>')
            else:
                points = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in segment)
                marks.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                             f'stroke-width="{width_px}" stroke-linejoin="round" '
                             f'stroke-linecap="round"'
                             + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
        return "\n".join(marks)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             'role="img" aria-label="RL reward and generated canvas validity by optimizer step">',
             '<rect width="100%" height="100%" fill="#171b24"/>',
             '<style>text{font-family:system-ui,sans-serif;fill:#e5e7eb} '
             '.muted{fill:#a9b0bf;font-size:14px}.title{font-size:23px;font-weight:700}</style>',
             '<text x="80" y="35" class="title">Text-paint RL: reward by optimizer step</text>',
             '<text x="80" y="58" class="muted">Primary: admitted effective-batch reward. '
             '0.5 is an effective tie with frozen SFT.</text>']
    for low, high in ((top, reward_bottom), (valid_top, valid_bottom)):
        for fraction in (0, .25, .5, .75, 1):
            yy = y(fraction, low, high)
            parts.append(f'<line x1="{left}" x2="{right}" y1="{yy:.1f}" y2="{yy:.1f}" '
                         'stroke="#343c4a" stroke-width="1"/>')
            parts.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" '
                         f'class="muted">{fraction:g}</text>')
    parts.append(f'<line x1="{left}" x2="{right}" y1="{y(.5,top,reward_bottom):.1f}" '
                 f'y2="{y(.5,top,reward_bottom):.1f}" stroke="#d4d9e2" '
                 'stroke-width="1.5" stroke-dasharray="6 6"/>')
    parts.append(poly("effective_reward", "#75b9ff", top, reward_bottom, width_px=3))
    parts.append(poly("trailing_5_reward", "#ffb26f", top, reward_bottom, width_px=4))
    parts.append(poly("generated_valid_fraction", "#c694ff", valid_top, valid_bottom,
                      width_px=3))
    parts.extend([
        '<text x="80" y="405" class="muted">Blue: admitted reward · Orange: trailing 5-step mean '
        '· Dashed: tie</text>',
        '<text x="80" y="440" class="muted">Purple: final-canvas validity by generation '
        'dispatch step (different episodes and denominator)</text>',
    ])
    tick_step = max(1, math.ceil(max_step / 10))
    for step in range(1, max_step + 1):
        if step != 1 and step != max_step and (step - 1) % tick_step:
            continue
        xx = x(step)
        parts.append(f'<text x="{xx:.1f}" y="{valid_bottom+27}" text-anchor="middle" '
                     f'class="muted">{step}</text>')
    parts.append('<text x="620" y="682" text-anchor="middle" class="muted">Step index '
                 '(optimizer above; generation dispatch below)</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def build(metrics: Path, traces: Path | None, output: Path) -> dict:
    rows = load_metrics(metrics)
    generated, generated_total = ({}, {"scope": "no trace stream supplied"})
    if traces is not None:
        generated, generated_total = load_generated_traces(traces, rows[-1]["step"])
    for row in rows:
        row.update(generated.get(row["step"], {}))
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "reward_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, sort_keys=True)
                                    if isinstance(value, dict) else value)
                             for key, value in row.items()})
    values = [row["effective_reward"] for row in rows]
    half = len(values) // 2
    summary = {
        "schema": "painter-reward-curve-v1",
        "metrics_sha256": digest(metrics),
        "traces_sha256": digest(traces) if traces else None,
        "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
        "admitted_batch_reward_mean": statistics.mean(values),
        "first_half_batch_mean": statistics.mean(values[:half]) if half else None,
        "second_half_batch_mean": statistics.mean(values[half:]) if half else None,
        "latest_batch_reward": values[-1],
        "latest_trailing_5_mean": rows[-1]["trailing_5_reward"],
        "generated_trace_audit": generated_total,
        "interpretation": "Training-batch reward is neither a held-out quality curve nor directly comparable across changing curriculum tasks.",
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2,
                                                    sort_keys=True) + "\n")
    (output / "reward_curve.svg").write_text(_chart(rows))
    note = html.escape(summary["interpretation"])
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Painter RL reward curve</title>'
        '<style>body{font:16px system-ui;background:#171b24;color:#e5e7eb;'
        'max-width:1250px;margin:24px auto;padding:0 20px}a{color:#75b9ff}'
        'img{width:100%}p{line-height:1.5}</style>'
        f'<h1>Painter RL reward curve · steps 1–{rows[-1]["step"]}</h1>'
        f'<p>{note}</p><img src="reward_curve.svg" alt="Reward and validity curves">'
        '<p><a href="reward_curve.csv">CSV</a> · '
        '<a href="summary.json">Summary and hashes</a></p>\n')
    return {key: value for key, value in summary.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.metrics, args.traces, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
