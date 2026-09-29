#!/usr/bin/env python3
"""Make a bounded, portable visual audit of saved RL rollout failures.

The full trace stream remains the source of truth. This gallery samples each
failure class across dispatch time and shows every turn for selected traces.
Generated traces can be speculative and must not be counted as optimizer data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import shutil

from profile_continuation import classify_turn_error, finite


def safe_source(path: object, root: Path) -> Path | None:
    if not isinstance(path, str):
        return None
    candidate = Path(path).resolve()
    if candidate.is_relative_to(root.resolve()) and candidate.is_file() and not candidate.is_symlink():
        return candidate
    return None


def evenly_spaced(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    return [items[i] for i in sorted(indices)]


def collect(stream: Path, root: Path, per_class: int) -> tuple[list[dict], dict]:
    records: list[dict] = []
    failure_buckets: dict[str, list[int]] = defaultdict(list)
    valid_buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    judge_disagreements: list[tuple[float, int]] = []
    counts: dict[str, int] = defaultdict(int)
    for line in stream.read_text().splitlines():
        if not line:
            continue
        for trace in json.loads(line).get("traces", []):
            info = trace.get("info") or {}
            turns = info.get("turns") or []
            if not turns:
                continue
            step = (info.get("dispatch") or {}).get("step")
            record = {"task_id": str(info.get("task_id") or "unknown"),
                      "trace_id": str(trace.get("id") or "unknown"),
                      "family": str(info.get("family") or "unknown"),
                      "dispatch_step": step if isinstance(step, int) else None,
                      "valid_final": info.get("final_canvas_valid") is True,
                      "score": finite(info.get("final_canvas_score")),
                      "judge_order_gap": None,
                      "reference": info.get("reference"), "turns": turns}
            evidence = info.get("score_evidence") or {}
            receipt = safe_source(evidence.get("receipt_path"), root) if isinstance(evidence, dict) else None
            if receipt and receipt.suffix == ".json":
                record["judge_order_gap"] = finite(json.loads(receipt.read_text()).get("order_gap"))
            index = len(records)
            records.append(record)
            counts["generated_traces"] += 1
            counts["valid_final_traces"] += record["valid_final"]
            seen = set()
            for turn in turns:
                if turn.get("valid") is True:
                    continue
                category = classify_turn_error(turn)
                counts["invalid_turn:" + category] += 1
                if category not in seen:
                    failure_buckets[category].append(index)
                    seen.add(category)
            if record["valid_final"] and record["score"] is not None:
                valid_buckets[record["family"]].append((record["score"], index))
            if record["judge_order_gap"] is not None and record["judge_order_gap"] >= 1 / 3:
                judge_disagreements.append((record["judge_order_gap"], index))
    selected: dict[int, set[str]] = defaultdict(set)
    for category, indices in sorted(failure_buckets.items()):
        ordered = sorted(indices, key=lambda i: (records[i]["dispatch_step"] or 0, i))
        for index in evenly_spaced(ordered, per_class):
            selected[index].add("failure:" + category)
    for family, scored in sorted(valid_buckets.items()):
        ordered = sorted(scored)
        for score, index in ordered[:2]:
            selected[index].add("valid-low:" + family)
        for score, index in ordered[-2:]:
            selected[index].add("valid-high:" + family)
    for gap, index in sorted(judge_disagreements, reverse=True)[:8]:
        selected[index].add("judge-order-disagreement")
    chosen = []
    for index in sorted(selected, key=lambda i: (records[i]["dispatch_step"] or 0, i)):
        record = dict(records[index])
        record["selection"] = sorted(selected[index])
        chosen.append(record)
    return chosen, {"denominator": "all saved generated traces with turns, including speculative work",
                    "counts": dict(sorted(counts.items())),
                    "selected_traces": len(chosen), "per_failure_class_limit": per_class,
                    "source_sha256": hashlib.sha256(stream.read_bytes()).hexdigest()}


def build(stream: Path, root: Path, output: Path, per_class: int) -> dict:
    chosen, summary = collect(stream, root, per_class)
    if output.exists():
        prior = output / "summary.json"
        if prior.is_file() and (output / "index.html").is_file():
            saved = json.loads(prior.read_text())
            if (saved.get("source_sha256") == summary["source_sha256"]
                    and saved.get("per_failure_class_limit") == per_class):
                return saved
        raise FileExistsError("audit output exists for a different or incomplete trace stream")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    cards = []
    index_rows = []
    for position, record in enumerate(chosen, start=1):
        brief_path = safe_source(record["reference"], root)
        brief = brief_path.read_text(errors="replace")[:1500] if brief_path and brief_path.suffix == ".txt" else ""
        turns_html = []
        for turn_index, turn in enumerate(record["turns"], start=1):
            image_path = safe_source(turn.get("canvas"), root)
            image_html = ""
            if image_path and image_path.suffix.lower() == ".png":
                name = f"{position:03d}-{turn_index:02d}.png"
                shutil.copyfile(image_path, output / "images" / name)
                image_html = f'<img loading="lazy" src="images/{name}" alt="Turn {turn_index} canvas">'
            errors = turn.get("error") or turn.get("renderer_errors") or []
            if isinstance(errors, str):
                errors = [errors]
            category = "valid" if turn.get("valid") is True else classify_turn_error(turn)
            turns_html.append(
                f'<section class="turn"><h4>Turn {turn_index}: {html.escape(category)}</h4>'
                f'{image_html}<p class="error">{html.escape(" | ".join(map(str, errors))[:1200])}</p>'
                f'<details><summary>Generated reply</summary><pre>{html.escape(str(turn.get("reply") or ""))}</pre></details></section>')
        score = record["score"]
        score_text = "unknown" if score is None else f"{score:.3f}"
        gap = record["judge_order_gap"]
        gap_text = "unknown" if gap is None else f"{gap:.3f}"
        heading = (f'#{position} · {record["task_id"]} · {record["family"]} · '
                   f'dispatch step {record["dispatch_step"]} · score {score_text} · order gap {gap_text}')
        cards.append(f'<article id="case-{position}"><h3>{html.escape(heading)}</h3>'
                     f'<p>Selected for: {html.escape(", ".join(record["selection"]))}</p>'
                     f'<p class="brief">{html.escape(brief)}</p>' + "".join(turns_html) + "</article>")
        index_rows.append({k: record[k] for k in ("task_id", "trace_id", "family", "dispatch_step",
                                                   "valid_final", "score", "judge_order_gap", "selection")})
    page = ("<!doctype html><meta charset='utf-8'><title>RL rollout audit</title>"
            "<style>body{font:16px system-ui;background:#191a1d;color:#eee;max-width:1400px;margin:auto;padding:2rem}"
            "article{background:#282a30;border:1px solid #555;border-radius:12px;margin:2rem 0;padding:1rem}"
            "section.turn{display:inline-block;vertical-align:top;width:min(31%,400px);min-width:290px;margin:.5%}"
            "img{width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:25rem;overflow:auto}"
            ".error{color:#ffb5aa;overflow-wrap:anywhere}.brief{white-space:pre-wrap;color:#c7d6ff}</style>"
            "<h1>RL rollout audit</h1><p>Sampled generated traces, including speculative work. "
            "They are not the admitted training-batch denominator.</p>" + "".join(cards))
    (output / "index.html").write_text(page)
    summary["cases"] = index_rows
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.per_class <= 20:
        raise SystemExit("per-class must be 1..20")
    print(json.dumps(build(args.stream, args.run_root, args.output, args.per_class),
                     sort_keys=True)[:500])


if __name__ == "__main__":
    main()
