"""Summarize saved rollout/action failures; never treat generated traces as gradients."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def error_class(value: str) -> str:
    e = value.lower()
    if "no complete javascript" in e:
        return "missing_program"
    if "syntaxerror" in e or "unexpected token" in e:
        return "javascript_syntax"
    if "render limit" in e or "timed out" in e:
        return "render_timeout"
    if "brush" in e and ("not found" in e or "not a function" in e or
                         "no brush or color" in e):
        return "brush_api"
    if "blank or nearly uniform" in e:
        return "blank_canvas"
    if "no canvas produced" in e:
        return "no_canvas"
    if "referenceerror" in e or "typeerror" in e or "rangeerror" in e:
        return "javascript_runtime"
    return "other"


def audit(stream: Path, judge_root: Path) -> dict:
    traces = []
    for line in stream.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for trace in record.get("traces", []):
            info = trace.get("info") or {}
            if info.get("turns"):
                traces.append(trace)
    if not traces:
        raise ValueError("no complete painting traces")
    statuses = collections.Counter()
    histories = collections.Counter()
    failures = collections.Counter()
    per_turn = [collections.Counter() for _ in range(3)]
    families = collections.defaultdict(lambda: collections.Counter())
    steps = collections.defaultdict(lambda: {"traces": 0, "valid": 0,
                                             "briefs": set(), "rewards": []})
    tokens = []
    for trace in traces:
        info = trace["info"]
        turns = info["turns"]
        valid = info.get("final_canvas_valid") is True
        statuses[info.get("reward_status") or "unknown"] += 1
        histories["".join("V" if turn.get("valid") else "X" for turn in turns)] += 1
        family = str(info.get("family") or "unknown")
        families[family]["traces"] += 1
        families[family]["valid" if valid else "invalid"] += 1
        step = (info.get("dispatch") or {}).get("step")
        if isinstance(step, int):
            group = steps[step]
            group["traces"] += 1
            group["valid"] += int(valid)
            group["briefs"].add(str(info.get("task_id")))
            score = info.get("final_canvas_score")
            if isinstance(score, (int, float)):
                group["rewards"].append(float(score))
        for index, turn in enumerate(turns):
            if index < len(per_turn):
                per_turn[index]["attempted"] += 1
                per_turn[index]["valid" if turn.get("valid") else "invalid"] += 1
            if not turn.get("valid"):
                failures[error_class(str(turn.get("error") or ""))] += 1
        for call in trace.get("calls") or []:
            value = (call.get("usage") or {}).get("completion_tokens")
            if isinstance(value, int):
                tokens.append(value)
    receipts = []
    for path in judge_root.glob("*/text-judge.json"):
        receipt = json.loads(path.read_text())
        if receipt.get("status") == "scored":
            receipts.append(receipt)
    gaps = collections.Counter(round(float(r.get("order_gap") or 0), 3)
                                for r in receipts)
    step_rows = {}
    for step, group in sorted(steps.items()):
        step_rows[str(step)] = {
            "generated_traces": group["traces"], "valid": group["valid"],
            "unique_briefs": len(group["briefs"]),
            "generated_trace_mean_reward": (statistics.mean(group["rewards"])
                                            if group["rewards"] else None),
        }
    return {
        "schema": "painter-rollout-audit-v1",
        "scope_note": "All saved generated traces, including possible speculative/stale traces; not the admitted gradient batch or an online learning curve.",
        "generated_traces": len(traces),
        "final_valid": sum(t["info"].get("final_canvas_valid") is True for t in traces),
        "reward_status": dict(statuses),
        "turn_validity_histories": dict(histories),
        "per_turn": [dict(c) for c in per_turn],
        "turn_error_classes": dict(failures),
        "by_family": {name: dict(counts) for name, counts in families.items()},
        "by_dispatch_step": step_rows,
        "completion_tokens_per_call": {
            "count": len(tokens), "median": statistics.median(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
        "visual_judge_receipts": len(receipts),
        "judge_order_gaps": {str(k): v for k, v in sorted(gaps.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.stream, args.judge_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"by_dispatch_step", "turn_validity_histories"}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
