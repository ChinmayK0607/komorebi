"""Calibrate a graded pairwise judge on controls and saved RL paintings."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from painter.text_pairwise_margin_judge import judge_pair


REVIEW_IDS = ("t600-425", "t600-542", "t600-426", "t600-512")


def cases(run: Path) -> list[dict]:
    validation = run / "validation/step-4-n16"
    all_cases = {row["id"]: row for row in json.loads(
        (validation / "cases.json").read_text())["cases"]}
    scored = {row["id"]: row for row in json.loads(
        (validation / "comparison/summary.json").read_text())["rows"]}
    selected = []
    for ident in REVIEW_IDS:
        row, result = all_cases[ident], scored[ident]
        if result["status"] != "scored":
            raise ValueError(f"missing reviewed pair {ident}")
        selected.append({"id": ident, "source": "reviewed_validation",
                         "old_score": result["score"],
                         "brief": Path(row["reference"]),
                         "anchor": Path(row["opponent"]),
                         "candidate": Path(result["canvas"])})
    train = {row["id"]: row for row in json.loads(
        (run / "anchors-frozen/train-manifest.json").read_text())["rows"]}
    traces = run / "async-output/text-pairwise-rl-20260928/monitors/file/traces/stream/00000.jsonl"
    seen = set()
    for line in traces.open():
        for trace in json.loads(line).get("traces", []):
            info = trace.get("info", {})
            ident = info.get("task_id")
            if ident in seen or ident not in train or not info.get("final_canvas_valid"):
                continue
            reward = (trace.get("rewards", {}).get("final_canvas_aesthetic") or {}).get("score")
            if reward != 0:
                continue
            row = train[ident]
            selected.append({"id": ident, "source": "first_run_scored_zero",
                             "old_score": 0.0, "brief": Path(row["reference"]),
                             "anchor": Path(row["opponent"]),
                             "candidate": Path(info["final_canvas"])})
            seen.add(ident)
            if len(seen) == 8:
                break
        if len(seen) == 8:
            break
    if len(seen) != 8:
        raise ValueError(f"found only {len(seen)} distinct scored-zero training pairs")
    return selected


async def run_calibration(run: Path, output: Path, key_file: Path) -> dict:
    items = cases(run)
    output.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(3)

    async def one(row: dict) -> dict:
        async with limit:
            result = await judge_pair(brief=row["brief"], anchor=row["anchor"],
                                      candidate=row["candidate"], key_file=key_file)
        receipt = output / (row["id"] + ".json")
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return {"id": row["id"], "source": row["source"],
                "old_score": row["old_score"], "margin_score": result["score"],
                "order_gap": result.get("order_gap"), "receipt": str(receipt)}

    rows = await asyncio.gather(*(one(row) for row in items))
    identity = await judge_pair(brief=items[0]["brief"], anchor=items[0]["anchor"],
                                candidate=items[0]["anchor"], key_file=key_file)
    scores = [row["margin_score"] for row in rows if row["source"] == "first_run_scored_zero"]
    summary = {"schema": "painter-text-margin-calibration-v1", "rows": rows,
               "identity_score": identity["score"],
               "zero_winner_distinct_scores": len(set(scores)),
               "zero_winner_score_range": [min(scores), max(scores)],
               "max_order_gap": max(row["order_gap"] for row in rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(run_calibration(args.run_root, args.output, args.key_file))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
