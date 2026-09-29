"""Directly compare the prior public step-16 paintings with this continuation."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from score_confirmatory_reserve import canvas, policy_traces, sha

try:
    from painter.text_pairwise_margin_judge import judge_pair
except ImportError:
    from text_pairwise_margin_judge import judge_pair


async def score(manifest: Path, prior_summary: Path, prior_images: Path,
                final_root: Path, output: Path, key_file: Path) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    ids = {row["id"] for row in cases}
    if len(cases) != 47 or len(ids) != 47:
        raise ValueError("expected 47 distinct frozen validation briefs")
    prior_rows = json.loads(prior_summary.read_text())["rows"]
    prior = {row["id"]: row for row in prior_rows}
    if set(prior) != ids or len(prior_rows) != len(prior):
        raise ValueError("prior step-16 control IDs differ from frozen validation")
    current = policy_traces(final_root, ids)
    output.mkdir(parents=True, exist_ok=False)
    limit = asyncio.Semaphore(8)

    async def one(case: dict) -> dict:
        ident = case["id"]
        brief = Path(case["reference"])
        if sha(brief) != case["reference_sha256"]:
            raise ValueError(f"brief hash mismatch: {ident}")
        record = {"id": ident, "family": case["family"]}
        if len(current[ident]) != 1:
            return {**record, "status": "censored", "reason": "trace_count"}
        final_trace = current[ident][0]
        if final_trace.get("service_failure"):
            return {**record, "status": "censored", "reason": "service_failure"}
        old_status = prior[ident]["status"]
        if old_status not in {"scored", "model_invalid"}:
            return {**record, "status": "censored", "reason": "prior_control_status"}
        old = (prior_images / f"{ident}-RL-step-16.png"
               if old_status == "scored" else None)
        if old is not None and (not old.is_file() or old.is_symlink()):
            raise FileNotFoundError(old)
        new = canvas(final_trace)
        record.update(prior_valid=old is not None, final_valid=new is not None)
        if old is None and new is None:
            return {**record, "status": "both_invalid", "score": 0.5}
        if old is None:
            return {**record, "status": "prior_invalid", "score": 1.0}
        if new is None:
            return {**record, "status": "final_invalid", "score": 0.0}
        async with limit:
            try:
                result = await judge_pair(brief=brief, anchor=old,
                                          candidate=new, key_file=key_file)
            except Exception as exc:
                return {**record, "status": "censored",
                        "reason": f"judge_failure_{type(exc).__name__}"}
        receipt = output / f"{ident}.json"
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return {**record, "status": "visual_judged", "score": result["score"],
                "order_gap": result.get("order_gap"), "receipt": str(receipt)}

    rows = await asyncio.gather(*(one(case) for case in cases))
    comparable = [row for row in rows if row["status"] != "censored"]
    visual = [row for row in rows if row["status"] == "visual_judged"]
    scores = [float(row["score"]) for row in comparable]
    interval = None
    if scores:
        rng = random.Random(20260929)
        boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores)
                      for _ in range(10000))
        interval = [boot[250], boot[9749]]
    summary = {
        "schema": "painter-direct-step16-validation-v1",
        "protocol": "text-paint-sft-pairwise-margin-v1",
        "manifest_sha256": sha(manifest),
        "prior_summary_sha256": sha(prior_summary),
        "expected": 47, "comparable": len(comparable),
        "censored": 47 - len(comparable),
        "prior_valid": sum(row.get("prior_valid", False) for row in rows),
        "final_valid": sum(row.get("final_valid", False) for row in rows),
        "both_valid": len(visual),
        "mean_pairwise_score": sum(scores) / len(scores) if scores else None,
        "bootstrap_95pct_interval": interval,
        "visual_only_mean": (sum(row["score"] for row in visual) / len(visual)
                             if visual else None),
        "order_disagreement": sum((row.get("order_gap") or 0) > 0.001
                                  for row in visual),
        "wins": sum(row["score"] > 0.5 for row in comparable),
        "ties": sum(row["score"] == 0.5 for row in comparable),
        "losses": sum(row["score"] < 0.5 for row in comparable),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2,
                                                    sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--prior-images", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(score(**vars(args)))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
