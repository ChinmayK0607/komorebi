"""Judge matched SFT and RL canvases directly on a frozen validation set."""

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


async def score(manifest: Path, sft_root: Path, rl_root: Path, output: Path,
                key_file: Path, expected_count: int) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    ids = {case["id"] for case in cases}
    if len(cases) != expected_count or len(ids) != expected_count:
        raise ValueError("frozen validation count or uniqueness mismatch")
    sft = policy_traces(sft_root, ids)
    rl = policy_traces(rl_root, ids)
    output.mkdir(parents=True, exist_ok=False)
    limit = asyncio.Semaphore(8)

    async def one(case: dict) -> dict:
        ident = case["id"]
        brief = Path(case["reference"])
        if sha(brief) != case["reference_sha256"]:
            raise ValueError(f"brief hash mismatch: {ident}")
        record = {"id": ident, "family": case["family"]}
        if len(sft[ident]) != 1 or len(rl[ident]) != 1:
            return {**record, "status": "censored", "reason": "trace_count",
                    "sft_traces": len(sft[ident]), "rl_traces": len(rl[ident])}
        a, b = sft[ident][0], rl[ident][0]
        if a.get("service_failure") or b.get("service_failure"):
            return {**record, "status": "censored", "reason": "service_failure"}
        ac, bc = canvas(a), canvas(b)
        record.update(sft_valid=ac is not None, rl_valid=bc is not None)
        if ac is None and bc is None:
            return {**record, "status": "both_invalid", "score": 0.5}
        if ac is None:
            return {**record, "status": "sft_invalid", "score": 1.0}
        if bc is None:
            return {**record, "status": "rl_invalid", "score": 0.0}
        async with limit:
            try:
                result = await judge_pair(brief=brief, anchor=ac,
                                          candidate=bc, key_file=key_file)
            except Exception as exc:
                return {**record, "status": "censored",
                        "reason": f"judge_failure_{type(exc).__name__}"}
        receipt = output / f"{ident}.json"
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return {**record, "status": "visual_judged", "score": result["score"],
                "order_gap": result.get("order_gap"), "receipt": str(receipt)}

    rows = await asyncio.gather(*(one(case) for case in cases))
    scored = [r for r in rows if r["status"] != "censored"]
    visual = [r for r in rows if r["status"] == "visual_judged"]
    scores = [float(r["score"]) for r in scored]
    interval = None
    if scores:
        rng = random.Random(20260929)
        boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores)
                      for _ in range(10000))
        interval = [boot[250], boot[9749]]
    summary = {
        "schema": "painter-direct-validation-v1", "protocol": "text-paint-sft-pairwise-margin-v1",
        "manifest_sha256": sha(manifest), "expected": len(cases),
        "comparable": len(scored), "censored": len(rows)-len(scored),
        "sft_valid": sum(r.get("sft_valid", False) for r in rows),
        "rl_valid": sum(r.get("rl_valid", False) for r in rows),
        "both_valid": len(visual),
        "mean_pairwise_score": sum(scores)/len(scores) if scores else None,
        "bootstrap_95pct_interval": interval,
        "visual_only_mean": (sum(r["score"] for r in visual)/len(visual) if visual else None),
        "order_disagreement": sum((r.get("order_gap") or 0) > 0.001 for r in visual),
        "strong_order_disagreement": sum((r.get("order_gap") or 0) >= 0.3 for r in visual),
        "wins": sum(r["score"] > 0.5 for r in scored),
        "ties": sum(r["score"] == 0.5 for r in scored),
        "losses": sum(r["score"] < 0.5 for r in scored),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    result = asyncio.run(score(**vars(args)))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
