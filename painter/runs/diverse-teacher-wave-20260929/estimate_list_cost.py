#!/usr/bin/env python3
"""Estimate public teacher-wave token spend when Gateway omits cost fields.

This is a dated list-rate estimate, never an invoice. Rerun with updated rates
when the provider catalog changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def calculate(candidates: dict, rates: dict[str, tuple[float, float]]) -> dict:
    rows = []
    for case in candidates["candidates"]:
        tier = case["tier"]
        if tier not in rates:
            raise ValueError(f"unknown model tier {tier}")
        turns = case["turns"]
        complete = bool(turns) and all(
            isinstance(turn.get(key), int) and turn[key] >= 0
            for turn in turns for key in ("prompt_tokens", "completion_tokens")
        )
        input_tokens = sum(turn.get("prompt_tokens") or 0 for turn in turns)
        output_tokens = sum(turn.get("completion_tokens") or 0 for turn in turns)
        if complete and case.get("total_tokens") != input_tokens + output_tokens:
            raise ValueError(f"usage mismatch for {case['id']}")
        input_rate, output_rate = rates[tier]
        rows.append({"id": case["id"], "tier": tier,
                     "usage_complete": complete,
                     "input_tokens": input_tokens, "output_tokens": output_tokens,
                     "known_usage_list_usd": round((input_tokens * input_rate
                                                     + output_tokens * output_rate) / 1_000_000, 8),
                     "provider_cost_usd": case.get("known_cost_usd")
                     if case.get("cost_complete") is True else None})
    return {"schema": "painter.teacher-list-cost-estimate.v1",
            "candidates": len(rows), "usage_complete": sum(row["usage_complete"] for row in rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "known_usage_list_usd": round(sum(row["known_usage_list_usd"] for row in rows), 8),
            "provider_cost_complete": all(row["provider_cost_usd"] is not None for row in rows),
            "note": "Dated list-rate estimate from reported tokens; missing provider charges remain unknown.",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flash-input", type=float, default=0.14)
    parser.add_argument("--flash-output", type=float, default=0.28)
    parser.add_argument("--pro-input", type=float, default=0.43)
    parser.add_argument("--pro-output", type=float, default=0.87)
    parser.add_argument("--rate-date", default="2026-09-29")
    args = parser.parse_args()
    rates = {"flash": (args.flash_input, args.flash_output),
             "pro": (args.pro_input, args.pro_output)}
    if any(rate < 0 for pair in rates.values() for rate in pair):
        raise ValueError("rates must be nonnegative USD per million tokens")
    report = calculate(json.loads(args.candidates.read_text()), rates)
    report["list_rate_usd_per_million"] = rates
    report["list_rate_checked_utc_date"] = args.rate_date
    report["pricing_sources"] = {
        "flash": "https://vercel.com/ai-gateway/models/mimo-v2.6-flash",
        "pro": "https://vercel.com/ai-gateway/models/mimo-v2.6-pro",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("candidates", "usage_complete",
                                                 "input_tokens", "output_tokens",
                                                 "known_usage_list_usd", "provider_cost_complete")}))


if __name__ == "__main__":
    main()
