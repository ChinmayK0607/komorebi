"""Direct, validity-inclusive pairwise review of a pilot checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import stat
import sys
from pathlib import Path

OLD = Path(__file__).resolve().parents[1] / "text-pairwise-rl-20260928"
sys.path.insert(0, str(OLD))
from score_confirmatory_reserve import canvas, policy_traces, sha  # noqa: E402

try:
    from painter.text_pairwise_grounded_judge import judge_pair
except ImportError:
    from text_pairwise_grounded_judge import judge_pair


def baseline_image(root: Path, ident: str, kind: str) -> Path:
    if kind == "SFT":
        return root / f"{ident}-SFT.png"
    if kind == "RL-step-16":
        return root / f"{ident}-RL-step-16.png"
    if kind == "midpoint":
        return root / ident / "final.png"
    raise ValueError("unsupported baseline kind")


async def score(*, manifest: Path, baseline_kind: str,
                baseline_summary: Path, baseline_images: Path,
                candidate_root: Path, output: Path, key_file: Path) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    ids = {row["id"] for row in cases}
    if not cases or len(ids) != len(cases) or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ident) is None for ident in ids):
        raise ValueError("manifest requires distinct safe case IDs")
    baseline_rows = json.loads(baseline_summary.read_text())["rows"]
    baseline = {row["id"]: row for row in baseline_rows}
    if not ids.issubset(baseline) or len(baseline_rows) != len(baseline):
        raise ValueError("baseline summary lacks manifest IDs or has duplicates")
    if (key_file.is_symlink() or not key_file.is_file()
            or stat.S_IMODE(key_file.stat().st_mode) != 0o600):
        raise PermissionError("judge key file must be owner-only and not a symlink")
    current = policy_traces(candidate_root, ids)
    if (output / "summary.json").is_file():
        raise FileExistsError(output / "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(8)

    async def one(case: dict) -> dict:
        ident = case["id"]
        brief = Path(case["reference"])
        if sha(brief) != case["reference_sha256"]:
            raise ValueError(f"brief hash mismatch: {ident}")
        row = {"id": ident, "family": case["family"]}
        traces = current[ident]
        if len(traces) != 1 or traces[0].get("service_failure"):
            return {**row, "status": "censored", "reason": "trace_or_service_failure"}
        old_status = baseline[ident]["status"]
        if old_status not in {"scored", "model_invalid"}:
            return {**row, "status": "censored", "reason": "baseline_status"}
        old = baseline_image(baseline_images, ident, baseline_kind) if old_status == "scored" else None
        if old is not None and (not old.is_file() or old.is_symlink()):
            raise FileNotFoundError(old)
        new = canvas(traces[0])
        row.update(baseline_valid=old is not None, candidate_valid=new is not None)
        if old is None and new is None:
            return {**row, "status": "both_invalid", "score": 0.5}
        if old is None:
            return {**row, "status": "baseline_invalid", "score": 1.0}
        if new is None:
            return {**row, "status": "candidate_invalid", "score": 0.0}
        receipt = output / f"{ident}.json"
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if result.get("protocol") != "text-paint-sft-pairwise-grounded-v1":
                raise ValueError(f"incompatible cached judge receipt: {ident}")
            return {**row, "status": "visual_judged", "score": result["score"],
                    "order_gap": result.get("order_gap"), "receipt": str(receipt)}
        async with limit:
            try:
                result = await judge_pair(brief=brief, anchor=old,
                                          candidate=new, key_file=key_file)
            except Exception as exc:
                return {**row, "status": "censored",
                        "reason": f"judge_failure_{type(exc).__name__}"}
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return {**row, "status": "visual_judged", "score": result["score"],
                "order_gap": result.get("order_gap"),
                "receipt": str(receipt)}

    rows = await asyncio.gather(*(one(case) for case in cases))
    comparable = [row for row in rows if row["status"] != "censored"]
    scores = [float(row["score"]) for row in comparable]
    interval = None
    if scores:
        rng = random.Random(20260930)
        boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores)
                      for _ in range(10_000))
        interval = [boot[250], boot[9749]]
    summary = {
        "schema": "painter-grounded-direct-pairwise-v1",
        "protocol": "text-paint-sft-pairwise-grounded-v1",
        "manifest_sha256": sha(manifest),
        "baseline_kind": baseline_kind,
        "baseline_summary_sha256": sha(baseline_summary),
        "expected": len(cases),
        "comparable": len(comparable),
        "censored": len(cases) - len(comparable),
        "baseline_valid": sum(row.get("baseline_valid", False) for row in rows),
        "candidate_valid": sum(row.get("candidate_valid", False) for row in rows),
        "both_valid": sum(row["status"] == "visual_judged" for row in rows),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "bootstrap_95pct_interval": interval,
        "order_gap_at_least_third": sum(
            (row.get("order_gap") or 0) >= 1 / 3 for row in rows),
        "wins": sum(row["score"] > 0.5 for row in comparable),
        "ties": sum(row["score"] == 0.5 for row in comparable),
        "losses": sum(row["score"] < 0.5 for row in comparable),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-kind", required=True,
                        choices=("SFT", "RL-step-16", "midpoint"))
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-images", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(score(**vars(args)))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
