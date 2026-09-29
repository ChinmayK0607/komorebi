"""Direct, first-sample SFT versus step-20 comparison on untouched reserves."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import stat
import sys
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402
from painter.text_pairwise_margin_judge import judge_pair  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_traces(root: Path, ids: set[str]) -> dict[str, list[dict]]:
    traces, errors, files = read_traces(root)
    if errors or not files:
        raise ValueError(f"missing or corrupt traces in {root}: {errors[:3]}")
    results: dict[str, list[dict]] = {ident: [] for ident in ids}
    for trace in traces:
        data, info = _trace_data(trace)
        ident = data.get("task_id") or data.get("name")
        if ident in results:
            results[ident].append(info)
    return results


def canvas(info: dict) -> Path | None:
    if not info.get("final_canvas_valid"):
        return None
    raw = info.get("final_canvas")
    path = Path(raw) if isinstance(raw, str) and raw else None
    if path is None or not path.is_file() or path.is_symlink():
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.width < 128 or image.height < 128:
                return None
            if max(ImageStat.Stat(image.resize((128, 128))).stddev) < 2.0:
                return None
    except (OSError, ValueError):
        return None
    return path


async def run(pool: Path, sft_root: Path, rl_root: Path, output: Path,
              key_file: Path) -> dict:
    if key_file.is_symlink() or not key_file.is_file() or stat.S_IMODE(key_file.stat().st_mode) != 0o600:
        raise PermissionError("judge key file must be owner-only and not a symlink")
    cases = json.loads(pool.read_text())["cases"]
    ids = {case["id"] for case in cases}
    if len(cases) != 14 or len(ids) != 14:
        raise ValueError("expected the frozen 14 distinct reserve cases")
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
        a_canvas, b_canvas = canvas(a), canvas(b)
        a_valid, b_valid = a_canvas is not None, b_canvas is not None
        record.update(sft_valid=a_valid, rl_valid=b_valid)
        if not a_valid and not b_valid:
            return {**record, "status": "both_invalid", "score": 0.5}
        if not a_valid:
            return {**record, "status": "sft_invalid", "score": 1.0}
        if not b_valid:
            return {**record, "status": "rl_invalid", "score": 0.0}
        async with limit:
            try:
                judgment = await judge_pair(brief=brief, anchor=a_canvas,
                                            candidate=b_canvas, key_file=key_file)
            except Exception as exc:
                return {**record, "status": "censored",
                        "reason": f"judge_failure_{type(exc).__name__}"}
        receipt = output / (ident + ".json")
        receipt.write_text(json.dumps(judgment, indent=2, sort_keys=True) + "\n")
        return {**record, "status": "visual_judged", "score": judgment["score"],
                "receipt": str(receipt)}

    rows = await asyncio.gather(*(one(case) for case in cases))
    scored = [row for row in rows if row["status"] != "censored"]
    visual = [row for row in rows if row["status"] == "visual_judged"]
    scores = [float(row["score"]) for row in scored]
    interval = None
    if scores:
        rng = random.Random(20260929)
        boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores)
                      for _ in range(10000))
        interval = [boot[250], boot[9749]]
    summary = {
        "schema": "painter-independent-reserve-comparison-v1",
        "protocol": "text-paint-sft-pairwise-margin-v1",
        "pool_sha256": sha(pool), "expected": len(cases),
        "comparable": len(scored), "censored": len(rows) - len(scored),
        "sft_valid": sum(row.get("sft_valid", False) for row in rows),
        "rl_valid": sum(row.get("rl_valid", False) for row in rows),
        "both_valid": len(visual),
        "mean_pairwise_score": sum(scores) / len(scores) if scores else None,
        "bootstrap_95pct_interval": interval,
        "mean_both_valid_visual_score": (
            sum(row["score"] for row in visual) / len(visual) if visual else None),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(args.pool, args.sft_root, args.rl_root,
                              args.output, args.key_file))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
