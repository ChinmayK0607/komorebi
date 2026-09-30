"""Score held-out RL canvases against frozen same-brief SFT baselines."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402
from painter.text_pairwise_judge import judge_pair as binary_judge_pair  # noqa: E402
from painter.text_pairwise_margin_judge import judge_pair as margin_judge_pair  # noqa: E402
from painter.text_pairwise_grounded_judge import judge_pair as grounded_judge_pair  # noqa: E402
from painter.text_pairwise_consistent_judge import judge_pair as consistent_judge_pair  # noqa: E402


def _judgment_error(judgment: object, protocol: str) -> str | None:
    """Return a stable censoring reason, or ``None`` for a valid score.

    A completed request can still yield an unresolved decision (for example
    ``status=uncertain`` with ``score=None``).  Keep its raw receipt, but do
    not let it enter numeric reward or summary calculations.
    """

    if not isinstance(judgment, dict):
        return "judge_result_not_object"
    if judgment.get("protocol") != protocol:
        return "protocol_mismatch"
    if judgment.get("status") not in {"scored", "identical_canvases"}:
        return "judge_status_uncertain"
    score = judgment.get("score")
    if score is None:
        return "score_missing"
    if isinstance(score, bool):
        return "score_boolean"
    if not isinstance(score, (int, float)):
        return "score_not_numeric"
    numeric = float(score)
    if not math.isfinite(numeric):
        return "score_nonfinite"
    if not 0.0 <= numeric <= 1.0:
        return "score_out_of_range"
    return None


def _resolved_judgment(judgment: object, protocol: str | None = None) -> bool:
    """Return whether a judge result has a usable finite pairwise score.

    ``protocol`` remains optional for compatibility with older local callers;
    the scoring path always supplies the selected protocol to bind receipts to
    the requested judge implementation.
    """

    if protocol is None:
        if not isinstance(judgment, dict):
            return False
        protocol = judgment.get("protocol")
    return _judgment_error(judgment, protocol) is None


async def score(manifest: Path, eval_root: Path, output: Path, key_file: Path,
                protocol: str = "text-paint-sft-pairwise-v1") -> dict:
    judges = {"text-paint-sft-pairwise-v1": binary_judge_pair,
              "text-paint-sft-pairwise-margin-v1": margin_judge_pair,
              "text-paint-sft-pairwise-grounded-v1": grounded_judge_pair,
              "text-paint-sft-pairwise-consistent-v1": consistent_judge_pair}
    if protocol not in judges:
        raise ValueError(f"unsupported judge protocol: {protocol}")
    judge_pair = judges[protocol]
    cases = json.loads(manifest.read_text())["cases"]
    traces, errors, files = read_traces(eval_root)
    if errors or not files:
        raise ValueError(f"missing or corrupt evaluation traces: {errors[:3]}")
    by_id: dict[str, list[dict]] = {row["id"]: [] for row in cases}
    for trace in traces:
        data, info = _trace_data(trace)
        task_id = data.get("task_id") or data.get("name")
        if task_id in by_id:
            by_id[task_id].append(info)
    output.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(8)

    async def one(row: dict) -> dict:
        record = {"id": row["id"], "family": row["family"]}
        episodes = by_id[row["id"]]
        if len(episodes) != 1:
            return {**record, "status": "censored", "reason": f"trace_count_{len(episodes)}"}
        info = episodes[0]
        if info.get("service_failure"):
            return {**record, "status": "censored", "reason": "service_failure"}
        if not info.get("final_canvas_valid"):
            return {**record, "status": "model_invalid", "score": 0.0}
        canvas = info.get("final_canvas")
        if not isinstance(canvas, str) or not Path(canvas).is_file():
            return {**record, "status": "censored", "reason": "missing_canvas_artifact"}
        brief, anchor = Path(row["reference"]), Path(row["opponent"])
        if not all(path.is_file() for path in (brief, anchor)):
            return {**record, "status": "censored", "reason": "missing_fixed_input"}
        async with limit:
            try:
                judgment = await judge_pair(brief=brief, anchor=anchor,
                                            candidate=Path(canvas), key_file=key_file)
            except Exception as exc:
                return {**record, "status": "censored",
                        "reason": f"judge_failure_{type(exc).__name__}"}
        receipt = output / (row["id"] + ".json")
        receipt.write_text(json.dumps(judgment, indent=2, sort_keys=True) + "\n")
        error = _judgment_error(judgment, protocol)
        if error is not None:
            return {**record, "status": "censored", "reason": error,
                    "judge_status": judgment.get("status") if isinstance(judgment, dict) else None,
                    "receipt": str(receipt), "canvas": canvas}
        return {**record, "status": "scored", "score": float(judgment["score"]),
                "receipt": str(receipt), "canvas": canvas}

    rows = await asyncio.gather(*(one(row) for row in cases))
    comparable = [row for row in rows if row["status"] in {"scored", "model_invalid"}]
    summary = {
        "schema": "painter-text-pairwise-validation-v1", "protocol": protocol,
        "manifest": str(manifest), "eval_root": str(eval_root),
        "expected": len(cases), "comparable": len(comparable),
        "censored": sum(row["status"] == "censored" for row in rows),
        "valid_candidates": sum(row["status"] == "scored" for row in rows),
        "wins": sum(row["score"] > .5 for row in comparable),
        "ties": sum(row["score"] == .5 for row in comparable),
        "losses": sum(row["score"] < .5 for row in comparable),
        "mean_score": (sum(row["score"] for row in comparable) / len(comparable)
                       if comparable else None),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--protocol", default="text-paint-sft-pairwise-v1")
    args = parser.parse_args()
    result = asyncio.run(score(**vars(args)))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
