"""Summarize matched canary trace validity, turns, tokens, and render cost."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _tokens, _trace_data, read_traces  # noqa: E402


POLICIES = ("sft", "rl", "mix025", "mix050")


def summarize(manifest: Path, eval_root: Path, output: Path) -> dict:
    ids = [row["id"] for row in json.loads(manifest.read_text())["cases"]]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("expected eight distinct canary cases")
    all_rows: list[dict] = []
    policies = {}
    for policy in POLICIES:
        traces, errors, files = read_traces(eval_root / policy)
        if errors or not files:
            raise ValueError(f"missing or corrupt {policy} traces: {errors[:2]}")
        rows: dict[str, dict] = {}
        for trace in traces:
            data, info = _trace_data(trace)
            ident = data.get("task_id") or data.get("name")
            if ident not in ids:
                continue
            if ident in rows:
                raise ValueError(f"duplicate trace: {policy}/{ident}")
            turns = info.get("turns") or []
            usage = _tokens(trace, turns)
            rows[ident] = {"id": ident, "policy": policy,
                           "final_canvas_valid": bool(info.get("final_canvas_valid")),
                           "turn_count": len(turns),
                           "valid_turns": sum(bool(t.get("valid")) for t in turns),
                           "no_change_turns": sum(bool(t.get("no_change")) for t in turns),
                           "render_seconds": sum(float(t.get("render_seconds") or 0) for t in turns),
                           "completion_tokens": usage["completion_tokens"],
                           "prompt_tokens": usage["prompt_tokens"],
                           "usage_complete": usage["complete"],
                           "stop_condition": trace.get("stop_condition")}
        if set(rows) != set(ids):
            raise ValueError(f"{policy}: {len(rows)}/8 traces")
        ordered = [rows[ident] for ident in ids]
        all_rows.extend(ordered)
        durations = [row["render_seconds"] for row in ordered]
        policies[policy] = {
            "cases": len(ordered),
            "final_valid": sum(row["final_canvas_valid"] for row in ordered),
            "valid_turns": sum(row["valid_turns"] for row in ordered),
            "no_change_turns": sum(row["no_change_turns"] for row in ordered),
            "render_seconds_total": sum(durations),
            "render_seconds_median_per_case": statistics.median(durations),
            "completion_tokens_total": sum(row["completion_tokens"] or 0 for row in ordered),
            "usage_complete_cases": sum(row["usage_complete"] for row in ordered),
        }
    result = {"schema": "matched-merge-canary-summary-v1",
              "case_count": len(ids), "policies": policies, "rows": all_rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(**vars(args))
    print(json.dumps(result["policies"], sort_keys=True))


if __name__ == "__main__":
    main()
