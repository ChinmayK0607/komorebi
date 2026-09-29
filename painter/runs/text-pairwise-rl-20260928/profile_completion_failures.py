"""Profile generated-turn completion and syntax failures from native traces.

This is speculative-inclusive generation evidence, not admitted-batch reward.
It records counts only; replies and image data are never copied to the report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


SYNTAX = re.compile(r"Unexpected end of input|Unexpected token|SyntaxError", re.I)
END_OF_INPUT = re.compile(r"Unexpected end of input", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["finish_reason"] for row in rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"turn_{row['turn']}"].append(row)
        groups[f"finish_{row['finish_reason']}"].append(row)

    def group(items: list[dict]) -> dict:
        tokens = [row["completion_tokens"] for row in items]
        return {
            "n": len(items),
            "mean_completion_tokens": round(statistics.mean(tokens), 2),
            "valid_actions": sum(row["valid"] for row in items),
            "syntax_related_errors": sum(row["syntax_error"] for row in items),
            "unexpected_end_of_input": sum(row["end_of_input"] for row in items),
        }

    return {
        "generated_turns": len(rows),
        "finish_reasons": dict(sorted(counts.items())),
        "groups": {key: group(items) for key, items in sorted(groups.items())},
    }


def profile(stream: Path) -> dict:
    rows: list[dict] = []
    generated_trajectories = 0
    mismatches = 0
    malformed_lines = 0
    with stream.open() as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            for trace in record.get("traces", []):
                generated_trajectories += 1
                turns = (trace.get("info") or {}).get("turns") or []
                calls = trace.get("calls") or []
                if len(turns) != len(calls):
                    mismatches += 1
                for turn_number, (turn, call) in enumerate(zip(turns, calls), 1):
                    error = str(turn.get("error") or "")
                    usage = call.get("usage") or {}
                    rows.append({
                        "turn": turn_number,
                        "finish_reason": str(call.get("finish_reason") or "missing"),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "valid": bool(turn.get("valid")),
                        "syntax_error": bool(SYNTAX.search(error)),
                        "end_of_input": bool(END_OF_INPUT.search(error)),
                    })
    if not rows:
        raise ValueError("native trace stream contains no aligned generated turns")
    return {
        "schema": "painter.generated-completion-failure-profile.v1",
        "denominator": "generated native traces, including speculative/stale work",
        "stream_sha256": sha256(stream),
        "generated_trajectories": generated_trajectories,
        "call_turn_alignment_mismatches": mismatches,
        "malformed_stream_lines": malformed_lines,
        **summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "generated_trajectories", "generated_turns", "finish_reasons",
        "call_turn_alignment_mismatches", "malformed_stream_lines")}, sort_keys=True))


if __name__ == "__main__":
    main()
