#!/usr/bin/env python3
"""Summarize matched painting traces, including censored second turns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POLICIES = ("baseline", "midpoint", "trained")


def summarize(evidence: Path) -> dict:
    manifest_path = evidence / "painter/eval-prep/eval-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {case["id"] for case in manifest["cases"]}
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = {"schema": "painter.brush-sft-eval-summary.v1",
              "case_count": len(expected), "manifest_sha256": manifest_sha, "policies": {}}
    for policy in POLICIES:
        policy_root = evidence / "eval" / policy
        receipt = json.loads((policy_root / "completion.json").read_text())
        if (receipt.get("status") != "completed" or receipt.get("case_count") != len(expected)
                or receipt.get("manifest_sha256") != manifest_sha):
            raise ValueError(f"{policy} completion does not match manifest")
        directory = receipt.get("result_dir")
        paths = ([policy_root / "results" / directory / "traces.jsonl"] if directory else
                 list((policy_root / "results").glob("*/traces.jsonl")))
        if len(paths) != 1 or not paths[0].is_file():
            raise ValueError(f"{policy} must select exactly one completed trace file")
        episodes = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        traces = [trace for episode in episodes for trace in episode.get("traces", [])]
        ids = [trace.get("info", {}).get("task_id") for trace in traces]
        if (len(episodes) != len(traces) or len(ids) != len(set(ids))
                or set(ids) != expected or not all(episode.get("ok") is True for episode in episodes)):
            raise ValueError(f"{policy} traces do not cover the frozen cases exactly once")
        infos = [trace["info"] for trace in traces]
        context_failures = []
        provider_failures = []
        for trace in traces:
            for call in trace.get("calls", []):
                error = call.get("error") or {}
                if not error:
                    continue
                case_id = trace["info"]["task_id"]
                provider_failures.append(case_id)
                if "maximum context length" in str(error.get("message", "")):
                    context_failures.append(case_id)
        first = [info["turns"][0] for info in infos]
        second = [info["turns"][1] if len(info["turns"]) > 1 else {} for info in infos]
        result["policies"][policy] = {
            "adapter_sha256": receipt["adapter_sha256"],
            "trace_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            "episodes_ok": len(episodes),
            "first_canvas_valid": sum(turn.get("valid") is True for turn in first),
            "second_turn_present": sum(bool(turn) for turn in second),
            "second_canvas_valid": sum(turn.get("valid") is True for turn in second),
            "final_canvas_valid": sum(info.get("final_canvas_valid") is True for info in infos),
            "provider_failure_cases": sorted(set(provider_failures)),
            "context_failure_cases": sorted(set(context_failures)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
