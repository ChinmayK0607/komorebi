#!/usr/bin/env python3
"""Score the final RL panel against matched earlier policies.

The SFT512 canvas remains the fixed RL reward opponent and is shown by the
review page, but it is never passed as the quality comparison baseline here.
This command is intentionally separate from node generation: it can run after
GPU release with the existing generic Flash Gateway and retains every invalid,
uncertain, failed, and censored case.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

try:
    from build_recovery_review import collect
except ModuleNotFoundError:
    _viewer_path = Path(__file__).resolve().with_name("build_recovery_review.py")
    _viewer_spec = importlib.util.spec_from_file_location("build_recovery_review", _viewer_path)
    if _viewer_spec is None or _viewer_spec.loader is None:
        raise
    _viewer = importlib.util.module_from_spec(_viewer_spec)
    sys.modules["build_recovery_review"] = _viewer
    _viewer_spec.loader.exec_module(_viewer)
    collect = _viewer.collect


PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
FORMAT_VERSION = "identity-joint-side-schema-v2-generic-v1"
MODEL = "xiaomi/mimo-v2.6-flash"
EXPECTED_CASES = 32
PRIMARY_COMPARISON = "final_rl_vs_start"
POSTWARMUP_COMPARISON = "final_rl_vs_postwarmup"


def _score_value(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise ValueError("recovery score requires exactly 32 heldout cases")
    return rows


def _path_for(data: dict[str, Any]) -> Path | None:
    value = data.get("path")
    return Path(value) if data.get("terminal_valid") and isinstance(value, str) and value else None


def _comparison_plan(view: dict[str, Any]) -> list[dict[str, str]]:
    """Return explicit quality comparisons in stable panel-label terms."""
    plan = [{
        "label": PRIMARY_COMPARISON,
        "anchor_policy": "starting_painter",
        "candidate_policy": "after_rl",
        "anchor_label": "Starting painter",
        "candidate_label": "After RL",
    }]
    if "after_code_sft" in view:
        plan.append({
            "label": POSTWARMUP_COMPARISON,
            "anchor_policy": "after_code_sft",
            "candidate_policy": "after_rl",
            "anchor_label": "After code SFT",
            "candidate_label": "After RL",
        })
    return plan


def _comparison_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row["comparisons"][label] for row in rows if label in row.get("comparisons", {})]
    scored = [row for row in selected if row.get("status") == "scored"]
    return {
        "label": label,
        "expected_cases": len(selected),
        "comparable_visual": len(scored),
        "mean_score": sum(float(row["score"]) for row in scored) / len(scored) if scored else None,
        "wins": sum(float(row["score"]) > 0.5 for row in scored),
        "ties": sum(float(row["score"]) == 0.5 for row in scored),
        "losses": sum(float(row["score"]) < 0.5 for row in scored),
        "invalid_or_censored": sum(row.get("status") != "scored" for row in selected),
    }


async def score(*, manifest: Path, baseline_root: Path, candidate_root: Path, output: Path, key_file: Path, concurrency: int = 8, postwarmup_root: Path | None = None) -> dict[str, Any]:
    if concurrency < 1 or concurrency > 8:
        raise ValueError("concurrency must be in [1, 8]")
    if key_file.is_symlink() or not key_file.is_file() or key_file.stat().st_mode & 0o077:
        raise PermissionError("judge key file must be owner-only and not a symlink")
    # Lazy import keeps controller-side contract tests free of model/render
    # dependencies. The judge itself only makes Gateway calls when this command
    # is explicitly invoked.
    try:
        from painter.text_pairwise_consistent_side_judge_v2_generic import judge_pair
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from painter.text_pairwise_consistent_side_judge_v2_generic import judge_pair

    rows = _manifest(manifest)
    review = collect(manifest=manifest, baseline_root=baseline_root, candidate_root=candidate_root, postwarmup_root=postwarmup_root)
    by_id = {row["id"]: row for row in review["rows"]}
    output.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        ident = row.get("id")
        if not isinstance(ident, str) or ident not in by_id:
            return {"id": ident, "status": "censored", "reason": "manifest_case_missing"}
        view = by_id[ident]
        starting_data = view["starting_painter"]
        rl_data = view["after_rl"]
        result: dict[str, Any] = {
            "id": ident,
            "family": row.get("family"),
            "starting_painter_status": str(starting_data.get("status", "case_missing")),
            "after_rl_status": str(rl_data.get("status", "case_missing")),
            "starting_painter_terminal_valid": bool(starting_data.get("terminal_valid")),
            "after_rl_terminal_valid": bool(rl_data.get("terminal_valid")),
            # Existing callers can continue reading the prior field names;
            # the explicit comparison labels below are authoritative.
            "flash12_status": str(starting_data.get("status", "case_missing")),
            "checkpoint4_status": str(rl_data.get("status", "case_missing")),
            "flash12_terminal_valid": bool(starting_data.get("terminal_valid")),
            "checkpoint4_terminal_valid": bool(rl_data.get("terminal_valid")),
        }
        if "after_code_sft" in view:
            result.update({
                "after_code_sft_status": str(view["after_code_sft"].get("status", "case_missing")),
                "after_code_sft_terminal_valid": bool(view["after_code_sft"].get("terminal_valid")),
            })
        brief = Path(str(row.get("reference", "")))
        if not brief.is_file():
            # The manifest may carry an admitted casebrief path under a
            # different key.  The viewer already resolved and read its text;
            # the judge still needs the original bytes, so try those paths.
            for key in ("casebrief", "brief"):
                value = row.get(key)
                if isinstance(value, str) and Path(value).is_file():
                    brief = Path(value)
                    break
        if not brief.is_file() and isinstance(view.get("brief"), str) and view["brief"]:
            brief = output / "briefs" / f"{ident}.txt"
            brief.parent.mkdir(parents=True, exist_ok=True)
            brief.write_text(view["brief"], encoding="utf-8")
        async def compare(spec: dict[str, str]) -> dict[str, Any]:
            anchor_data = view[spec["anchor_policy"]]
            candidate_data = view[spec["candidate_policy"]]
            anchor = _path_for(anchor_data)
            candidate = _path_for(candidate_data)
            comparison: dict[str, Any] = {
                "comparison": spec["label"],
                "anchor_policy": spec["anchor_policy"],
                "candidate_policy": spec["candidate_policy"],
                "anchor_label": spec["anchor_label"],
                "candidate_label": spec["candidate_label"],
                "anchor_status": str(anchor_data.get("status", "case_missing")),
                "candidate_status": str(candidate_data.get("status", "case_missing")),
                "anchor_terminal_valid": bool(anchor_data.get("terminal_valid")),
                "candidate_terminal_valid": bool(candidate_data.get("terminal_valid")),
            }
            receipt = output / f"{ident}-{spec['label']}.json"
            if spec["label"] == PRIMARY_COMPARISON:
                # Preserve the old per-case receipt name for existing API users.
                receipt = output / f"{ident}.json"
            if not brief.is_file():
                return {**comparison, "status": "censored", "reason": "missing_brief"}
            if anchor is None and candidate is None:
                return {**comparison, "status": "invalid_both", "reason": "no_valid_terminal_canvases"}
            if anchor is None:
                return {**comparison, "status": "baseline_invalid", "reason": f"no_valid_{spec['anchor_policy']}_terminal_canvas"}
            if candidate is None:
                return {**comparison, "status": "candidate_invalid", "reason": f"no_valid_{spec['candidate_policy']}_terminal_canvas"}
            async with limit:
                try:
                    judgment = await judge_pair(
                        brief=brief,
                        anchor=anchor,
                        candidate=candidate,
                        key_file=key_file,
                        model=MODEL,
                        reasoning={"effort": "medium"},
                    )
                except Exception as exc:
                    return {**comparison, "status": "censored", "reason": f"judge_failure_{type(exc).__name__}"}
            receipt.write_text(json.dumps(judgment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if judgment.get("protocol") != PROTOCOL or judgment.get("format_version") != FORMAT_VERSION:
                return {**comparison, "status": "censored", "reason": "judge_protocol_mismatch", "receipt": str(receipt)}
            if judgment.get("status") not in {"scored", "identical_canvases"} or not _score_value(judgment.get("score")):
                return {**comparison, "status": "censored", "reason": "judge_uncertain", "receipt": str(receipt)}
            return {**comparison, "status": "scored", "score": float(judgment["score"]), "receipt": str(receipt)}

        comparisons = await asyncio.gather(*(compare(spec) for spec in _comparison_plan(view)))
        result["comparisons"] = {item["comparison"]: item for item in comparisons}
        primary = result["comparisons"][PRIMARY_COMPARISON]
        return {**result, "status": primary["status"], **({key: primary[key] for key in ("score", "receipt") if key in primary}), **({"reason": primary["reason"]} if "reason" in primary else {})}

    scored_rows = await asyncio.gather(*(one(row) for row in rows))
    comparison_labels = [PRIMARY_COMPARISON]
    if postwarmup_root is not None:
        comparison_labels.append(POSTWARMUP_COMPARISON)
    comparisons = {label: _comparison_summary(scored_rows, label) for label in comparison_labels}
    primary = comparisons[PRIMARY_COMPARISON]
    summary = {
        "schema": "Diverse-flow-matched-score-v2",
        "status": "complete",
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "judge_model": MODEL,
        "judge_effort": "medium",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "expected": EXPECTED_CASES,
        "comparable_visual": primary["comparable_visual"],
        "mean_score": primary["mean_score"],
        "wins": primary["wins"],
        "ties": primary["ties"],
        "losses": primary["losses"],
        "comparison_labels": comparison_labels,
        "comparisons": comparisons,
        "validity": {
            "expected_cases": EXPECTED_CASES,
            "valid_starting_painter_terminal": sum(bool(row["starting_painter_terminal_valid"]) for row in scored_rows),
            "valid_after_rl_terminal": sum(bool(row["after_rl_terminal_valid"]) for row in scored_rows),
            "valid_flash12_terminal": sum(bool(row["flash12_terminal_valid"]) for row in scored_rows),
            "valid_checkpoint4_terminal": sum(bool(row["checkpoint4_terminal_valid"]) for row in scored_rows),
            "comparable_visual_pairs": primary["comparable_visual"],
            "invalid_or_censored_cases": sum(row["status"] != "scored" for row in scored_rows),
        },
        "invalid_or_censored": primary["invalid_or_censored"],
        "baseline_policy": "Starting painter",
        "candidate_policy": "After RL",
        "sft512_reward_opponent": "retained for reward/display only",
        "rows": scored_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--postwarmup-root", "--third-policy-root", dest="postwarmup_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    summary = asyncio.run(score(**vars(args)))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
