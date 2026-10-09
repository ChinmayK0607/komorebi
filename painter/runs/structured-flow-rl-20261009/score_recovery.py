#!/usr/bin/env python3
"""Score checkpoint4 directly against the cached Flash12 panel.

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
EXPECTED_CASES = 16


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
        raise ValueError("recovery score requires exactly 16 heldout cases")
    return rows


async def score(*, manifest: Path, baseline_root: Path, candidate_root: Path, output: Path, key_file: Path, concurrency: int = 8) -> dict[str, Any]:
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
    review = collect(manifest=manifest, baseline_root=baseline_root, candidate_root=candidate_root)
    by_id = {row["id"]: row for row in review["rows"]}
    output.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        ident = row.get("id")
        if not isinstance(ident, str) or ident not in by_id:
            return {"id": ident, "status": "censored", "reason": "manifest_case_missing"}
        view = by_id[ident]
        flash_data = view["flash12"]
        candidate_data = view["checkpoint4"]
        flash = Path(flash_data["path"]) if flash_data.get("terminal_valid") and flash_data.get("path") else None
        candidate = Path(candidate_data["path"]) if candidate_data.get("terminal_valid") and candidate_data.get("path") else None
        flash_status = str(flash_data.get("status", "case_missing"))
        candidate_status = str(candidate_data.get("status", "case_missing"))
        result: dict[str, Any] = {
            "id": ident,
            "family": row.get("family"),
            "flash12_status": flash_status,
            "checkpoint4_status": candidate_status,
            "flash12_terminal_valid": bool(view["flash12"].get("terminal_valid")),
            "checkpoint4_terminal_valid": bool(view["checkpoint4"].get("terminal_valid")),
        }
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
        if not brief.is_file():
            return {**result, "status": "censored", "reason": "missing_brief"}
        if flash is None and candidate is None:
            return {**result, "status": "invalid_both", "reason": "no_valid_terminal_canvases"}
        if flash is None:
            return {**result, "status": "baseline_invalid", "reason": "no_valid_flash12_terminal_canvas"}
        if candidate is None:
            return {**result, "status": "candidate_invalid", "reason": "no_valid_checkpoint4_terminal_canvas"}
        async with limit:
            try:
                judgment = await judge_pair(
                    brief=brief,
                    anchor=flash,
                    candidate=candidate,
                    key_file=key_file,
                    model=MODEL,
                    reasoning={"effort": "medium"},
                )
            except Exception as exc:
                return {**result, "status": "censored", "reason": f"judge_failure_{type(exc).__name__}"}
        receipt = output / f"{ident}.json"
        receipt.write_text(json.dumps(judgment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if judgment.get("protocol") != PROTOCOL or judgment.get("format_version") != FORMAT_VERSION:
            return {**result, "status": "censored", "reason": "judge_protocol_mismatch", "receipt": str(receipt)}
        if judgment.get("status") not in {"scored", "identical_canvases"} or not _score_value(judgment.get("score")):
            return {**result, "status": "censored", "reason": "judge_uncertain", "receipt": str(receipt)}
        return {**result, "status": "scored", "score": float(judgment["score"]), "receipt": str(receipt)}

    scored_rows = await asyncio.gather(*(one(row) for row in rows))
    visual = [row for row in scored_rows if row["status"] == "scored"]
    summary = {
        "schema": "structured-flow-checkpoint4-matched-score-v1",
        "status": "complete",
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "judge_model": MODEL,
        "judge_effort": "medium",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "expected": EXPECTED_CASES,
        "comparable_visual": len(visual),
        "mean_score": sum(float(row["score"]) for row in visual) / len(visual) if visual else None,
        "wins": sum(float(row["score"]) > 0.5 for row in visual),
        "ties": sum(float(row["score"]) == 0.5 for row in visual),
        "losses": sum(float(row["score"]) < 0.5 for row in visual),
        "validity": {
            "expected_cases": EXPECTED_CASES,
            "valid_flash12_terminal": sum(bool(row["flash12_terminal_valid"]) for row in scored_rows),
            "valid_checkpoint4_terminal": sum(bool(row["checkpoint4_terminal_valid"]) for row in scored_rows),
            "comparable_visual_pairs": len(visual),
            "invalid_or_censored_cases": sum(row["status"] != "scored" for row in scored_rows),
        },
        "invalid_or_censored": sum(row["status"] != "scored" for row in scored_rows),
        "baseline_policy": "Flash12 starting policy",
        "sft512_reward_opponent": "retained for reward/anchor display only",
        "rows": scored_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    summary = asyncio.run(score(**vars(args)))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
