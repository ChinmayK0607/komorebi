"""Run a small repeated old-vs-draft judge check on frozen painting pairs.

Preparation freezes exact brief/image bytes and the calibration/confirmation
selection split.  Running sends the same pair and brief to the unchanged
side-keyed v1 judge and the un-deployed v2 draft for each requested repeat.
Each judge already presents both image orders and censors unresolved
consensus.  This runner records the two methods separately; it never changes
training reward or treats a selected calibration panel as an accuracy set.

The key file is passed through to the existing Gateway implementation.  It is
never opened, logged, copied, or included in a manifest by this runner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable

try:
    from painter import text_pairwise_consistent_side_judge as old_judge
    from painter import text_pairwise_consistent_side_judge_v2 as draft_judge
except ImportError:  # pragma: no cover - direct execution from a checkout
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_consistent_side_judge as old_judge
    from painter import text_pairwise_consistent_side_judge_v2 as draft_judge


SCHEMA = "text-rl16-frozen-pair-repeat-input-v1"
RECEIPT_SCHEMA = "text-rl16-frozen-pair-repeat-receipt-v1"
SUMMARY_SCHEMA = "text-rl16-frozen-pair-repeat-summary-v1"
METHODS = ("old", "draft")
SPLITS = ("calibration", "confirmation")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_content_sha256", None)
    return sha256_text(_canonical(copy))


def _safe_id(value: Any) -> str:
    value = str(value)
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe pair identifier: {value!r}")
    return value


def _path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("pair path must be a nonempty string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve()


def _load_pair_rows(source: Path) -> tuple[list[dict[str, Any]], str]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows, note = value, "The selected panel is descriptive; it is not a random accuracy sample."
    elif isinstance(value, dict) and isinstance(value.get("pairs"), list):
        rows, note = value["pairs"], value.get("selection_note", "")
    else:
        raise ValueError("pairs input must be a list or an object with pairs")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("pairs input requires a nonempty selection_note")
    return rows, note.strip()


def _source_identity(*, brief_sha256: str, anchor: Path, candidate: Path) -> dict[str, Any]:
    anchor_sha = sha256_file(anchor)
    candidate_sha = sha256_file(candidate)
    anchor_id = old_judge.image_id(anchor)
    candidate_id = old_judge.image_id(candidate)
    if anchor_id != draft_judge.image_id(anchor) or candidate_id != draft_judge.image_id(candidate):
        raise ValueError("old and draft image identity implementations differ")
    if anchor_id == candidate_id and anchor_sha != candidate_sha:
        raise ValueError("image identity prefix collision")
    return {
        "brief_sha256": brief_sha256,
        "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
        "image_sha256": {"anchor": anchor_sha, "candidate": candidate_sha},
    }


def prepare_manifest(*, pairs_file: Path, output: Path, repeat_count: int = 2) -> dict[str, Any]:
    """Freeze a small pair list without making provider calls."""

    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    rows, selection_note = _load_pair_rows(pairs_file)
    if not rows:
        raise ValueError("pairs input is empty")
    frozen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("each pair must be an object")
        pair_id = _safe_id(raw.get("pair_id", raw.get("trace_id", "")))
        if pair_id in seen:
            raise ValueError(f"duplicate pair identifier: {pair_id}")
        seen.add(pair_id)
        split = raw.get("split")
        if split not in SPLITS:
            raise ValueError(f"pair {pair_id} must use split calibration or confirmation")
        anchor = _path(raw.get("anchor_path"), base=pairs_file.parent)
        candidate = _path(raw.get("candidate_path"), base=pairs_file.parent)
        brief_path_value = raw.get("brief_path")
        brief_text = raw.get("brief")
        if brief_path_value is not None:
            brief_path = _path(brief_path_value, base=pairs_file.parent)
            brief = brief_path.read_text(encoding="utf-8").strip()
            stored_brief_path: str | None = str(brief_path)
        elif isinstance(brief_text, str):
            brief_path = None
            brief = brief_text.strip()
            stored_brief_path = None
        else:
            raise ValueError(f"pair {pair_id} needs brief_path or brief")
        if not brief or len(brief) > 4000:
            raise ValueError(f"pair {pair_id} has an invalid brief")
        identity = _source_identity(brief_sha256=sha256_text(brief), anchor=anchor, candidate=candidate)
        frozen.append({
            "pair_id": pair_id,
            "split": split,
            "brief": brief if brief_path is None else None,
            "brief_path": stored_brief_path,
            "brief_sha256": identity["brief_sha256"],
            "anchor_path": str(anchor),
            "candidate_path": str(candidate),
            "image_ids": identity["image_ids"],
            "image_sha256": identity["image_sha256"],
            "source_note": str(raw.get("source_note", "")).strip() or None,
        })
    splits = {row["split"] for row in frozen}
    if splits != set(SPLITS):
        raise ValueError("manifest must contain both calibration and confirmation pairs")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "protocols": {
            "old": old_judge.PROTOCOL,
            "draft": draft_judge.PROTOCOL,
        },
        "models": {"old": old_judge.MODEL, "draft": draft_judge.MODEL},
        "prompts": {
            "old_sha256": sha256_text(old_judge.PROMPT),
            "draft_sha256": sha256_text(draft_judge.PROMPT),
        },
        "format_versions": {
            "old": old_judge.FORMAT_VERSION,
            "draft": draft_judge.FORMAT_VERSION,
        },
        "source_format_sha256": {
            "old": old_judge.SOURCE_FORMAT_SHA256,
            "draft": draft_judge.SOURCE_FORMAT_SHA256,
        },
        "repeat_count": repeat_count,
        "selection_note": selection_note,
        "scope_note": (
            "Calibration and confirmation are selected frozen-image descriptions, not an accuracy estimate. "
            "Confirmation is held out from prompt selection; no root-review preference is sent to either judge."
        ),
        "pairs": frozen,
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "input-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected repeat manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("repeat manifest integrity mismatch")
    if not isinstance(manifest.get("repeat_count"), int) or manifest["repeat_count"] < 1:
        raise ValueError("repeat manifest repeat_count is invalid")
    if not isinstance(manifest.get("pairs"), list) or not manifest["pairs"]:
        raise ValueError("repeat manifest has no pairs")
    if {row.get("split") for row in manifest["pairs"]} != set(SPLITS):
        raise ValueError("repeat manifest must contain both calibration and confirmation")
    expected = {
        "old": (old_judge.PROTOCOL, old_judge.MODEL, old_judge.PROMPT,
                old_judge.FORMAT_VERSION, old_judge.SOURCE_FORMAT_SHA256),
        "draft": (draft_judge.PROTOCOL, draft_judge.MODEL, draft_judge.PROMPT,
                   draft_judge.FORMAT_VERSION, draft_judge.SOURCE_FORMAT_SHA256),
    }
    for method, (protocol, model, prompt, fmt, source_sha) in expected.items():
        if manifest.get("protocols", {}).get(method) != protocol:
            raise ValueError(f"{method} protocol changed since preparation")
        if manifest.get("models", {}).get(method) != model:
            raise ValueError(f"{method} model changed since preparation")
        if manifest.get("prompts", {}).get(f"{method}_sha256") != sha256_text(prompt):
            raise ValueError(f"{method} prompt changed since preparation")
        if manifest.get("format_versions", {}).get(method) != fmt:
            raise ValueError(f"{method} format changed since preparation")
        if manifest.get("source_format_sha256", {}).get(method) != source_sha:
            raise ValueError(f"{method} source changed since preparation")
    return manifest


def _pair_paths(pair: dict[str, Any]) -> tuple[str, Path, Path, str]:
    anchor = _path(pair.get("anchor_path"), base=Path.cwd())
    candidate = _path(pair.get("candidate_path"), base=Path.cwd())
    if pair.get("brief_path"):
        brief_path = _path(pair["brief_path"], base=Path.cwd())
        brief = brief_path.read_text(encoding="utf-8").strip()
    else:
        brief = pair.get("brief")
    if not isinstance(brief, str) or not brief.strip() or len(brief.strip()) > 4000:
        raise ValueError(f"invalid brief for {pair.get('pair_id')}")
    brief = brief.strip()
    if sha256_text(brief) != pair.get("brief_sha256"):
        raise ValueError(f"brief changed for {pair.get('pair_id')}")
    actual = _source_identity(brief_sha256=pair["brief_sha256"], anchor=anchor, candidate=candidate)
    if actual["image_ids"] != pair.get("image_ids") or actual["image_sha256"] != pair.get("image_sha256"):
        raise ValueError(f"image identity changed for {pair.get('pair_id')}")
    return brief, anchor, candidate, pair["pair_id"]


def _identity(manifest: dict[str, Any], pair: dict[str, Any], method: str, repeat: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"], "split": pair["split"], "method": method,
        "repeat": repeat, "protocol": manifest["protocols"][method],
        "model": manifest["models"][method],
        "prompt_sha256": manifest["prompts"][f"{method}_sha256"],
        "format_version": manifest["format_versions"][method],
        "source_format_sha256": manifest["source_format_sha256"][method],
        "brief_sha256": pair["brief_sha256"],
        "image_ids": pair["image_ids"], "image_sha256": pair["image_sha256"],
    }


def _safe_failure(exc: BaseException) -> str:
    name = type(exc).__name__
    return "judge_failure_" + (name if name.replace("_", "").isalnum() else "ProviderError")


def _exception_status(exc: BaseException) -> str:
    """Keep local implementation/input faults out of the judge denominator."""

    if isinstance(exc, (AttributeError, AssertionError, FileNotFoundError,
                        ImportError, KeyError, PermissionError, TypeError, ValueError)):
        return "implementation_error"
    return "censored"


async def _one(*, manifest: dict[str, Any], pair: dict[str, Any], method: str,
               repeat: int, output: Path, key_file: Path,
               sem: asyncio.Semaphore) -> dict[str, Any]:
    brief, anchor, candidate, pair_id = _pair_paths(pair)
    expected = _identity(manifest, pair, method, repeat)
    path = output / "receipts" / method / f"{pair_id}--repeat-{repeat}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("schema") != RECEIPT_SCHEMA or cached.get("identity") != expected:
            raise ValueError(f"stale receipt: {method}/{pair_id}/repeat-{repeat}")
        return cached
    judge = old_judge if method == "old" else draft_judge
    async with sem:
        started = time.monotonic()
        try:
            # Always judge the normalized, hash-bound text.  This avoids a
            # source brief's optional trailing newline changing the receipt
            # identity while the visible prompt remains the same.
            brief_path = _brief_temp(brief, output, pair_id)
            result = await judge.judge_pair(brief=brief_path,
                                            anchor=anchor, candidate=candidate,
                                            key_file=key_file)
            if result.get("protocol") != judge.PROTOCOL or result.get("format_version") != judge.FORMAT_VERSION:
                raise ValueError("judge protocol or format mismatch")
            if (result.get("brief_sha256") != pair["brief_sha256"]
                    or result.get("image_ids") != pair["image_ids"]
                    or result.get("image_sha256") != pair["image_sha256"]):
                raise ValueError("judge result image or brief identity mismatch")
            accepted = result.get("status") in {"scored", "identical_canvases"}
            receipt = {
                "schema": RECEIPT_SCHEMA, "identity": expected,
                "status": "scored" if accepted else "censored",
                "reason": None if accepted else result.get("consensus", "incomplete_evidence"),
                "elapsed_seconds_runner": time.monotonic() - started,
                "judge": result,
            }
        except Exception as exc:
            receipt = {
                "schema": RECEIPT_SCHEMA, "identity": expected,
                "status": _exception_status(exc),
                "reason": _safe_failure(exc), "elapsed_seconds_runner": time.monotonic() - started,
            }
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
    return receipt


def _brief_temp(brief: str, output: Path, pair_id: str) -> Path:
    """Materialize inline briefs outside the tracked source surface."""

    path = output / "briefs" / f"{pair_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"inline brief path is a symlink for {pair_id}")
    if path.is_file() and path.read_text(encoding="utf-8") != brief:
        raise ValueError(f"inline brief changed for {pair_id}")
    if not path.is_file():
        path.write_text(brief, encoding="utf-8")
    return path


def _read_receipts(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        for method in METHODS:
            for repeat in range(1, manifest["repeat_count"] + 1):
                path = output / "receipts" / method / f"{pair['pair_id']}--repeat-{repeat}.json"
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(path)
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("schema") != RECEIPT_SCHEMA or row.get("identity") != _identity(manifest, pair, method, repeat):
                    raise ValueError(f"receipt identity mismatch: {method}/{pair['pair_id']}/repeat-{repeat}")
                row["_pair"] = pair
                rows.append(row)
    return rows


def _decision(row: dict[str, Any]) -> tuple[str, str] | None:
    result = row.get("judge") or {}
    if row.get("status") != "scored" or result.get("status") not in {"scored", "identical_canvases"}:
        return None
    winner, preference = result.get("winner_id"), result.get("preference")
    if not isinstance(winner, str) or not isinstance(preference, str):
        return None
    return winner, preference


def _usage(result: dict[str, Any]) -> tuple[int, float, dict[str, int]]:
    calls = []
    for view in list(result.get("views") or []) + list(result.get("recheck_views") or []):
        calls.extend(view.get("calls") or [])
    totals: dict[str, int] = {}
    cost = 0.0
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(name), (int, float)) and not isinstance(usage.get(name), bool):
                totals[name] = totals.get(name, 0) + int(usage[name])
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            cost += float(usage["cost"])
    return len(calls), cost, totals


def _order(row: dict[str, Any]) -> tuple[bool | None, list[str]]:
    result = row.get("judge") or {}
    votes = result.get("initial_votes")
    if not isinstance(votes, list) or len(votes) != 2 or not all(isinstance(v, str) for v in votes):
        return None, []
    return votes[0] == votes[1], votes


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "receipt_count": len(rows), "scored_count": 0, "censored_count": 0,
        "implementation_error_count": 0, "uncertain_count": 0,
        "winner_counts": {}, "preference_counts": {},
        "initial_order_valid_count": 0, "initial_order_agreement_count": 0,
        "initial_order_conflict_count": 0, "api_call_count": 0,
        "reported_cost_usd": 0.0, "usage": {},
    }
    for row in rows:
        result = row.get("judge") or {}
        decision = _decision(row)
        if decision is None:
            if row.get("status") == "implementation_error":
                out["implementation_error_count"] += 1
            else:
                out["censored_count"] += 1
                if result.get("status") == "uncertain":
                    out["uncertain_count"] += 1
        else:
            out["scored_count"] += 1
            winner, preference = decision
            out["winner_counts"][winner] = out["winner_counts"].get(winner, 0) + 1
            out["preference_counts"][preference] = out["preference_counts"].get(preference, 0) + 1
        agreement, votes = _order(row)
        if agreement is not None:
            out["initial_order_valid_count"] += 1
            if agreement:
                out["initial_order_agreement_count"] += 1
            else:
                out["initial_order_conflict_count"] += 1
        calls, cost, usage = _usage(result)
        out["api_call_count"] += calls
        out["reported_cost_usd"] += cost
        for name, value in usage.items():
            out["usage"][name] = out["usage"].get(name, 0) + value
    return out


def _repeat_metrics(rows: list[dict[str, Any]], repeat_count: int) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["_pair"]["pair_id"]].append(row)
    eligible = winner_consistent = exact_consistent = magnitude_only = 0
    unresolved = 0
    implementation_errors = 0
    for pair_rows in by_pair.values():
        pair_rows.sort(key=lambda row: row["identity"]["repeat"])
        decisions = [_decision(row) for row in pair_rows]
        if len(pair_rows) != repeat_count or any(decision is None for decision in decisions):
            unresolved += 1
            implementation_errors += any(row.get("status") == "implementation_error" for row in pair_rows)
            continue
        eligible += 1
        assert all(decision is not None for decision in decisions)
        winners = {decision[0] for decision in decisions}
        exact = len(set(decisions)) == 1
        winner_consistent += len(winners) == 1
        exact_consistent += exact
        magnitude_only += len(winners) == 1 and not exact
    denom = eligible or 0
    return {
        "pair_count": len(by_pair), "repeat_count": repeat_count,
        "all_repeats_scored_pairs": eligible,
        "unresolved_or_censored_pairs": unresolved,
        "implementation_error_pairs": implementation_errors,
        "winner_consistent_pairs": winner_consistent,
        "exact_decision_consistent_pairs": exact_consistent,
        "magnitude_only_change_pairs": magnitude_only,
        "winner_consistency_rate": winner_consistent / denom if denom else None,
        "exact_decision_consistency_rate": exact_consistent / denom if denom else None,
    }


def _paired_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        ident = row["identity"]
        by_key[(ident["pair_id"], ident["repeat"])][ident["method"]] = row
    paired = winner_agreement = exact_agreement = winner_flip = unresolved = implementation_errors = 0
    for methods in by_key.values():
        old, draft = methods.get("old"), methods.get("draft")
        if old is None or draft is None:
            continue
        old_decision, draft_decision = _decision(old), _decision(draft)
        if old_decision is None or draft_decision is None:
            unresolved += 1
            implementation_errors += any(row.get("status") == "implementation_error" for row in (old, draft))
            continue
        paired += 1
        if old_decision[0] == draft_decision[0]:
            winner_agreement += 1
        else:
            winner_flip += 1
        exact_agreement += old_decision == draft_decision
    return {
        "paired_scored_repeats": paired,
        "unresolved_or_censored_repeats": unresolved,
        "implementation_error_repeats": implementation_errors,
        "winner_agreement": winner_agreement,
        "winner_flip": winner_flip,
        "exact_decision_agreement": exact_agreement,
        "winner_agreement_rate": winner_agreement / paired if paired else None,
        "exact_decision_agreement_rate": exact_agreement / paired if paired else None,
    }


def summarize(*, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    repeats: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["identity"]["method"] == method]
        metrics[method] = {
            split: _metrics([row for row in method_rows if row["_pair"]["split"] == split])
            for split in SPLITS
        }
        metrics[method]["total"] = _metrics(method_rows)
        repeats[method] = {
            split: _repeat_metrics(
                [row for row in method_rows if row["_pair"]["split"] == split],
                manifest["repeat_count"],
            )
            for split in SPLITS
        }
        repeats[method]["total"] = _repeat_metrics(method_rows, manifest["repeat_count"])
    for split in SPLITS:
        paired[split] = _paired_metrics([row for row in rows if row["_pair"]["split"] == split])
    paired["total"] = _paired_metrics(rows)
    compact_rows = []
    for row in rows:
        result = row.get("judge") or {}
        decision = _decision(row)
        order_agreement, order_votes = _order(row)
        compact_rows.append({
            "pair_id": row["identity"]["pair_id"], "split": row["identity"]["split"],
            "method": row["identity"]["method"], "repeat": row["identity"]["repeat"],
            "status": row["status"], "judge_status": result.get("status"),
            "winner_id": decision[0] if decision else None,
            "preference": decision[1] if decision else None,
            "score": result.get("score") if decision else None,
            "initial_order_agreement": order_agreement, "initial_votes": order_votes,
        })
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "protocols": manifest["protocols"], "models": manifest["models"],
        "prompt_sha256": manifest["prompts"], "format_versions": manifest["format_versions"],
        "pair_count": len(manifest["pairs"]), "repeat_count": manifest["repeat_count"],
        "selection_note": manifest["selection_note"], "scope_note": manifest["scope_note"],
        "metrics": metrics, "repeat_consistency": repeats,
        "total_repeat_consistency": {method: repeats[method]["total"] for method in METHODS},
        "paired_old_vs_draft": paired, "rows": compact_rows,
    }


async def run(*, manifest_path: Path, output: Path, key_file: Path,
              concurrency: int = 4) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    manifest = read_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    tasks: list[Awaitable[dict[str, Any]]] = []
    for pair in manifest["pairs"]:
        for method in METHODS:
            for repeat in range(1, manifest["repeat_count"] + 1):
                tasks.append(_one(manifest=manifest, pair=pair, method=method,
                                  repeat=repeat, output=output, key_file=key_file,
                                  sem=sem))
    await asyncio.gather(*tasks)
    rows = _read_receipts(output, manifest)
    summary = summarize(manifest=manifest, rows=rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, help="JSON pair list for --prepare")
    parser.add_argument("--manifest", type=Path, help="prepared input-manifest.json for --run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, help="Gateway credential file for --run")
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.pairs is None:
            parser.error("--pairs is required with --prepare-only")
        manifest = prepare_manifest(pairs_file=args.pairs, output=args.output,
                                    repeat_count=args.repeat_count)
        print(json.dumps({"manifest": str(args.output / "input-manifest.json"),
                          "pairs": len(manifest["pairs"])}, sort_keys=True))
        return 0
    if args.manifest is None or args.key_file is None:
        parser.error("--manifest and --key-file are required with --run")
    summary = asyncio.run(run(manifest_path=args.manifest, output=args.output,
                              key_file=args.key_file, concurrency=args.concurrency))
    print(json.dumps({"pairs": summary["pair_count"], "repeat_count": summary["repeat_count"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
