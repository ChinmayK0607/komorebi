"""Replay saved identity-judge responses with lossless schema normalization.

This is an offline repair pass for provider responses that the live strict
parser censored.  It never contacts the provider and never edits the source
receipts.  Only known-key whitespace and the explicit ``_preference`` alias
are normalized; extra top-level fields are recorded and ignored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "judge-alignment-identity-replay-receipt-v1"
SUMMARY_SCHEMA = "judge-alignment-identity-replay-summary-v1"
MAGNITUDES = ("tied", "slight", "moderate", "clear")
FIDELITY = {"clear", "partial", "missing"}
TOP_FIELDS = {"assessments", "reason", "winner_id", "preference"}
ASSESSMENT_FIELDS = {"brief_fidelity", "visible_evidence", "weaknesses"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _strip_json_fence(raw: str) -> tuple[str, list[str]]:
    # The replay is intentionally narrower than a permissive response parser:
    # fences are not a known-key repair and therefore remain censored.
    return raw.strip(), []


def _known_key_map(value: dict[str, Any], known: set[str], *, aliases: set[str] = set(),
                   allow_unknown: bool = False) -> tuple[dict[str, Any], list[str], list[str]]:
    """Map only known keys, rejecting canonical/alias collisions."""

    result: dict[str, Any] = {}
    transformations: list[str] = []
    ignored: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("non-string JSON key")
        canonical = key if key in known or key in aliases else None
        if canonical is None and key.strip() in known:
            canonical = key.strip()
            transformations.append(f"trimmed_key:{key!r}->{canonical}")
        if canonical in aliases:
            if canonical != "_preference":
                raise ValueError("unsupported key alias")
            canonical = "preference"
            transformations.append("alias:_preference->preference")
        if canonical is None:
            if allow_unknown:
                ignored.append(key)
                transformations.append(f"ignored_top_key:{key}")
                continue
            raise ValueError(f"unknown nested key:{key}")
        if canonical in result:
            raise ValueError(f"conflicting key alias:{canonical}")
        result[canonical] = item
    return result, transformations, ignored


def _validate_assessment(value: Any, ids: tuple[str, str], transformations: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("assessment is not an object")
    assessments, changes, _ = _known_key_map(value, ASSESSMENT_FIELDS)
    transformations.extend(changes)
    if set(assessments) != ASSESSMENT_FIELDS:
        raise ValueError("assessment fields are incomplete")
    fidelity = assessments["brief_fidelity"]
    evidence = assessments["visible_evidence"]
    weaknesses = assessments["weaknesses"]
    if fidelity not in FIDELITY:
        raise ValueError("invalid brief_fidelity")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("empty visible_evidence")
    # Deliberately do not turn a scalar into a one-item list.  That changes the
    # wire protocol's shape and would conceal a non-lossless repair.
    if not isinstance(weaknesses, list) or not weaknesses or not all(isinstance(item, str) and item.strip() for item in weaknesses):
        raise ValueError("weaknesses must be a nonempty string list")
    return {key: assessments[key] for key in ("brief_fidelity", "visible_evidence", "weaknesses")}


def normalize_response(raw: str, ids: tuple[str, str]) -> dict[str, Any]:
    """Normalize and strictly validate one saved assistant response."""

    text, transformations = _strip_json_fence(raw)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    normalized, changes, ignored = _known_key_map(value, TOP_FIELDS, aliases={"_preference"}, allow_unknown=True)
    transformations.extend(changes)
    if set(normalized) != TOP_FIELDS:
        raise ValueError("response required fields are incomplete")
    assessments = normalized["assessments"]
    if not isinstance(assessments, dict) or set(assessments) != set(ids):
        # IDs are opaque: do not trim, case-fold, or otherwise repair them.
        raise ValueError("response image IDs do not match call IDs")
    checked = {ident: _validate_assessment(assessments[ident], ids, transformations) for ident in ids}
    reason = normalized["reason"]
    winner = normalized["winner_id"]
    preference = normalized["preference"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("empty comparison reason")
    if winner not in (*ids, "tie"):
        raise ValueError("invalid winner_id")
    if preference not in MAGNITUDES:
        raise ValueError("invalid preference")
    if (winner == "tie") != (preference == "tied"):
        raise ValueError("winner and preference disagree")
    return {
        "assessments": checked,
        "reason": reason,
        "winner_id": winner,
        "preference": preference,
        "normalization": {"transformations": transformations, "ignored_top_keys": ignored},
    }


def _call_ids(call: dict[str, Any]) -> tuple[str, str]:
    image_ids = call.get("image_ids")
    if not isinstance(image_ids, dict) or set(image_ids) != {"A", "B"}:
        raise ValueError("call image IDs are incomplete")
    ids = (image_ids["A"], image_ids["B"])
    if not all(isinstance(ident, str) and ident for ident in ids) or ids[0] == ids[1]:
        raise ValueError("call image IDs are invalid")
    return ids


def replay_call(call: dict[str, Any]) -> dict[str, Any]:
    """Return a provenance record retaining the untouched call object."""

    record: dict[str, Any] = {
        "raw_call": copy.deepcopy(call),
        "usage": copy.deepcopy(call.get("usage")),
        "valid": False,
        "assessment": None,
        "normalization": {"transformations": [], "ignored_top_keys": []},
    }
    if isinstance(call.get("elapsed_seconds"), (int, float)):
        record["elapsed_seconds"] = call["elapsed_seconds"]
    try:
        if not isinstance(call, dict):
            raise ValueError("call is not an object")
        ids = _call_ids(call)
        raw = call.get("raw_response")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("raw_response is missing")
        normalized = normalize_response(raw, ids)
        record["valid"] = True
        record["assessment"] = normalized
        record["normalization"] = normalized.pop("normalization")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        record["error"] = str(exc).split(":", 1)[0]
    return record


def _score(winner: str, preference: str, anchor_id: str, candidate_id: str) -> float:
    if winner == "tie":
        return 0.5
    delta = MAGNITUDES.index(preference) / 6
    return 0.5 + delta if winner == candidate_id else 0.5 - delta


def _replay_view(view: dict[str, Any]) -> dict[str, Any]:
    calls = view.get("calls") if isinstance(view, dict) else None
    if not isinstance(calls, list):
        calls = []
    all_calls = [replay_call(call) for call in calls]
    selected_index = next((index for index, record in enumerate(all_calls) if record["valid"]), None)
    selected = all_calls[selected_index]["assessment"] if selected_index is not None else None
    return {
        "assessment": selected,
        "calls": all_calls,
        "selected_call_index": selected_index,
        "source_view_assessment_present": isinstance(view.get("assessment"), dict) if isinstance(view, dict) else False,
    }


def replay_judge(judge: dict[str, Any]) -> dict[str, Any]:
    """Replay saved views without creating a missing fresh view."""

    result = copy.deepcopy(judge)
    if judge.get("status") == "identical_canvases":
        result["status"] = "identical_canvases"
        result["replay_pending_fresh"] = False
        return result
    views = judge.get("views") if isinstance(judge, dict) else None
    if not isinstance(views, list):
        views = []
    replayed_views = [_replay_view(view) for view in views]
    result["views"] = replayed_views
    anchor_id = (judge.get("image_ids") or {}).get("anchor")
    candidate_id = (judge.get("image_ids") or {}).get("candidate")
    initial = replayed_views[:2]
    votes = [view["assessment"]["winner_id"] for view in initial if view.get("assessment")]
    initial_conflict = len(votes) == 2 and votes[0] != votes[1]
    fresh_source = None
    fresh_original = judge.get("fresh_view")
    if isinstance(fresh_original, dict):
        fresh_source = _replay_view(fresh_original)
    if len(initial) < 2 or len(votes) != 2 or not anchor_id or not candidate_id:
        result.update({"status": "uncertain", "consensus": "incomplete_initial_evidence", "winner_id": None, "preference": None, "score": None, "conflict": False, "fresh_view": fresh_source, "replay_pending_fresh": False})
        return result
    if votes[0] == votes[1]:
        selected = initial[0]["assessment"]
        preference = min((view["assessment"]["preference"] for view in initial), key=MAGNITUDES.index)
        if selected["winner_id"] == "tie":
            preference = "tied"
        result.update({"status": "replay_scored", "consensus": "initial_agreement", "winner_id": selected["winner_id"], "preference": preference, "score": _score(selected["winner_id"], preference, anchor_id, candidate_id), "conflict": False, "fresh_view": fresh_source, "replay_pending_fresh": False})
        return result
    if fresh_source and fresh_source.get("assessment"):
        fresh_assessment = fresh_source["assessment"]
        all_votes = votes + [fresh_assessment["winner_id"]]
        counts = {winner: all_votes.count(winner) for winner in set(all_votes)}
        majority = next((winner for winner, count in counts.items() if count >= 2), None)
        if majority is not None:
            winning = [assessment["preference"] for view in (*initial, fresh_source) for assessment in ([view["assessment"]] if view.get("assessment") and view["assessment"]["winner_id"] == majority else [])]
            preference = min(winning, key=MAGNITUDES.index)
            if majority == "tie":
                preference = "tied"
            result.update({"status": "replay_conflict_resolved", "consensus": "saved_two_of_three", "winner_id": majority, "preference": preference, "score": _score(majority, preference, anchor_id, candidate_id), "conflict": initial_conflict, "fresh_view": fresh_source, "replay_pending_fresh": False})
            return result
    # There is no saved valid third vote.  Do not treat two order views as a
    # majority or silently manufacture a retry.
    if fresh_original is None:
        consensus = "pending_fresh"
        pending = True
    elif fresh_source and fresh_source.get("assessment") is None:
        consensus = "incomplete_saved_fresh"
        pending = False
    else:
        consensus = "no_majority"
        pending = False
    result.update({"status": "uncertain", "consensus": consensus, "winner_id": None, "preference": None, "score": None, "conflict": initial_conflict, "fresh_view": fresh_source, "replay_pending_fresh": pending})
    return result


def replay_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    original = copy.deepcopy(receipt)
    judge = receipt.get("judge")
    if not isinstance(judge, dict):
        return {"schema": SCHEMA, "status": "censored", "reason": "missing_judge", "original_receipt_sha256": sha256_bytes(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()), "original_receipt": original}
    replayed = replay_judge(judge)
    outer_status = "scored" if replayed.get("status") in {"replay_scored", "replay_conflict_resolved", "identical_canvases"} else "censored"
    return {
        "schema": SCHEMA,
        "status": outer_status,
        "reason": None if outer_status == "scored" else replayed.get("consensus"),
        "identity": copy.deepcopy(receipt.get("identity")),
        "original_status": receipt.get("status"),
        "original_receipt_sha256": sha256_bytes(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()),
        "judge": replayed,
    }


def _receipt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    root = input_path / "receipts" if (input_path / "receipts").is_dir() else input_path
    return sorted(path for path in root.glob("*.json") if path.is_file() and not path.is_symlink())


def run(*, input_path: Path, output: Path, limit: int | None = None) -> dict[str, Any]:
    files = _receipt_files(input_path)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no receipt JSON files under {input_path}")
    receipts_dir = output / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in files:
        original = json.loads(source.read_text(encoding="utf-8"))
        replayed = replay_receipt(original)
        replayed["source_receipt"] = source.name
        destination = receipts_dir / source.name
        destination.write_text(json.dumps(replayed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(replayed)
    metrics = {"receipts": len(rows), "scored": sum(row["status"] == "scored" for row in rows), "censored": sum(row["status"] == "censored" for row in rows), "pending_fresh": sum((row.get("judge") or {}).get("replay_pending_fresh") is True for row in rows), "valid_calls": 0, "invalid_calls": 0, "normalization_transformations": 0}
    for row in rows:
        for view in ((row.get("judge") or {}).get("views") or []):
            for call in view.get("calls") or []:
                metrics["valid_calls"] += bool(call.get("valid"))
                metrics["invalid_calls"] += not bool(call.get("valid"))
                metrics["normalization_transformations"] += len((call.get("normalization") or {}).get("transformations") or [])
    summary = {"schema": SUMMARY_SCHEMA, "input": str(input_path), "output": str(output), "metrics": metrics, "scope_note": "Offline lossless response normalization only; no new provider calls or fresh votes."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="identity receipt directory or one receipt JSON")
    parser.add_argument("--output", type=Path, required=True, help="new normalized-replay evidence directory")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(input_path=args.input, output=args.output, limit=args.limit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
