"""Build a readable frozen-image report from consistent-judge receipts.

The report fixes the manifest anchor on the left and candidate on the right.
It performs no judging, network access, asset fetching, or image swapping.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any


ROLE_NAMES = {
    "frozen SFT": "Frozen SFT",
    "SFT": "Matched SFT",
    "RL-step-16": "RL step 16",
    "pilot-step-24": "RL step 24",
}
SCHEMA = "judge-alignment-consistent-report-v1"
RECEIPT_SCHEMA = "judge-alignment-consistent-receipt-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_content_sha256", None)
    return hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_report_image(gallery: Path, relative: str) -> Path:
    path = (gallery.parent / relative).resolve()
    root = gallery.parent.resolve()
    if path.is_symlink() or not path.is_file() or (root not in path.parents and path != root):
        raise ValueError(f"missing or escaping gallery image: {relative}")
    return path


def _receipt_files(path: Path) -> list[Path]:
    root = path / "receipts" if (path / "receipts").is_dir() else path
    return sorted(p for p in root.glob("*.json") if p.is_file() and not p.is_symlink())


def _load_inputs(manifest_path: Path, receipts_path: Path, gallery: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("manifest_content_sha256"):
        raise ValueError("manifest lacks immutable content hash")
    if manifest["manifest_content_sha256"] != _manifest_digest(manifest):
        raise ValueError("manifest content hash changed")
    if manifest.get("gallery_html_sha256") and _sha256(gallery) != manifest["gallery_html_sha256"]:
        raise ValueError("gallery HTML hash changed")
    pair_by_id = {pair["pair_id"]: pair for pair in manifest.get("pairs", [])}
    if len(pair_by_id) != len(manifest.get("pairs", [])):
        raise ValueError("manifest contains duplicate pair IDs")
    for pair in pair_by_id.values():
        for role, image in pair["images"].items():
            actual = _safe_report_image(gallery, image["path"])
            if _sha256(actual) != image["sha256"]:
                raise ValueError(f"gallery image hash changed: {pair['pair_id']} {role}")
        if pair.get("brief_sha256") and hashlib.sha256(pair["brief"].encode()).hexdigest() != pair["brief_sha256"]:
            raise ValueError(f"brief hash changed: {pair['pair_id']}")
    receipts: dict[str, dict[str, Any]] = {}
    for path in _receipt_files(receipts_path):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"unsupported consistent receipt schema: {path.name}")
        identity = receipt.get("identity") or {}
        pair_id = identity.get("pair_id")
        if pair_id not in pair_by_id or pair_id in receipts:
            raise ValueError(f"receipt pair identity invalid: {path.name}")
        if identity.get("manifest_content_sha256") != manifest["manifest_content_sha256"]:
            raise ValueError(f"receipt manifest binding invalid: {path.name}")
        pair = pair_by_id[pair_id]
        expected_images = {role: pair["images"][role]["sha256"] for role in (pair["anchor_role"], pair["candidate_role"])}
        if identity.get("image_sha256") != expected_images:
            raise ValueError(f"receipt image binding invalid: {path.name}")
        if pair.get("brief_sha256") and identity.get("brief_sha256") not in {None, pair["brief_sha256"]}:
            raise ValueError(f"receipt brief binding invalid: {path.name}")
        if receipt.get("status") not in {"scored", "censored"}:
            raise ValueError(f"receipt status invalid: {path.name}")
        receipts[pair_id] = receipt
    return manifest, receipts


def _side(judge: dict[str, Any], winner: str | None) -> str:
    if winner == "tie":
        return "tie"
    ids = judge.get("image_ids") or {}
    if not isinstance(winner, str) or not winner or not isinstance(ids, dict) or not ids.get("anchor") or not ids.get("candidate"):
        return "unknown"
    if winner == ids.get("anchor"):
        return "LEFT"
    if winner == ids.get("candidate"):
        return "RIGHT"
    return "unknown"


def _votes(judge: dict[str, Any], field: str) -> list[str]:
    direct = judge.get(field)
    if isinstance(direct, list):
        return [value if value in {"LEFT", "RIGHT", "tie", "unknown"} else _side(judge, str(value)) for value in direct]
    views = judge.get("views" if field == "initial_votes" else "recheck_views") or []
    return [_side(judge, (view.get("assessment") or {}).get("winner_id")) for view in views if view.get("assessment")]


def _verdict(judge: dict[str, Any]) -> dict[str, Any]:
    status = judge.get("status")
    winner = _side(judge, judge.get("winner_id"))
    if status != "scored" or winner == "unknown":
        headline = "Uncertain"
    elif judge.get("consensus") == "initial_agreement":
        headline = "Judge agrees: " + (winner if winner != "tie" else "tie")
    elif judge.get("acceptance") == "recheck_consensus" or judge.get("consensus") in {"three_of_four", "four_of_four"}:
        headline = "Judge agrees after recheck: " + (winner if winner != "tie" else "tie")
    else:
        headline = "Judge agrees: " + (winner if winner != "tie" else "tie")
    return {"headline": headline, "winner": winner, "status": status, "initial_votes": _votes(judge, "initial_votes"), "recheck_votes": _votes(judge, "recheck_votes"), "consensus": judge.get("consensus"), "acceptance": judge.get("acceptance"), "score": judge.get("score")}


def _flatten_calls(judge: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for group, views in (("initial", judge.get("views") or []), ("recheck", judge.get("recheck_views") or [])):
        for view_index, view in enumerate(views, 1):
            for call_index, call in enumerate(view.get("calls") or [], 1):
                usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
                flattened.append({
                    "view": f"{group}-{view_index}", "attempt": call.get("attempt", call_index),
                    "valid": bool(call.get("assessment")),
                    "elapsed_seconds": call.get("elapsed_seconds"),
                    "usage": usage,
                    "provider_error": call.get("provider_error"),
                    "http_status": call.get("http_status"),
                    "parse_error": call.get("parse_error"),
                })
    return flattened


def _call_summary(judge: dict[str, Any]) -> dict[str, Any]:
    calls = _flatten_calls(judge)
    usage: dict[str, float] = {}
    reported_cost: dict[str, float] = {}
    for call in calls:
        for key, value in call["usage"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target = reported_cost if key in {"cost", "gateway_cost", "market_cost"} else usage
                target[key] = target.get(key, 0) + value
    return {"calls": calls, "call_count": len(calls), "valid_calls": sum(call["valid"] for call in calls),
            "failed_calls": sum(not call["valid"] for call in calls),
            "latency_seconds": sum(float(call["elapsed_seconds"] or 0) for call in calls),
            "usage": usage, "reported_cost": reported_cost}


def _assessment(judge: dict[str, Any], preferred_winner: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    views = list(judge.get("views") or []) + list(judge.get("recheck_views") or [])
    for view in views:
        assessment = view.get("assessment") or {}
        if assessment and (preferred_winner is None or assessment.get("winner_id") == preferred_winner):
            raw = assessment.get("assessments") or {}
            ids = judge.get("image_ids") or {}
            return assessment.get("reason"), {"LEFT": raw.get(ids.get("anchor"), {}), "RIGHT": raw.get(ids.get("candidate"), {})}
    return None, None


def _previous(receipt: dict[str, Any] | None, pair: dict[str, Any]) -> dict[str, Any] | None:
    if not receipt:
        return None
    version = (receipt.get("versions") or {}).get("new") or {}
    result = version.get("judge") or {}
    views = result.get("views") or []
    choices = []
    for index, view in enumerate(views):
        label = view.get("label") or {}
        winner = label.get("winner")
        if winner == "tie":
            choices.append("tie")
        elif winner in {"A", "B"}:
            role = pair["anchor_role"] if (winner == "A") == (index == 0) else pair["candidate_role"]
            choices.append("LEFT" if role == pair["anchor_role"] else "RIGHT")
    if len(choices) != 2 or choices[0] != choices[1]:
        headline = "Uncertain"
    elif choices[0] == "tie":
        headline = "Judge agrees: tie"
    else:
        headline = "Judge agrees: " + choices[0]
    return {"headline": headline, "choices": choices, "reason": ((views[0].get("label") or {}).get("reason") if views else None)}


def _human_feedback(manifest: dict[str, Any], pair: dict[str, Any], verdict: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    feedback = (manifest.get("human_feedback") or {}).get(pair["case_id"], {})
    for item in feedback.get("explicit_preferences", []):
        if {item.get("preferred_role"), item.get("other_role")} != {pair["anchor_role"], pair["candidate_role"]}:
            continue
        judge_winner = verdict.get("winner") if verdict.get("status") == "scored" else "unknown"
        selected.append({
            "case_id": pair["case_id"], "preferred_role": item.get("preferred_role"), "other_role": item.get("other_role"),
            "judge_winner": judge_winner,
            "agreement": "agree" if judge_winner == ("LEFT" if item.get("preferred_role") == pair["anchor_role"] else "RIGHT") else "disagree" if judge_winner in {"LEFT", "RIGHT"} else "uncertain",
        })
    return selected


def build_summary(*, manifest: dict[str, Any], receipts: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = []
    metrics = {"pairs": len(manifest.get("pairs", [])), "initial_valid": 0, "initial_conflicts": 0, "accepted_2_of_2": 0, "accepted_3_or_4_of_4": 0, "uncertain": 0, "unknown": 0, "human_selected": 0, "human_agree": 0, "human_disagree": 0, "human_uncertain": 0, "api_calls": 0, "valid_calls": 0, "failed_calls": 0, "latency_seconds": 0.0, "usage": {}, "reported_cost": {}}
    selected_all = []
    for pair in manifest.get("pairs", []):
        receipt = receipts.get(pair["pair_id"])
        judge = (receipt or {}).get("judge") or {}
        verdict = _verdict(judge) if receipt else {"headline": "Uncertain", "winner": "unknown", "initial_votes": [], "recheck_votes": [], "consensus": None, "acceptance": None, "status": "missing"}
        calls = _call_summary(judge)
        initial_valid = len(verdict["initial_votes"]) == 2 and all(vote in {"LEFT", "RIGHT", "tie"} for vote in verdict["initial_votes"])
        if initial_valid:
            metrics["initial_valid"] += 1
        if (judge.get("initial_conflict") and initial_valid) or (initial_valid and verdict["initial_votes"][0] != verdict["initial_votes"][1]):
            metrics["initial_conflicts"] += 1
        if verdict["status"] == "scored" and verdict["consensus"] == "initial_agreement":
            metrics["accepted_2_of_2"] += 1
        elif verdict["status"] == "scored" and verdict["consensus"] in {"three_of_four", "four_of_four"}:
            metrics["accepted_3_or_4_of_4"] += 1
        else:
            metrics["uncertain"] += 1
            metrics["unknown"] += 1
        metrics["api_calls"] += calls["call_count"]
        metrics["valid_calls"] += calls["valid_calls"]
        metrics["failed_calls"] += calls["failed_calls"]
        metrics["latency_seconds"] += calls["latency_seconds"]
        for key, value in calls["usage"].items():
            metrics["usage"][key] = metrics["usage"].get(key, 0) + value
        for key, value in calls["reported_cost"].items():
            metrics["reported_cost"][key] = metrics["reported_cost"].get(key, 0) + value
        reason, assessments = _assessment(judge, judge.get("winner_id"))
        human = _human_feedback(manifest, pair, verdict)
        selected_all.extend(human)
        metrics["human_selected"] += len(human)
        for item in human:
            metrics["human_" + item["agreement"]] += 1
        previous_display = _previous((previous or {}).get(pair["pair_id"]), pair) if (previous or {}).get(pair["pair_id"]) else None
        rows.append({"pair_id": pair["pair_id"], "case_id": pair["case_id"], "family": pair.get("family"), "brief": pair.get("brief"), "anchor_role": pair["anchor_role"], "candidate_role": pair["candidate_role"], "images": pair["images"], "image_ids": judge.get("image_ids") or {}, "verdict": verdict, "reason": reason, "assessments": assessments or {}, "calls": calls, "human_selected": human, "previous": previous_display})
    return {"schema": SCHEMA, "manifest_content_sha256": manifest.get("manifest_content_sha256"), "pair_count": len(rows), "metrics": metrics, "human_selected": selected_all, "pairs": rows}


def _img_src(report: Path, gallery: Path, relative: str) -> str:
    return Path(os.path.relpath(gallery.parent / relative, report.parent)).as_posix()


def _display_text(value: Any, image_ids: dict[str, Any]) -> str:
    text = str(value)
    replacements = (
        (image_ids.get("anchor"), "LEFT painting"),
        (image_ids.get("candidate"), "RIGHT painting"),
    )
    for ident, label in replacements:
        if isinstance(ident, str) and ident:
            text = text.replace(ident, label)
    return text


def write_report(summary: dict[str, Any], *, output: Path, gallery: Path) -> Path:
    names = ROLE_NAMES
    cards = []
    for index, pair in enumerate(summary["pairs"], 1):
        left = pair["anchor_role"]
        right = pair["candidate_role"]
        images = []
        for side, role in (("LEFT", left), ("RIGHT", right)):
            images.append(f"<figure><img src='{html.escape(_img_src(output / 'report.html', gallery, pair['images'][role]['path']))}' alt='{side}'><figcaption>{side} · {html.escape(names.get(role, role))}</figcaption></figure>")
        assessments = pair.get("assessments") or {}
        details = []
        for side, role in (("LEFT", left), ("RIGHT", right)):
            assessment = assessments.get(side, {})
            image_ids = pair.get("image_ids") or {}
            evidence = _display_text(assessment.get("visible_evidence", "Unavailable"), image_ids)
            weaknesses = _display_text("; ".join(assessment.get("weaknesses", [])) or "Unavailable", image_ids)
            details.append(f"<div><b>{side} evidence:</b> {html.escape(evidence)}<br><b>{side} weaknesses:</b> {html.escape(weaknesses)}</div>")
        human = "".join(f"<aside>Human selection: {html.escape(ROLE_NAMES.get(item['preferred_role'], item['preferred_role']))} preferred over {html.escape(ROLE_NAMES.get(item['other_role'], item['other_role']))}.</aside>" for item in pair.get("human_selected", []))
        reason = _display_text(pair.get("reason") or "No comparison reason available.", pair.get("image_ids") or {})
        previous = f"<p><b>Previous judge:</b> {html.escape(pair['previous']['headline'])}</p>" if pair.get("previous") else ""
        technical = html.escape(json.dumps({"calls": pair["calls"]["call_count"], "valid_calls": pair["calls"]["valid_calls"], "failed_calls": pair["calls"]["failed_calls"], "latency_seconds": pair["calls"]["latency_seconds"], "usage": pair["calls"]["usage"], "reported_cost": pair["calls"]["reported_cost"], "initial_votes": pair["verdict"]["initial_votes"], "recheck_votes": pair["verdict"]["recheck_votes"]}, sort_keys=True))
        cards.append(f"<section><h2>Pair {index}: {html.escape(pair['case_id'])}</h2><details><summary>Painting brief</summary><p>{html.escape(pair['brief'])}</p></details>{human}<div class='images'>{''.join(images)}</div><p class='verdict'>{html.escape(pair['verdict']['headline'])}</p>{previous}<details open><summary>Why this result</summary><p>{html.escape(reason)}</p>{''.join(details)}</details><details><summary>Technical details</summary><pre>{technical}</pre></details></section>")
    metrics = summary["metrics"]
    report = """<!doctype html><html><head><meta charset='utf-8'><title>Consistent judge report</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui,sans-serif;padding:24px;overflow-x:auto}main{max-width:1320px;min-width:900px;margin:auto}section,.meta{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:18px;margin:18px 0}h1,h2{color:#f9fafb}.images{display:grid;grid-template-columns:repeat(2,minmax(420px,1fr));gap:18px}.images figure{margin:0;background:#111827;padding:10px;border-radius:10px}.images img{display:block;width:100%;height:auto;max-height:600px;object-fit:contain;background:white;border-radius:7px;cursor:zoom-in}.images img:active{transform:scale(1.35);position:relative;z-index:2}.verdict{font-size:1.15em;font-weight:700;color:#bfdbfe}details{margin-top:10px;line-height:1.5}summary{cursor:pointer;color:#bfdbfe}aside{border-left:3px solid #fbbf24;color:#fde68a;padding:8px 12px;margin:8px 0}pre{white-space:pre-wrap;color:#cbd5e1}</style></head><body><main>"""
    report += f"<h1>Consistent judge report</h1><div class='meta'><p>Each pair keeps the first painting on the LEFT and the second painting on the RIGHT. The judge sees both paintings together. ‘Judge agrees after recheck’ means both image orders were rechecked and at least three of four votes agreed; ‘Uncertain’ means the evidence did not meet that threshold.</p><p>Pairs: {metrics['pairs']} · Initial valid: {metrics['initial_valid']} · Initial conflicts: {metrics['initial_conflicts']} · Accepted 2/2: {metrics['accepted_2_of_2']} · Accepted 3/4 or 4/4: {metrics['accepted_3_or_4_of_4']} · Uncertain: {metrics['uncertain']} · Recorded human selections: {metrics['human_selected']} (agree {metrics['human_agree']}, disagree {metrics['human_disagree']}, uncertain {metrics['human_uncertain']})</p><details><summary>Batch usage</summary><pre>{html.escape(json.dumps({'api_calls': metrics['api_calls'], 'valid_calls': metrics['valid_calls'], 'failed_calls': metrics['failed_calls'], 'latency_seconds': metrics['latency_seconds'], 'usage': metrics['usage'], 'reported_cost': metrics['reported_cost']}, indent=2, sort_keys=True))}</pre></details></div>" + "".join(cards) + "</main></body></html>\n"
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def build(*, manifest_path: Path, receipts_path: Path, gallery: Path, output: Path, previous_receipts: Path | None = None) -> dict[str, Any]:
    manifest, receipts = _load_inputs(manifest_path, receipts_path, gallery)
    previous = {}
    if previous_receipts:
        for path in _receipt_files(previous_receipts):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            pair_id = (receipt.get("identity") or {}).get("pair_id")
            if pair_id:
                previous[pair_id] = receipt
    summary = build_summary(manifest=manifest, receipts=receipts, previous=previous)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, output=output, gallery=gallery)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-receipts", type=Path)
    args = parser.parse_args(argv)
    summary = build(manifest_path=args.manifest, receipts_path=args.receipts, gallery=args.gallery, output=args.output, previous_receipts=args.previous_receipts)
    print(json.dumps({"output": str(args.output), "pairs": summary["pair_count"], "metrics": summary["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
