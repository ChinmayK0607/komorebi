"""Run the identity-keyed joint-sheet judge against the frozen 36-pair panel."""

from __future__ import annotations

import argparse
import asyncio
import html
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_identity_judge as identity_judge
    from painter.text_pairwise_margin_judge import MODEL
except ImportError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_identity_judge as identity_judge
    from painter.text_pairwise_margin_judge import MODEL


def _load_helpers():
    path = Path(__file__).with_name("judge_alignment.py")
    spec = importlib.util.spec_from_file_location("judge_alignment_identity_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ja = _load_helpers()
SCHEMA = "judge-alignment-identity-input-v1"
RECEIPT_SCHEMA = "judge-alignment-identity-receipt-v1"
SUMMARY_SCHEMA = "judge-alignment-identity-summary-v1"
IDENTITY_MODULE_SOURCE_SHA256 = identity_judge.SOURCE_FORMAT_SHA256
RUNNER_SOURCE_SHA256 = ja.sha256_file(Path(__file__))


def _digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_content_sha256", None)
    return ja.sha256_text(json.dumps(copy, sort_keys=True, separators=(",", ":")))


def _source(pass1_output: Path) -> dict[str, Any]:
    manifest = ja._read_manifest(pass1_output)
    if len(manifest.get("pairs", [])) != 36:
        raise ValueError("identity calibration requires all 36 frozen pairs")
    return manifest


def _baseline(pass1_output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in manifest["pairs"]:
        path = pass1_output / "receipts" / f"{pair['pair_id']}--repeat-1.json"
        if not path.is_file() or path.is_symlink():
            missing.append(pair["pair_id"])
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != ja.RECEIPT_SCHEMA or receipt.get("identity") != ja._pair_identity(manifest, pair, 1):
            raise ValueError(f"pass-1 receipt identity mismatch: {pair['pair_id']}")
        rows.append(receipt)
    if missing:
        raise FileNotFoundError("missing pass-1 receipts: " + ", ".join(missing))
    return rows


def prepare_manifest(*, pass1_output: Path, output: Path) -> dict[str, Any]:
    source = _source(pass1_output)
    _baseline(pass1_output, source)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source_pass1_manifest_content_sha256": source["manifest_content_sha256"],
        "source_pass1_protocol": source["protocol"],
        "protocol": identity_judge.PROTOCOL,
        "model": MODEL,
        "prompt_sha256": ja.sha256_text(identity_judge.PROMPT),
        "format_version": identity_judge.FORMAT_VERSION,
        "source_format_sha256": identity_judge.SOURCE_FORMAT_SHA256,
        "source_module_sha256": IDENTITY_MODULE_SOURCE_SHA256,
        "source_runner_sha256": RUNNER_SOURCE_SHA256,
        "gallery_html_sha256": source["gallery_html_sha256"],
        "gallery_path": source.get("gallery_path"),
        "cases": source["cases"], "pairs": source["pairs"],
        "pair_plan": source["pair_plan"], "human_feedback": source["human_feedback"],
        "historical_score_note": source["historical_score_note"],
        "scope_note": "Identity/evidence-first joint-sheet calibration; no training reward integration.",
    }
    manifest["manifest_content_sha256"] = _digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_manifest(output: Path) -> dict[str, Any]:
    path = output / "input-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("manifest_content_sha256") != _digest(manifest):
        raise ValueError("identity manifest integrity mismatch")
    if len(manifest.get("pairs", [])) != 36:
        raise ValueError("identity calibration requires all 36 pairs")
    if manifest.get("prompt_sha256") != ja.sha256_text(identity_judge.PROMPT) or manifest.get("protocol") != identity_judge.PROTOCOL:
        raise ValueError("identity prompt/protocol changed since preparation")
    if manifest.get("format_version") != identity_judge.FORMAT_VERSION or manifest.get("source_format_sha256") != IDENTITY_MODULE_SOURCE_SHA256:
        raise ValueError("identity source format changed since preparation")
    if manifest.get("source_module_sha256") != IDENTITY_MODULE_SOURCE_SHA256 or manifest.get("source_runner_sha256") != RUNNER_SOURCE_SHA256:
        raise ValueError("identity judge or runner source changed since preparation")
    return manifest


def _identity(manifest: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "source_pass1_manifest_content_sha256": manifest["source_pass1_manifest_content_sha256"],
        "protocol": manifest["protocol"], "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"], "format_version": manifest["format_version"],
        "source_format_sha256": manifest["source_format_sha256"],
        "source_module_sha256": manifest["source_module_sha256"],
        "source_runner_sha256": manifest["source_runner_sha256"],
        "pair_id": pair["pair_id"], "repeat": 1, "brief_sha256": pair["brief_sha256"],
        "image_sha256": {role: pair["images"][role]["sha256"] for role in (pair["anchor_role"], pair["candidate_role"])},
    }


def _safe_failure(exc: Exception) -> str:
    name = type(exc).__name__
    return "judge_failure_" + name if name.replace("_", "").isalnum() else "judge_failure_ProviderError"


async def _one(*, manifest: dict[str, Any], pair: dict[str, Any], gallery: Path,
               output: Path, key_file: Path, sem: asyncio.Semaphore,
               index: int, total: int) -> dict[str, Any]:
    ja._verify_pair_files(pair, gallery)
    brief = ja._materialize_brief(pair, output)
    expected = _identity(manifest, pair)
    path = output / "receipts" / f"{pair['pair_id']}--repeat-1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("schema") != RECEIPT_SCHEMA or cached.get("identity") != expected:
            raise ValueError(f"stale identity receipt: {pair['pair_id']}")
        print(f"[{index}/{total}] cached {pair['pair_id']}", flush=True)
        return cached
    async with sem:
        started = time.monotonic()
        try:
            result = await identity_judge.judge_pair(
                brief=brief,
                anchor=gallery.parent / pair["images"][pair["anchor_role"]]["path"],
                candidate=gallery.parent / pair["images"][pair["candidate_role"]]["path"],
                key_file=key_file,
            )
            if result.get("protocol") != identity_judge.PROTOCOL or result.get("format_version") != identity_judge.FORMAT_VERSION:
                raise ValueError("identity protocol or format mismatch")
            valid_statuses = {"scored", "conflict_resolved", "identical_canvases"}
            outer_status = "scored" if result.get("status") in valid_statuses else "censored"
            receipt = {"schema": RECEIPT_SCHEMA, "identity": expected, "status": outer_status, "reason": result.get("consensus") if outer_status == "censored" else None, "elapsed_seconds_runner": time.monotonic() - started, "judge": result}
        except Exception as exc:
            receipt = {"schema": RECEIPT_SCHEMA, "identity": expected, "status": "censored", "reason": _safe_failure(exc), "elapsed_seconds_runner": time.monotonic() - started}
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
    print(f"[{index}/{total}] {receipt['status']} {pair['pair_id']}", flush=True)
    return receipt


def _side(pair: dict[str, Any], role: str | None) -> str:
    if role == pair["anchor_role"]:
        return "LEFT"
    if role == pair["candidate_role"]:
        return "RIGHT"
    return "tie" if role == "tie" else "unknown"


def _previous(receipt: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    version = (receipt.get("versions") or {}).get("new") or {}
    if version.get("status") != "scored":
        return {"status": "censored", "reason": version.get("reason")}
    result = version.get("judge") or {}
    views = result.get("views", [])
    choices = []
    for index, view in enumerate(views):
        winner = (view.get("label") or {}).get("winner")
        if winner not in {"A", "B", "tie"}:
            choices.append("unknown")
        elif winner == "tie":
            choices.append("tie")
        else:
            role = pair["anchor_role"] if (winner == "A") == (index == 0) else pair["candidate_role"]
            choices.append(_side(pair, role))
    if len(choices) != 2 or choices[0] != choices[1]:
        headline = "Uncertain — inconsistent judge choices"
    elif choices[0] == "tie":
        headline = "No clear preference"
    else:
        headline = "Prefers " + choices[0]
    return {"status": "scored", "headline": headline, "choices": choices, "reason": (views[0].get("label") or {}).get("reason") if views else None, "technical": {"protocol": result.get("protocol"), "score": result.get("score"), "order_gap": result.get("order_gap")}}


def _call_technical(result: dict[str, Any], ids: dict[str, Any]) -> dict[str, Any]:
    all_views = list(result.get("views") or [])
    if result.get("fresh_view"):
        all_views.append(result["fresh_view"])
    calls = [call for view in all_views for call in (view.get("calls") or [])]
    usage: dict[str, int] = {}
    for call in calls:
        values = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = values.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[name] = usage.get(name, 0) + int(value)
    latency = sum(float(call.get("elapsed_seconds", 0.0)) for call in calls if isinstance(call.get("elapsed_seconds"), (int, float)))
    return {"protocol": result.get("protocol"), "preference": result.get("preference"),
            "consensus": result.get("consensus"), "conflict": result.get("conflict"),
            "image_ids": ids, "api_call_count": len(calls),
            "api_latency_seconds": latency, "usage": usage}


def _revised(receipt: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    result = receipt.get("judge") or {}
    ids = result.get("image_ids", {})
    id_to_side = {ids.get("anchor"): "LEFT", ids.get("candidate"): "RIGHT"}
    views = result.get("views", [])
    choices = []
    for view in views:
        winner = (view.get("assessment") or {}).get("winner_id")
        choices.append("tie" if winner == "tie" else id_to_side.get(winner, "unknown"))
    if receipt.get("status") != "scored":
        return {"status": "censored", "headline": "Could not judge", "choices": choices, "reason": receipt.get("reason"),
                "technical": _call_technical(result, ids)}
    if result.get("status") == "uncertain":
        headline = "Could not judge — incomplete or conflicting evidence"
    elif result.get("winner_id") == "tie":
        headline = "No clear preference"
    else:
        winner_side = id_to_side.get(result.get("winner_id"), "unknown")
        headline = "Prefers " + winner_side if winner_side != "unknown" else "Could not judge — incomplete identity"
        if result.get("conflict") and winner_side != "unknown":
            headline = "Initial views disagreed; recheck prefers " + winner_side
    reason_view = views[0] if views else {}
    if result.get("conflict") and result.get("fresh_view"):
        # The recheck is the decisive evidence when it supplies the two-of-three
        # vote.  Keep the reason tied to a visible assessment and expose its
        # ID-to-side mapping in technical details instead of rewriting prose.
        fresh = result["fresh_view"]
        if (fresh.get("assessment") or {}).get("winner_id") == result.get("winner_id"):
            reason_view = fresh
    reason = ((reason_view.get("assessment") or {}).get("reason") if reason_view else None)
    return {"status": result.get("status"), "headline": headline, "choices": choices, "reason": reason,
            "technical": _call_technical(result, ids)}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"left": 0, "right": 0, "tie": 0, "uncertain": 0, "censored": 0,
              "initial_valid": 0, "initial_agreement": 0, "initial_conflict": 0,
              "recovered_conflict": 0, "remaining_conflict": 0,
              "api_call_count": 0, "api_latency_seconds": 0.0, "usage": {}}
    for row in rows:
        technical = row.get("technical") or {}
        choices = row.get("choices") or []
        if len(choices) == 2 and all(choice in {"LEFT", "RIGHT", "tie"} for choice in choices):
            counts["initial_valid"] += 1
            if choices[0] == choices[1]:
                counts["initial_agreement"] += 1
            else:
                counts["initial_conflict"] += 1
        counts["api_call_count"] += int(technical.get("api_call_count", 0) or 0)
        counts["api_latency_seconds"] += float(technical.get("api_latency_seconds", 0.0) or 0.0)
        for name, value in (technical.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counts["usage"][name] = counts["usage"].get(name, 0) + value
        headline = row.get("headline", "")
        if technical.get("conflict") and row.get("status") == "conflict_resolved":
            counts["recovered_conflict"] += 1
        if row.get("status") == "censored":
            counts["censored"] += 1
            if technical.get("conflict"):
                counts["remaining_conflict"] += 1
        elif headline.startswith("Uncertain") or headline.startswith("Could not judge"):
            counts["uncertain"] += 1
            if technical.get("conflict"):
                counts["remaining_conflict"] += 1
        elif headline == "No clear preference":
            counts["tie"] += 1
        elif headline.startswith("Initial views disagreed;"):
            if "LEFT" in headline:
                counts["left"] += 1
            elif "RIGHT" in headline:
                counts["right"] += 1
            else:
                counts["uncertain"] += 1
        elif headline == "Prefers LEFT":
            counts["left"] += 1
        elif headline == "Prefers RIGHT":
            counts["right"] += 1
        else:
            counts["uncertain"] += 1
    return counts


def summarize(*, manifest: dict[str, Any], pairs: list[dict[str, Any]], baseline: list[dict[str, Any]], revised: list[dict[str, Any]], gallery: Path) -> dict[str, Any]:
    baseline_by_id = {r["identity"]["pair_id"]: r for r in baseline}
    revised_by_id = {r["identity"]["pair_id"]: r for r in revised}
    rows = []
    for pair in pairs:
        rows.append({"pair_id": pair["pair_id"], "case_id": pair["case_id"], "family": pair["family"], "anchor_role": pair["anchor_role"], "candidate_role": pair["candidate_role"], "images": pair["images"], "brief": pair["brief"], "human_feedback": manifest["human_feedback"].get(pair["case_id"], {}), "previous": _previous(baseline_by_id[pair["pair_id"]], pair), "revised": _revised(revised_by_id[pair["pair_id"]], pair)})
    return {"schema": SUMMARY_SCHEMA, "protocol": identity_judge.PROTOCOL, "model": MODEL, "prompt_sha256": manifest["prompt_sha256"], "format_version": manifest["format_version"], "manifest_content_sha256": manifest["manifest_content_sha256"], "source_pass1_manifest_content_sha256": manifest["source_pass1_manifest_content_sha256"], "pair_count": len(rows), "previous_metrics": _metrics([row["previous"] for row in rows]), "revised_metrics": _metrics([row["revised"] for row in rows]), "pairs": rows, "scope_note": manifest["scope_note"]}


def _img(report: Path, gallery: Path, relative: str) -> str:
    import os
    return Path(os.path.relpath(gallery.parent / relative, report.parent)).as_posix()


def write_report(summary: dict[str, Any], *, gallery: Path, output: Path) -> Path:
    names = {"frozen SFT": "SFT reference", "RL-step-16": "Earlier RL", "pilot-step-24": "Latest RL", "SFT": "SFT comparison"}
    cards = []
    for index, pair in enumerate(summary["pairs"], 1):
        images = []
        for side, role in (("LEFT", pair["anchor_role"]), ("RIGHT", pair["candidate_role"])):
            images.append(f"<div class='image'><img src='{html.escape(_img(output / 'report.html', gallery, pair['images'][role]['path']))}' alt='{side}'><b>{side} · {html.escape(names.get(role, role))}</b></div>")
        details = f"<details><summary>Pair details</summary><p>Case ID: {html.escape(pair['case_id'])}<br>Family: {html.escape(pair['family'])}<br>LEFT role: {html.escape(pair['anchor_role'])}<br>RIGHT role: {html.escape(pair['candidate_role'])}</p></details>"
        feedback = pair.get("human_feedback") or {}
        feedback_html = f"<aside><b>Your feedback:</b> {html.escape(feedback['note'])}</aside>" if feedback.get("note") else ""
        method_html = []
        for label, row in (("Previous judge", pair["previous"]), ("Revised judge", pair["revised"])):
            mapping = "<p>First comparison: A = LEFT, B = RIGHT. Swapped comparison: A = RIGHT, B = LEFT.</p>" if label == "Previous judge" else ""
            reason = f"<details><summary>Why</summary>{mapping}<p>{html.escape(row['reason'])}</p></details>" if row.get("reason") else ""
            technical = f"<details><summary>Technical details</summary><p>{html.escape(str(row.get('technical')))}</p><p>Order choices: {html.escape(str(row.get('choices')))}</p></details>"
            method_html.append(f"<div class='method'><h3>{label}</h3><p class='verdict {'bad' if row.get('status') == 'censored' or row.get('headline','').startswith('Uncertain') else ''}'>{html.escape(row.get('headline', 'Could not judge'))}</p>{reason}{technical}</div>")
        cards.append(f"<section><h2>Pair {index}</h2><details class='brief'><summary>Painting brief</summary><p>{html.escape(pair['brief'])}</p></details>{feedback_html}<div class='images'>{''.join(images)}</div><div class='methods'>{''.join(method_html)}</div>{details}</section>")
    previous, revised = summary["previous_metrics"], summary["revised_metrics"]
    report = """<!doctype html><html><head><meta charset='utf-8'><title>Identity evidence calibration</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui,sans-serif;padding:28px}main{max-width:1240px;margin:auto}h1{color:#f9fafb}h2{margin:0 0 6px}.meta,section{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:18px;margin:18px 0}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:14px}.image{background:#111827;border-radius:10px;padding:10px}.image img{display:block;width:100%;max-width:600px;aspect-ratio:1/1;object-fit:cover;border-radius:7px;margin-bottom:8px;cursor:zoom-in}.image img:active{transform:scale(1.25);position:relative;z-index:2}.methods{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:16px}.method{border-top:1px solid #4b5563;padding:12px 0}.method h3{color:#bfdbfe}.verdict{font-size:1.1em;font-weight:650}.bad{color:#fca5a5}.brief,.meta p{color:#cbd5e1;line-height:1.45}details{margin-top:9px;color:#cbd5e1;line-height:1.45}summary{cursor:pointer;color:#bfdbfe}aside{border-left:3px solid #fbbf24;padding:8px 12px;color:#fde68a}@media(max-width:760px){body{padding:14px}.images,.methods{grid-template-columns:1fr}}</style></head><body><main>"""
    report += f"<h1>Identity evidence calibration</h1><div class='meta'><p>Both methods judge the same two paintings together. The Revised judge uses stable image identities, evidence-first assessments, and a bounded conflict check; the Previous judge is the saved pass-1 result.</p><p>LEFT/RIGHT are the displayed pair. An Uncertain result means the judge choices were inconsistent or had no majority.</p><details><summary>Batch details</summary><p>Pairs: {summary['pair_count']}<br>Previous: {previous}<br>Revised: {revised}</p></details></div>" + "".join(cards) + "</main></body></html>\n"
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


async def run(*, pass1_output: Path, output: Path, gallery: Path, key_file: Path, concurrency: int = 6) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    manifest = _read_manifest(output)
    source = _source(pass1_output)
    if source["manifest_content_sha256"] != manifest["source_pass1_manifest_content_sha256"]:
        raise ValueError("pass-1 manifest changed since preparation")
    if ja.sha256_file(gallery) != manifest["gallery_html_sha256"]:
        raise ValueError("gallery changed since preparation")
    baseline = _baseline(pass1_output, source)
    sem = asyncio.Semaphore(concurrency)
    revised = await asyncio.gather(*(_one(manifest=manifest, pair=pair, gallery=gallery, output=output, key_file=key_file, sem=sem, index=index, total=len(manifest["pairs"])) for index, pair in enumerate(manifest["pairs"], 1)))
    summary = summarize(manifest=manifest, pairs=manifest["pairs"], baseline=baseline, revised=revised, gallery=gallery)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, gallery=gallery, output=output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    if args.prepare_only:
        manifest = prepare_manifest(pass1_output=args.pass1_output, output=args.output)
        print(json.dumps({"manifest": str(args.output / "input-manifest.json"), "pairs": len(manifest["pairs"])}, sort_keys=True))
        return 0
    result = asyncio.run(run(pass1_output=args.pass1_output, output=args.output, gallery=args.gallery, key_file=args.key_file, concurrency=args.concurrency))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
