"""Run the separate-image presentation ablation against pass-1's 36 pairs.

The pass-1 receipts are a required frozen baseline.  This runner makes no old
judge calls and refuses to start the treatment until every pass-1 pair receipt
is present and bound to the unchanged pass-1 manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_grounded_v2_split_judge as split_judge
    from painter.text_pairwise_margin_judge import MODEL
except ImportError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_grounded_v2_split_judge as split_judge
    from painter.text_pairwise_margin_judge import MODEL


TARGET_SCHEMA = "judge-alignment-split-input-v1"
RECEIPT_SCHEMA = "judge-alignment-split-receipt-v1"
SUMMARY_SCHEMA = "judge-alignment-split-summary-v1"
FORMAT_VERSION = split_judge.FORMAT_VERSION
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_pass1_module():
    path = Path(__file__).with_name("judge_alignment.py")
    spec = importlib.util.spec_from_file_location("judge_alignment_pass1_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ja = _load_pass1_module()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_content_sha256", None)
    return ja.sha256_text(json.dumps(copy, sort_keys=True, separators=(",", ":")))


def _safe_id(value: str) -> None:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe split receipt identifier: {value!r}")


def _pass1_manifest(pass1_output: Path) -> dict[str, Any]:
    manifest = ja._read_manifest(pass1_output)
    if len(manifest.get("pairs", [])) != 36:
        raise ValueError("split ablation requires exactly all 36 pass-1 pairs")
    return manifest


def _baseline_receipts(pass1_output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in manifest["pairs"]:
        _safe_id(pair["pair_id"])
        path = pass1_output / "receipts" / f"{pair['pair_id']}--repeat-1.json"
        if not path.is_file() or path.is_symlink():
            missing.append(pair["pair_id"])
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        identity = receipt.get("identity", {})
        if receipt.get("schema") != ja.RECEIPT_SCHEMA:
            raise ValueError(f"unexpected pass-1 receipt schema: {pair['pair_id']}")
        if identity != ja._pair_identity(manifest, pair, 1):
            raise ValueError(f"pass-1 receipt identity mismatch: {pair['pair_id']}")
        new_result = (receipt.get("versions") or {}).get("new") or {}
        if new_result.get("status") == "scored":
            result = new_result.get("judge") or {}
            if result.get("protocol") != manifest["protocol"]:
                raise ValueError(f"pass-1 v2 protocol mismatch: {pair['pair_id']}")
        receipts.append(receipt)
    if missing:
        raise FileNotFoundError("missing pass-1 receipts: " + ", ".join(missing))
    if len(receipts) != 36:
        raise ValueError("pass-1 baseline is incomplete")
    return receipts


def prepare_manifest(*, pass1_output: Path, output: Path) -> dict[str, Any]:
    source = _pass1_manifest(pass1_output)
    manifest: dict[str, Any] = {
        "schema": TARGET_SCHEMA,
        "source_pass1_manifest_content_sha256": source["manifest_content_sha256"],
        "source_pass1_protocol": source["protocol"],
        "protocol": split_judge.PROTOCOL,
        "model": MODEL,
        "prompt_sha256": ja.sha256_text(split_judge.PROMPT),
        "format_version": FORMAT_VERSION,
        "gallery_html_sha256": source["gallery_html_sha256"],
        "gallery_path": source.get("gallery_path"),
        "cases": source["cases"],
        "pairs": source["pairs"],
        "pair_plan": source["pair_plan"],
        "human_feedback": source["human_feedback"],
        "historical_score_note": source["historical_score_note"],
        "presentation_note": "API-presentation ablation: separate original PNG inputs versus pass-1 v2 contact sheet; no prompt change.",
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _read_manifest(output: Path) -> dict[str, Any]:
    path = output / "input-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TARGET_SCHEMA:
        raise ValueError("unsupported split ablation manifest")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("split ablation manifest content hash mismatch")
    if len(manifest.get("pairs", [])) != 36:
        raise ValueError("split ablation requires exactly 36 pairs")
    if manifest.get("protocol") != split_judge.PROTOCOL:
        raise ValueError("split ablation protocol mismatch")
    if manifest.get("model") != MODEL:
        raise ValueError("split ablation model mismatch")
    if manifest.get("prompt_sha256") != ja.sha256_text(split_judge.PROMPT):
        raise ValueError("split ablation prompt mismatch")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("split ablation format mismatch")
    return manifest


def _identity(manifest: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "source_pass1_manifest_content_sha256": manifest["source_pass1_manifest_content_sha256"],
        "protocol": manifest["protocol"],
        "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"],
        "format_version": manifest["format_version"],
        "pair_id": pair["pair_id"],
        "repeat": 1,
        "brief_sha256": pair["brief_sha256"],
        "image_sha256": {
            pair["anchor_role"]: pair["images"][pair["anchor_role"]]["sha256"],
            pair["candidate_role"]: pair["images"][pair["candidate_role"]]["sha256"],
        },
    }


def _verify_pair(pair: dict[str, Any], gallery: Path) -> None:
    ja._verify_pair_files(pair, gallery)


def _safe_failure(exc: Exception) -> str:
    name = type(exc).__name__
    if not SAFE_NAME.fullmatch(name):
        name = "ProviderError"
    return "judge_failure_" + name


async def _one_pair(*, manifest: dict[str, Any], pair: dict[str, Any], gallery: Path,
                    output: Path, key_file: Path, semaphore: asyncio.Semaphore,
                    index: int, total: int) -> dict[str, Any]:
    identity = _identity(manifest, pair)
    _verify_pair(pair, gallery)
    brief = ja._materialize_brief(pair, output)
    receipt_path = output / "receipts" / f"{pair['pair_id']}--repeat-1.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.is_file():
        if receipt_path.is_symlink():
            raise ValueError(f"receipt path is symlink: {receipt_path}")
        cached = json.loads(receipt_path.read_text(encoding="utf-8"))
        if cached.get("schema") != RECEIPT_SCHEMA or cached.get("identity") != identity:
            raise ValueError(f"stale split receipt: {pair['pair_id']}")
        print(f"[{index}/{total}] cached {pair['pair_id']}", flush=True)
        return cached
    anchor = gallery.parent / pair["images"][pair["anchor_role"]]["path"]
    candidate = gallery.parent / pair["images"][pair["candidate_role"]]["path"]
    async with semaphore:
        started = time.monotonic()
        try:
            result = await split_judge.judge_pair(
                brief=brief, anchor=anchor, candidate=candidate, key_file=key_file
            )
            if result.get("protocol") != split_judge.PROTOCOL or result.get("format_version") != FORMAT_VERSION:
                raise ValueError("split judge protocol or format mismatch")
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "identity": identity,
                "status": "scored",
                "elapsed_seconds_runner": time.monotonic() - started,
                "judge": result,
            }
        except Exception as exc:
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "identity": identity,
                "status": "censored",
                "reason": _safe_failure(exc),
                "elapsed_seconds_runner": time.monotonic() - started,
            }
    partial = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(receipt_path)
    print(f"[{index}/{total}] {receipt['status']} {pair['pair_id']}", flush=True)
    return receipt


def _split_flat(receipt: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != "scored":
        return {"status": receipt.get("status"), "reason": receipt.get("reason")}
    return ja._flatten_version({"status": "scored", "judge": receipt.get("judge")}, pair)


def _side_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    gaps = [float(row["order_gap"]) for row in scored if row.get("order_gap") is not None]
    latencies = [float(view["elapsed_seconds"]) for row in scored for view in row.get("views", []) if view.get("elapsed_seconds") is not None]
    label_rows = [row for row in scored if len(row.get("normalized_label_votes", [])) == 2]
    tokens: dict[str, int] = {}
    costs: list[float] = []
    for row in scored:
        for view in row.get("views", []):
            usage = view.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    tokens[key] = tokens.get(key, 0) + int(usage[key])
            observed = ja._reported_cost(usage)
            if observed is not None:
                costs.append(observed[1])
    return {
        "status_counts": {"scored": len(scored), "censored": len(rows) - len(scored)},
        "order_disagreement": {
            "count": sum(gap + 1e-9 >= 1 / 3 for gap in gaps),
            "denominator": len(gaps),
            "mean_order_gap": statistics.fmean(gaps) if gaps else None,
        },
        "two_order_label_consensus": {
            "consensus_count": sum(bool(row.get("two_order_label_consensus")) for row in label_rows),
            "disagreement_count": sum(bool(row.get("two_order_label_disagreement")) for row in label_rows),
            "denominator": len(label_rows),
        },
        "api_latency_seconds": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "provider_usage": tokens,
        "provider_reported_cost": sum(costs) if costs else None,
        "provider_reported_cost_observations": len(costs),
    }


def summarize(*, manifest: dict[str, Any], pairs: list[dict[str, Any]],
              baseline: list[dict[str, Any]], split: list[dict[str, Any]],
              gallery: Path) -> dict[str, Any]:
    baseline_summary = ja.summarize(
        manifest={**manifest, "schema": ja.SCHEMA, "protocol": manifest["source_pass1_protocol"],
                  "old_protocol": manifest["source_pass1_protocol"], "prompt_sha256": ja.sha256_text(ja.judge.PROMPT),
                  "old_prompt_sha256": ja.sha256_text(ja.old_judge.PROMPT),
                  "manifest_content_sha256": manifest["source_pass1_manifest_content_sha256"]},
        pairs=pairs,
        receipts=baseline,
        gallery=gallery,
        output=Path("."),
        repeats=1,
    )
    baseline_by_id = {row["pair_id"]: row for row in baseline_summary["pairs"]}
    split_by_id = {receipt["identity"]["pair_id"]: receipt for receipt in split}
    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        baseline_row = baseline_by_id[pair["pair_id"]]
        baseline_v2 = baseline_row["repeats"][0]["new"]
        split_v2 = _split_flat(split_by_id[pair["pair_id"]], pair)
        pair_rows.append({
            "pair_id": pair["pair_id"], "case_id": pair["case_id"],
            "family": pair["family"], "group": pair["group"], "brief": pair["brief"],
            "anchor_role": pair["anchor_role"], "candidate_role": pair["candidate_role"],
            "images": pair["images"], "human_feedback": manifest["human_feedback"].get(pair["case_id"], {}),
            "baseline_same_v2": baseline_v2, "split_presentation_v2": split_v2,
        })
    baseline_rows = [row["baseline_same_v2"] for row in pair_rows]
    split_rows = [row["split_presentation_v2"] for row in pair_rows]
    return {
        "schema": SUMMARY_SCHEMA,
        "comparison": "API-presentation ablation: SAMEv2 contact sheet baseline versus separate original PNG inputs",
        "protocol": split_judge.PROTOCOL,
        "model": MODEL,
        "prompt_sha256": manifest["prompt_sha256"],
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "source_pass1_manifest_content_sha256": manifest["source_pass1_manifest_content_sha256"],
        "gallery_html_sha256": manifest["gallery_html_sha256"],
        "pair_count": len(pairs),
        "baseline": _side_metrics(baseline_rows),
        "split_presentation": _side_metrics(split_rows),
        "pairs": pair_rows,
        "no_fresh_old_calls": True,
        "no_training_or_reward_claim": True,
    }


def _img_src(report: Path, gallery: Path, relative: str) -> str:
    import os
    return Path(os.path.relpath(gallery.parent / relative, report.parent)).as_posix()


def write_report(summary: dict[str, Any], *, gallery: Path, output: Path) -> Path:
    role_names = {
        "frozen SFT": "SFT reference",
        "RL-step-16": "Earlier RL",
        "pilot-step-24": "Latest RL",
        "SFT": "SFT comparison",
    }

    def side_for_role(pair: dict[str, Any], role: Any) -> str:
        if role == pair["anchor_role"]:
            return "LEFT"
        if role == pair["candidate_role"]:
            return "RIGHT"
        return "tie" if role == "tie" else "unknown"

    def order_choice(pair: dict[str, Any], view: dict[str, Any], index: int) -> str:
        winner = view.get("winner")
        if winner == "tie":
            return "tie"
        if winner not in {"A", "B"}:
            return "unknown"
        # The first request displays anchor/candidate as A/B; the second
        # reverses those images.  State the mapping instead of rewriting text.
        role = pair["anchor_role"] if (winner == "A") == (index == 0) else pair["candidate_role"]
        return side_for_role(pair, role)

    def headline(pair: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[str]]:
        if row.get("status") != "scored":
            return "Could not judge", []
        choices = [order_choice(pair, view, index) for index, view in enumerate(row.get("views", []))]
        if len(choices) != 2 or any(choice == "unknown" for choice in choices):
            return "Uncertain — incomplete order evidence", choices
        if choices[0] != choices[1]:
            return "Uncertain — changed choice when paintings swapped", choices
        return ("No clear preference" if choices[0] == "tie" else f"Prefers {choices[0]}"), choices

    def method_block(pair: dict[str, Any], label: str, row: dict[str, Any]) -> str:
        title, choices = headline(pair, row)
        if row.get("status") != "scored":
            return f"<div class='method'><h3>{html.escape(label)}</h3><p class='verdict bad'>{html.escape(title)}</p><p class='muted'>The provider result was censored; this is not a tie.</p></div>"
        views = row.get("views", [])
        choice_text = " · ".join(
            f"{'First comparison' if index == 0 else 'Swapped comparison'}: {html.escape(choice)}"
            for index, choice in enumerate(choices)
        )
        rationale = []
        mappings = (("First comparison (LEFT=A, RIGHT=B)", views[0] if len(views) > 0 else None),
                    ("Swapped comparison (LEFT=B, RIGHT=A)", views[1] if len(views) > 1 else None))
        for view_index, (mapping, view) in enumerate(mappings):
            if not view:
                continue
            details = []
            if view.get("reason"):
                details.append(f"<p>{html.escape(view['reason'])}</p>")
            if view.get("visible_evidence_A"):
                a_side = "LEFT" if view_index == 0 else "RIGHT"
                details.append(f"<p><b>{a_side} evidence:</b> {html.escape(view['visible_evidence_A'])}</p>")
            if view.get("visible_evidence_B"):
                b_side = "RIGHT" if view_index == 0 else "LEFT"
                details.append(f"<p><b>{b_side} evidence:</b> {html.escape(view['visible_evidence_B'])}</p>")
            if details:
                rationale.append(f"<details><summary>{html.escape(mapping)}</summary>{''.join(details)}</details>")
        cap_note = ""
        if (row.get("candidate_margin_cap") is not None
                and row.get("raw_pairwise_score") is not None
                and row.get("candidate_margin_cap") < row.get("raw_pairwise_score")
                and row.get("score") != row.get("raw_pairwise_score")):
            cap_note = "<p class='notice'>Reward was limited because the subject was unclear.</p>"
        technical = (
            f"score={row.get('score')!r}; raw score={row.get('raw_pairwise_score')!r}; "
            f"cap={row.get('candidate_margin_cap')!r}; order gap={row.get('order_gap')!r}; "
            f"votes={row.get('normalized_label_votes')!r}; protocol={row.get('judge_protocol')!r}"
        )
        prominent_choices = f"<p class='choices'>{choice_text}</p>" if title.startswith("Uncertain") else ""
        return f"<div class='method'><h3>{html.escape(label)}</h3><p class='verdict'>{html.escape(title)}</p>{cap_note}{prominent_choices}{''.join(rationale)}<details class='technical'><summary>Technical details</summary><p>{html.escape(technical)}</p></details></div>"

    blocks: list[str] = []
    for pair_index, pair in enumerate(summary["pairs"], 1):
        feedback = pair.get("human_feedback") or {}
        note = f"<aside><b>Your feedback:</b> {html.escape(feedback['note'])}</aside>" if feedback.get("note") else ""
        images = []
        for side, role in (("LEFT", pair["anchor_role"]), ("RIGHT", pair["candidate_role"])):
            row = pair["images"][role]
            label = role_names.get(role, role)
            images.append(f"<div class='image'><img src='{html.escape(_img_src(output / 'report.html', gallery, row['path']))}' alt='{html.escape(side + ' ' + label)}'><div><b>{side} · {html.escape(label)}</b></div></div>")
        brief = f"<details class='brief'><summary>Painting brief</summary><p>{html.escape(pair['brief'])}</p></details>"
        pair_details = f"<details class='pair-details'><summary>Pair details</summary><p>Case ID: {html.escape(pair['case_id'])}<br>Family: {html.escape(pair['family'])}<br>Group: {html.escape(pair['group'])}<br>LEFT source role: {html.escape(pair['anchor_role'])}<br>RIGHT source role: {html.escape(pair['candidate_role'])}</p></details>"
        methods = method_block(pair, "Shown together", pair["baseline_same_v2"]) + method_block(pair, "Shown separately", pair["split_presentation_v2"])
        blocks.append(f"<section><h2>Pair {pair_index}</h2>{brief}{note}<div class='images'>{''.join(images)}</div><div class='methods'>{methods}</div>{pair_details}</section>")
    together = summary["baseline"]
    separately = summary["split_presentation"]
    report = """<!doctype html><html><head><meta charset='utf-8'><title>Same judge, two ways of showing the paintings</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui,sans-serif;padding:28px}main{max-width:1240px;margin:auto}h1{color:#f9fafb;margin:0 0 8px}h2{margin:0 0 6px}h2 small{font-size:.65em;color:#93c5fd}.meta,section{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:18px;margin:18px 0}.meta p,.muted{color:#cbd5e1;line-height:1.45}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:14px}.image{background:#111827;border-radius:10px;padding:10px}.image img{display:block;width:100%;max-width:600px;aspect-ratio:1/1;object-fit:cover;border-radius:7px;margin-bottom:8px;cursor:zoom-in}.image img:active{transform:scale(1.25);position:relative;z-index:2}.methods{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:16px}.method{border-top:1px solid #4b5563;padding:12px 0}.method h3{margin:0 0 5px;color:#bfdbfe}.verdict{font-size:1.1em;font-weight:650;margin:6px 0}.verdict.bad,.bad{color:#fca5a5}.choices{color:#fcd34d;font-weight:600}.notice{color:#fde68a;background:#292014;border-left:3px solid #fbbf24;padding:8px 10px}.brief{color:#cbd5e1;line-height:1.45}.brief summary,summary{cursor:pointer;color:#bfdbfe}.method details{margin-top:9px;color:#cbd5e1;line-height:1.45}.technical{font-size:.88em;color:#9ca3af}aside{border-left:3px solid #fbbf24;padding:8px 12px;color:#fde68a;margin-top:12px}@media(max-width:760px){body{padding:14px}.images,.methods{grid-template-columns:1fr}}</style></head><body><main>"""
    report += f"<h1>Same judge, two ways of showing the paintings</h1><div class='meta'><p>Same judge and same instructions. Together = both paintings in one image; separately = two individual images. Each comparison repeated with images swapped. Uncertain means inconsistent judge choices, not bad paintings.</p><p><b>Legend:</b> LEFT and RIGHT are the two displayed paintings. Friendly labels: SFT reference, Earlier RL, Latest RL, and SFT comparison.</p><details class='run-details'><summary>Run details</summary><p>{summary['pair_count']} matched pairs; these counts describe this batch, not judge accuracy. Shown together scored {together['status_counts']['scored']}/{summary['pair_count']}; shown separately scored {separately['status_counts']['scored']}/{summary['pair_count']}.</p><p>Source checkpoint IDs, family/group assignments, raw roles, and technical judge values are available within each pair's collapsed details.</p></details></div>" + "".join(blocks) + "</main></body></html>\n"
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


async def run(*, pass1_output: Path, output: Path, gallery: Path, key_file: Path,
              concurrency: int = 6) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    manifest = _read_manifest(output)
    source = _pass1_manifest(pass1_output)
    if source["manifest_content_sha256"] != manifest["source_pass1_manifest_content_sha256"]:
        raise ValueError("pass-1 manifest changed since split preparation")
    if ja.sha256_file(gallery) != manifest["gallery_html_sha256"]:
        raise ValueError("gallery changed since split preparation")
    baseline = _baseline_receipts(pass1_output, source)
    pairs = manifest["pairs"]
    sem = asyncio.Semaphore(concurrency)
    split = await asyncio.gather(*(
        _one_pair(manifest=manifest, pair=pair, gallery=gallery, output=output,
                  key_file=key_file, semaphore=sem, index=index, total=len(pairs))
        for index, pair in enumerate(pairs, 1)
    ))
    summary = summarize(manifest=manifest, pairs=pairs, baseline=baseline,
                        split=split, gallery=gallery)
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
        print(json.dumps({"manifest": str(args.output / "input-manifest.json"), "pairs": len(manifest["pairs"]), "format_version": FORMAT_VERSION}, sort_keys=True))
        return 0
    result = asyncio.run(run(pass1_output=args.pass1_output, output=args.output,
                             gallery=args.gallery, key_file=args.key_file,
                             concurrency=args.concurrency))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
