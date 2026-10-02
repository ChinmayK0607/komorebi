"""Build a static, fixed-side report for the three-model frozen-image panel."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("threeway_runner_for_report", HERE / "judge_threeway_ablation.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _relative_image(path: Path, report_dir: Path) -> str:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return _esc(os.path.relpath(resolved, report_dir.resolve()))


def _load_calls(manifest: dict[str, Any], receipt_dir: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for pair, model, round_index in runner._work_items(manifest):
        for order in ("canonical", "reverse"):
            path = runner._call_paths(receipt_dir, model, pair["pair_id"], round_index, order)
            identity = runner._identity(manifest, pair, model, round_index, order)
            if path.is_file():
                value = runner._load_existing(path, identity, runner.CALL_SCHEMA)
                if value is not None:
                    result[(pair["pair_id"], model, round_index, order)] = value
    return result


def _load_rounds(manifest: dict[str, Any], receipt_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for pair, model, round_index in runner._work_items(manifest):
        path = runner._round_path(receipt_dir, model, pair["pair_id"], round_index)
        identity = runner._round_identity(manifest, pair, model, round_index)
        if path.is_file():
            value = runner._load_existing(path, identity, runner.ROUND_SCHEMA)
            if value is not None:
                result[(pair["pair_id"], model, round_index)] = value
    return result


def _token_text(call: dict[str, Any]) -> str:
    usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
    return "prompt={} completion={} reasoning={} cost={}".format(
        usage.get("prompt_tokens", "missing"), usage.get("completion_tokens", "missing"),
        usage.get("reasoning_tokens", "missing"), usage.get("cost", "missing"),
    )


def _call_card(row: dict[str, Any] | None, order: str) -> str:
    if row is None:
        return f"<div class='call missing'><strong>{_esc(order)}</strong>: missing receipt</div>"
    call = row.get("call") if isinstance(row.get("call"), dict) else {}
    status = row.get("status", "unknown")
    identity = row.get("identity", {})
    reason = call.get("parse_error") or call.get("provider_error") or row.get("reason") or ""
    label = " · ".join([
        f"status={_esc(status)}",
        f"requested={_esc(call.get('requested_model', identity.get('model')))}",
        f"returned={_esc(call.get('returned_model', 'missing'))}",
        f"finish={_esc(call.get('finish_reason', 'missing'))}",
        f"elapsed={_esc(call.get('elapsed_seconds', 'missing'))}s",
        _esc(_token_text(call)),
    ] + ([f"reason={_esc(reason)}"] if reason else []))
    body = f"<p>{label}</p>"
    assessment = call.get("assessment")
    if isinstance(assessment, dict):
        winner = assessment.get("winner_id")
        image_ids = call.get("image_ids") or {}
        winner_text = "TIE" if winner == "tie" else (
            "LEFT" if winner == image_ids.get("LEFT") else
            "RIGHT" if winner == image_ids.get("RIGHT") else "unknown"
        )
        body += f"<p><strong>Winner {winner_text}</strong> · preference {_esc(assessment.get('preference'))} · {_esc(assessment.get('reason'))}</p>"
        for side_name, image_key in (("LEFT", "LEFT"), ("RIGHT", "RIGHT")):
            panel = (assessment.get("assessments") or {}).get(image_ids.get(image_key))
            if not isinstance(panel, dict):
                continue
            weaknesses = "".join(f"<li>{_esc(item)}</li>" for item in panel.get("weaknesses", []))
            body += (
                f"<div class='assessment'><strong>{side_name}</strong> · "
                f"{_esc(panel.get('brief_fidelity'))}: {_esc(panel.get('visible_evidence'))}"
                f"<ul>{weaknesses}</ul></div>"
            )
    return f"<div class='call {status}'><strong>{_esc(order)}</strong>{body}</div>"


def _model_card(*, manifest: dict[str, Any], pair: dict[str, Any], model: str,
                calls: dict[tuple[str, str, int, str], dict[str, Any]],
                rounds: dict[tuple[str, str, int], dict[str, Any]]) -> str:
    repeat_ids = set(manifest["fixed_repeat_pair_ids"])
    round_count = 1 + (manifest["fixed_repeat_rounds"] if pair["pair_id"] in repeat_ids else 0)
    sections: list[str] = []
    for round_index in range(round_count):
        round_row = rounds.get((pair["pair_id"], model, round_index))
        status = round_row.get("status", "missing") if round_row else "missing"
        reason = round_row.get("reason") if round_row else "missing receipt"
        sections.append(
            "<details {}><summary>Round {} · {}{}</summary>{}</details>".format(
                "open" if round_index == 0 else "",
                "primary" if round_index == 0 else f"fixed repeat {round_index}",
                _esc(status), f" · {_esc(reason)}" if reason else "",
                _call_card(calls.get((pair["pair_id"], model, round_index, "canonical")), "canonical LEFT/RIGHT")
                + _call_card(calls.get((pair["pair_id"], model, round_index, "reverse")), "reverse RIGHT/LEFT"),
            )
        )
    return f"<article class='judge-card'><h3>{_esc(model)}</h3>{''.join(sections)}</article>"


def _summary_html(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "<p class='pending'>No complete call set is present. This is a preparation panel; no labels or accuracy are shown.</p>"
    rows: list[str] = []
    for model, metrics in summary["by_model"].items():
        order = metrics["order_consistency"]
        cc = metrics["tokens"]["completion"]["per_call"]
        cp = metrics["tokens"]["completion"]["per_pair"]
        rc = metrics["tokens"]["reasoning"]["per_call"]
        rp = metrics["tokens"]["reasoning"]["per_pair"]
        lc = metrics["latency_seconds"]["per_call"]
        lps = metrics["latency_seconds"]["per_pair_service_seconds"]
        lpw = metrics["latency_seconds"]["per_pair_wall_clock_seconds"]
        rows.append(
            "<tr><th>{}</th><td>{}/{}</td><td>{}/{}/{}</td><td>{}/{}/{}</td>"
            "<td>{}/{}/{}</td><td>{}/{}/{}</td><td>{}/{}/{}</td><td>{}/{}</td><td>{}</td></tr>".format(
                _esc(model), metrics["valid_call_count"], metrics["call_count"],
                _esc(lc["mean"]), _esc(lc["median"]), _esc(lc["p90"]),
                _esc(lps["mean"]), _esc(lps["median"]), _esc(lps["p90"]),
                _esc(lpw["mean"]), _esc(lpw["median"]), _esc(lpw["p90"]),
                _esc(cc["mean"]), _esc(cc["median"]), _esc(cc["p90"]),
                _esc(cp["mean"]), _esc(cp["median"]), _esc(cp["p90"]),
                _esc(rc["mean"]), _esc(rc["median"]), _esc(rc["p90"]),
                _esc(rp["mean"]), _esc(rp["median"]), _esc(rp["p90"]),
                order["agreeing_rounds"], order["rounds_with_two_valid_orders"],
                _esc(metrics["reported_cost"]["sum"]),
            )
        )
    cross = summary.get("cross_model_agreement", {})
    cross_html = (
        "<p>Cross-model agreement on complete pair-rounds: "
        f"{_esc(cross.get('all_models_agree', 0))}/{_esc(cross.get('complete_pair_rounds', 0))} "
        f"({_esc(cross.get('all_models_agree_rate'))}); disagreements {_esc(cross.get('model_disagreement', 0))}."
        " This is a descriptive selected-panel statistic.</p>"
    )
    return (
        "<table><thead><tr><th>Model</th><th>valid calls/total</th>"
        "<th>latency s/call mean med p90</th><th>service s/pair mean med p90</th>"
        "<th>wall s/pair mean med p90</th>"
        "<th>completion/call mean med p90</th><th>completion/pair mean med p90</th>"
        "<th>reasoning/call mean med p90</th><th>reasoning/pair mean med p90</th>"
        "<th>order agree/valid</th><th>reported cost</th></tr></thead>"
        + "".join(rows)
        + "</table>" + cross_html
        + "<p>Service seconds sum the two order-call durations; wall seconds use the maximum order duration per parallel round. Missing usage fields remain missing and are excluded from observed-token statistics; partial pairs are excluded from complete per-pair token summaries.</p>"
    )


def build_report(*, manifest_path: Path, output: Path, receipt_dir: Path | None = None,
                 panel_only: bool = False) -> Path:
    manifest = runner.load_manifest(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output.parent
    receipt_root = receipt_dir or report_dir.parent
    calls = {} if panel_only else _load_calls(manifest, receipt_root)
    rounds = {} if panel_only else _load_rounds(manifest, receipt_root)
    expected_calls = sum(
        2 * (1 + (manifest["fixed_repeat_rounds"] if pair["pair_id"] in set(manifest["fixed_repeat_pair_ids"]) else 0))
        for pair in manifest["pairs"]
    ) * len(manifest["models"])
    summary = None
    if not panel_only and len(calls) == expected_calls:
        rows = list(calls.values())
        for row in rows:
            row["_pair"] = next(pair for pair in manifest["pairs"] if pair["pair_id"] == row["identity"]["pair_id"])
        summary = runner.summarize(manifest=manifest, rows=rows)
    cards: list[str] = []
    for pair in manifest["pairs"]:
        left, right = pair["left"], pair["right"]
        human = pair.get("historical_human_preference")
        human_html = (
            f"<p class='historical'>Historical user preference: {_esc(human)}. "
            "This selected-panel context is not a new gold label or accuracy estimate.</p>"
            if human else ""
        )
        model_cards = "" if panel_only else "".join(
            _model_card(manifest=manifest, pair=pair, model=model, calls=calls, rounds=rounds)
            for model in manifest["models"]
        )
        cards.append(
            f"<section class='pair'><h2>{_esc(pair['pair_id'])} · {_esc(pair.get('case_id'))}</h2>"
            f"<p class='meta'>{_esc(pair.get('family'))} · {_esc(pair.get('selection_note'))}</p>"
            f"<p class='brief'>{_esc(pair['brief'])}</p>{human_html}<div class='images'>"
            f"<figure><img src='{_relative_image(Path(left['path']), report_dir)}' alt='LEFT {_esc(pair['case_id'])}'><figcaption><strong>LEFT</strong> · {_esc(left['label'])}<br><code>{_esc(left['image_id'])}</code><br>{_esc(left['sha256'])}</figcaption></figure>"
            f"<figure><img src='{_relative_image(Path(right['path']), report_dir)}' alt='RIGHT {_esc(pair['case_id'])}'><figcaption><strong>RIGHT</strong> · {_esc(right['label'])}<br><code>{_esc(right['image_id'])}</code><br>{_esc(right['sha256'])}</figcaption></figure>"
            f"</div><div class='judges'>{model_cards}</div></section>"
        )
    document = """<!doctype html>
<html><head><meta charset='utf-8'><title>Three-judge frozen-image ablation</title>
<style>
body{font:14px system-ui,sans-serif;color:#202124;background:#f6f7f9;margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{margin:0 0 8px}.meta,.pending{color:#59636e}.brief{font-size:15px;line-height:1.45}.pair{background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:16px;margin:18px 0;box-shadow:0 1px 3px #00000012}.images{display:grid;grid-template-columns:1fr 1fr;gap:18px}.images figure{margin:0}.images img{width:100%;height:340px;object-fit:contain;background:#f1f3f5;border:1px solid #c8ced6}.images figcaption{font-size:12px;color:#4d5660;padding-top:6px;word-break:break-word}.images code{font-size:11px}.judges{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:15px}.judge-card{border:1px solid #d8dde3;border-radius:8px;padding:10px;background:#fbfcfd}.judge-card h3{font-size:15px;margin:0 0 8px;word-break:break-all}details{border-top:1px solid #e3e6ea;padding:6px 0}summary{cursor:pointer}.call{border-left:3px solid #bbc3cc;background:#f5f7f9;padding:6px;margin:7px 0;font-size:12px}.call.valid{border-left-color:#2d8a57}.call.censored{border-left-color:#c47c20}.call.missing{border-left-color:#a23939}.call p{margin:3px 0;line-height:1.35}.assessment{margin:5px 0;padding:5px;background:#fff;border:1px solid #e0e4e8}.assessment ul{margin:3px 0 3px 18px;padding:0}.historical{background:#fff5d6;border-left:3px solid #d49b22;padding:6px}.summary{background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:14px;margin-bottom:20px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d8dde3;padding:6px;text-align:left;white-space:nowrap}th{background:#eef1f4}@media(max-width:950px){.judges{grid-template-columns:1fr}.images{grid-template-columns:1fr}.images img{height:300px}}
</style></head><body><main>
<h1>Three-judge frozen-image ablation</h1>
<p class='meta'>Fixed LEFT/RIGHT display. Every judge request uses the same joint sheet in canonical and reversed order. The report never swaps images or invents a tie from censored evidence.</p>
<p class='meta'>Prompt SHA256: <code>__PROMPT__</code> · manifest SHA256: <code>__MANIFEST__</code> · requested reasoning: <code>__REASONING__</code></p>
<p class='meta'>__CAVEAT__</p>
<div class='summary'><h2>Summary</h2>__SUMMARY__</div>
__CARDS__
</main></body></html>
"""
    document = document.replace("__PROMPT__", _esc(manifest["prompt_sha256"])).replace(
        "__MANIFEST__", _esc(manifest["manifest_content_sha256"])
    ).replace("__REASONING__", _esc(json.dumps(manifest["reasoning"], sort_keys=True))).replace(
        "__CAVEAT__", _esc(manifest["reasoning_caveat"])
    ).replace("__SUMMARY__", _summary_html(summary)).replace("__CARDS__", "".join(cards))
    output.write_text(document, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args(argv)
    build_report(manifest_path=args.manifest, output=args.output,
                 receipt_dir=args.receipt_dir, panel_only=args.panel_only)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
