"""Build a portable offline report for the Flash-versus-Pro judge panel.

The report consumes a completed model-comparison manifest, summary, and raw
receipts.  It verifies the frozen source identities and image bytes before
copying anything, preserves every receipt, and makes no provider or renderer
call.  The panel is a consistency diagnostic without gold labels; it does not
present a win rate or an accuracy estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_consistent_side_judge_v2_model as transport
except ImportError:  # pragma: no cover - direct execution from a checkout
    import sys
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_consistent_side_judge_v2_model as transport


METHODS = ("baseline", "candidate")
MODEL_LABELS = {
    "xiaomi/mimo-v2.6-flash": "MiMo Flash",
    "xiaomi/mimo-v2.6-pro": "MiMo Pro",
}
MANIFEST_SCHEMA = "text-rl16-frozen-pair-model-repeat-input-v1"
SUMMARY_SCHEMA = "text-rl16-frozen-pair-model-repeat-summary-v1"
RECEIPT_SCHEMA = "text-rl16-frozen-pair-model-repeat-receipt-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _brief(pair: dict[str, Any]) -> str:
    if isinstance(pair.get("brief"), str):
        value = pair["brief"].strip()
    else:
        path = Path(str(pair.get("brief_path", "")))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty brief: {pair.get('pair_id')}")
    if hashlib.sha256(value.encode("utf-8")).hexdigest() != pair.get("brief_sha256"):
        raise ValueError(f"brief identity mismatch: {pair.get('pair_id')}")
    return value


def _validate_source_pair(pair: dict[str, Any]) -> None:
    for role, path_key in (("anchor", "anchor_path"), ("candidate", "candidate_path")):
        path = Path(str(pair.get(path_key, "")))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != pair["image_sha256"].get(role):
            raise ValueError(f"{role} image hash mismatch: {pair['pair_id']}")
        if transport.image_id(path) != pair["image_ids"].get(role):
            raise ValueError(f"{role} image ID mismatch: {pair['pair_id']}")


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected model calibration manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("manifest integrity mismatch")
    expected_fields = {
        "protocol": transport.PROTOCOL,
        "format_version": transport.FORMAT_VERSION,
        "source_format_sha256": transport.SOURCE_FORMAT_SHA256,
        "rubric_source_format_sha256": transport.RUBRIC_SOURCE_FORMAT_SHA256,
        "prompt_sha256": transport.PROMPT_SHA256,
        "temperature": transport.TEMPERATURE,
        "max_tokens": transport.MAX_TOKENS,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(f"manifest {field} differs from the reviewed transport")
    if manifest.get("reasoning") != {"mode": "provider_default", "request_field_omitted": True}:
        raise ValueError("manifest reasoning configuration differs")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(METHODS):
        raise ValueError("manifest must bind baseline and candidate models")
    if set(models.values()) != set(MODEL_LABELS):
        raise ValueError("report requires the verified MiMo Flash and MiMo Pro model panel")
    if not isinstance(manifest.get("repeat_count"), int) or manifest["repeat_count"] < 1:
        raise ValueError("manifest repeat_count is invalid")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest has no pairs")
    by_id: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
            raise ValueError("manifest pair is malformed")
        if pair["pair_id"] in by_id:
            raise ValueError(f"duplicate pair: {pair['pair_id']}")
        _brief(pair)
        _validate_source_pair(pair)
        by_id[pair["pair_id"]] = pair
    return by_id


def _validate_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("unexpected model calibration summary schema")
    for field in ("manifest_content_sha256", "models",
                  "protocol", "format_version", "source_format_sha256", "prompt_sha256"):
        if summary.get(field) != manifest.get(field):
            raise ValueError(f"summary {field} differs from manifest")
    if summary.get("pair_count") != len(manifest.get("pairs", [])):
        raise ValueError("summary pair count differs from manifest")
    if summary.get("repeat_count") != manifest.get("repeat_count"):
        raise ValueError("summary repeat count differs from manifest")
    if summary.get("gold_labels") is not False:
        raise ValueError("summary gold-label scope is not explicit")


def _expected_identity(manifest: dict[str, Any], pair: dict[str, Any],
                       method: str, repeat: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"], "split": pair["split"],
        "method": method, "repeat": repeat,
        "model": manifest["models"][method],
        "model_config_sha256": manifest["model_config_sha256"],
        "protocol": manifest["protocol"], "format_version": manifest["format_version"],
        "source_format_sha256": manifest["source_format_sha256"],
        "rubric_source_format_sha256": manifest["rubric_source_format_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "brief_sha256": pair["brief_sha256"], "image_ids": pair["image_ids"],
        "image_sha256": pair["image_sha256"],
    }


def _load_receipt(*, root: Path, manifest: dict[str, Any], pair: dict[str, Any],
                  method: str, repeat: int) -> tuple[dict[str, Any] | None, Path]:
    path = root / "receipts" / method / f"{pair['pair_id']}--repeat-{repeat}.json"
    if not path.is_file() or path.is_symlink():
        return None, path
    receipt = _read_json(path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"receipt schema mismatch: {path}")
    if receipt.get("identity") != _expected_identity(manifest, pair, method, repeat):
        raise ValueError(f"receipt identity mismatch: {path}")
    return receipt, path


def _winner_label(pair: dict[str, Any], winner_id: Any) -> str:
    if winner_id == "tie":
        return "Tie"
    ids = pair["image_ids"]
    if winner_id == ids.get("anchor"):
        return "SFT"
    if winner_id == ids.get("candidate"):
        return "RL"
    return f"Unknown ID ({winner_id})"


def _view_rows(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for index, view in enumerate(result.get("views") or [], 1):
        rows.append((f"Order view {index}", view))
    for index, view in enumerate(result.get("recheck_views") or [], 1):
        rows.append((f"Recheck view {index}", view))
    return rows


def _orientation_note(pair: dict[str, Any], view: dict[str, Any]) -> str:
    """Describe accepted request orientation; never infer missing metadata."""

    ids = pair["image_ids"]
    for call in view.get("calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("assessment"), dict):
            continue
        image_ids = call.get("image_ids")
        if not isinstance(image_ids, dict):
            continue
        left_id, right_id = image_ids.get("A"), image_ids.get("B")
        if left_id == ids.get("anchor") and right_id == ids.get("candidate"):
            orientation = "LEFT = SFT, RIGHT = RL"
        elif left_id == ids.get("candidate") and right_id == ids.get("anchor"):
            orientation = "LEFT = RL, RIGHT = SFT"
        else:
            continue
        return f"In this judge request: {orientation}. The review images above stay SFT left / RL right."
    return "In this judge request: LEFT/RIGHT orientation unavailable from accepted call metadata. The review images above stay SFT left / RL right."


def _assessment_html(pair: dict[str, Any], assessment: dict[str, Any]) -> str:
    assessments = assessment.get("assessments")
    if not isinstance(assessments, dict):
        return "<p class='uncertain'>No accepted panel assessments were recorded.</p>"
    cards: list[str] = []
    ids = pair["image_ids"]
    for ident, label in ((ids["anchor"], "SFT"), (ids["candidate"], "RL")):
        value = assessments.get(ident)
        if not isinstance(value, dict):
            cards.append(f"<div class='assessment'><h5>{label}</h5><p>Assessment unavailable.</p></div>")
            continue
        weaknesses = value.get("weaknesses")
        if isinstance(weaknesses, list) and weaknesses:
            weakness_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in weaknesses)
        else:
            weakness_html = "<li>Weaknesses unavailable.</li>"
        cards.append(
            f"<div class='assessment'><h5>{label}</h5>"
            f"<p><b>Fidelity:</b> {html.escape(str(value.get('brief_fidelity', 'unknown')))}</p>"
            f"<p><b>Visible evidence:</b> {html.escape(str(value.get('visible_evidence', '')))}</p>"
            f"<p><b>Weaknesses:</b></p><ul>{weakness_html}</ul></div>"
        )
    reason = assessment.get("reason")
    reason_html = f"<p><b>Reason:</b> {html.escape(str(reason))}</p>" if reason else ""
    return reason_html + "<div class='assessments'>" + "".join(cards) + "</div>"


def _repeat_html(pair: dict[str, Any], repeat: int,
                 receipt: dict[str, Any] | None, receipt_link: str) -> str:
    title = f"Repeat {repeat}"
    if receipt is None:
        return f"<article class='repeat'><h4>{title}</h4><p class='uncertain'>Missing receipt — no judgment was recorded and no score is available.</p></article>"
    status = receipt.get("status")
    result = receipt.get("judge") if isinstance(receipt.get("judge"), dict) else {}
    if status == "scored" and result.get("status") in {"scored", "identical_canvases"}:
        score = result.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "missing"
        headline = f"Relative RL score {score_text} · winner {_winner_label(pair, result.get('winner_id'))} · {result.get('preference', 'unknown')}"
        css = "scored"
    elif status == "implementation_error":
        headline = f"Could not judge — local implementation error ({receipt.get('reason', 'unknown')}); no score"
        css = "uncertain"
    elif result.get("status") == "uncertain":
        headline = f"Uncertain — {result.get('consensus', 'no accepted consensus')}; no score"
        css = "uncertain"
    else:
        headline = f"Could not judge — censored ({receipt.get('reason', 'unknown')}); no score"
        css = "uncertain"
    requested = result.get("requested_model", receipt.get("identity", {}).get("model", "unknown"))
    returned = result.get("returned_models")
    returned_text = ", ".join(str(value) for value in returned) if isinstance(returned, list) and returned else "unavailable"
    details: list[str] = []
    for label, view in _view_rows(result):
        assessment = view.get("assessment")
        orientation = html.escape(_orientation_note(pair, view))
        if isinstance(assessment, dict):
            details.append(f"<section><h5>{label}</h5><p class='scope'>{orientation}</p>{_assessment_html(pair, assessment)}</section>")
        else:
            details.append(f"<section><h5>{label}</h5><p class='scope'>{orientation}</p><p>No parsed assessment was recorded for this view.</p></section>")
    votes = (result.get("initial_votes") or []) + (result.get("recheck_votes") or [])
    vote_html = f"<p><b>Returned order votes:</b> {html.escape(' · '.join(_winner_label(pair, vote) for vote in votes))}</p>" if votes else ""
    identity_json = html.escape(json.dumps(receipt.get("identity", {}), indent=2, sort_keys=True))
    details_html = "".join(details) if details else "<p>No parsed assessment was recorded.</p>"
    return (
        f"<article class='repeat'><h4>{title}</h4><p class='{css}'>{html.escape(headline)}</p>"
        f"<p class='scope'><b>Requested model:</b> {html.escape(str(requested))}<br>"
        f"<b>Returned model identity:</b> {html.escape(returned_text)}</p>{vote_html}"
        f"<details><summary>Weaknesses, reasons, and request orientation</summary>{details_html}</details>"
        f"<details><summary>Receipt identity (IDs and hashes)</summary><pre>{identity_json}</pre></details>"
        f"<p><a href='{html.escape(receipt_link)}'>Raw receipt JSON</a></p></article>"
    )


def _report_html(*, manifest: dict[str, Any], cards: list[str]) -> str:
    selection = html.escape(str(manifest.get("selection_note", "")))
    scope = html.escape(str(manifest.get("scope_note", "")))
    style = (
        "body{font:15px system-ui,sans-serif;line-height:1.45;margin:0;background:#f3f4f6;color:#172033}"
        "main{max-width:1280px;margin:0 auto;padding:24px}header,.pair{background:white;border:1px solid #d1d5db;border-radius:12px;padding:20px;margin:0 0 18px}"
        "h1{margin-top:0}.scope{color:#4b5563;font-size:.94em}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}"
        ".images figure{margin:0;background:#f9fafb;border:1px solid #d1d5db;padding:10px;min-width:0}.images img{display:block;width:100%;height:auto}"
        ".images figcaption{font-weight:700;margin-top:7px}.models{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}"
        ".model{border-top:1px solid #d1d5db;padding-top:12px}.repeat{border:1px solid #d1d5db;border-radius:8px;padding:12px;margin:10px 0}"
        ".scored{color:#166534;font-weight:700}.uncertain{color:#9a3412;font-weight:700}.brief{background:#f9fafb;border-left:4px solid #64748b;padding:10px}"
        ".assessments{display:grid;grid-template-columns:1fr 1fr;gap:12px}.assessment{background:#f9fafb;padding:8px;border-radius:6px}.assessment h5{margin:0 0 5px}"
        "details{margin-top:8px}summary{cursor:pointer;color:#1d4ed8}pre{white-space:pre-wrap;overflow:auto;font-size:12px;background:#111827;color:#e5e7eb;padding:10px;border-radius:6px}"
        "a{color:#1d4ed8}@media(max-width:800px){.models,.assessments{grid-template-columns:1fr}}"
    )
    intro = (
        "<header><h1>Frozen judge model panel</h1>"
        "<p>Each card uses the same frozen painting brief and the same two images. The displayed review stays SFT left / RL right. "
        "MiMo Flash and MiMo Pro are separate judge models using the same v2 rubric, request schema, temperature, and repeated side-order protocol.</p>"
        "<p>Scores are relative preference for RL: 0 means SFT wins clearly, 0.5 is a tie, and 1 means RL wins clearly. "
        "This is not an absolute painting-quality grade. Missing, censored, and uncertain repeats carry no score.</p>"
        f"<p class='scope'><b>Scope:</b> {selection}<br>{scope}<br>"
        "This selected diagnostic panel has no gold labels and does not establish judge accuracy, a win rate, or aggregate reward improvement.</p>"
        "<p class='scope'><a href='manifest.json'>Input manifest</a> · <a href='summary.json'>Machine summary</a></p></header>"
    )
    return "<!doctype html><html><head><meta charset='utf-8'><title>Frozen judge model panel</title><style>" + style + "</style></head><body><main>" + intro + "".join(cards) + "</main></body></html>\n"


def build_report(*, manifest_path: Path, summary_path: Path, output: Path) -> Path:
    """Validate all evidence first, then copy it and write ``report.html``."""

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    pairs = _validate_manifest(manifest)
    _validate_summary(summary, manifest)
    root = summary_path.parent.resolve()
    repeat_count = int(manifest["repeat_count"])
    prepared: list[tuple[dict[str, Any], str, dict[str, dict[int, tuple[dict[str, Any] | None, Path]]]]] = []
    for pair in pairs.values():
        brief = _brief(pair)
        method_receipts: dict[str, dict[int, tuple[dict[str, Any] | None, Path]]] = {method: {} for method in METHODS}
        for method in METHODS:
            for repeat in range(1, repeat_count + 1):
                method_receipts[method][repeat] = _load_receipt(
                    root=root, manifest=manifest, pair=pair, method=method, repeat=repeat,
                )
        prepared.append((pair, brief, method_receipts))

    output.mkdir(parents=True, exist_ok=True)
    _copy(manifest_path, output / "manifest.json")
    _copy(summary_path, output / "summary.json")
    cards: list[str] = []
    for index, (pair, brief, method_receipts) in enumerate(prepared, 1):
        pair_id = pair["pair_id"]
        _copy(Path(pair["anchor_path"]), output / "images" / f"{pair_id}-sft.png")
        _copy(Path(pair["candidate_path"]), output / "images" / f"{pair_id}-rl.png")
        model_blocks: list[str] = []
        for method in METHODS:
            model = manifest["models"][method]
            repeat_blocks: list[str] = []
            for repeat in range(1, repeat_count + 1):
                receipt, receipt_path = method_receipts[method][repeat]
                receipt_link = f"receipts/{method}/{receipt_path.name}"
                if receipt is not None:
                    _copy(receipt_path, output / "receipts" / method / receipt_path.name)
                repeat_blocks.append(_repeat_html(pair, repeat, receipt, receipt_link))
            model_blocks.append(
                f"<section class='model'><h3>{html.escape(MODEL_LABELS[model])}</h3>"
                f"<p class='scope'><b>Model ID:</b> {html.escape(model)}</p>{''.join(repeat_blocks)}</section>"
            )
        cards.append(
            f"<section class='pair'><h2>Pair {index} · <code>{html.escape(pair_id)}</code></h2>"
            f"<p class='brief'><b>Painting brief:</b> {html.escape(brief)}</p>"
            f"<div class='images'><figure><img src='images/{html.escape(pair_id)}-sft.png' alt='SFT painting, left'>"
            f"<figcaption>SFT · LEFT</figcaption></figure><figure><img src='images/{html.escape(pair_id)}-rl.png' alt='RL painting, right'>"
            f"<figcaption>RL · RIGHT</figcaption></figure></div>"
            f"<div class='models'>{''.join(model_blocks)}</div></section>"
        )
    report = output / "report.html"
    report.write_text(_report_html(manifest=manifest, cards=cards), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(manifest_path=args.manifest, summary_path=args.summary, output=args.output)
    print(json.dumps({"report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
