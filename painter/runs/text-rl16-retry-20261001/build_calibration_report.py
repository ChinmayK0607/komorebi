"""Build an offline HTML review of a frozen old-vs-draft judge run.

The report consumes only the prepared input manifest, the completed summary,
and saved JSON receipts. It makes no provider, renderer, or model calls. It
copies source PNGs and raw receipts into the report directory so the HTML can
be reviewed beside its exact evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


METHODS = ("old", "draft")
METHOD_LABELS = {"old": "Previous judge", "draft": "Revised judge"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    actual = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if actual != pair.get("brief_sha256"):
        raise ValueError(f"brief identity mismatch: {pair.get('pair_id')}")
    return value


def _expected_identity(manifest: dict[str, Any], pair: dict[str, Any],
                       method: str, repeat: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "pair_id": pair["pair_id"], "split": pair["split"],
        "method": method, "repeat": repeat,
        "protocol": manifest["protocols"][method],
        "model": manifest["models"][method],
        "prompt_sha256": manifest["prompts"][f"{method}_sha256"],
        "format_version": manifest["format_versions"][method],
        "source_format_sha256": manifest["source_format_sha256"][method],
        "brief_sha256": pair["brief_sha256"],
        "image_ids": pair["image_ids"], "image_sha256": pair["image_sha256"],
    }


def _validate_source_images(pair: dict[str, Any]) -> None:
    for role, key in (("anchor", "anchor_path"), ("candidate", "candidate_path")):
        path = Path(pair[key])
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != pair["image_sha256"][role]:
            raise ValueError(f"{role} image identity mismatch: {pair['pair_id']}")


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != "text-rl16-frozen-pair-repeat-input-v1":
        raise ValueError("unexpected frozen calibration manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("manifest integrity mismatch")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest has no pairs")
    by_id: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
            raise ValueError("manifest pair is malformed")
        pair_id = pair["pair_id"]
        if pair_id in by_id:
            raise ValueError(f"duplicate pair: {pair_id}")
        _brief(pair)
        _validate_source_images(pair)
        by_id[pair_id] = pair
    return by_id


def _validate_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    if summary.get("schema") != "text-rl16-frozen-pair-repeat-summary-v1":
        raise ValueError("unexpected frozen calibration summary schema")
    if summary.get("manifest_content_sha256") != manifest.get("manifest_content_sha256"):
        raise ValueError("summary and manifest identities differ")
    if summary.get("pair_count") != len(manifest.get("pairs", [])):
        raise ValueError("summary pair count differs from manifest")
    if summary.get("repeat_count") != manifest.get("repeat_count"):
        raise ValueError("summary repeat count differs from manifest")


def _load_receipt(*, root: Path, manifest: dict[str, Any], pair: dict[str, Any],
                  method: str, repeat: int) -> tuple[dict[str, Any] | None, Path]:
    path = root / "receipts" / method / f"{pair['pair_id']}--repeat-{repeat}.json"
    if not path.is_file() or path.is_symlink():
        return None, path
    receipt = _read_json(path)
    expected = _expected_identity(manifest, pair, method, repeat)
    if receipt.get("schema") != "text-rl16-frozen-pair-repeat-receipt-v1":
        raise ValueError(f"receipt schema mismatch: {path}")
    if receipt.get("identity") != expected:
        raise ValueError(f"receipt identity mismatch: {path}")
    return receipt, path


def _side_name(pair: dict[str, Any], winner_id: Any) -> str:
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


def _assessment_html(pair: dict[str, Any], assessment: dict[str, Any]) -> str:
    ids = pair["image_ids"]
    cards: list[str] = []
    for ident, label in ((ids["anchor"], "SFT"), (ids["candidate"], "RL")):
        value = assessment.get("assessments", {}).get(ident)
        if not isinstance(value, dict):
            continue
        weaknesses = value.get("weaknesses") or []
        weakness_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in weaknesses)
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
        return f"<article class='repeat'><h4>{title}</h4><p class='uncertain'>Missing receipt — no judgment was recorded.</p></article>"
    status = receipt.get("status")
    result = receipt.get("judge") if isinstance(receipt.get("judge"), dict) else {}
    if status == "scored" and result.get("status") in {"scored", "identical_canvases"}:
        score = result.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "missing"
        headline = f"Score {score_text} · winner {_side_name(pair, result.get('winner_id'))} · {result.get('preference', 'unknown')}"
        css = "scored"
    elif status == "implementation_error":
        headline = f"Could not judge — local implementation error ({receipt.get('reason', 'unknown')})"
        css = "uncertain"
    elif result.get("status") == "uncertain":
        headline = "Uncertain — no accepted consensus; no score"
        css = "uncertain"
    else:
        headline = f"Could not judge — censored ({receipt.get('reason', 'unknown')})"
        css = "uncertain"
    details: list[str] = []
    for label, view in _view_rows(result):
        assessment = view.get("assessment")
        if isinstance(assessment, dict):
            details.append(f"<section><h5>{label}</h5>{_assessment_html(pair, assessment)}</section>")
    votes = (result.get("initial_votes") or []) + (result.get("recheck_votes") or [])
    vote_html = f"<p><b>Returned order votes:</b> {html.escape(' · '.join(_side_name(pair, vote) for vote in votes))}</p>" if votes else ""
    identity_json = html.escape(json.dumps(receipt.get("identity", {}), indent=2, sort_keys=True))
    details_html = "".join(details) if details else "<p>No parsed assessment was recorded.</p>"
    return (
        f"<article class='repeat'><h4>{title}</h4><p class='{css}'>{html.escape(headline)}</p>"
        f"{vote_html}<details><summary>Visible assessments and reasons</summary>{details_html}</details>"
        f"<details><summary>Receipt identity (IDs and hashes)</summary><pre>{identity_json}</pre></details>"
        f"<p><a href='{html.escape(receipt_link)}'>Raw receipt JSON</a></p></article>"
    )


def _report_html(*, manifest: dict[str, Any], cards: list[str]) -> str:
    selection = html.escape(str(manifest.get("selection_note", "")))
    scope = html.escape(str(manifest.get("scope_note", "")))
    style = "body{font:15px system-ui,sans-serif;line-height:1.45;margin:0;background:#f3f4f6;color:#172033}main{max-width:1280px;margin:0 auto;padding:24px}header,.pair{background:white;border:1px solid #d1d5db;border-radius:12px;padding:20px;margin:0 0 18px}h1{margin-top:0}.scope{color:#4b5563;font-size:.94em}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.images figure{margin:0;background:#f9fafb;border:1px solid #d1d5db;padding:10px}.images img{display:block;width:100%;height:auto}.images figcaption{font-weight:700;margin-top:7px}.methods{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.method{border-top:1px solid #d1d5db;padding-top:12px}.repeat{border:1px solid #d1d5db;border-radius:8px;padding:12px;margin:10px 0}.scored{color:#166534;font-weight:700}.uncertain{color:#9a3412;font-weight:700}.brief{background:#f9fafb;border-left:4px solid #64748b;padding:10px}.assessments{display:grid;grid-template-columns:1fr 1fr;gap:12px}.assessment{background:#f9fafb;padding:8px;border-radius:6px}.assessment h5{margin:0 0 5px}details{margin-top:8px}summary{cursor:pointer;color:#1d4ed8}pre{white-space:pre-wrap;overflow:auto;font-size:12px;background:#111827;color:#e5e7eb;padding:10px;border-radius:6px}a{color:#1d4ed8}@media(max-width:800px){.methods,.assessments{grid-template-columns:1fr}}"
    intro = (
        "<header><h1>Frozen pair judge calibration</h1>"
        "<p>Each card keeps the frozen SFT painting on the left and the RL rollout on the right. "
        "“Previous judge” is the unchanged v1 rubric on fresh repeated calls; “Revised judge” is the v2 draft on fresh repeated calls. "
        "Each method has two repeated judgments when receipts are available. Missing, censored, and uncertain results carry no score.</p>"
        "<p>Scores are relative preference for RL: 0 means SFT wins clearly, 0.5 is a tie, and 1 means RL wins clearly. "
        "A score is not an absolute painting-quality grade.</p>"
        f"<p class='scope'><b>Scope:</b> {selection}<br>{scope}<br>"
        "This selected diagnostic panel is not a blind human-accuracy benchmark and does not establish aggregate reward improvement.</p>"
        "<p class='scope'><a href='manifest.json'>Input manifest</a> · <a href='summary.json'>Machine summary</a></p></header>"
    )
    return "<!doctype html><html><head><meta charset='utf-8'><title>Frozen pair judge calibration</title><style>" + style + "</style></head><body><main>" + intro + "".join(cards) + "</main></body></html>\n"


def build_report(*, manifest_path: Path, summary_path: Path, output: Path) -> Path:
    """Validate identities, preserve evidence, and write ``report.html``."""

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    pairs = _validate_manifest(manifest)
    _validate_summary(summary, manifest)
    root = summary_path.parent.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _copy(manifest_path, output / "manifest.json")
    _copy(summary_path, output / "summary.json")
    cards: list[str] = []
    repeat_count = int(manifest["repeat_count"])
    for index, pair in enumerate(pairs.values(), 1):
        pair_id = pair["pair_id"]
        brief = _brief(pair)
        _copy(Path(pair["anchor_path"]), output / "images" / f"{pair_id}-sft.png")
        _copy(Path(pair["candidate_path"]), output / "images" / f"{pair_id}-rl.png")
        method_blocks: dict[str, list[str]] = {method: [] for method in METHODS}
        for method in METHODS:
            for repeat in range(1, repeat_count + 1):
                receipt, receipt_path = _load_receipt(
                    root=root, manifest=manifest, pair=pair, method=method, repeat=repeat,
                )
                receipt_link = f"receipts/{method}/{receipt_path.name}"
                if receipt is not None:
                    _copy(receipt_path, output / "receipts" / method / receipt_path.name)
                method_blocks[method].append(_repeat_html(pair, repeat, receipt, receipt_link))
        cards.append(
            f"<section class='pair'><h2>Pair {index} · <code>{html.escape(pair_id)}</code></h2>"
            f"<p class='brief'><b>Painting brief:</b> {html.escape(brief)}</p>"
            f"<div class='images'><figure><img src='images/{html.escape(pair_id)}-sft.png' alt='SFT painting, left'>"
            f"<figcaption>SFT · LEFT</figcaption></figure><figure><img src='images/{html.escape(pair_id)}-rl.png' alt='RL rollout, right'>"
            f"<figcaption>RL rollout · RIGHT</figcaption></figure></div>"
            f"<div class='methods'><section class='method'><h3>{METHOD_LABELS['old']}</h3>{''.join(method_blocks['old'])}</section>"
            f"<section class='method'><h3>{METHOD_LABELS['draft']}</h3>{''.join(method_blocks['draft'])}</section></div></section>"
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
