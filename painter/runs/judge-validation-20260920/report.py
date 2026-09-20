"""Write a post-review JSON and static HTML gallery for the validation run."""
from __future__ import annotations

import argparse
import html
import json
import tempfile
from pathlib import Path

from validate import (JUDGES, JUDGMENT_KEYS, _canonical, _label_path,
                      _parse_label_file, _sha, validate)

FOLLOWUP_MANIFESTS = {
    "forward": "separate-calls-manifest.json",
    "reversed": "separate-calls-reversed-manifest.json",
}
FOLLOWUP_LABELS = {
    "forward": "separate-calls-labels.json",
    "reversed": "separate-calls-reversed-labels.json",
}


def _write(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def _valid_label(row: object) -> bool:
    return (isinstance(row, dict) and set(row) == JUDGMENT_KEYS and
            row.get("winner") in {"A", "B", "tie", "uncertain"} and
            row.get("confidence") in {"high", "medium", "low"} and
            row.get("acceptable_A") in {"yes", "no", "uncertain"} and
            row.get("acceptable_B") in {"yes", "no", "uncertain"} and
            isinstance(row.get("observations"), dict) and
            set(row["observations"]) == {"reference", "A", "B"})


def _image_tag(path: str | None) -> str:
    if not path:
        return "<span>missing</span>"
    return f'<img src="{html.escape(Path(path).as_uri())}" loading="lazy">'


def _label_markup(label: dict | None) -> str:
    if not label:
        return "<p class='missing'>label missing</p>"
    return (f"<p><b>winner:</b> {html.escape(str(label.get('winner')))} "
            f"<b>confidence:</b> {html.escape(str(label.get('confidence')))} "
            f"<b>acceptable A/B:</b> {html.escape(str(label.get('acceptable_A')))} / "
            f"{html.escape(str(label.get('acceptable_B')))}</p>"
            f"<p><b>reason:</b> {html.escape(str(label.get('reason')))}</p>"
            f"<dl>" + "".join(
                f"<dt>{side}</dt><dd>{html.escape(str(label.get('observations', {}).get(side, '')))}</dd>"
                for side in ("reference", "A", "B")) + "</dl>")


def _manifest_items(path: Path) -> tuple[list[dict], list[str]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"{path.name}: invalid JSON ({exc})"]
    if not isinstance(doc, list):
        return [], [f"{path.name}: manifest must be a list"]
    errors, seen = [], set()
    for row in doc:
        if not isinstance(row, dict) or set(row) != {"id", "reference", "A", "B"}:
            errors.append(f"{path.name}: manifest item schema mismatch")
            continue
        if row["id"] in seen:
            errors.append(f"{path.name}: duplicate id {row['id']}")
        seen.add(row["id"])
        for side in ("reference", "A", "B"):
            image = Path(row[side]) if isinstance(row[side], str) else Path("<invalid>")
            if not image.is_absolute() or image.suffix.lower() != ".png" or not image.is_file():
                errors.append(f"{path.name}: invalid image path for {row['id']}/{side}")
    return doc, errors


def followup_audit(root: str | Path) -> dict:
    """Audit the separate-call controls without touching main validation.json."""
    root = Path(root).resolve()
    private = json.loads((root / "artifacts" / "private.json").read_text(encoding="utf-8"))
    views = {}
    for source_id, pair in private.get("pairs", {}).items():
        if pair.get("kind") != "control":
            continue
        for public_id, view in pair.get("views", {}).items():
            views[public_id] = {"source_id": source_id, "pair": pair, "view": view}
    sections, all_errors, all_missing = {}, [], []
    followup_ids = []
    for name, manifest_name in FOLLOWUP_MANIFESTS.items():
        manifest_path = root / manifest_name
        if not manifest_path.exists():
            sections[name] = {"manifest": manifest_name, "labels": FOLLOWUP_LABELS[name],
                              "records": [], "labeled_views": 0, "expected_views": 0,
                              "errors": [], "missing": [f"{name}:manifest"]}
            all_missing.append(f"{name}:manifest")
            continue
        items, errors = _manifest_items(manifest_path)
        item_map, seen_paths = {}, set()
        for item in items:
            ident = item.get("id")
            if ident not in views:
                errors.append(f"{manifest_name}: id is not a control view: {ident}")
                continue
            if ident in followup_ids:
                errors.append(f"{manifest_name}: view appears in both follow-up manifests: {ident}")
            followup_ids.append(ident)
            item_map[ident] = item
            info = views[ident]
            for side in ("reference", "A", "B"):
                path = item[side]
                if path != info["view"].get("files", {}).get(side):
                    errors.append(f"{manifest_name}:{ident}/{side}: path not bound")
                if path in seen_paths:
                    errors.append(f"{manifest_name}: duplicate image path {path}")
                seen_paths.add(path)
                image = Path(path)
                if image.is_file() and info["view"].get("hashes", {}).get(side) != _sha(image):
                    errors.append(f"{manifest_name}:{ident}/{side}: hash mismatch")
        label_path = root / FOLLOWUP_LABELS[name]
        labels, label_errors = {}, []
        if label_path.exists():
            doc, label_errors = _parse_label_file(label_path, set(item_map))
            rows = doc.get("judgments", []) if isinstance(doc, dict) else []
            labels = {row["id"]: row for row in rows if _valid_label(row) and row["id"] in item_map}
            errors.extend(label_errors)
        else:
            all_missing.append(f"{name}:labels")
        records = []
        for ident, item in item_map.items():
            info = views[ident]
            label = labels.get(ident)
            if not label:
                all_missing.append(f"{name}:{ident}")
            else:
                expected_images = [item[x] for x in ("reference", "A", "B")]
                if label.get("reviewed_images") != expected_images:
                    errors.append(f"{name}:{ident}: reviewed_images do not match manifest")
            expected = private["controls"][info["source_id"]]["expected_winner"]
            actual = None
            if label and label.get("winner") in {"A", "B", "tie"}:
                actual = _canonical(label, info["view"]["orientation"])[0]
            correct = bool(label and label.get("confidence") != "low" and actual == expected)
            records.append({"id": ident, "source_id": info["source_id"],
                            "judge_orientation": info["view"]["orientation"],
                            "expected_winner": expected, "actual_winner": actual,
                            "correct": correct, "reference": item.get("reference"),
                            "A": item.get("A"), "B": item.get("B"), "label": label})
        sections[name] = {"manifest": manifest_name, "labels": FOLLOWUP_LABELS[name],
                          "records": records, "labeled_views": len(labels),
                          "expected_views": len(item_map), "errors": errors}
        all_errors.extend(errors)
    expected_followup_ids = set(views)
    if set(followup_ids) != expected_followup_ids or len(followup_ids) != len(expected_followup_ids):
        all_errors.append("follow-up manifests must cover every private control view exactly once")
    for source_id in private.get("controls", {}):
        source_views = [views[ident] for ident in followup_ids if ident in views and
                        views[ident]["source_id"] == source_id]
        if len(source_views) != 2 or {x["view"].get("orientation") for x in source_views} != {"ab", "ba"}:
            all_errors.append(f"follow-up views for control {source_id} must be one ab and one ba")
    record_list = [r for section in sections.values() for r in section["records"]]
    source_ids = set(private.get("controls", {}))
    by_source = {source_id: [r for r in record_list if r["source_id"] == source_id]
                 for source_id in source_ids}
    correct_controls = sum(bool(rows) and len(rows) == 2 and all(r["correct"] for r in rows)
                           for rows in by_source.values())
    expected_views = sum(section["expected_views"] for section in sections.values())
    labeled_views = sum(section["labeled_views"] for section in sections.values())
    correct_views = sum(r["correct"] for r in record_list)
    status = "pending" if all_missing else "fail" if all_errors or correct_controls != len(source_ids) else "pass"
    result = {"status": status, "overall_pass": status == "pass",
              "total_controls": len(source_ids), "expected_views": expected_views,
              "labeled_views": labeled_views, "correct_views": correct_views,
              "correct_controls": correct_controls, "all_controls_correct": correct_controls == len(source_ids),
              "errors": all_errors, "missing": all_missing, "sections": sections,
              "interpretation": "This independent one-image-per-call follow-up audits the inspection path; it does not rehabilitate or rewrite the original 24-view result."}
    _write(root / "separate-calls-report.json", result)
    return result


def build_report(root: str | Path) -> dict:
    root = Path(root).resolve()
    validation_path = root / "validation.json"
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation = validate(root)
    followup = followup_audit(root)
    private = json.loads((root / "artifacts" / "private.json").read_text(encoding="utf-8"))
    manifests = {}
    for judge in JUDGES:
        manifests[judge] = {x["id"]: x for x in json.loads((root / f"{judge}.json").read_text())}
    labels = {}
    for judge in JUDGES:
        path = _label_path(root, judge)
        if path:
            try:
                labels[judge] = {x["id"]: x for x in json.loads(path.read_text()).get("judgments", [])}
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                labels[judge] = {}
    cards, report_rows = [], []
    for source_id, pair in private.get("pairs", {}).items():
        row = {"id": source_id, "kind": pair.get("kind"), "views": []}
        if source_id in private.get("controls", {}):
            row["expected_winner"] = private["controls"][source_id].get("expected_winner")
        for public_id, view in pair.get("views", {}).items():
            judge, manifest = view.get("judge"), manifests.get(view.get("judge"), {}).get(public_id, {})
            label = labels.get(judge, {}).get(public_id)
            view_row = {"id": public_id, "judge": judge, "orientation": view.get("orientation"),
                        "reference": manifest.get("reference"), "A": manifest.get("A"),
                        "B": manifest.get("B"), "label": label}
            if label and label.get("winner") in {"A", "B"}:
                canonical_winner = label["winner"]
                if view.get("orientation") == "ba":
                    canonical_winner = "A" if canonical_winner == "B" else "B"
                view_row["canonical_winner"] = canonical_winner
            if source_id in private.get("controls", {}):
                expected = private["controls"][source_id].get("expected_winner")
                view_row["control_correct"] = bool(
                    label and label.get("confidence") != "low" and
                    (label.get("winner") == "tie" if expected == "tie" else
                     view_row.get("canonical_winner") == expected))
            row["views"].append(view_row)
            label_text = _label_markup(label)
            expected = (f"<p class='control'><b>control expected winner:</b> "
                        f"{html.escape(str(row['expected_winner']))}</p>" if "expected_winner" in row else "")
            correctness = (f" <b>control view:</b> {html.escape(str(view_row.get('control_correct')))}"
                          if "control_correct" in view_row else "")
            cards.append(f"<article><h2>{html.escape(source_id)} <small>{html.escape(str(judge))}</small></h2>"
                         f"{expected}<div class='images'><figure><figcaption>reference</figcaption>{_image_tag(manifest.get('reference'))}</figure>"
                         f"<figure><figcaption>A</figcaption>{_image_tag(manifest.get('A'))}</figure>"
                         f"<figure><figcaption>B</figcaption>{_image_tag(manifest.get('B'))}</figure></div>{label_text}{correctness}</article>")
        report_rows.append(row)
    followup_cards = []
    for name, section in followup["sections"].items():
        for record in section["records"]:
            label = record["label"]
            followup_cards.append(
                f"<article class='followup'><h2>follow-up {html.escape(name)} · "
                f"{html.escape(record['source_id'])} <small>{html.escape(record['id'])}</small></h2>"
                f"<p class='control'><b>expected:</b> {html.escape(str(record['expected_winner']))} "
                f"<b>actual canonical:</b> {html.escape(str(record['actual_winner']))} "
                f"<b>correct:</b> {html.escape(str(record['correct']))}</p>"
                f"<div class='images'><figure><figcaption>reference</figcaption>{_image_tag(record.get('reference'))}</figure>"
                f"<figure><figcaption>A</figcaption>{_image_tag(record.get('A'))}</figure>"
                f"<figure><figcaption>B</figcaption>{_image_tag(record.get('B'))}</figure></div>"
                f"{_label_markup(label)}</article>")
    report = {"validation": validation, "pairs": report_rows,
              "separate_calls_followup": followup,
              "interpretation": "Real-pair agreement is descriptive for this small validation sample; it is not evidence of artistic judge mastery."}
    _write(root / "report.json", report)
    page = """<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Luna judge validation</title>
<style>body{font:15px/1.45 system-ui;background:#f4f1e9;color:#292822;margin:24px}article{background:#fff;padding:14px;margin:18px 0;border-radius:8px}h2{margin:0 0 8px}small{font-weight:normal;color:#666}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.images figure{margin:0}.images img{width:100%;background:#eee}.images figcaption{font-weight:700;margin-bottom:4px}.control{background:#fff2c2;padding:5px}.missing{color:#a33}dt{font-weight:700}dd{margin:0 0 5px}.followup{border:3px solid #607d8b}@media(max-width:800px){.images{grid-template-columns:1fr}}</style>
<h1>Luna low visual inspection validation</h1><p>Controls are revealed here for audit. Real-pair agreement is descriptive for a small sample and does not establish judge mastery. Self-reported paths alone do not prove visual inspection.</p>
""" + "\n".join(cards) + "<h2>Independent one-image-per-call follow-up</h2>" + "\n".join(followup_cards)
    (root / "report.html").write_text(page, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root), indent=2))
