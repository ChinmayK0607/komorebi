"""Create a portable, full-case visual comparison of fixed SFT and RL policies."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402


def _parse_named(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError("expected LABEL:PATH")
    label, raw = value.split(":", 1)
    if not label or not raw or any(c in label for c in "/\\"):
        raise ValueError("unsafe or empty label/path")
    return label, Path(raw)


def _eval_rows(root: Path, ids: set[str]) -> dict[str, dict]:
    traces, errors, files = read_traces(root)
    if errors or not files:
        raise ValueError(f"missing or corrupt traces in {root}: {errors[:3]}")
    found: dict[str, dict] = {}
    for trace in traces:
        data, info = _trace_data(trace)
        ident = data.get("task_id") or data.get("name")
        if ident in ids:
            if ident in found:
                raise ValueError(f"duplicate trace for {ident} in {root}")
            found[ident] = info
    return found


def build(manifest: Path, evaluations: list[tuple[str, Path]],
          summaries: dict[str, Path], output: Path,
          static: list[tuple[str, Path]] | None = None) -> dict:
    static = static or []
    cases = json.loads(manifest.read_text())["cases"]
    ids = {row["id"] for row in cases}
    if len(ids) != len(cases):
        raise ValueError("duplicate validation ID")
    if len({label for label, _ in evaluations + static}) != len(evaluations + static):
        raise ValueError("duplicate policy label")
    info = {label: _eval_rows(root, ids) for label, root in evaluations}
    grades = {}
    for label, path in summaries.items():
        summary = json.loads(path.read_text())
        grades[label] = {row["id"]: row for row in summary["rows"]}
        if set(grades[label]) != ids:
            raise ValueError(f"summary case IDs differ for {label}")
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)

    def image_cell(label: str, ident: str, source: Path | None,
                   status: str, score: float | None = None) -> str:
        score_text = f" · score {score:.3f}" if score is not None else ""
        caption = (f"<strong>{html.escape(label)}</strong> · "
                   f"{html.escape(status)}{score_text}")
        if source is None or not source.is_file():
            return f"<div class='cell'><div class='missing'>No valid canvas</div><p>{caption}</p></div>"
        target = images / f"{ident}-{label}.png"
        shutil.copyfile(source, target)
        return (f"<div class='cell'><img loading='lazy' src='images/{html.escape(target.name)}' "
                f"alt='{html.escape(ident + ' ' + label)}'><p>{caption}</p></div>")

    cards = []
    for row in cases:
        ident = row["id"]
        brief = Path(row["reference"]).read_text().strip()
        cells = [image_cell("frozen SFT", ident, Path(row["opponent"]), "fixed anchor")]
        for label, folder in static:
            grade = grades.get(label, {}).get(ident)
            if grade is None:
                raise ValueError(f"missing static control grade for {label}/{ident}")
            status = "valid" if grade["status"] == "scored" else "invalid"
            source = folder / f"{ident}-{label}.png"
            if status == "valid" and not source.is_file():
                raise FileNotFoundError(source)
            cells.append(image_cell(label, ident, source if status == "valid" else None,
                                    status, grade.get("score")))
        for label, _ in evaluations:
            result = info[label].get(ident, {})
            value = result.get("final_canvas")
            source = Path(value) if isinstance(value, str) and value else None
            status = "valid" if result.get("final_canvas_valid") else (
                "censored" if result.get("service_failure") or not result else "invalid")
            grade = grades.get(label, {}).get(ident, {})
            cells.append(image_cell(label, ident, source if status == "valid" else None,
                                    status, grade.get("score")))
        cards.append(f"<section><h2>{html.escape(ident)} · {html.escape(row['family'])}</h2>"
                     f"<p class='brief'>{html.escape(brief)}</p><div class='row'>"
                     + "".join(cells) + "</div></section>")
    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Text painting checkpoint comparison</title><style>
body{background:#141414;color:#eee;font:16px system-ui;margin:0;padding:24px}
h1{margin-top:0}section{background:#222;border:1px solid #444;border-radius:14px;
margin:20px 0;padding:20px}h2{font-size:20px;margin:0 0 8px}.brief{max-width:1100px}
.row{display:flex;gap:14px;overflow-x:auto}.cell{flex:0 0 290px}
img,.missing{width:290px;height:290px;object-fit:contain;background:#fff}
.missing{display:grid;place-items:center;background:#333;color:#bbb}
.cell p{margin:8px 0 0;color:#bbb}strong{color:#fff}
</style></head><body><h1>Text painting: frozen SFT vs sampled checkpoints</h1>
<p>Same held-out briefs and generation settings. Invalid and censored cases remain visible.
Scores are pairwise preferences against the fixed SFT anchor, when available.</p>
""" + "\n".join(cards) + "</body></html>"
    (output / "index.html").write_text(page)
    receipt = {"cases": len(cases), "policies": [label for label, _ in static + evaluations],
               "output": str(output / "index.html")}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval", action="append", required=True, type=_parse_named)
    parser.add_argument("--static", action="append", default=[], type=_parse_named,
                        help="LABEL:PATH to already verified public gallery images")
    parser.add_argument("--summary", action="append", default=[], type=_parse_named)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.eval, dict(args.summary),
                           args.output, static=args.static)))


if __name__ == "__main__":
    main()
