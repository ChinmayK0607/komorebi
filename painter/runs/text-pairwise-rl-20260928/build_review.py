"""Build a portable visual review of the frozen SFT text-painting baseline."""

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


def build(pool: Path, eval_root: Path, output: Path) -> dict:
    cases = json.loads(pool.read_text())["cases"]
    traces, errors, files = read_traces(eval_root)
    by_id: dict[str, list[dict]] = {row["id"]: [] for row in cases}
    for trace in traces:
        data, info = _trace_data(trace)
        task_id = data.get("task_id") or data.get("name")
        if task_id in by_id:
            by_id[task_id].append(info)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    cards = []
    summary = {"cases": len(cases), "traces": len(traces), "trace_errors": errors,
               "valid_final": 0, "invalid_final": 0, "censored": 0}
    for row in cases:
        episodes = by_id[row["id"]]
        brief = (pool.parent / row["reference"]).read_text().strip()
        status = "censored"
        image_html = "<div class='missing'>No final painting</div>"
        turns = []
        if len(episodes) == 1:
            info = episodes[0]
            turns = info.get("turns", [])
            status = "valid" if info.get("final_canvas_valid") else "invalid"
            value = info.get("final_canvas")
            canvas = Path(value) if isinstance(value, str) and value else None
            if status == "valid" and canvas and canvas.is_file():
                target = image_dir / (row["id"] + ".png")
                shutil.copyfile(canvas, target)
                image_html = f"<img loading='lazy' src='images/{html.escape(target.name)}' alt='{html.escape(row['id'])} painting'>"
            elif status == "valid":
                status = "censored"
        summary[{"valid": "valid_final", "invalid": "invalid_final", "censored": "censored"}[status]] += 1
        failures = ", ".join(str(turn.get("failure_kind", "")) for turn in turns if not turn.get("valid"))
        cards.append(
            f"<article><div class='meta'><b>{html.escape(row['id'])}</b> · {html.escape(row['family'])} "
            f"· {html.escape(row['split'])} · <strong class='{status}'>{status}</strong> · "
            f"{len(turns)} turns</div><p>{html.escape(brief)}</p>{image_html}"
            f"<small>Failed turns: {html.escape(failures or 'none')}</small></article>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'><title>Text painting SFT baseline</title>
<style>body{font-family:system-ui;background:#151515;color:#eee;max-width:1500px;margin:auto;padding:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}
article{background:#232323;padding:16px;border-radius:12px}img{width:100%;aspect-ratio:1;object-fit:contain;background:white}
.meta,small{color:#bbb}.valid{color:#8ee7a1}.invalid{color:#ff9a91}.censored{color:#eccc82}
p{min-height:5em}.missing{height:280px;display:grid;place-items:center;background:#333}
</style></head><body><h1>Frozen step-512 SFT text baseline</h1><p>Matched three-turn sampling;
the image is the final valid canvas, not a teacher painting. This gallery is visual evidence,
not a quality score.</p><div class='grid'>""" + "\n".join(cards) + "</div></body></html>"
    (output / "index.html").write_text(page)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
