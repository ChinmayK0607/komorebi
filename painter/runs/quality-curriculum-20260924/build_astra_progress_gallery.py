#!/usr/bin/env python3
"""Build one inspectable gallery from collected, hash-verified Astra renders."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "collected/quality-curriculum-20260924"
OUT = BASE / "astra-high-100/progress.html"


def main() -> None:
    manifest = json.loads((HERE / "reference-manifest-astra-100.json").read_text())
    reference_ids = {row["id"] for row in manifest["references"]}
    # A helper-only repair supersedes its failed original render. Other later
    # passes remain separate corrections rather than replacing first-paint rows.
    rows = {}
    for wave in (1, 2, 3, 4):
        for summary_path in sorted((BASE / f"astra-high-wave{wave}-results").glob("*/run-summary.json")):
            summary = json.loads(summary_path.read_text())
            if summary.get("wave", 1) != wave or summary.get("shard") != summary_path.parent.name:
                raise ValueError(f"render summary identity mismatch: {summary_path}")
            for status in summary["statuses"]:
                ident = status["reference_id"]
                if ident not in reference_ids:
                    raise ValueError(f"development reference found in teacher gallery: {ident}")
                if not status["valid"]:
                    continue
                if ident in rows and summary["shard"] != "repair-new-b":
                    continue
                episode = summary_path.parent / "episodes" / ident
                if not (episode / "turn-01.png").is_file():
                    raise ValueError(f"valid row lacks canvas: {ident}")
                rows[ident] = (wave, summary["shard"], episode)

    cards = []
    for ident, (wave, shard, episode) in sorted(rows.items()):
        def figure(filename: str, label: str) -> str:
            path = episode / filename
            if not path.is_file():
                return ""
            rel = (Path("..") / path.relative_to(BASE)).as_posix()
            return (f'<figure><a href="{escape(rel)}"><img loading="lazy" src="{escape(rel)}" '
                    f'alt="{escape(ident)} {escape(label)}"></a><figcaption>{escape(label)}</figcaption></figure>')
        figures = figure("reference.jpg", "Reference") + figure("prior.png", "Prior canvas") + \
            figure("turn-01.png", "Astra high render")
        cards.append(f'<article id="{escape(ident)}"><h2>{escape(ident)} · wave {wave}/{escape(shard)}</h2>'
                     f'<div class="images">{figures}</div></article>')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Astra high painter progress</title>
<style>body{{font:16px system-ui;background:#1b1d20;color:#eee;margin:24px}}
h1{{margin-bottom:4px}}p{{color:#bfc3c7;max-width:80ch}}article{{background:#292d32;padding:16px;margin:24px 0;border-radius:12px}}
h2{{font-size:1rem;margin:0 0 12px}}.images{{display:flex;gap:12px;overflow-x:auto}}
figure{{margin:0;min-width:260px;max-width:420px}}img{{width:100%;height:410px;object-fit:contain;background:#e7e7df}}
figcaption{{color:#ccc;margin-top:5px}}</style></head><body><h1>Astra high painter candidates · {len(rows)}/100 rendered</h1>
<p>Reference, previous canvas when available, and actual Linux render. These are candidates; renderer validity does not mean visual acceptance or SFT admission.</p>
{''.join(cards)}</body></html>'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(OUT)


if __name__ == "__main__":
    main()
