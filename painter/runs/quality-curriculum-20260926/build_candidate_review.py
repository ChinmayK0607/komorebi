#!/usr/bin/env python3
"""Build a local source/prompt review page for unrendered teacher candidates."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
OUT = POOL / "review/index.html"


def local_path(batch: Path, value: str) -> Path:
    return ROOT / value if value.startswith("painter/") else batch / value


def href(target: Path) -> str:
    return quote(Path(os.path.relpath(target, OUT.parent)).as_posix(), safe="/.-_")


def collect() -> list[dict]:
    cards = []
    for manifest_path in sorted(POOL.glob("*/manifest.json")):
        batch = manifest_path.parent
        if "static-repair" in batch.name:
            continue  # Alternative programs do not add new reference coverage.
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("items", manifest.get("samples", manifest.get("entries", [])))
        model = manifest.get("author_model", manifest.get("model", "unknown"))
        for row in rows:
            reference = row.get("reference_file") or row.get("reference_path")
            prompt = row.get("prompt_file") or row.get("prompt_path")
            program = row.get("program_file") or row.get("program_path")
            if reference:
                input_file = local_path(batch, reference)
                source = row.get("source", row)
                cards.append({"id": row["id"], "batch": batch.name, "kind": "image",
                              "model": model, "category": row.get("corrected_category", row.get("category", "")),
                              "difficulty": row.get("difficulty", "unlabelled"),
                              "description": row.get("intended_composition", row.get("plan", "")),
                              "reference": input_file,
                              "program": local_path(batch, program),
                              "source_url": source.get("source_url", ""),
                              "license": source.get("license_name", source.get("license", ""))})
            else:
                input_file = local_path(batch, prompt)
                cards.append({"id": row["id"], "batch": batch.name, "kind": "text",
                              "model": model, "category": row.get("category", ""),
                              "difficulty": row.get("difficulty", "unlabelled"),
                              "description": input_file.read_text().strip(),
                              "reference": None, "program": local_path(batch, program),
                              "source_url": "", "license": ""})
    return cards


def build() -> dict:
    cards = collect()
    kinds = Counter(card["kind"] for card in cards)
    blocks = []
    for card in cards:
        visual = (f'<img loading="lazy" src="{href(card["reference"])}" alt="reference for {html.escape(card["id"])}">'
                  if card["reference"] else '<div class="textmark">TEXT PROMPT</div>')
        source = (f' · <a href="{html.escape(card["source_url"], quote=True)}">source</a>'
                  if card["source_url"] else "")
        blocks.append(
            f'<article data-kind="{card["kind"]}" data-model="{html.escape(card["model"])}" '
            f'data-batch="{html.escape(card["batch"])}">'
            f'<div class="visual">{visual}</div><div class="body">'
            f'<div class="meta">{html.escape(card["batch"])} · {html.escape(card["model"])} · '
            f'{html.escape(str(card["difficulty"]))}</div>'
            f'<h2>{html.escape(card["id"])}</h2><p class="category">{html.escape(str(card["category"]))}</p>'
            f'<p>{html.escape(card["description"])}</p>'
            f'<p class="links"><a href="{href(card["program"])}">program</a>{source}'
            f'{" · " + html.escape(str(card["license"])) if card["license"] else ""}</p>'
            f'</div></article>'
        )
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Teacher candidate source review</title>
<style>
body{{margin:0;background:#ede9df;color:#242722;font:16px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#f7f4ed;border-bottom:1px solid #c9c4b9;padding:14px 22px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:4px 0}}p{{margin:7px 0}}
.status{{color:#6b3e30;font-weight:650}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin-top:9px}}
select,input{{font:inherit;padding:6px 8px;border:1px solid #aaa79e;border-radius:5px;background:white}}
input{{min-width:240px}}main{{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px}}
article{{background:white;border:1px solid #d4cec0;border-radius:9px;overflow:hidden;box-shadow:0 2px 8px #00000012}}
.visual{{height:210px;background:#ddd9d0;display:grid;place-items:center}}.visual img{{width:100%;height:100%;object-fit:contain}}
.textmark{{font-size:14px;letter-spacing:.12em;color:#6d685d}}.body{{padding:13px 16px}}
.meta,.category,.links{{font-size:13px;color:#666b62}}.links a{{color:#315b63}}article[hidden]{{display:none}}
</style>
<header><h1>Teacher candidate source review</h1>
<div class="status">Unrendered programs only. This page does not show paintings or imply visual approval.</div>
<div>{len(cards)} distinct inputs · {kinds['text']} text · {kinds['image']} image</div>
<div class="controls"><select id="kind"><option value="">All inputs</option><option>text</option><option>image</option></select>
<select id="model"><option value="">Both teachers</option><option>gpt-5.6-sol</option><option>gpt-6-astra</option></select>
<select id="batch"><option value="">All batches</option>{''.join(f'<option>{html.escape(batch)}</option>' for batch in sorted({card['batch'] for card in cards}))}</select>
<input id="search" placeholder="Search subject, category, batch"><span id="shown"></span></div></header>
<main>{''.join(blocks)}</main>
<script>
const cards=[...document.querySelectorAll('article')];
function filter(){{const k=document.querySelector('#kind').value,m=document.querySelector('#model').value,
b=document.querySelector('#batch').value,q=document.querySelector('#search').value.toLowerCase();let n=0;for(const c of cards){{
const ok=(!k||c.dataset.kind===k)&&(!m||c.dataset.model===m)&&(!b||c.dataset.batch===b)&&c.textContent.toLowerCase().includes(q);
c.hidden=!ok;if(ok)n++;}}document.querySelector('#shown').textContent=n+' shown';}}
for(const id of ['kind','model','batch','search'])document.getElementById(id).addEventListener('input',filter);filter();
</script></html>'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    return {"path": str(OUT), "count": len(cards), "text": kinds["text"], "image": kinds["image"]}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
