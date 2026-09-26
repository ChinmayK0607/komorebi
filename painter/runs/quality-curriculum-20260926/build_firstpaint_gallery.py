#!/usr/bin/env python3
"""Build one paged, searchable review for collected teacher first paints."""

from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
RENDERS = POOL / "rendered-cloud"
DEST = RENDERS / "teacher600-gallery"


def source_rows(batch: str) -> dict[str, dict]:
    source = POOL / batch / "source.tar.gz"
    if not source.is_file():
        raise FileNotFoundError(f"missing original source bundle: {batch}")
    with tarfile.open(source, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    return {row["id"]: row for row in manifest["rows"]}


def collect() -> list[dict]:
    rows = []
    source_cache = {}
    for summary_file in sorted(RENDERS.glob("*/run-summary.json")):
        summary = json.loads(summary_file.read_text())
        if summary.get("schema") != "painter.teacher500-render.v1":
            continue
        batch = summary["batch"]
        if batch not in source_cache:
            source_cache[batch] = source_rows(batch)
        for status in summary["statuses"]:
            ident = status["id"]
            source = source_cache[batch][ident]
            if status.get("role", source.get("role")) != "first_paint_candidate":
                continue
            episode = summary_file.parent / "episodes" / ident
            prompt = (episode / "input.txt").read_text() if status["mode"] == "text_to_image" else None
            input_path = episode / ("input.txt" if prompt is not None else "input.jpg")
            if not input_path.is_file():
                raise FileNotFoundError(f"missing collected input: {episode}")
            canvas = episode / "canvas.png"
            if status["valid"] and not canvas.is_file():
                raise ValueError(f"valid row lacks canvas: {episode}")
            rows.append({"id": ident, "batch": batch, "run_id": summary["run_id"],
                         "mode": status["mode"], "category": source.get("category") or "uncategorized",
                         "prompt": prompt, "reference": None if prompt is not None else f"../{summary['run_id']}/episodes/{ident}/input.jpg",
                         "canvas": f"../{summary['run_id']}/episodes/{ident}/canvas.png" if canvas.is_file() else None,
                         "program": f"../{summary['run_id']}/episodes/{ident}/program.js",
                         "valid": bool(status["valid"]), "error": status.get("error_code"),
                         "elapsed_seconds": status.get("elapsed_seconds"),
                         "input_sha256": status["input_sha256"],
                         "program_sha256": status["program_sha256"],
                         "canvas_sha256": status.get("canvas_sha256")})
    keys = [(r["mode"], r["input_sha256"], r["program_sha256"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("same first-paint source appears in multiple collected runs")
    return sorted(rows, key=lambda r: (r["mode"], r["category"], r["id"]))


def build() -> Path:
    rows = collect()
    DEST.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["mode"] for r in rows)
    valid = sum(r["valid"] for r in rows)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Teacher 600 render review</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#15181a;color:#f2f3f1;font:16px/1.45 system-ui}}
header{{position:sticky;top:0;background:#15181af5;z-index:2;padding:18px 22px;border-bottom:1px solid #465054}}
h1{{font-size:1.5rem;margin:0 0 4px}}.muted{{color:#b9c6c7}}.controls{{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}}
input,select,button{{font:inherit;border:1px solid #66767b;border-radius:8px;background:#252d30;color:#fff;padding:8px}}
input{{min-width:260px;flex:1}}button{{cursor:pointer}}button:disabled{{opacity:.45;cursor:auto}}
main{{max-width:1680px;margin:0 auto;padding:20px}}article{{background:#22292c;border:1px solid #364145;border-radius:12px;margin:0 0 18px;padding:16px}}
article.bad{{border-color:#8b5550}}h2{{font-size:1.08rem;margin:0 0 8px}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0}}
img,pre,.missing{{box-sizing:border-box;width:100%;height:450px;background:#eae9e3;color:#15181a;border-radius:6px}}
img{{object-fit:contain}}pre{{padding:18px;overflow:auto;white-space:pre-wrap}}
.missing{{display:grid;place-items:center;font-weight:600}}figcaption{{color:#b9c6c7;margin:6px 0}}
.meta{{color:#aebcbd;font-size:.86rem;overflow-wrap:anywhere}}a{{color:#9ed7df}}.pager{{display:flex;gap:10px;align-items:center;justify-content:center;padding:8px 0 25px}}
@media(max-width:850px){{.pair{{grid-template-columns:1fr}}img,pre,.missing{{height:360px}}}}
</style>
<header><h1>Teacher first-paint renders</h1>
<div class="muted">{len(rows)} collected raw tasks · {valid} renderer-valid · {len(rows)-valid} failed/censored · {counts['text_to_image']} text · {counts['image_to_image']} photo. Validity is not visual admission.</div>
<div class="controls"><input id="search" placeholder="Search subject, category, batch or prompt" aria-label="Search paintings">
<select id="mode" aria-label="Input mode"><option value="">Both modes</option><option value="text_to_image">Text prompt</option><option value="image_to_image">Photo reference</option></select>
<select id="status" aria-label="Render status"><option value="">All statuses</option><option value="valid">Valid</option><option value="failed">Failed/censored</option></select></div></header>
<main><div id="results"></div><div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div></main>
<script>const rows={payload};let page=0;const size=40;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function draw(){{const q=document.querySelector('#search').value.toLowerCase().trim(),mode=document.querySelector('#mode').value,status=document.querySelector('#status').value;
const found=rows.filter(r=>(!mode||r.mode===mode)&&(!status||(status==='valid'?r.valid:!r.valid))&&(!q||[r.id,r.batch,r.category,r.prompt].join(' ').toLowerCase().includes(q)));
const pages=Math.max(1,Math.ceil(found.length/size));page=Math.min(page,pages-1);const shown=found.slice(page*size,(page+1)*size);
document.querySelector('#results').innerHTML=shown.map(r=>{{const left=r.reference?`<a href="${{esc(r.reference)}}"><img loading="lazy" src="${{esc(r.reference)}}" alt="reference"></a>`:`<pre>${{esc(r.prompt)}}</pre>`;
const right=r.canvas?`<a href="${{esc(r.canvas)}}"><img loading="lazy" src="${{esc(r.canvas)}}" alt="rendered painting"></a>`:`<div class="missing">No canvas · ${{esc(r.error||'unknown error')}}</div>`;
return `<article class="${{r.valid?'':'bad'}}"><h2>${{esc(r.id)}} · ${{esc(r.category)}} · ${{r.valid?'valid':'failed/censored'}}</h2><div class="pair"><figure>${{left}}<figcaption>${{r.reference?'Photo reference':'Text prompt'}}</figcaption></figure><figure>${{right}}<figcaption>Painted canvas</figcaption></figure></div><div class="meta">${{esc(r.batch)}} · ${{esc(r.run_id)}} · ${{r.elapsed_seconds===null?'time unavailable':r.elapsed_seconds+' s'}} · <a href="${{esc(r.program)}}">program</a></div></article>`;}}).join('');
document.querySelector('#page').textContent=`${{found.length}} matches · page ${{page+1}} of ${{pages}}`;
document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=page>=pages-1;}}
for(const id of ['search','mode','status'])document.querySelector('#'+id).addEventListener('input',()=>{{page=0;draw()}});
document.querySelector('#prev').onclick=()=>{{page--;draw();window.scrollTo(0,0)}};
document.querySelector('#next').onclick=()=>{{page++;draw();window.scrollTo(0,0)}};draw();</script></html>'''
    path = DEST / "index.html"
    path.write_text(html)
    (DEST / "coverage.json").write_text(json.dumps({"collected": len(rows), "valid": valid,
                                                       "failed_or_censored": len(rows) - valid,
                                                       "by_mode": counts}, indent=2, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    print(build())
