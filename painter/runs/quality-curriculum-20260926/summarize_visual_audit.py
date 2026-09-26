#!/usr/bin/env python3
"""Validate provisional visual labels and build a reviewable first-paint gallery."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from prepare_visual_audit import OUT, RENDERS


LABELS = {"A", "B", "C", "U"}
PARENT_REVIEW = Path(__file__).with_name("PARENT_A_REVIEW.json")


def rel(path: str) -> str:
    return "../" + str(Path(path).relative_to(RENDERS))


def validate() -> tuple[list[dict], dict]:
    full = []
    missing = []
    for index in range(6):
        shard = json.loads((OUT / f"shard-{index}.json").read_text())
        labels_path = OUT / f"labels-shard-{index}.json"
        if not labels_path.is_file():
            missing.append(index)
            continue
        labels = json.loads(labels_path.read_text())
        expected = {row["audit_id"]: row for row in shard}
        if len(labels) != len(shard) or len({x["audit_id"] for x in labels}) != len(labels):
            raise ValueError(f"shard {index} labels incomplete or duplicated")
        for label in labels:
            row = expected.get(label["audit_id"])
            if (row is None or label["id"] != row["id"]
                    or label["canvas_sha256"] != row["canvas_sha256"]
                    or label["label"] not in LABELS
                    or len(label.get("observed_subject", "").strip()) < 3
                    or len(label.get("reason", "").strip()) < 25):
                raise ValueError(f"invalid label identity or evidence: {label.get('audit_id')}")
            if row["mode"] == "image_to_image" and not label.get("reference_fidelity_if_photo"):
                raise ValueError(f"missing reference fidelity: {label['audit_id']}")
            full.append({**row, **label, "judge": "gpt-5.6-luna/low", "shard": index})
    full.sort(key=lambda x: x["audit_id"])
    counts = Counter(row["label"] for row in full)
    parent = json.loads(PARENT_REVIEW.read_text())
    nominated = {row["audit_id"]: row for row in full if row["label"] == "A"}
    if set(parent) != set(nominated):
        raise ValueError("parent reviews must cover exactly every Luna A nomination")
    for audit_id, review in parent.items():
        if review["canvas_sha256"] != nominated[audit_id]["canvas_sha256"]:
            raise ValueError(f"parent review hash mismatch: {audit_id}")
        nominated[audit_id]["parent_review"] = review
    summary = {"schema": "painter.teacher600-visual-audit-summary.v1", "labelled": len(full),
               "reviewable": 595, "missing_shards": missing, "counts": dict(counts),
               "by_mode": {mode: dict(Counter(x["label"] for x in full if x["mode"] == mode))
                           for mode in ("text_to_image", "image_to_image")},
               "parent_review_status": "all_A_nominations_reviewed", "parent_reviewed_A": len(parent),
               "sft_admitted": sum(x["decision"] == "admit_direct_sft" for x in parent.values()),
               "note": "Luna labels are provisional and mode-confounded; only parent-reviewed admissions may enter SFT."}
    return full, summary


def render(rows: list[dict], summary: dict) -> Path:
    OUT.mkdir(exist_ok=True)
    payload = []
    for row in rows:
        payload.append({"audit_id": row["audit_id"], "id": row["id"], "mode": row["mode"],
                        "category": row["category"], "prompt": row["prompt"],
                        "reference": rel(row["reference_path"]) if row["reference_path"] else None,
                        "canvas": rel(row["canvas_path"]), "label": row["label"],
                        "reason": row["reason"], "fidelity": row.get("reference_fidelity_if_photo"),
                        "confidence": row.get("confidence"), "route": row["canvas_route"],
                        "chosen_run_id": row["chosen_run_id"],
                        "parent_review": row.get("parent_review")})
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher 600 visual audit</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:#14191c;color:#f4f5f4;font:16px/1.4 system-ui}}
header{{position:sticky;top:0;background:#14191cef;z-index:2;padding:16px 22px;border-bottom:1px solid #46545a}}
h1{{margin:0;font-size:1.5rem}}.sub{{color:#b6c4c8}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}}
input,select,button{{font:inherit;padding:8px;border-radius:7px;border:1px solid #718087;background:#253038;color:white}}
input{{min-width:260px;flex:1}}main{{max-width:1700px;margin:auto;padding:18px}}
article{{background:#212a2e;border:1px solid #3a494e;border-radius:10px;margin-bottom:18px;padding:15px}}
article.A{{border-color:#73bb82}}article.B{{border-color:#bcab66}}article.C{{border-color:#a86666}}
h2{{margin:0 0 6px;font-size:1.09rem}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
figure{{margin:0}}img,pre{{box-sizing:border-box;width:100%;height:430px;border-radius:6px;background:#eee;color:#15191b}}
img{{object-fit:contain}}pre{{white-space:pre-wrap;overflow:auto;padding:16px}}figcaption,.meta{{color:#bdc9cb;font-size:.88rem}}
.reason{{padding:10px 0 2px}}a{{color:#a8dbe3}}.pager{{display:flex;gap:12px;justify-content:center;align-items:center;padding:10px}}
@media(max-width:850px){{.pair{{grid-template-columns:1fr}}img,pre{{height:350px}}}}</style>
<header><h1>Teacher 600 visual audit</h1><div class="sub">{summary['labelled']}/595 provisional Luna labels · A {summary['counts'].get('A',0)} · B {summary['counts'].get('B',0)} · C {summary['counts'].get('C',0)} · U {summary['counts'].get('U',0)}. Mode-confounded first pass; {summary['sft_admitted']} direct SFT admissions after parent review.</div>
<div class="controls"><input id="search" placeholder="Search ID, subject, category, feedback"><select id="grade"><option value="">All labels</option><option>A</option><option>B</option><option>C</option><option>U</option></select><select id="mode"><option value="">Both input modes</option><option value="image_to_image">Photo reference</option><option value="text_to_image">Text prompt</option></select></div></header>
<main><div id="results"></div><div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div></main>
<script>const rows={data};let page=0;const size=30;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function draw(){{const q=document.querySelector('#search').value.toLowerCase().trim(),grade=document.querySelector('#grade').value,mode=document.querySelector('#mode').value;
const found=rows.filter(r=>(!grade||r.label===grade)&&(!mode||r.mode===mode)&&(!q||[r.id,r.audit_id,r.category,r.reason].join(' ').toLowerCase().includes(q)));
const pages=Math.max(1,Math.ceil(found.length/size));page=Math.min(page,pages-1);const shown=found.slice(page*size,(page+1)*size);
document.querySelector('#results').innerHTML=shown.map(r=>{{const left=r.reference?`<a href="${{esc(r.reference)}}"><img loading="lazy" src="${{esc(r.reference)}}" alt="reference"></a>`:`<pre>${{esc(r.prompt)}}</pre>`;
return `<article class="${{r.label}}"><h2>${{esc(r.audit_id)}} · ${{esc(r.id)}} · provisional ${{esc(r.label)}} · ${{esc(r.category)}}</h2><div class="pair"><figure>${{left}}<figcaption>${{r.reference?'Reference photo':'Text prompt'}}</figcaption></figure><figure><a href="${{esc(r.canvas)}}"><img loading="lazy" src="${{esc(r.canvas)}}" alt="painted canvas"></a><figcaption>Painted canvas · ${{esc(r.route)}}</figcaption></figure></div><div class="reason">Luna: ${{esc(r.reason)}}${{r.fidelity?' · Fidelity: '+esc(r.fidelity):''}}</div>${{r.parent_review?`<div class="reason"><strong>Parent verdict: ${{esc(r.parent_review.decision)}}</strong> — ${{esc(r.parent_review.reason)}}</div>`:''}}<div class="meta">${{esc(r.chosen_run_id)}} · confidence ${{esc(r.confidence)}}</div></article>`;}}).join('');
document.querySelector('#page').textContent=`${{found.length}} matches · page ${{page+1}} of ${{pages}}`;document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=page>=pages-1;}}
for(const id of ['search','grade','mode'])document.querySelector('#'+id).addEventListener('input',()=>{{page=0;draw()}});
document.querySelector('#prev').onclick=()=>{{page--;draw();window.scrollTo(0,0)}};document.querySelector('#next').onclick=()=>{{page++;draw();window.scrollTo(0,0)}};draw();</script></html>'''
    target = OUT / "index.html"
    target.write_text(html)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "labels-all.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return target


if __name__ == "__main__":
    rows, summary = validate()
    path = render(rows, summary)
    print(json.dumps({"gallery": str(path), **summary}, sort_keys=True))
