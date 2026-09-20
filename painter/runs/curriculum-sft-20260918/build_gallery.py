"""Display existing renders; performs no painting or inference."""
import argparse
import html
import json
from pathlib import Path


def build(data):
    data=Path(data);rows=[json.loads(s) for s in (data/'manifest.jsonl').read_text().splitlines()];cards=[]
    for r in rows:
        images=[]
        for i,plan in enumerate(r['plans'],1):
            rel=f"{r['id']}/turn-{i}.png"
            picture=f'<img loading="lazy" src="{rel}" alt="{html.escape(plan)}">' if (data/rel).exists() else '<p>Render missing or failed.</p>'
            images.append(f'<figure>{picture}<figcaption>{i}. {html.escape(plan)}</figcaption><a href="{r["programs"][i-1]}">Program</a></figure>')
        cards.append(f'<article><h2>{r["id"]} · level {r["level"]} · {r["split"]}</h2><div>{"".join(images)}</div></article>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Painting curriculum — first demonstrations</title><style>body{font:16px/1.5 system-ui;margin:30px;background:#f5f2eb;color:#292923}article{background:white;padding:18px;margin:25px 0}article>div{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}img{width:100%}figcaption{min-height:3em}h1,p{max-width:1000px}@media(max-width:750px){article>div{grid-template-columns:1fr}}</style><h1>First teacher demonstrations</h1><p>Quality before speed. These are assistant-authored procedural demonstrations of primitive shapes and simple compositions, rendered on a rented Linux node. The final canvas is the synthetic reference for that example; earlier canvases teach construction and revision. They are not photo-conditioned teacher paintings, student outputs, or evidence of generalisation. Development examples hold out parameter bands. Real-photo examples enter subsequent rungs.</p>'''+''.join(cards)
    (data/'index.html').write_text(page);return data/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');a=p.parse_args();print(build(a.data))
