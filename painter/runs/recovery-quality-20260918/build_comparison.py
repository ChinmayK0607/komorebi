"""Show existing reference/base/trained files; no local rendering or inference."""
import argparse,json,html
from pathlib import Path

def build(root):
 root=Path(root);refs=json.loads((root/'evaluation-references.json').read_text());out=root/'comparison';out.mkdir(exist_ok=True);cards=[]
 for row in refs:
  for k in range(8):
   cols=[f'<figure><img loading="lazy" src="../data/{row["reference"]}"><figcaption>Reference</figcaption></figure>']
   any_result=False
   for model in ['base','trained']:
    folder=root/'evaluation'/model/row['id']/f'sample-{k:02d}';summary=folder/'summary.json'
    if summary.exists():
     any_result=True;s=json.loads(summary.read_text());canvas=folder/Path(s['canvas']).parent.name/'painting.png' if s.get('canvas') else None
     rel='../'+str(canvas.relative_to(root)) if canvas and canvas.exists() else None
     pic=f'<img loading="lazy" src="{rel}">' if rel else '<p>No valid canvas</p>'
     detail=f'{model} · {len(s.get("turns",[]))} turns · {s.get("generated_tokens",0):,} generated tokens · '+('finished' if s.get('finished') else 'unfinished/censored')
     cols.append(f'<figure>{pic}<figcaption>{html.escape(detail)}</figcaption></figure>')
    else:cols.append(f'<figure><p>{model}: pending</p></figure>')
   if any_result:cards.append(f'<article data-k="{k}"><h2>{row["family"]} · sample {k}</h2><div>{"".join(cols)}</div></article>')
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded curriculum — base and SFT</title><style>body{font:16px/1.5 system-ui;background:#f4f1e9;color:#292822;margin:28px}article{padding:18px;background:white;margin:24px 0}article>div{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{width:100%}p{max-width:1000px}select{padding:8px}article[data-k]:not([data-k="0"]){display:none}</style><h1>Expanded painting curriculum</h1><p>Reference → base 27B → SFT adapter. These are development examples from held-out parameter bands in procedural families. They do not establish photograph generalisation. Failed and unfinished episodes remain visible. Visual quality has not been automatically graded here.</p><label>Sample <select id="sample">'''+''.join(f'<option value="{k}">{k}</option>' for k in range(8))+'''<option value="all">All available</option></select></label>'''+''.join(cards)+'''<script>document.querySelector('#sample').onchange=e=>document.querySelectorAll('article').forEach(x=>x.style.display=e.target.value==='all'||x.dataset.k===e.target.value?'block':'none')</script>'''
 (out/'index.html').write_text(page);return out/'index.html'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(build(a.root))
