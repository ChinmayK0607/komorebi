"""Display preserved images and judge receipts; no local rendering/inference."""
import argparse,json,html
from pathlib import Path

def build(root):
 root=Path(root).resolve();out=root/'comparison';out.mkdir(exist_ok=True);cards=[]
 def local(path):return root/str(path).removeprefix('/root/painter/')
 def figure(path,label):
  f=local(path) if path else None
  pic=f'<a href="../{f.relative_to(root)}"><img loading="lazy" src="../{f.relative_to(root)}"></a>' if f and f.exists() else '<div class="pending">Blank / unavailable</div>'
  return '<figure>'+pic+'<figcaption>'+html.escape(label)+'</figcaption></figure>'
 tasks=json.loads((root/'evaluation-tasks.json').read_text())
 for task in tasks:
  for k in range(2):
   columns=[figure(task['reference'],'Reference'),figure(task.get('canvas'),'Starting canvas')]
   for model in ['parent','trained']:
    f=root/'evaluation'/model/task['id']/f'sample-{k:02d}/summary.json'
    if f.exists():
     s=json.loads(f.read_text());label=f'{model} · {len(s.get("turns",[]))} turns · {s.get("generated_tokens",0):,} output tokens · '+('finished' if s.get('finished') else 'unfinished')
     columns.append(figure(s.get('canvas'),label))
    else:columns.append(figure(None,model+': pending'))
   cards.append(f'<article data-kind="{task["kind"]}" data-k="{k}"><h2>{html.escape(task["family"])} · {task["kind"]} · sample {k}</h2><div>{"".join(columns)}</div></article>')
 judge=root/'judging/summary.json';judgment='<p>Pairwise judging pending.</p>'
 if judge.exists():
  j=json.loads(judge.read_text());judgment=f'<p>Frozen judge, both A/B orders: {j["wins"]} trained wins, {j["ties"]} ties, {j["losses"]} losses, {j["masked"]} masked. These are model judgments, not a human-qualified quality win rate.</p>'
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery curriculum comparison</title><style>body{font:16px/1.5 system-ui;background:#f4f1e9;color:#292822;margin:24px}article{padding:16px;background:white;margin:20px 0}article>div{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%}.pending{aspect-ratio:1;background:#faf8f2;display:grid;place-items:center}select{padding:8px}article{display:none}article[data-kind="paint"][data-k="0"]{display:block}@media(max-width:800px){article>div{grid-template-columns:repeat(2,1fr)}}</style><h1>Recovery quality curriculum</h1><p>Reference → starting canvas → previous adapter → recovery adapter. Both models use the same updated scaffold. These twelve references were excluded from training. This tests procedural reference reconstruction, repair, and keeping completed paintings; photograph generalisation is not established. Click any image to inspect it at full size.</p>'''+judgment+'''<label>Task <select id="kind"><option>paint</option><option>repair</option><option>finish</option><option>all</option></select></label> <label>Sample <select id="sample"><option>0</option><option>1</option><option>all</option></select></label>'''+''.join(cards)+'''<script>function filter(){let kind=document.querySelector('#kind').value,k=document.querySelector('#sample').value;document.querySelectorAll('article').forEach(x=>x.style.display=(kind==='all'||x.dataset.kind===kind)&&(k==='all'||x.dataset.k===k)?'block':'none')}document.querySelector('#kind').onchange=filter;document.querySelector('#sample').onchange=filter</script>'''
 (out/'index.html').write_text(page);return out/'index.html'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(build(a.root))
