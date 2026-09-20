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
  for k in range(4):
   columns=[figure(task['reference'],'Reference'),figure(task.get('canvas'),'Starting canvas')]
   for model in ['parent','midpoint','trained']:
    f=root/'evaluation'/model/task['id']/f'sample-{k:02d}/summary.json'
    if f.exists():
     s=json.loads(f.read_text());label=f'{model} · {len(s.get("turns",[]))} turns · {s.get("generated_tokens",0):,} output tokens · '+('finished' if s.get('finished') else 'unfinished')
     columns.append(figure(s.get('canvas'),label))
    else:columns.append(figure(None,model+': pending'))
   group='New composition' if task['id'].startswith('v3-') else 'Previous reference'
   cards.append(f'<article data-kind="{task["kind"]}" data-k="{k}"><h2>{html.escape(task["family"])} · {group} · {task["kind"]} · sample {k}</h2><p>{html.escape(task["id"])}</p><div>{"".join(columns)}</div></article>')
 judge=root/'luna-summary.json';judgment='<p>Luna visual review pending.</p>'
 if judge.exists():
  j=json.loads(judge.read_text());judgment=f'<p>Painting tasks: final step72 versus parent: {j["wins"]} wins, {j["ties"]} ties, {j["losses"]} losses, {j["masked"]} uncertain or low-confidence pairs. Luna inspected all 384 blinded painting, repair and finish comparisons. These are model judgments.</p>'
  judgment+='<table><thead><tr><th>Candidate vs anchor</th><th>Reference group</th><th>Wins</th><th>Losses</th><th>Ties</th><th>Masked</th></tr></thead><tbody>'
  for comparison,title in [('parent_vs_midpoint','Midpoint vs parent'),('parent_vs_trained','Final vs parent'),('midpoint_vs_trained','Final vs midpoint')]:
   for group,label in [('previous','Previous references'),('new_composition','New compositions')]:
    x=j['comparisons'][comparison]['painting_by_group'][group];judgment+=f'<tr><td>{title}</td><td>{label} (painting)</td><td>{x["wins"]}</td><td>{x["losses"]}</td><td>{x["ties"]}</td><td>{x["masked"]}</td></tr>'
  judgment+='</tbody></table><p>This run requires observing a newly painted canvas before finishing. Inspect the rollout records to distinguish useful revisions from finishing without changes.</p>'
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Contract-quality comparison</title><style>body{font:16px/1.5 system-ui;background:#f4f1e9;color:#292822;margin:24px}article{padding:16px;background:white;margin:20px 0}article>div{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}figure{margin:0}img{width:100%}.pending{aspect-ratio:1;background:#faf8f2;display:grid;place-items:center}select{padding:8px}article{display:none}article[data-kind="paint"][data-k="0"]{display:block}@media(max-width:800px){article>div{grid-template-columns:repeat(2,1fr)}}</style><h1>Contract-quality checkpoint comparison</h1><p>Reference → starting canvas → recovery initializer → step24 → step72. All three checkpoints use the same scaffold. These old and new development references were excluded from training. This tests procedural reference reconstruction, repair, and keeping completed paintings; photograph generalisation is not established. Click any image to inspect it at full size.</p>'''+judgment+'''<label>Task <select id="kind"><option>paint</option><option>repair</option><option>finish</option><option>all</option></select></label> <label>Sample <select id="sample"><option>0</option><option>1</option><option>2</option><option>3</option><option>all</option></select></label>'''+''.join(cards)+'''<script>function filter(){let kind=document.querySelector('#kind').value,k=document.querySelector('#sample').value;document.querySelectorAll('article').forEach(x=>x.style.display=(kind==='all'||x.dataset.kind===kind)&&(k==='all'||x.dataset.k===k)?'block':'none')}document.querySelector('#kind').onchange=filter;document.querySelector('#sample').onchange=filter</script>'''
 (out/'index.html').write_text(page);return out/'index.html'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(build(a.root))
