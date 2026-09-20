"""Independent node-only representation parity experiment; no learning dependency."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent;assert sys.platform=='linux'
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(P/'browsers')
node=next((P/'renderer-env').glob('lib/python*/site-packages/playwright/driver/node'))
out=P/'compiler-parity';out.mkdir(exist_ok=True);out.chmod(0o777)
canonical=out/'canonical.jsonl'
if not canonical.exists():subprocess.run([str(node),str(P/'compile_teacher.cjs'),str(P/'compiler-fixtures.jsonl'),str(canonical)],check=True)
original={r['id']:r for r in map(json.loads,(P/'compiler-fixtures.jsonl').read_text().splitlines())}
compiled=list(map(json.loads,canonical.read_text().splitlines()));jobs=[]
for r in compiled:
 for version,source in [('original',original[r['id']]['programs']['reference']),('canonical',r['programs']['reference'])]:
  file=out/(r['id']+'-'+version+'.js');file.write_text(source);file.chmod(0o644);jobs.append(file)
def one(source):
 png=source.with_suffix('.png');receipt=source.with_suffix('.json')
 if not receipt.exists():
  subprocess.run(['/usr/sbin/runuser','-u','painter','--',str(P/'renderer-env/bin/python'),str(P/'vendor/integrations/watercolour/renderer.py'),str(source),'--output',str(png),'--backend','swiftshader','--timeout','120'],capture_output=True,timeout=145)
 return source.name,json.loads(receipt.read_text()) if receipt.exists() else {'valid':False}
with ThreadPoolExecutor(max_workers=6) as pool:results=dict(pool.map(one,jobs))
rows=[]
for r in compiled:
 a=out/(r['id']+'-original.png');b=out/(r['id']+'-canonical.png')
 rows.append(dict(id=r['id'],original_valid=results[a.with_suffix('.js').name]['valid'],canonical_valid=results[b.with_suffix('.js').name]['valid'],identical_png=a.exists() and b.exists() and a.read_bytes()==b.read_bytes(),**r['compilation']['reference']))
report=dict(references=len(rows),identical_renders=sum(r['identical_png'] for r in rows),rows=rows,training_dependency=False)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
