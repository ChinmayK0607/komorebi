"""Node-only materialization/rendering of direct-final diversity examples."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from render_demos import atomic


def run(root,workers=12):
    assert sys.platform=='linux','Painting rendering belongs on the node.'
    root=Path(root).resolve();data=root/'diversity';os.umask(0)
    rows=[json.loads(x) for x in (data/'programs.jsonl').read_text().splitlines()]
    preview=[]
    for family in sorted({r['primary_family'] for r in rows}):
        candidates=[r for r in rows if r['primary_family']==family]
        for layout in ['single','overlap','tilted']:
            preview.append(next(r['id'] for r in candidates if r['layout']==layout))
    preview=set(preview);jobs=[];metadata=[]
    for row in rows:
        folder=data/row['id'];folder.mkdir(exist_ok=True);folder.chmod(0o777)
        for kind,source in row['programs'].items():
            path=folder/(kind+'.js');path.write_text(source);path.chmod(0o644)
            jobs.append((0 if row['id'] in preview else 1,row['id'],kind,path))
        metadata.append({k:v for k,v in row.items() if k!='programs'}|
                        {'programs':{k:str(Path(row['id'])/(k+'.js')) for k in row['programs']}})
    jobs.sort(key=lambda j:j[:3]);(data/'materialized-manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metadata))
    atomic(data/'review-selection.json',dict(ids=sorted(preview),reason='Every family: single, overlapping scene, tilted; selected before renders'))
    def one(job):
        priority,identifier,kind,source=job;png=source.with_suffix('.png');receipt=source.with_suffix('.json')
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if receipt.exists() and png.exists():
            old=json.loads(receipt.read_text())
            if old.get('valid') and old.get('source_sha256')==sha and old.get('png_sha256')==hashlib.sha256(png.read_bytes()).hexdigest():
                return dict(id=identifier,kind=kind,valid=True,resumed=True)
        for p in [receipt,png]:
            if p.exists():p.rename(p.with_name(p.name+f'.previous-{time.time_ns()}'))
        command=['/usr/sbin/runuser','-u','painter','--',str(root/'renderer-env/bin/python'),str(root/'vendor/integrations/watercolour/renderer.py'),str(source),'--output',str(png),'--backend','swiftshader','--timeout','120']
        result=subprocess.run(command,text=True,capture_output=True,timeout=145)
        evidence=json.loads(receipt.read_text()) if receipt.exists() else dict(valid=False,error=result.stderr[-1000:])
        return dict(id=identifier,kind=kind,valid=bool(evidence.get('valid')),resumed=False,error=evidence.get('error'))
    results=[];review_ready=False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:record=f.result()
            except Exception as exc:record=dict(id=job[1],kind=job[2],valid=False,error=f'{type(exc).__name__}: {exc}')
            results.append(record)
            atomic(data/'render-progress.json',dict(completed=len(results),expected=len(jobs),valid=sum(x['valid'] for x in results),workers=workers))
            if not review_ready and sum(x['id'] in preview for x in results)==len(preview)*3:
                atomic(data/'review-ready.json',dict(ids=sorted(preview),valid=sum(x['valid'] for x in results if x['id'] in preview),expected=len(preview)*3));review_ready=True
    valid={(x['id'],x['kind']) for x in results if x['valid']}
    accepted=[r for r in metadata if all((r['id'],k) in valid for k in r['programs'])]
    (data/'rendered-manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
    atomic(data/'render-summary.json',dict(references=len(rows),accepted_references=len(accepted),expected=len(jobs),valid=len(valid),complete=len(results)==len(jobs),records=results,visual_review_pending=True))
    return dict(references=len(rows),accepted_references=len(accepted),valid=len(valid),expected=len(jobs))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    print(json.dumps(run(args.root,args.workers)))
