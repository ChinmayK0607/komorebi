"""Resume individual teacher renders on a Linux node; export actual SFT examples."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SYSTEM='''Reproduce the supplied reference as an attractive painting using p5.brush JavaScript.
Use p5.js 2.3.2 and p5.brush 2.2.1 with a 600x600 WEBGL canvas. Coordinates initially have origin at the center; translate(-300,-300) once in draw() for a top-left origin.
You may use variables, functions, loops, paths and layered brushwork.
Useful brush API: brush.noStroke(); brush.noWash(); brush.fill("#688a9b",180); brush.fillBleed(0.03,"out"); brush.fillTexture(0.2,0.1); brush.polygon([[x1,y1],[x2,y2],[x3,y3]]).
For lines: brush.noFill(); brush.noWash(); brush.set("cpencil","#554d44",0.8); brush.line(x1,y1,x2,y2). Reset inactive paint modes when switching. Prefer brush.fill for polygons; wash-to-pencil transitions have lost a pending face in this runtime.
Inspect the reference and current canvas. Preserve correct structure while improving the painting.
Do not embed images, copy pixels, write labels in place of objects, or access external resources.
There is no target stroke count or speed reward in this quality phase. Finish when the painting is good.
Return a brief visual plan followed by a complete sketch inside a javascript code block.
Use setup() and draw(); finish draw() with noLoop(). You will receive the actual render to revise.
When the painting is complete, include FINISHED on its own line outside the code block.'''


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def image(path):return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()))


def main(data,renderer,python,workers=4):
    if sys.platform!='linux':raise RuntimeError('Render on the node; local work is code and metadata only.')
    data=Path(data).resolve();renderer=Path(renderer).resolve();rows=[json.loads(s) for s in (data/'manifest.jsonl').read_text().splitlines()]
    records=[]
    def one(relative):
        source=data/relative;png=source.with_suffix('.png');receipt=source.with_suffix('.json')
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if png.exists() and receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous.get('valid') and previous.get('source_sha256')==sha and previous.get('png_sha256')==hashlib.sha256(png.read_bytes()).hexdigest():
                return dict(source=relative,valid=True,resumed=True)
        # Preserve failures before a retry; never change the already recorded response.
        for f in (png,receipt):
            if f.exists():f.rename(f.with_name(f.name+f'.previous-{time.time_ns()}'))
        job=subprocess.run(['/usr/sbin/runuser','-u','painter','--',str(Path(python).absolute()),str(renderer),str(source),
           '--output',str(png),'--backend','swiftshader','--timeout','120'],capture_output=True,text=True,timeout=140)
        result=json.loads(receipt.read_text()) if receipt.exists() else dict(valid=False,error=job.stderr[-1500:])
        return dict(source=relative,valid=bool(result.get('valid')),resumed=False,error=result.get('error'))
    futures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in rows:
            for source in row['programs']:futures[pool.submit(one,source)]=source
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as exc:result=dict(source=futures[future],valid=False,error=f'{type(exc).__name__}: {exc}')
            records.append(result);atomic(data/'render-progress.json',dict(completed=len(records),expected=len(rows)*3,valid=sum(r['valid'] for r in records)))
    valid={r['source'] for r in records if r['valid']};sft=[]
    (data/'rendered-manifest.jsonl').write_text(''.join(json.dumps(dict(row, validity='rendered_valid' if all(s in valid for s in row['programs']) else 'render_failed'))+'\n' for row in rows))
    for row in rows:
        if not all(s in valid for s in row['programs']):continue
        ref=data/row['reference']
        for n,source in enumerate(row['programs']):
            parts=[dict(type='text',text='REFERENCE'),image(ref)]
            if n:
                previous=data/row['programs'][n-1]
                parts += [dict(type='text',text='CURRENT CANVAS'),image(previous.with_suffix('.png')),
                          dict(type='text',text='Current program:\n'+previous.read_text())]
            parts.append(dict(type='text',text='Inspect the images and paint the next complete version.'))
            answer=row['plans'][n]+'\n```javascript\n'+(data/source).read_text()+'```'+('\nFINISHED' if n==len(row['programs'])-1 else '')
            sft.append(dict(id=f"{row['id']}-turn-{n+1}",source_group=row['source_group'],split=row['split'],level=row['level'],
                family=row['family'],reference_kind=row['reference_kind'],teacher=row['teacher'],
                messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),dict(role='user',content=parts),
                          dict(role='assistant',content=[dict(type='text',text=answer)])]))
    for split in ('train','dev'):
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sft if r['split']==split))
    summary=dict(complete=len(records)==len(rows)*3,expected=len(rows)*3,valid=sum(r['valid'] for r in records),
        records=records,sft_train=sum(r['split']=='train' for r in sft),sft_dev=sum(r['split']=='dev' for r in sft),
        visual_review_pending=True,quality_claim=False)
    atomic(data/'render-summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--renderer',required=True);p.add_argument('--python',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.data,a.renderer,a.python,a.workers)
