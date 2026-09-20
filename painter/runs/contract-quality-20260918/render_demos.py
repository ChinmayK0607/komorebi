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

from contract import SYSTEM, NEXT_VERSION, NO_CHANGE, identity, paint_target, finish_target


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
            answer=paint_target(row['plans'][n], (data/source).read_text())
            sft.append(dict(contract=identity(), id=f"{row['id']}-turn-{n+1}",source_group=row['source_group'],split=row['split'],level=row['level'],
                family=row['family'],reference_kind=row['reference_kind'],teacher=row['teacher'],
                messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),dict(role='user',content=parts),
                          dict(role='assistant',content=[dict(type='text',text=answer)])]))
        final=data/row['programs'][-1]
        parts=[dict(type='text',text='REFERENCE'),image(ref),
               dict(type='text',text='CURRENT CANVAS'),image(final.with_suffix('.png')),
               dict(type='text',text='Current program:\n'+final.read_text()),
               dict(type='text',text=NEXT_VERSION)]
        sft.append(dict(id=row['id']+'-finish', contract=identity(), source_group=row['source_group'],
            split=row['split'], level=row['level'], family=row['family'],
            reference_kind=row['reference_kind'], teacher=row['teacher'],
            messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),
                      dict(role='user',content=parts),
                      dict(role='assistant',content=[dict(type='text',text=finish_target())])]))
    for split in ('train','dev'):
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sft if r['split']==split))
    summary=dict(complete=len(records)==len(rows)*3,expected=len(rows)*3,valid=sum(r['valid'] for r in records),
        records=records,sft_train=sum(r['split']=='train' for r in sft),sft_dev=sum(r['split']=='dev' for r in sft),
        visual_review_pending=True,quality_claim=False)
    atomic(data/'render-summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--renderer',required=True);p.add_argument('--python',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.data,a.renderer,a.python,a.workers)
