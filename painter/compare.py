"""Persist every judge reply; resume pairs independently. Execute on the node."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
from client import completion
from pairwise import PROMPT,reconcile
from render_demos import atomic,image

SCHEMA=dict(type='object',properties=dict(winner=dict(type='string',enum=['A','B','tie','uncertain']),
    acceptable_A=dict(type='boolean'),acceptable_B=dict(type='boolean'),reason=dict(type='string')),
    required=['winner','acceptable_A','acceptable_B','reason'],additionalProperties=False)


def compare(row,endpoint,model,out):
    paths={k:Path(row[k]).resolve(strict=True) for k in ['reference','candidate','anchor']}
    binding=dict(model=model,endpoint=endpoint,prompt=PROMPT,task=row.get('task','Reference-conditioned painting.'),images={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    folder=Path(out)/key;folder.mkdir(parents=True,exist_ok=True);atomic(folder/'binding.json',binding)
    answers=[]
    for order in [('candidate','anchor'),('anchor','candidate')]:
        name='AB' if order[0]=='candidate' else 'BA';saved=folder/(name+'.json')
        if saved.exists():result=json.loads(saved.read_text())
        else:
            parts=[dict(type='text',text=PROMPT+'\nTask: '+row.get('task','Reference-conditioned painting.')),image(paths['reference']),image(paths[order[0]]),image(paths[order[1]])]
            result=completion(endpoint,model,[dict(role='user',content=parts)],1024,temperature=0,schema=SCHEMA);atomic(saved,result)
        try:
            choice=result['response']['choices'][0]
            answer=choice['message']['content'] if choice['finish_reason']=='stop' else None
        except (KeyError,TypeError,IndexError):answer=None
        answers.append(answer)
    result=dict(id=row['id'],**reconcile(*answers),evidence=str(folder));atomic(folder/'result.json',result);return result


if __name__=='__main__':
    if sys.platform!='linux':raise RuntimeError('Run judging on the node.')
    p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--concurrency',type=int,default=4);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.pairs).read_text().splitlines()];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);results=[]
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures={pool.submit(compare,r,a.endpoint,a.model,out):r for r in rows}
        for f in as_completed(futures):
            try:r=f.result()
            except Exception as exc:r=dict(id=futures[f]['id'],reward=None,mask=True,error=f'{type(exc).__name__}: {exc}')
            results.append(r);atomic(out/'progress.json',dict(completed=len(results),expected=len(rows),results=results))
    atomic(out/'summary.json',dict(complete=len(results)==len(rows),results=results))
