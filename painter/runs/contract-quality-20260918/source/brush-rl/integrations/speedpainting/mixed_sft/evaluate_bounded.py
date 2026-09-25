#!/usr/bin/env python3
"""Collect four fixed matched-seed model outputs; validate JSON without rendering."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from urllib.error import HTTPError
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.request import Request,urlopen

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_replay',HERE.parent/'replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
spec=importlib.util.spec_from_file_location('bounded_contracts',HERE.parent/'contracts.py')
contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)


def sha(raw):return hashlib.sha256(raw).hexdigest()

def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def validate_manifest(manifest):
    if manifest.get('version')!=1 or len(manifest.get('examples',[]))!=4:raise ValueError('Exactly four frozen held-out examples required')
    ids=set()
    for row in manifest['examples']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',row['id']) or row['id'] in ids:raise ValueError('Unsafe or duplicate example ID')
        ids.add(row['id'])
        if row.get('split') not in ('val','validation','test'):raise ValueError('Held-out split required')
        messages=row['messages']
        if [m['role'] for m in messages]!=['system','user']:raise ValueError('Only system/user prompts allowed; no target answer')
        if any(not isinstance(m['content'],list) for m in messages):raise ValueError('Typed content lists required')
        urls=[p['image_url']['url'] for m in messages for p in m['content'] if p.get('type')=='image_url']
        if len(urls)!=1 or not urls[0].startswith(('data:image/png;base64,','data:image/jpeg;base64,')):raise ValueError('One embedded reference required')
        budget=row['budget']
        if type(budget)!=int or budget<=0 or budget>256 or budget%4:raise ValueError('Invalid fixed host budget')
        if contracts.requested_contract(row) != 'bounded-brush-v1':
            raise ValueError('Legacy four-case evaluation supports bounded-brush-v1 only')
    return manifest


def freeze(dataset,output,budget=64,provenance=None):
    raw=(dataset/'validation.jsonl').read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    rows.sort(key=lambda r:sha(r['id'].encode()))
    if len(rows)<4:raise ValueError('Need four validation examples')
    groups = None
    if provenance is not None:
        records=[json.loads(line) for line in provenance.read_text().splitlines() if line.strip()]
        groups={r['id']:r['reference_photo'] for r in records}
        if any(r['id'] not in groups for r in rows):raise ValueError('Incomplete photo provenance')
    selected=[]
    seen_groups=set()
    for row in rows:
        group=groups[row['id']] if groups else row['id']
        if group in seen_groups:continue
        seen_groups.add(group)
        messages=copy.deepcopy(row['messages'])
        if messages[-1]['role']!='assistant':raise ValueError('Expected dataset assistant target to remove')
        messages.pop()
        selected.append({'id':row['id'],'split':'validation','messages':messages,'budget':row.get('budget',budget)})
        if len(selected)==4:break
    manifest=validate_manifest({'version':1,'validation_file_sha256':sha(raw),'examples':selected})
    if provenance is not None:
        manifest.update(provenance_sha256=sha(provenance.read_bytes()),distinct_reference_photo_groups=len(seen_groups))
    if output.exists():raise ValueError('Refusing to overwrite frozen panel')
    output.parent.mkdir(parents=True,exist_ok=True);write(output,manifest)
    return manifest


def evaluate(manifest,base_url,model,output,max_tokens=24576,concurrency=4,request_timeout=1800):
    if not 1<=concurrency<=4 or not 1<=request_timeout<=1800:raise ValueError("Invalid concurrency/deadline")
    if not 1<=max_tokens<=30000:raise ValueError("Invalid completion budget")
    validate_manifest(manifest)
    if output.exists():raise ValueError('Use a fresh evaluation output directory')
    output.mkdir(parents=True)
    settings={'seed':42,'temperature':.3,'max_tokens':max_tokens,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
    write(output/'evaluation-manifest.json',{'input_manifest_sha256':sha(encode(manifest)),'model':model,'sampling':settings,'concurrency':concurrency,'request_timeout_seconds':request_timeout,'examples':manifest['examples'],'rendered':False})
    def run_one(row):
        started=time.monotonic()
        folder=output/row['id'];folder.mkdir()
        payload={'model':model,'messages':row['messages'],**settings}
        write(folder/'request.json',payload)
        result={'id':row['id'],'budget':row['budget'],'valid':False,'rendered':False,'infrastructure_error':None}
        try:
            request=Request(base_url.rstrip('/')+'/chat/completions',data=encode(payload),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=request_timeout) as response:response_raw=response.read()
            (folder/'response.raw').write_bytes(response_raw)
            response_body=json.loads(response_raw)
            write(folder/'response.json',response_body)
            choice=response_body['choices'][0];text=choice['message'].get('content')
            if not isinstance(text,str):raise ValueError('Missing text content')
            (folder/'raw.txt').write_text(text)
            result.update(finish_reason=choice.get('finish_reason'),usage=response_body.get('usage'),raw_sha256=sha(text.encode()))
        except HTTPError as exc:
            (folder/'http-error.raw').write_bytes(exc.read())
            result['infrastructure_error']=type(exc).__name__+': '+str(exc)
        except Exception as exc:
            result['infrastructure_error']=type(exc).__name__+': '+str(exc)
        else:
            try:
                if result.get('finish_reason')=='length':raise replay.InvalidActions('Completion token limit reached; truncated output is invalid even if JSON parses')
                doc,costs=replay.validate(text,row['budget'],expected_contract='bounded-brush-v1')
                write(folder/'actions.json',doc)
                result.update(valid=True,used_budget=sum(costs),action_count=len(doc['actions']))
            except replay.InvalidActions as exc:
                result['validation_error']=str(exc)
        result['elapsed_seconds']=time.monotonic()-started
        write(folder/'result.json',result)
        return result
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results=list(executor.map(run_one,manifest['examples']))
    summary={'model':model,'examples':len(results),'valid':sum(r['valid'] for r in results),'infrastructure_errors':sum(r['infrastructure_error'] is not None for r in results),'results':results,'note':'Format validity only; no rendered visual quality was evaluated. Seed matching does not guarantee bitwise inference reproducibility.'}
    write(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--freeze-from',type=Path);p.add_argument('--provenance',type=Path);p.add_argument('--budget',type=int,default=64);p.add_argument('--base-url');p.add_argument('--model');p.add_argument('--output',type=Path);p.add_argument('--max-tokens',type=int,default=24576);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--request-timeout',type=int,default=1800);a=p.parse_args()
    if a.freeze_from:
        freeze(a.freeze_from,a.manifest,a.budget,a.provenance);print(str(a.manifest.resolve()));return
    if not a.base_url or not a.model or not a.output:p.error('Evaluation needs --base-url --model --output')
    result=evaluate(json.loads(a.manifest.read_text()),a.base_url,a.model,a.output,a.max_tokens,a.concurrency,a.request_timeout);print(json.dumps({'valid':result['valid'],'examples':4,'infrastructure_errors':result['infrastructure_errors']}))
    if result['infrastructure_errors']:raise SystemExit(1)

if __name__=='__main__':main()
