"""The same reference -> program -> canvas -> revision loop for teachers and students."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from client import completion
from render_demos import SYSTEM,atomic,image


def sketch(reply):
    blocks=re.findall(r'```(?:javascript|js)?\s*\n(.*?)```',reply,re.S)
    if not blocks:raise ValueError('No complete JavaScript code block')
    return blocks[-1]


def run(reference,output,endpoint,model,renderer,renderer_python,config,seed=917):
    if sys.platform!='linux':raise RuntimeError('Run models and painting renders on the node.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    reference=Path(reference).resolve(strict=True);limits=config['operational_limits'];start=time.monotonic()
    binding=dict(reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),model=model,endpoint=endpoint,seed=seed,
      config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),system=SYSTEM)
    bound=output/'binding.json'
    if bound.exists() and json.loads(bound.read_text())!=binding:raise ValueError('Use a new output directory for a changed rollout')
    atomic(bound,binding);canvas=None;program='';turns=[];finished=False;feedback='Inspect the images and paint the next complete version.'
    for index in range(limits['max_turns']):
        if time.monotonic()-start>=limits['episode_timeout_seconds']:break
        turn=output/f'turn-{index+1}';turn.mkdir(exist_ok=True);os.chmod(output,0o755);os.chmod(turn,0o777)
        content=[dict(type='text',text='REFERENCE'),image(reference)]
        if canvas:content += [dict(type='text',text='CURRENT CANVAS'),image(canvas),dict(type='text',text='Current program:\n'+program)]
        content.append(dict(type='text',text=feedback))
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
        saved=turn/'response.json'
        result=json.loads(saved.read_text()) if saved.exists() else completion(endpoint,model,messages,limits['max_tokens_per_response'],seed=seed+index)
        atomic(saved,result);record=dict(turn=index+1,valid=False,seconds=result.get('seconds'))
        try:
            choice=result['response']['choices'][0];raw=choice['message']['content'];record['tokens']=result['response'].get('usage',{}).get('completion_tokens')
            if choice['finish_reason']!='stop':raise ValueError('Operational token cutoff')
            next_program=sketch(raw);source=turn/'program.js';source.write_text(next_program);os.chmod(source,0o644)
            png=turn/'painting.png';receipt=turn/'painting.json'
            if not png.exists() or not receipt.exists():
                for f in (png,receipt):
                    if f.exists():f.rename(f.with_name(f.name+'.previous-'+str(time.time_ns())))
                subprocess.run(['/usr/sbin/runuser','-u','painter','--',str(Path(renderer_python).absolute()),str(Path(renderer).resolve()),
                    str(source),'--output',str(png),'--timeout',str(limits['render_timeout_seconds']),'--backend','swiftshader'],check=False,capture_output=True,timeout=limits['render_timeout_seconds']+20)
            receipt_data=json.loads(receipt.read_text())
            if not receipt_data.get('valid'):
                record['renderer_errors']=receipt_data.get('errors') or receipt_data.get('error')
                raise ValueError('Renderer: '+str(record['renderer_errors'])[:1600])
            if receipt_data.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('Render belongs to a different source')
            canvas=png;program=next_program;record.update(valid=True,canvas=str(canvas),render_seconds=receipt_data.get('painting_seconds'))
            finished=bool(re.search(r'^FINISHED\s*$',re.sub(r'```.*?```','',raw,flags=re.S),re.M))
            feedback='Inspect the images and paint the next complete version.'
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'[:2000]
            feedback='The previous proposal did not complete successfully: '+record['error']+'. Correct it; the last valid canvas is unchanged.'
        turns.append(record);atomic(turn/'result.json',record)
        atomic(output/'progress.json',dict(turns=turns,finished=finished))
        if finished:break
    summary=dict(model=model,reference=str(reference),finished=finished,canvas=str(canvas) if canvas else None,valid_canvas=canvas is not None,
        operational_cutoff=not finished,seconds=time.monotonic()-start,turns=turns,generated_tokens=sum(r.get('tokens') or 0 for r in turns),
        speed_reward=0.,quality_judged=False)
    atomic(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','output','endpoint','model','renderer','renderer-python']:p.add_argument('--'+name,required=True)
    p.add_argument('--config',default=str(Path(__file__).with_name('campaign.json')));p.add_argument('--seed',type=int,default=917);a=p.parse_args()
    print(json.dumps(run(a.reference,a.output,a.endpoint,a.model,a.renderer,a.renderer_python,json.loads(Path(a.config).read_text()),a.seed),indent=2))
