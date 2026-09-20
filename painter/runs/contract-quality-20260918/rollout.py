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
from render_demos import atomic,image
from contract import SYSTEM,NEXT_VERSION,NO_CHANGE,identity


def sketch(reply):
    blocks=re.findall(r'```[ \t]*(?:javascript|js)?[ \t]*\r?\n(.*?)```',reply,re.S)
    if not blocks:raise ValueError('No complete JavaScript code block')
    return blocks[-1]


def run(reference,output,endpoint,model,renderer,renderer_python,config,seed=917,initial_canvas=None,initial_program=None):
    if sys.platform!='linux':raise RuntimeError('Run models and painting renders on the node.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    reference=Path(reference).resolve(strict=True);limits=config['operational_limits'];start=time.monotonic()
    binding=dict(harness_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('rollout.py','client.py','contract.py')},reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),model=model,endpoint=endpoint,seed=seed,
      contract=identity(),renderer_assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(renderer).parent/'assets').glob('*.js'))},renderer_sha256=hashlib.sha256(Path(renderer).read_bytes()).hexdigest(),
      config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),system=SYSTEM)
    if bool(initial_canvas)!=bool(initial_program):raise ValueError('Initial canvas and program must be supplied together')
    if initial_canvas:
        initial_canvas=Path(initial_canvas).resolve(strict=True);initial_program=Path(initial_program).resolve(strict=True)
        binding.update(initial_canvas_sha256=hashlib.sha256(initial_canvas.read_bytes()).hexdigest(),initial_program_sha256=hashlib.sha256(initial_program.read_bytes()).hexdigest())
    bound=output/'binding.json'
    if bound.exists() and json.loads(bound.read_text())!=binding:raise ValueError('Use a new output directory for a changed rollout')
    atomic(bound,binding);canvas=initial_canvas;program=initial_program.read_text() if initial_program else '';turns=[];finished=False;feedback=NEXT_VERSION
    for index in range(limits['max_turns']):
        if time.monotonic()-start>=limits['episode_timeout_seconds']:break
        turn=output/f'turn-{index+1}';turn.mkdir(exist_ok=True);os.chmod(output,0o755);os.chmod(turn,0o777)
        content=[dict(type='text',text='REFERENCE'),image(reference)]
        if canvas:content += [dict(type='text',text='CURRENT CANVAS'),image(canvas),dict(type='text',text='Current program:\n'+program)]
        content.append(dict(type='text',text=feedback))
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
        saved=turn/'response.json'
        result=json.loads(saved.read_text()) if saved.exists() else completion(endpoint,model,messages,limits['max_tokens_per_response'],seed=seed+index)
        atomic(saved,result);record=dict(turn=index+1,valid=False,seconds=result.get('seconds'),observed_canvas=str(canvas) if canvas else None)
        try:
            choice=result['response']['choices'][0];raw=choice['message']['content'];record['tokens']=result['response'].get('usage',{}).get('completion_tokens')
            if choice['finish_reason']!='stop':raise ValueError('Operational token cutoff')
            stop=bool(re.search(r'^FINISHED\s*$',re.sub(r'```.*?```','',raw,flags=re.S),re.M))
            if stop and '```' not in raw:
                if canvas is None:raise ValueError('Cannot finish without a current canvas; return a complete painting program')
                record.update(valid=True,action='finish',finished=True,canvas=str(canvas),no_change=True,render_seconds=0.)
                finished=True;turns.append(record);atomic(turn/'result.json',record)
                atomic(output/'progress.json',dict(turns=turns,finished=True));break
            next_program=sketch(raw);source=turn/'program.js';source.write_text(next_program);os.chmod(source,0o644)
            if canvas is not None and next_program.strip()==program.strip():
                record.update(valid=True,action='paint',finished=stop,canvas=str(canvas),no_change=True,render_seconds=0.)
                finished=stop;feedback=NO_CHANGE;turns.append(record);atomic(turn/'result.json',record)
                atomic(output/'progress.json',dict(turns=turns,finished=finished))
                if finished:break
                continue
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
            if receipt_data.get('png_sha256')!=hashlib.sha256(png.read_bytes()).hexdigest():raise ValueError('Render PNG hash mismatch')
            unchanged=canvas is not None and hashlib.sha256(canvas.read_bytes()).digest()==hashlib.sha256(png.read_bytes()).digest()
            canvas=png;program=next_program;record.update(valid=True,action='paint',finished=False,finish_deferred=stop,canvas=str(canvas),no_change=unchanged,render_seconds=receipt_data.get('painting_seconds'))
            # Even code + FINISHED must first return its actual canvas to the model.
            finished=False
            feedback=NO_CHANGE if unchanged else NEXT_VERSION
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'[:2000]
            feedback='The previous proposal did not complete successfully: '+record['error']+'. Correct it; the last valid canvas is unchanged.'
        turns.append(record);atomic(turn/'result.json',record)
        atomic(output/'progress.json',dict(turns=turns,finished=finished))
        if finished:break
    summary=dict(contract=identity(),model=model,reference=str(reference),finished=finished,canvas=str(canvas) if canvas else None,valid_canvas=canvas is not None,
        operational_cutoff=not finished,seconds=time.monotonic()-start,turns=turns,generated_tokens=sum(r.get('tokens') or 0 for r in turns),
        speed_reward=0.,quality_judged=False)
    atomic(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','output','endpoint','model','renderer','renderer-python']:p.add_argument('--'+name,required=True)
    p.add_argument('--config',default=str(Path(__file__).with_name('campaign.json')));p.add_argument('--seed',type=int,default=917);a=p.parse_args()
    print(json.dumps(run(a.reference,a.output,a.endpoint,a.model,a.renderer,a.renderer_python,json.loads(Path(a.config).read_text()),a.seed),indent=2))
