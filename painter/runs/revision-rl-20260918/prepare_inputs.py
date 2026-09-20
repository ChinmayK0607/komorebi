"""Stage the existing portable node bundle and training-only canvas states."""
from pathlib import Path
import json,hashlib,tarfile,shutil
P=Path(__file__).resolve().parent
old=P.parent/'contract-quality-20260918'
evidence=old/'collected-final/evidence'
root=P/'inputs';root.mkdir(exist_ok=True)
with tarfile.open(old/'run-inputs.tar.gz') as t:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
 t.extractall(root,filter='data')
heldout={t['source_group'] for t in json.loads((evidence/'evaluation-tasks.json').read_text())}
tasks=json.loads((evidence/'training-state-tasks.json').read_text())
assert len(tasks)==24 and not heldout & {t['source_group'] for t in tasks}
def source(path):
 return evidence/str(path).removeprefix('/root/painter/')
rows=[]
for i,t in enumerate(sorted(tasks,key=lambda x:(x['family'],x['id']))):
 s=json.loads((evidence/'training-states/parent'/t['id']/'sample-00/summary.json').read_text())
 turns=[v for v in s['turns'] if v.get('valid') and v.get('action')=='paint' and source(v['canvas']).is_file()]
 # Alternate early and final observed states within each family. Never select
 # by development judgments or rewrite the student canvas into a teacher one.
 selected=turns[0] if i%2==0 else turns[-1]
 canvas=source(selected['canvas']);program=canvas.with_name('program.js')
 assert program.is_file()
 folder=root/'rl-states'/f'state-{i:02d}';folder.mkdir(parents=True,exist_ok=True)
 paths={}
 for name,f in [('reference.png',source(t['reference'])),('canvas.png',canvas),('program.js',program)]:
  dst=folder/name;shutil.copyfile(f,dst);paths[name]=str(dst.relative_to(root))
 rows.append(dict(id=f'state-{i:02d}',source_group=t['source_group'],family=t['family'],split='train',reference=paths['reference.png'],canvas=paths['canvas.png'],program=paths['program.js'],provenance=dict(run='contract-quality-20260918',policy='recovery-parent',task=t['id'],turn=selected['turn'])))
(root/'rl-states.json').write_text(json.dumps(rows,indent=2)+'\n')
receipt=json.loads((evidence/'train-output/contract-quality-20260918/artifacts/adapters/step_24/hf-upload.json').read_text())
experiment=dict(run_name=P.name,model='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d2dd7c60c0',initial_adapter=dict(repo=receipt['repo_id'],revision=receipt['commit'],prefix=receipt['path'],sha256=receipt['files']['adapter_model.safetensors']['sha256']),initializer_status='provisional step24 pilot; not global promotion',states=24,samples_per_state=8,temperature=1.0,max_tokens=8192,optimizer='Prime RL',learning_rate=1e-6,max_epochs=1,groups_per_update=2,judge='gpt-5.6-luna',judge_reasoning_effort='low',speed_reward=0,training_state_groups=sorted({r['source_group'] for r in rows}),heldout_groups=sorted(heldout))
experiment['revision']='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
(root/'EXPERIMENT.json').write_text(json.dumps(experiment,indent=2)+'\n');(P/'EXPERIMENT.json').write_text(json.dumps(experiment,indent=2)+'\n')
# Never ship a prior pipeline that could accidentally restart SFT.
(root/'pipeline.py').unlink(missing_ok=True)
for name in ['setup_training.sh','download_model.py','stage_model.py','wait_server.py']:
 shutil.copyfile(old/name,root/name)
# Async rollout workers need the renderer and reward protocol on the node
# itself. Keep these beside rl-states.json in the portable input bundle so a
# fresh hardware provider cannot start with a missing scoring import.
assert (root/'contract.py').is_file(), 'Portable archive must include the pinned contract'
for name in ['quality_signal.py','render_actions.py','taskset_impl.py','reward_queue.py']:
 shutil.copyfile(P/name, root/name)
manifest={str(f.relative_to(root)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in root.rglob('*') if f.is_file() and f.name!='input-manifest.json'}
(root/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(states=len(rows),families=len({r['family'] for r in rows}),heldout_overlap=[],files=len(manifest),initializer=experiment['initial_adapter'])))
