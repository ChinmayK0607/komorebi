"""Compact direct-coordinate training pool plus training-only student repairs."""
import json,hashlib
from pathlib import Path
from contract import validate_teacher_program,identity

def build(root):
 root=Path(root);old=json.loads((root/'recovery-pools.json').read_text())
 pools={'paint':old['paint'],'repair':list(old['repair']),'finish':old['finish']}
 # The current parent supplies states on training references only; development
 # rollouts never enter this pool. Exact reference programs supply the correction.
 lookup={r['source_group']:r for r in old['paint'] if not r.get('canvas')}
 fresh=[]
 for path in sorted((root/'training-states/parent').glob('*/sample-*/summary.json')):
  state=json.loads(path.read_text());identifier=path.parent.parent.name
  ref=lookup[identifier.removesuffix('-paint')]
  for turn in state.get('turns',[]):
   if not turn.get('valid') or turn.get('action')!='paint':continue
   directory=path.parent/f'turn-{turn["turn"]}';program=directory/'program.js';canvas=directory/'painting.png'
   if not program.exists() or not canvas.exists():continue
   if canvas.read_bytes()==(root/ref['reference']).read_bytes():continue
   row=dict(ref,id=ref['source_group']+f'-fresh-{path.parent.name}-t{turn["turn"]}',kind='repair',
       program=str(program.relative_to(root)),canvas=str(canvas.relative_to(root)),
       feedback='Inspect the reference and this actual student canvas. Correct the remaining geometry or appearance while preserving correct parts.',
       provenance=str((directory/'result.json').relative_to(root)))
   fresh.append(row)
 # Give fresh states half the repair pool, without confusing repeated draws with
 # independent references. Dataset IDs are made unique by the existing sampler.
 if fresh:pools['repair']=fresh*max(1,len(old['repair'])//len(fresh))+old['repair']
 manifest={r['id']:r for r in [json.loads(s) for s in (root/'data/manifest.jsonl').read_text().splitlines()]}
 validation=[]
 for task in json.loads((root/'legacy-evaluation-tasks.json').read_text()):
  ref=manifest[task['source_group']]
  validation.append(dict(task,split='dev',target_program='data/'+ref['programs'][-1],plan='Match the complete reference geometry, placement, values and brushwork.'))
 train_groups={r['source_group'] for pool in pools.values() for r in pool};dev_groups={r['source_group'] for r in json.loads((root/'evaluation-tasks.json').read_text())}
 assert not train_groups&dev_groups
 for pool in pools.values():
  for r in pool:
   validate_teacher_program((root/r['target_program']).read_text())
   for key in ['reference','target_program','canvas','program']:
    if r.get(key):assert (root/r[key]).is_file(),r[key]
 (root/'contract-data-receipt.json').write_text(json.dumps(dict(contract=identity(),pool_counts={k:len(v) for k,v in pools.items()},fresh_student_repairs=len(fresh),training_groups=len(train_groups),evaluation_groups=len(dev_groups),overlap=[],teacher_scope='Existing compact procedural teachers across 12 families; no photograph generalisation claim'),indent=2)+'\n')
 return pools,validation
