"""Reference-group-safe oracle corrections from preserved student rollouts.

Build lightweight descriptors locally; materialize multimodal rows on the node.
No inference, image processing, or painting rendering happens here.
"""
import argparse,hashlib,json,random
from pathlib import Path
from render_demos import SYSTEM,NEXT_VERSION,NO_CHANGE,image


def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    root=Path(root).resolve();manifest=read(root/'data/manifest.jsonl')
    old_ids={f.name for f in (root/'previous-evaluation/trained').iterdir() if f.is_dir()}
    lookup={r['id']:r for r in manifest};assert old_ids<=lookup.keys()
    train=[r for r in manifest if r['split']=='train' or r['id'] in old_ids]
    development=[r for r in manifest if r['split']=='dev' and r['id'] not in old_ids]
    selected={}
    for r in development:selected.setdefault(r['family'],r)
    assert len(selected)==12
    pools={'repair':[],'paint':[],'finish':[]};seen=set()

    def row(ref,kind,identifier,canvas=None,program=None,feedback=NEXT_VERSION):
        return dict(id=identifier,source_group=ref['source_group'],family=ref['family'],level=ref['level'],
            split='train',kind=kind,reference='data/'+ref['reference'],canvas=canvas,program=program,
            feedback=feedback,target_program='data/'+ref['programs'][-1],teacher='exact_reference_program',
            plan='Match the reference silhouette, proportions and connected parts; preserve correct structure and complete the missing shapes.')

    for ref in train:
        for stage in [None,0]:
            program='data/'+ref['programs'][stage] if stage is not None else None
            canvas=str(Path(program).with_suffix('.png')) if program else None
            pools['paint'].append(row(ref,'paint',ref['id']+f'-paint-{stage}',canvas,program))
        pools['finish'].append(row(ref,'finish',ref['id']+'-finish','data/'+ref['reference'],'data/'+ref['programs'][-1]))

    for model in ['trained','base']:
        for f in sorted((root/'previous-evaluation'/model).glob('*/sample-*/summary.json')):
            ref=lookup[f.parent.parent.name];assert ref['id'] in old_ids
            summary=json.loads(f.read_text());current_canvas=None;current_program=None;previous_source=None
            for t in summary.get('turns',[]):
                folder=f.parent/f'turn-{t["turn"]}'
                if t.get('valid'):
                    current_canvas=str((folder/'painting.png').relative_to(root))
                    current_program=str((folder/'program.js').relative_to(root))
                    if not (root/current_canvas).is_file():continue
                    source=(root/current_program).read_text()
                    feedback=NO_CHANGE if source==previous_source else NEXT_VERSION
                    previous_source=source
                else:
                    feedback='The previous proposal did not complete successfully: '+t.get('error','invalid program')+'. Correct it; the last valid canvas is unchanged.'
                key=(ref['id'],sha(root/current_program) if current_program else None,feedback)
                if key in seen:continue
                seen.add(key)
                item=row(ref,'repair',ref['id']+f'-{model}-{f.parent.name}-t{t["turn"]}',current_canvas,current_program,feedback)
                item['provenance']=str((folder/'result.json').relative_to(root))
                pools['repair'].append(item)

    # Limit repeated examples from a single source reference, without selecting
    # on future evaluation quality. Keep deterministic family-balanced pools.
    rng=random.Random(9182);repair=[]
    for family in sorted(selected):
        subset=[x for x in pools['repair'] if x['family']==family];rng.shuffle(subset);repair.extend(subset[:32])
    pools['repair']=repair
    train_groups={r['source_group'] for r in train};dev_groups={r['source_group'] for r in development}
    assert not train_groups&dev_groups
    tasks=[]
    for ref in selected.values():
        for task in ['paint','repair','finish']:
            stage=None if task=='paint' else 0 if task=='repair' else 2
            program='data/'+ref['programs'][stage] if stage is not None else None
            tasks.append(dict(id=ref['id']+'-'+task,reference='data/'+ref['reference'],source_group=ref['source_group'],family=ref['family'],kind=task,
                canvas=str(Path(program).with_suffix('.png')) if program else None,program=program))
    write(root/'recovery-pools.json',pools);write(root/'evaluation-tasks.json',tasks)
    write(root/'evaluation-references.json',list(selected.values()))
    split=dict(retired_development_ids=sorted(old_ids),training_references=len(train),remaining_development_references=len(development),
        evaluation_reference_ids=[r['id'] for r in selected.values()],training_groups=sorted(train_groups),development_groups=sorted(dev_groups),
        source_group_overlap=[],counts={k:len(v) for k,v in pools.items()},note='Former evaluation references supplying student states are training-only in this run.')
    write(root/'recovery-data-receipt.json',split)
    return split


def materialize(root,row):
    root=Path(root);parts=[dict(type='text',text='REFERENCE'),image(root/row['reference'])]
    if row.get('canvas'):
        parts += [dict(type='text',text='CURRENT CANVAS'),image(root/row['canvas']),dict(type='text',text='Current program:\n'+(root/row['program']).read_text())]
    parts.append(dict(type='text',text=row.get('feedback',NEXT_VERSION)))
    answer='The current canvas already matches the reference. Keep the painting unchanged.\nFINISHED' if row['kind']=='finish' else row['plan']+'\n```javascript\n'+(root/row['target_program']).read_text()+'```\nFINISHED'
    return {k:v for k,v in row.items() if k not in ('canvas','program','target_program','reference','feedback','plan')}|{'messages':[dict(role='system',content=[dict(type='text',text=SYSTEM)]),dict(role='user',content=parts),dict(role='assistant',content=[dict(type='text',text=answer)])]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(build(a.root),indent=2))
