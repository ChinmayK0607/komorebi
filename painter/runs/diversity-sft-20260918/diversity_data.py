"""Build reference-disjoint descriptors; only the node materializes images."""
import json
from pathlib import Path
from render_demos import NEXT_VERSION


FEEDBACK={
    'missing_parts':'The current canvas is missing some of the reference parts. Locate what is missing, preserve correct parts, and finish the reference faithfully.',
    'off_canvas':'The current painting is displaced and may be clipped by the canvas. Correct its placement against the reference and preserve the intended object geometry.',
    'wrong_proportions':'The current painting has incorrect width or height. Compare its silhouette and proportions to the reference, then correct the geometry.',
}


def target_plan(row):
    # Affine transforms can mirror the lighting and turn circular silhouettes
    # into ellipses. Describe the rendered target without contradicting it.
    parts=[]
    for obj in row['objects']:
        description=obj['parts'].replace(
            'round silhouette, attached shadow and upper-left highlight',
            'rounded volume, attached shadow and surface highlight matching the reference')
        description=description.replace('both vertical sides','both long sides')
        description=description.replace('a round fruit','a rounded fruit')
        parts.append(f'Paint {obj["family"]} near ({obj["center"][0]},{obj["center"][1]}): {description}.')
    return ' '.join(parts)


def build(root):
    root=Path(root);data=root/'diversity'
    rows=[json.loads(s) for s in (data/'rendered-manifest.jsonl').read_text().splitlines()]
    review=json.loads((root/'teacher-review.json').read_text())
    assert review['status']=='reviewed','Teacher images must be visually reviewed before training.'
    excluded=set(review.get('excluded_ids',[]));excluded_families=set(review.get('excluded_primary_families',[]))
    excluded_layouts=set(review.get('excluded_layouts',[]));rows=[r for r in rows if r['id'] not in excluded and r['primary_family'] not in excluded_families and r['layout'] not in excluded_layouts]
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    assert not {r['source_group'] for r in train}&{r['source_group'] for r in dev}
    pools={k:[] for k in ['paint','repair','finish','replay']}
    def item(row,kind,corruption=None):
        initial='reference' if kind=='finish' else corruption
        return dict(id=row['id']+'-'+kind+('-'+corruption if corruption else ''),source_group=row['source_group'],family=row['primary_family'],level=0 if row['primary_family'] in ['box','sphere','ribbons','cylinder'] and row['object_count']==1 else 1,
            split=row['split'],kind=kind,reference='diversity/'+row['programs']['reference'].replace('.js','.png'),
            canvas='diversity/'+row['programs'][initial].replace('.js','.png') if initial else None,
            program='diversity/'+row['programs'][initial] if initial else None,
            target_program='diversity/'+row['programs']['reference'],plan=target_plan(row),
            feedback=FEEDBACK[corruption] if corruption else NEXT_VERSION,
            teacher='known-program reference; complete targets; stratified Luna review',layout=row['layout'],object_count=row['object_count'])
    for row in train:
        pools['paint'].append(item(row,'paint'));pools['finish'].append(item(row,'finish'))
        for corruption in row['programs']:
            if corruption!='reference':pools['repair'].append(item(row,'repair',corruption))
    old=json.loads((root/'recovery-pools.json').read_text())
    for row in old['paint']:
        # Revisit exact full-reference targets from prior TRAINING data only.
        if row.get('canvas'):continue
        pools['replay'].append(dict(row,kind='replay'))
    legacy_tasks=json.loads((root/'legacy-evaluation-tasks.json').read_text())
    old_manifest={r['id']:r for r in [json.loads(s) for s in (root/'data/manifest.jsonl').read_text().splitlines()]}
    validation=[]
    for task in legacy_tasks:
        ref=old_manifest[task['id'].rsplit('-',1)[0]]
        validation.append(dict(task,split='dev',target_program='data/'+ref['programs'][-1],plan='Reconstruct the complete reference, preserving its placement, proportions and connected parts.'))
    layouts=['single','pair','overlap','offset','tilted']
    selected=[]
    for n,family in enumerate(sorted({r['primary_family'] for r in dev})):
        candidates=[r for r in dev if r['primary_family']==family]
        selected.append(next((r for r in candidates if r['layout']==layouts[n%len(layouts)]),candidates[0]))
    tasks=list(legacy_tasks)
    for row in selected:
        desc=item(row,'paint');validation.append(desc)
        tasks.append({k:v for k,v in desc.items() if k in ['id','reference','source_group','family','kind','canvas','program','layout','object_count']})
    train_groups={r['source_group'] for pool in pools.values() for r in pool};dev_groups={r['source_group'] for r in validation}
    assert not train_groups&dev_groups
    (root/'evaluation-tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (root/'diversity-data-receipt.json').write_text(json.dumps(dict(pool_counts={k:len(v) for k,v in pools.items()},train_references=len(train),remaining_development_references=len(dev),validation_examples=len(validation),evaluation_tasks=len(tasks),evaluation_references=sorted(dev_groups),source_group_overlap=[],teacher_review=review),indent=2)+'\n')
    return pools,validation
