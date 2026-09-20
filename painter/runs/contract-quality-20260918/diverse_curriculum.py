"""Build compact, unrendered teacher records; rendering stays on the node.

All supervision targets the complete reference program. Corrupted canvases are
inputs for repair, never intermediate assistant targets. Existing evaluated
references are not imported into this training pool.
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from demonstrations import PRELUDE
from expand_demonstrations import FAMILIES, expanded

PALETTE = ['#b96448', '#477687', '#ba9347', '#648269', '#9b657e', '#666da0',
           '#ab7650', '#507e78', '#bd847b', '#85905c', '#7b6384', '#6b859e']
PARTS = {
    'sphere': 'round silhouette, attached shadow and upper-left highlight',
    'box': 'three faces sharing the same corners',
    'cylinder': 'elliptical rim attached to both vertical sides',
    'ribbons': 'complete flowing bands with consistent spacing',
    'flower': 'radial petals meeting the center, a stem and an attached leaf',
    'cup': 'an open handle attached to the body and an elliptical opening',
    'bus': 'separate wheels attached beneath the body and aligned windows',
    'house': 'a roof seated on connected walls, with the door touching the ground',
    'tree': 'a canopy connected to branches and a rooted trunk',
    'vase': 'an open neck connected to shoulders and a wider body',
    'sailboat': 'two sails attached to the mast above a complete hull',
    'still_life': 'a round fruit beside a taller cylindrical vessel',
}

HELPERS = ('let frame=[1,0,0,1,0,0];\n'
           'function mapped(p){return [frame[0]*p[0]+frame[2]*p[1]+frame[4],'
           'frame[1]*p[0]+frame[3]*p[1]+frame[5]];}\n' + PRELUDE)
HELPERS = HELPERS.replace('brush.polygon(points);', 'brush.polygon(points.map(mapped));')
HELPERS = HELPERS.replace('brush.line(x1,y1,x2,y2);',
    'const a=mapped([x1,y1]),b=mapped([x2,y2]);brush.line(a[0],a[1],b[0],b[1]);')


def corners(frame):
    a,b,c,d,x,y = frame
    return [(a*u+c*v+x, b*u+d*v+y) for u in [-220,220] for v in [-220,220]]


def make_frame(cx, cy, radius, aspect, angle, mirrored):
    """Conservative recipe bounds stay inside the canvas, including rotations."""
    ax, ay = math.sqrt(aspect), 1/math.sqrt(aspect)
    co, si = math.cos(angle), math.sin(angle)
    sx = -ax if mirrored else ax
    raw = [co*sx, si*sx, -si*ay, co*ay]
    extent_x = 220*(abs(raw[0])+abs(raw[2]))
    extent_y = 220*(abs(raw[1])+abs(raw[3]))
    scale = min(radius/max(extent_x,extent_y),
                (cx-25)/extent_x, (575-cx)/extent_x,
                (cy-25)/extent_y, (575-cy)/extent_y)
    result = [round(v*scale,6) for v in raw] + [cx,cy]
    assert all(24 <= x <= 576 and 24 <= y <= 576 for x,y in corners(result))
    return result


def object_spec(family, seed, split, center, radius, mode):
    rng = random.Random(seed)
    # A new seed range and explicit geometric transforms; never reuse a held-out
    # rollout/reference as a training example.
    _, source = expanded(family, seed, 'train')
    bodies = []
    petals = rng.choice([4,5,6,7,8,9,10,12])
    windows = rng.choice([2,3,4,5,6,7])
    pigment = rng.choice(PALETTE)
    for program in source:
        body = program.split('function draw(){translate(-300,-300);\n',1)[1]
        body = body.removesuffix('\nnoLoop();}\n')
        body = re.sub(r'const x=\d+,y=\d+,s=\d+', 'const x=0,y=0,s=100', body)
        body = re.sub(r'pigment="#[0-9a-f]+"', 'pigment="'+pigment+'"', body)
        body = body.replace('brush.spline(pts,.5)', 'brush.spline(pts.map(mapped),.5)')
        if family == 'flower':
            body = re.sub(r'for\(let i=0;i<\d+;i\+\+\)\{const a=i\*2\*Math.PI/\d+;',
                f'for(let i=0;i<{petals};i++){{const a=i*2*Math.PI/{petals};', body)
        elif family == 'bus':
            body = re.sub(r',n=\d+,roof=', f',n={windows},roof=', body)
        elif family == 'ribbons':
            count = rng.choice([2,3,4]) if not bodies else ribbon_count
            ribbon_count = count
            body = body.replace('j<3', f'j<{count}')
            # Four bands fit the same conservative bounding box.
            body = body.replace('j*.7*s', f'j*{1.4/max(1,count-1):.5f}*s')
        bodies.append(body)
    aspect = rng.choice([.55,.7,.85,1.,1.2,1.5,1.8])
    if split == 'dev':
        aspect = rng.choice([.625,.775,.925,1.1,1.35,1.65])
    angle = rng.choice([-.18,0.,0.,.18]) if mode == 'tilted' else 0.
    cx,cy = center
    transform = make_frame(cx,cy,radius,aspect,angle,rng.random()<.5)
    extra = f' Exactly {petals} radial petals.' if family=='flower' else (
            f' {windows} aligned passenger windows, plus the small front window.' if family=='bus' else '')
    return dict(family=family, seed=seed, frame=transform, aspect=aspect,
                angle=angle, center=[cx,cy], pigment=pigment, bodies=bodies,
                parts=PARTS[family]+extra)


def program(objects, corruption=None):
    blocks=[]
    for index, item in enumerate(objects):
        frame = list(item['frame'])
        body = item['bodies'][0 if corruption=='missing_parts' and index==0 else 2]
        if index==0 and corruption=='off_canvas':
            frame[4] += 310
            frame[5] += 220
        if index==0 and corruption=='wrong_proportions':
            frame[0] *= .45
            frame[1] *= .45
            frame[2] *= 1.25
            frame[3] *= 1.25
        blocks.append('{\nframe='+json.dumps(frame,separators=(',',':'))+';\n'+body+'\n}')
    return HELPERS+'\nfunction draw(){translate(-300,-300);\n'+'\n'.join(blocks)+'\nnoLoop();}\n'


def build(output, per_family=192):
    assert per_family >= 12
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];hashes=set()
    for family_index,family in enumerate(FAMILIES):
        for variant in range(per_family):
            split='dev' if variant%6==5 else 'train'
            seed=9200000+family_index*10000+variant
            rng=random.Random(seed)
            # Keep simple cases dominant; explicitly include compositional
            # transfer. Dev uses new parameter combinations, not new families.
            # Split must not determine layout: cycle layouts independently.
            mode=['single','single','offset','pair','overlap','tilted'][(variant//6+variant)%6]
            if mode in ['single','offset','tilted']:
                count=1
                cx=rng.randint(190,410) if mode=='offset' else rng.randint(265,335)
                cy=rng.randint(235,365)
                slots=[((cx,cy),rng.randint(175,225))]
            elif mode=='pair':
                count=2;slots=[((rng.randint(165,195),rng.randint(270,330)),150),
                              ((rng.randint(395,425),rng.randint(270,330)),145)]
            else:
                count=rng.choice([2,3]);slots=[((250,270),180),((375,365),155),((165,380),120)][:count]
            objects=[]
            for n,(center,radius) in enumerate(slots):
                second=rng.choice(FAMILIES) if n else family
                objects.append(object_spec(second,seed+n*3571,split,center,radius,mode))
            identifier=f'v3-{family}-{variant:04d}'
            final=program(objects)
            checksum=hashlib.sha256(final.encode()).hexdigest()
            assert checksum not in hashes
            hashes.add(checksum)
            distortion=rng.choice(['off_canvas','wrong_proportions'])
            plans=[f'Paint {x["family"]} near ({x["center"][0]},{x["center"][1]}): {x["parts"]}.' for x in objects]
            records.append(dict(id=identifier,source_group=identifier,split=split,
                primary_family=family,layout=mode,object_count=count,
                objects=[{k:v for k,v in x.items() if k!='bodies'} for x in objects],
                reference_kind='procedural_known_program',seed=seed,
                programs={'reference':final,'missing_parts':program(objects,'missing_parts'),
                          distortion:program(objects,distortion)},
                target_plan=' '.join(plans),target_sha256=checksum,
                supervision='blank to final; each corrupted canvas to final; keep completed reference',
                render_status='pending_node_render',visual_review_status='pending',
                generalization_scope='held-out parameter combinations and scenes; not photographs or unseen recipe families'))
    path=output/'programs.jsonl'
    path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
    summary=dict(references=len(records),training_references=sum(r['split']=='train' for r in records),
        development_references=sum(r['split']=='dev' for r in records),
        programs=sum(len(r['programs']) for r in records),
        proposed_sft_examples=len(records)*4,rendered_programs=0,
        primary_families=dict(Counter(r['primary_family'] for r in records)),
        layouts_by_split={s:dict(Counter(r['layout'] for r in records if r['split']==s)) for s in ['train','dev']},
        repair_categories_by_split={s:dict(Counter(k for r in records if r['split']==s
            for k in r['programs'] if k!='reference')) for s in ['train','dev']},
        object_counts=dict(Counter(r['object_count'] for r in records)),
        source_group_overlap=[],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target_policy='Complete-reference targets only; corrupted programs are inputs, never desired outputs.',
        status='prepared_unrendered',note='No model inference, painting rendering or training was run by this builder.')
    (output/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--per-family',type=int,default=192);args=parser.parse_args()
    print(json.dumps(build(args.output,args.per_family),indent=2))
