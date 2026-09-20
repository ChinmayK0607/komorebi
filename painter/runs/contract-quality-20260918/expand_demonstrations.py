"""A larger procedural skill curriculum; no image rendering on the controller."""
import argparse,json,random,re
from pathlib import Path
from demonstrations import PRELUDE,recipe

FAMILIES=['sphere','box','cylinder','ribbons','flower','cup','bus','house','tree','vase','sailboat','still_life']

def expanded(kind,seed,split):
    rng=random.Random(seed)
    if kind in FAMILIES[:6]:
        plans,programs=recipe(kind,seed,split)
        # Changes the scene, not merely the renderer's random seed.
        hue=rng.choice(['#b96448','#477687','#ba9347','#648269','#9b657e','#666da0','#ab7650','#507e78'])
        programs=[re.sub(r'pigment="#[0-9a-f]+"',f'pigment="{hue}"',s) for s in programs]
        if kind=='flower':
            petals=rng.choice([5,7,9] if split=='train' else [6,8])
            programs=[s.replace('i<7','i<'+str(petals)).replace('i*2*Math.PI/7','i*2*Math.PI/'+str(petals)) for s in programs]
        elif kind=='ribbons':
            wave=rng.choice([.22,.30,.38] if split=='train' else [.26,.34])
            programs=[s.replace('.3*s*Math.sin',f'{wave}*s*Math.sin') for s in programs]
        return plans,programs
    x=rng.choice([245,300,355] if split=='train' else [270,330]);y=rng.randint(270,320)
    s=rng.choice([74,90,106] if split=='train' else [82,98]);pigment=rng.choice(['#b96448','#477687','#ba9347','#648269','#9b657e','#666da0'])
    n=rng.choice([3,5] if split=='train' else [4]);roof=rng.choice([.9,1.3] if split=='train' else [1.1])
    common=f'const x={x},y={y},s={s},pigment="{pigment}",n={n},roof={roof};'
    shadow='ellipseMark(x+16,y+s*.8,s*1.65,s*.18,"#777879",65);'
    if kind=='bus':
        steps=[shadow+'polygonMark([[x-s*1.7,y-s*.75],[x+s*1.45,y-s*.75],[x+s*1.7,y-s*.4],[x+s*1.7,y+s*.55],[x-s*1.7,y+s*.55]],pigment,240,"flat");',
         'for(let i=0;i<n;i++){const q=x-s*1.5+i*s*2.65/n,w=s*2.25/n;polygonMark([[q,y-s*.56],[q+w,y-s*.56],[q+w,y-s*.05],[q,y-s*.05]],"#405b65",225,"flat");}for(const q of [-1.05,1.08]){ellipseMark(x+s*q,y+s*.52,s*.32,s*.32,"#464449",235,"flat");}',
         'for(const q of [-1.05,1.08]){ellipseMark(x+s*q,y+s*.52,s*.14,s*.14,"#a6a3a0",190,"flat");}pencilLine(x-s*1.5,y+s*.12,x+s*1.55,y+s*.12,"#725346",.6);polygonMark([[x+s*1.36,y-s*.51],[x+s*1.55,y-s*.30],[x+s*1.55,y-s*.07],[x+s*1.36,y-s*.07]],"#b1c6c1",175,"flat");']
        plans=['Block in the long bus body above its ground shadow.','Place aligned windows and two wheels attached to the body.','Resolve the wheel hubs, front window and a restrained body seam.']
    elif kind=='house':
        steps=[shadow+'polygonMark([[x-s,y-s*.45],[x+s,y-s*.45],[x+s,y+s*.8],[x-s,y+s*.8]],pigment,210,"flat");polygonMark([[x+s,y-s*.45],[x+s*1.45,y-s*.72],[x+s*1.45,y+s*.49],[x+s,y+s*.8]],"#79736b",145,"flat");',
         'polygonMark([[x-s*1.18,y-s*.42],[x,y-s*(.45+roof)],[x+s*1.18,y-s*.42]],"#825b4c",230,"flat");polygonMark([[x,y-s*(.45+roof)],[x+s*.45,y-s*(.72+roof)],[x+s*1.58,y-s*.70],[x+s*1.18,y-s*.42]],"#615b57",200,"flat");',
         'polygonMark([[x+s*.12,y+s*.1],[x+s*.58,y+s*.1],[x+s*.58,y+s*.8],[x+s*.12,y+s*.8]],"#575c54",200,"flat");polygonMark([[x-s*.70,y-s*.1],[x-s*.24,y-s*.1],[x-s*.24,y+s*.31],[x-s*.70,y+s*.31]],"#637f88",200,"flat");pencilLine(x-s*.47,y-s*.08,x-s*.47,y+s*.28,"#eddfbc",.6);']
        plans=['Establish connected front and side walls in perspective.','Seat the roof on both walls with a consistent ridge.','Attach the door to ground level and add one inset window.']
    elif kind=='tree':
        steps=[shadow+'polygonMark([[x-s*.13,y-s*.25],[x+s*.13,y-s*.25],[x+s*.20,y+s*.85],[x-s*.20,y+s*.85]],"#796049",210,"flat");pencilLine(x,y+s*.2,x-s*.55,y-s*.55,"#76654c",3);pencilLine(x,y,x+s*.48,y-s*.7,"#76654c",2.6);',
         'for(const q of [[-.7,-.55,.68],[-.15,-1,.78],[.6,-.65,.7],[.05,-.4,.9]]){ellipseMark(x+s*q[0],y+s*q[1],s*q[2],s*q[2]*.8,pigment,150);}',
         'ellipseMark(x-s*.40,y-s*.96,s*.46,s*.30,"#c6be6b",80);ellipseMark(x+s*.50,y-s*.38,s*.42,s*.27,"#436452",65);pencilLine(x-s*.05,y+s*.3,x-s*.04,y+s*.7,"#cfb48b",1);']
        plans=['Root the trunk and two connected main branches.','Build an overlapping canopy that meets the branches.','Separate lit foliage from the darker lower canopy and finish the trunk.']
    elif kind=='vase':
        steps=[shadow+'polygonMark([[x-s*.37,y-s*.9],[x+s*.37,y-s*.9],[x+s*.3,y-s*.4],[x+s*.67,y+s*.25],[x+s*.45,y+s*.8],[x-s*.45,y+s*.8],[x-s*.67,y+s*.25],[x-s*.3,y-s*.4]],pigment,225,"flat");',
         'polygonMark([[x+s*.13,y-s*.75],[x+s*.3,y-s*.75],[x+s*.23,y-s*.35],[x+s*.52,y+s*.28],[x+s*.32,y+s*.68],[x+s*.11,y+s*.65]],"#625e69",90);ellipseMark(x,y-s*.9,s*.37,s*.13,"#7e756b",170,"flat");',
         'pencilLine(x-s*.19,y-s*.6,x-s*.14,y-s*.24,"#ede0bd",1.2);for(let i=0;i<3;i++){const q=y+s*(.22+i*.14);pencilLine(x-s*.35,q,x+s*.28,q,"#d0b484",.7);}']
        plans=['Establish the connected neck, shoulders and rounded body of the vase.','Model the right side and add the elliptical opening.','Add a narrow light edge and a few curved-surface decoration marks.']
    elif kind=='sailboat':
        steps=['ellipseMark(x,y+s*.8,s*1.7,s*.16,"#72979d",75);polygonMark([[x-s*1.5,y+s*.38],[x+s*1.45,y+s*.38],[x+s*.95,y+s*.85],[x-s*.95,y+s*.85]],pigment,225,"flat");',
         'pencilLine(x,y-s*1.6,x,y+s*.4,"#726657",1.8);polygonMark([[x-s*.08,y-s*1.55],[x-s*.08,y+s*.18],[x-s*1.3,y+s*.18]],"#bfc5bf",150,"flat");polygonMark([[x+s*.08,y-s*1.2],[x+s*.08,y+s*.18],[x+s*1.24,y+s*.18]],"#d4b680",150,"flat");',
         'pencilLine(x-s*1.38,y+s*.42,x+s*1.31,y+s*.42,"#705e51",.7);for(let i=0;i<4;i++){const q=y+s*(.97+i*.15);pencilLine(x-s*(1.15-i*.12),q,x+s*(.95-i*.10),q,"#8ca8a6",.6);}']
        plans=['Place a boat hull on a quiet strip of water.','Connect the mast and two triangular sails to the hull.','Clarify the gunwale and add short reflections beneath it.']
    else:
        steps=['ellipseMark(x,y+s*.95,s*1.75,s*.22,"#7d7c79",70);polygonMark([[x-s*1.15,y-s*.85],[x-s*.12,y-s*.85],[x-s*.12,y+s*.6],[x-s*1.15,y+s*.6]],"#6d929b",190,"flat");ellipseMark(x-s*.63,y-s*.85,s*.515,s*.17,"#c6c1aa",200,"flat");',
         'ellipseMark(x+s*.65,y+s*.33,s*.68,s*.68,pigment,210);ellipseMark(x+s*.91,y+s*.56,s*.36,s*.35,"#7a6252",65);',
         'ellipseMark(x+s*.35,y+s*.04,s*.20,s*.17,"#efe0ad",120);pencilLine(x+s*.59,y-s*.27,x+s*.66,y-s*.42,"#6b6748",1.2);pencilLine(x-s*.92,y-s*.51,x-s*.92,y+s*.38,"#d7debc",1.0);']
        plans=['Set a tall cylinder behind a shared ground shadow.','Place a round fruit in front, with correct overlap and darker lower volume.','Add the fruit stem and small highlights while preserving the relative scale.']
    programs=[PRELUDE+'\nfunction draw(){translate(-300,-300);\n'+common+'\n'+'\n'.join(steps[:i])+'\nnoLoop();}\n' for i in range(1,4)]
    return plans,programs

def build(output,variants=32):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for n,kind in enumerate(FAMILIES):
        for variant in range(variants):
            split='dev' if variant%4==3 else 'train';seed=918000+n*1000+variant
            ident=f'v2-{kind}-{variant:03d}';folder=output/ident;folder.mkdir(exist_ok=True)
            plans,programs=expanded(kind,seed,split)
            for i,s in enumerate(programs,1):(folder/f'turn-{i}.js').write_text(s)
            rows.append(dict(id=ident,source_group=ident,split=split,level=0 if n<4 else 1,family=kind,seed=seed,reference_kind='procedural_known_program',teacher='assistant-authored expanded recipe',plans=plans,programs=[f'{ident}/turn-{i}.js' for i in (1,2,3)],reference=f'{ident}/turn-3.png',attempts=0,acceptable=0,validity='awaiting_node_render',generalisation_scope='held-out geometric parameter bands, not unseen template families'))
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variants',type=int,default=32);a=p.parse_args();r=build(a.output,a.variants);print(json.dumps({'references':len(r),'programs':len(r)*3}))
