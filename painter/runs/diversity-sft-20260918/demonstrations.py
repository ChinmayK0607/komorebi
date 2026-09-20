"""Procedural teacher programs for the first two rungs; photographs stay distinct."""
import argparse
import hashlib
import json
import random
from pathlib import Path

PRELUDE='''function setup(){createCanvas(600,600,WEBGL);pixelDensity(1);randomSeed(17);noiseSeed(17);brush.seed(17);background(248,245,237);}
function polygonMark(points,color,alpha=210,medium="watercolour"){
  brush.noStroke();brush.noWash();
  if(medium==="flat"){brush.fill(color,alpha);brush.fillBleed(0,"out");brush.fillTexture(0.12,0.05);}
  else{brush.fill(color,alpha);brush.fillBleed(0.06,"out");brush.fillTexture(0.32,0.16);}
  brush.polygon(points);
}
function ellipseMark(x,y,rx,ry,color,alpha=210,medium="watercolour"){
  const points=[];for(let i=0;i<48;i++){const a=2*Math.PI*i/48;points.push([x+rx*Math.cos(a),y+ry*Math.sin(a)]);}
  polygonMark(points,color,alpha,medium);
}
function pencilLine(x1,y1,x2,y2,color="#554d44",weight=0.6){brush.noFill();brush.noWash();brush.set("cpencil",color,weight);brush.line(x1,y1,x2,y2);}
'''


def recipe(kind,seed,split):
    rng=random.Random(seed)
    # Development holds out parameter bands, not just identical programs with a new seed.
    x=rng.choice([245,300,355] if split=='train' else [270,330]);y=rng.randint(275,325)
    size=rng.choice([78,96,114] if split=='train' else [87,105]);hue=rng.choice(['#c8704a','#688a9b','#b89645','#7c8d56'])
    common=f'const x={x},y={y},s={size},pigment="{hue}";'
    shadow='ellipseMark(x+18,y+s*.8,s*1.12,s*.20,"#79797c",80);'
    if kind=='sphere':
        steps=[shadow+'ellipseMark(x,y,s,s,pigment,225);',
          'for(let i=0;i<9;i++){ellipseMark(x+s*.22+i*1.8,y+s*.18,s*(.76-i*.055),s*(.80-i*.05),"#574f55",22);}',
          'ellipseMark(x-s*.31,y-s*.36,s*.34,s*.30,"#fff0bf",90);ellipseMark(x-s*.35,y-s*.40,s*.13,s*.12,"#fff9e6",150);']
        plans=['Establish the sphere silhouette and a soft contact shadow.','Build the darker lower-right volume with layered washes.','Add a restrained upper-left light and keep a soft transition.']
    elif kind=='box':
        steps=[shadow+'polygonMark([[x-s,y-s*.45],[x,y-s],[x+s,y-s*.45],[x,y]],pigment,240,"flat");',
          'polygonMark([[x-s,y-s*.45],[x,y],[x,y+s],[x-s,y+s*.5]],pigment,205,"flat");polygonMark([[x,y],[x+s,y-s*.45],[x+s,y+s*.5],[x,y+s]],"#645b55",215,"flat");',
          'pencilLine(x-s,y-s*.45,x,y-s,"#e0cba1");pencilLine(x,y-s,x+s,y-s*.45,"#e0cba1");pencilLine(x,y,x,y+s,"#514c43");']
        plans=['Place the top plane and ground shadow.','Connect the two side planes with shared vertices and different values.','Clarify a few lit edges without outlining every surface.']
    elif kind=='cylinder':
        steps=[shadow+'polygonMark([[x-s*.65,y-s*.7],[x+s*.65,y-s*.7],[x+s*.65,y+s*.6],[x-s*.65,y+s*.6]],pigment,225,"flat");ellipseMark(x,y+s*.6,s*.65,s*.23,pigment,225,"flat");',
          'for(let i=0;i<8;i++){const z=x+s*.25+i*s*.045;polygonMark([[z,y-s*.7],[z+s*.045,y-s*.7],[z+s*.045,y+s*.65],[z,y+s*.65]],"#565053",35+i*9,"flat");}ellipseMark(x,y-s*.7,s*.65,s*.23,"#e3d3b6",235,"flat");',
          'pencilLine(x-s*.48,y-s*.4,x-s*.48,y+s*.48,"#eddbc3",1.4);ellipseMark(x,y-s*.70,s*.49,s*.12,"#8b7d69",70);']
        plans=['Establish the cylinder body and its contact with the ground.','Connect the elliptical top to a shaded vertical body.','Add a narrow highlight and preserve the top ellipse.']
    elif kind=='ribbons':
        steps=['for(let j=0;j<3;j++){const points=[];for(let i=0;i<=24;i++){let t=i/24;points.push([x-1.7*s+3.4*s*t,y-s+j*.7*s+.3*s*Math.sin(t*Math.PI*2)]);}for(let i=24;i>=0;i--){let t=i/24;points.push([x-1.7*s+3.4*s*t,y-s+j*.7*s+.16*s+.3*s*Math.sin(t*Math.PI*2)]);}polygonMark(points,pigment,180);}',
          'for(let j=0;j<3;j++){brush.noFill();brush.noWash();brush.set("cpencil","#6e625a",.8);const pts=[];for(let i=0;i<=8;i++){let t=i/8;pts.push([x-1.7*s+3.4*s*t,y-s+j*.7*s+.07*s+.3*s*Math.sin(t*Math.PI*2)]);}brush.spline(pts,.5);}',
          'for(let j=0;j<3;j++){ellipseMark(x-1.48*s,y-s+j*.7*s+.22*s,s*.10,s*.06,"#e7cba1",70);}']
        plans=['Paint three flowing ribbons with consistent spacing.','Follow their curvature with one fine stroke per ribbon.','Add small light accents while keeping the broad marks dominant.']
    elif kind=='flower':
        steps=['pencilLine(x,y+s*.1,x-12,y+s*1.75,"#6b7950",2.4);polygonMark([[x-8,y+s],[x-s*.8,y+s*.55],[x-s*.45,y+s*1.25]],"#7d9553",190);',
          'for(let i=0;i<7;i++){const a=i*2*Math.PI/7;const cx=x+s*.52*Math.cos(a),cy=y+s*.52*Math.sin(a),pts=[];for(let j=0;j<32;j++){const t=j*2*Math.PI/32,u=s*.64*Math.cos(t),v=s*.30*Math.sin(t);pts.push([cx+u*Math.cos(a)-v*Math.sin(a),cy+u*Math.sin(a)+v*Math.cos(a)]);}polygonMark(pts,pigment,160);}',
          'ellipseMark(x,y,s*.23,s*.21,"#b79442",220);for(let i=0;i<13;i++){const a=i*2.4,r=s*.18*Math.sqrt((i+.5)/13);ellipseMark(x+Math.cos(a)*r,y+Math.sin(a)*r,2.2,2,"#665344",170);}']
        plans=['Anchor the stem and a connected leaf.','Arrange overlapping petals around the same center.','Resolve the flower center with a few varied dark accents.']
    elif kind=='cup':
        steps=[shadow+'const handle=[];for(let i=0;i<=24;i++){const a=-Math.PI/2+i*Math.PI/24;handle.push([x+s*.4+s*.8*Math.cos(a),y+s*.55*Math.sin(a)]);}for(let i=24;i>=0;i--){const a=-Math.PI/2+i*Math.PI/24;handle.push([x+s*.4+s*.5*Math.cos(a),y+s*.32*Math.sin(a)]);}polygonMark(handle,pigment,225,"flat");',
          'polygonMark([[x-s*.65,y-s*.7],[x+s*.60,y-s*.7],[x+s*.43,y+s*.65],[x-s*.45,y+s*.65]],pigment,235,"flat");ellipseMark(x,y-s*.7,s*.63,s*.22,"#c8bba4",235,"flat");ellipseMark(x,y-s*.70,s*.48,s*.13,"#635442",240,"flat");',
          'polygonMark([[x-s*.44,y-s*.38],[x-s*.22,y-s*.34],[x-s*.16,y+s*.4],[x-s*.32,y+s*.44]],"#eee1c5",135);pencilLine(x-s*.4,y+s*.63,x+s*.4,y+s*.63,"#5f5955",.6);']
        plans=['Place a shadow and an open handle attached to the cup.','Paint the body in front of the handle and show the opening.','Use one broad highlight to suggest ceramic volume.']
    else:raise ValueError(kind)
    programs=[PRELUDE+'\nfunction draw(){translate(-300,-300);\n'+common+'\n'+'\n'.join(steps[:i])+'\nnoLoop();}\n' for i in range(1,4)]
    return plans,programs


def build(output,variants=8):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    kinds=['sphere','box','cylinder','ribbons','flower','cup']
    for n,kind in enumerate(kinds):
        for variant in range(variants):
            split='dev' if variant%4==3 else 'train';seed=917000+n*1000+variant
            ident=f'{kind}-{variant:03d}';folder=output/ident;folder.mkdir(exist_ok=True)
            plans,programs=recipe(kind,seed,split)
            for i,source in enumerate(programs,1):(folder/f'turn-{i}.js').write_text(source)
            row=dict(id=ident,source_group=ident,split=split,level=0 if n<4 else 1,family=kind,
               reference_kind='procedural_known_program',teacher='assistant-authored recipe with deterministic parameter variation',
               seed=seed,programs=[f'{ident}/turn-{i}.js' for i in (1,2,3)],plans=plans,
               reference=f'{ident}/turn-3.png',attempts=0,acceptable=0,
               validity='awaiting_node_render',generalisation_scope='held-out parameter bands; not unseen object families')
            rows.append(row)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--variants',type=int,default=8);args=a.parse_args()
    rows=build(args.output,args.variants);print(json.dumps(dict(references=len(rows),teacher_programs=len(rows)*3)))
