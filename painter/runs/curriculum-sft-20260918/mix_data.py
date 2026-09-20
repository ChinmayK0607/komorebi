"""Materialize a disclosed SFT mixture using measured assistant-token counts."""
import argparse,collections,copy,json,random
from pathlib import Path


def mix(pools,weights,epochs=2,seed=917):
    if not pools.get('current'):raise ValueError('Current-level training examples required')
    if epochs<1:raise ValueError('Need positive epoch count')
    if weights.get('current',0)<=0 or any(v<0 for v in weights.values()):raise ValueError('Current pool must have positive weight and all weights must be nonnegative')
    for pool in pools.values():
        for r in pool:
            if r.get('split')!='train':raise ValueError('Evaluation reference entered training mixture')
            if not isinstance(r.get('supervised_tokens'),int) or r['supervised_tokens']<=0:raise ValueError('Measure supervised token counts with the actual training tokenizer first')
    active={k:weights[k] for k,v in pools.items() if v and weights.get(k,0)>0};den=sum(active.values());active={k:v/den for k,v in active.items()}
    rng=random.Random(seed);queues={};cursors=collections.Counter();tokens=collections.Counter();exposures=collections.Counter();rows=[]
    for k,pool in pools.items():
        queues[k]=[]
        for _ in range(epochs if k=='current' else 1):
            shuffled=list(pool);rng.shuffle(shuffled);queues[k].extend(shuffled)
    while cursors['current']<len(queues['current']):
        total=sum(tokens.values());group=max(active,key=lambda k:active[k]*(total+1)-tokens[k])
        if cursors[group]==len(queues[group]):
            cursors[group]=0;rng.shuffle(queues[group])
        row=copy.deepcopy(queues[group][cursors[group]]);cursors[group]+=1
        source=row.get('source_example_id',row['id']);exposures[group+':'+source]+=1
        row.update(source_example_id=source,id=f'{source}__mix_{len(rows):07d}',mixture_group=group)
        tokens[group]+=row['supervised_tokens'];rows.append(row)
    return rows,dict(requested_fractions=weights,effective_target_fractions=active,
        missing_pools=[k for k in weights if not pools.get(k)],actual_tokens=dict(tokens),
        actual_fractions={k:v/sum(tokens.values()) for k,v in tokens.items()},exposures=dict(exposures),
        current_passes=epochs,note='Measured-token mixture; earlier/general examples can repeat. Repetition is not new diversity.')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--current',required=True);a.add_argument('--review');a.add_argument('--retention');a.add_argument('--output',required=True);a.add_argument('--epochs',type=int,default=2);args=a.parse_args()
    pools={k:[json.loads(s) for s in Path(v).read_text().splitlines()] if v else [] for k,v in [('current',args.current),('review',args.review),('retention',args.retention)]}
    rows,report=mix(pools,{'current':.7,'review':.2,'retention':.1},args.epochs);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'mixture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
