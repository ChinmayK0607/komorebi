"""Audit full processor-expanded sequences, masks, and mixture without truncating."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import tomllib
from scheduled_data import load_exact_split
from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.model import setup_processor, setup_tokenizer
from renderers.base import create_renderer
from train import StrictSFTDataset

def audit(config):
    tokenizer=setup_tokenizer(config.tokenizer)
    renderer=create_renderer(tokenizer,config.renderer)
    renderer._processor=setup_processor(config.model)
    rows=[];groups=collections.defaultdict(lambda: {'examples':0,'assistant_tokens':0,'total_tokens':0,'padded_tokens':0})
    for split in ('train','validation'):
        dataset=load_exact_split(config.data.name,split)
        wrapper=StrictSFTDataset(dataset,renderer,seq_len=2**62,multimodal=True,loss_mask_config=config.data.loss_mask)
        for row in dataset:
            sample=wrapper._process(row);total=len(sample['input_ids']);supervised=sum(sample['loss_mask'])
            has_image=any(p.get('type')=='image_url' for m in row['messages'] for p in m['content'])
            image_types=sample.get('mm_token_type_ids') or [0]*total
            if has_image and (not sample.get('mm_kwargs') or not any(image_types)): raise ValueError(f'{row["id"]}: missing image tensors')
            # Causal-shift loss_mask[i] describes token i+1, while mm types describe input token i.
            if any(sample['loss_mask'][i] and image_types[i+1] for i in range(total-1)): raise ValueError('Image token contributes to loss')
            if not 0<supervised<total:raise ValueError('Invalid assistant-only loss mask')
            kind=row.get('mixture_group') or row.get('task_family') or row.get('task_type') or row.get('kind') or ('image_unspecified' if has_image else 'text_unspecified')
            padded=((total+255)//256)*256
            record={'id':row['id'],'split':split,'group':kind,'has_image':has_image,'total_tokens':total,'assistant_tokens':supervised,
                    'image_tokens':sum(bool(x) for x in image_types),'padded_tokens':padded,'fits':total<=config.data.seq_len}
            rows.append(record);group=groups[split+'/'+kind];group['examples']+=1
            for key in ('assistant_tokens','total_tokens','padded_tokens'):group[key]+=record[key]
    for key,value in groups.items():
        denominator=sum(v['assistant_tokens'] for k,v in groups.items() if k.split('/')[0]==key.split('/')[0])
        value['assistant_token_fraction']=value['assistant_tokens']/denominator
    return {'seq_len':config.data.seq_len,'all_fit':all(r['fits'] for r in rows),'max_tokens':max(r['total_tokens'] for r in rows),
       'mask':'assistant_only','padding_multiple':256,'groups':dict(groups),'examples':rows,
       'dataset_sha256':{s:hashlib.sha256((Path(config.data.name)/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation')}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=SFTConfig.model_validate(tomllib.loads(a.config.read_text()));report=audit(config)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='examples'}))
    if not report['all_fit']:raise SystemExit('Oversize target: shorten explicitly or increase context; never truncate')
