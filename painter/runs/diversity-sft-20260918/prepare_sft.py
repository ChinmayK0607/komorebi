"""Prepare complete-target Prime SFT data on the node, using the real processor."""
import os,sys,json,hashlib,copy,math,random
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['PRL_OUTPUT_DIR']=str(P/'train-output');os.environ['CAPACITY_STATE_COMPLETE']='1'
assert sys.platform=='linux'
B=P/'source/brush-rl';M=B/'integrations/speedpainting/mixed_sft'
sys.path[:0]=[str(P),str(M),str(B/'integrations/prime/watercolour_sft'),str(B/'scripts')]
import tomli_w
from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.model import setup_tokenizer,setup_processor
from renderers.base import create_renderer
from train import StrictSFTDataset
from datasets import Dataset
from model_profiles import profile,verify_snapshot
from mix_data import mix
from upload_checkpoints import check_repository
from huggingface_hub import HfApi

def write(path,v):path.write_text(json.dumps(v,indent=2)+'\n')
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
model=Path((P/'model-path.txt').read_text().strip());verify_snapshot(model)
experiment=json.loads((P/'EXPERIMENT.json').read_text());name=experiment['run_name'];dest=P/'sft-data';dest.mkdir(exist_ok=True)
setup=B/'runs'/(name+'-setup');setup.mkdir(parents=True,exist_ok=True)
common=dict(type='sft',name=str(dest),batch_size=8,micro_batch_size=1,seq_len=16384,num_workers=1,seed=918,loss_mask=dict(system=False,user=False,assistant=True,tool=False))
config=dict(max_steps=experiment['max_steps'],run=dict(name=name),deployment=dict(gpus_per_node=1,num_train_gpus=1),model=dict(name=str(model),impl='custom',compile='None',attn='auto',optimization_dtype='bfloat16',reduce_dtype='bfloat16',vlm=dict(vision_encoder_attr='model.visual',language_model_attr='model.language_model',freeze_vision_encoder=True),lora=dict(rank=16,alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']),ac=dict(freq=1),ac_offloading='None',optim_cpu_offload=False),renderer=dict(name='qwen3.5',enable_thinking=False),data=dict(**common,splits=['train'],shuffle=True),val=dict(interval=experiment['checkpoint_interval'],eval_on_start=True,data=dict(**common,splits=['validation'],shuffle=False)),optim=dict(type='adamw',lr=experiment['learning_rate'],betas1=.9,betas2=.999,weight_decay=.01,max_norm=1.),scheduler=dict(type='constant'),ckpt=dict(interval=experiment['checkpoint_interval'],keep_last=1),monitors=dict(file={}))
resolved=SFTConfig.model_validate(copy.deepcopy(config));(setup/'resolved.toml').write_text(tomli_w.dumps(config))
check_repository(experiment['checkpoint_repo'],name);assert not HfApi().repo_info(experiment['checkpoint_repo']).private
from recovery_data import materialize
from diversity_data import build
pools,validation_descriptors=build(P);retention=read(P/'retention.jsonl')
train=[materialize(P,r) for rows in pools.values() for r in rows]
validation=[materialize(P,r) for r in validation_descriptors]
assert not {r['source_group'] for r in train}&{r['source_group'] for r in validation}
tok=setup_tokenizer(resolved.tokenizer);renderer=create_renderer(tok,resolved.renderer);renderer._processor=setup_processor(resolved.model)
wrapper=StrictSFTDataset(Dataset.from_list([train[0]]),renderer,seq_len=16384,multimodal=True,loss_mask_config=resolved.data.loss_mask)
audits=[];accepted={g:[] for g in experiment['sampling_fractions']};valid=[];rejected=[];prefixes={}
allrows=[(r,r['kind'],False) for r in train]+[(r,'retention',False) for r in retention]+[(r,r['kind'],True) for r in validation]
for i,(row,group,is_val) in enumerate(allrows):
 try:
  sample=wrapper._process(row)
  learned=[t for t,m in zip(sample['target_ids'],sample['loss_mask']) if m]
  raw=row['messages'][-1]['content'][0]['text'];expected=tok.encode(raw,add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]
  if learned!=expected:raise ValueError('Assistant loss mask does not cover exactly the complete target and EOS')
  record=dict(id=row['id'],group=group,validation=is_val,total_tokens=len(sample['input_ids'])+1,supervised_tokens=len(learned),images=len((sample.get('mm_kwargs') or {}).get('image_grid_thw',[])))
  audits.append(record);row.update(supervised_tokens=len(learned),mixture_group=group,task_family=group)
  if is_val:valid.append(row)
  else:accepted[group].append(row)
  # One example for each image-count regime gives a direct serving-stack check.
  key=str(record['images'])
  if key not in prefixes:
   from prime_rl.trainer.sft.data import _drop_null_fields
   prefix=list(renderer.render_ids(_drop_null_fields(row['messages'][:-1]),add_generation_prompt=True))
   mm=sample.get('mm_kwargs') or {};pixels=mm.get('pixel_values')
   prefixes[key]=dict(row=row,tokens=prefix,pixel_sha256=hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest() if pixels is not None else None,image_grid=mm['image_grid_thw'].tolist() if 'image_grid_thw' in mm else [])
 except Exception as exc:rejected.append(dict(id=row['id'],group=group,error=f'{type(exc).__name__}: {exc}'))
 if i%50==0:write(P/'preparation-progress.json',dict(done=i+1,total=len(allrows),accepted=len(audits),rejected=len(rejected)))
# Execution/data faults exclude the affected rows; a wrong target mask is a
# correctness problem, never silently treated as training material.
write(P/'preprocessing-rejections.json',rejected)
if not all(accepted.values()) or not valid:raise RuntimeError('A required curriculum pool is empty; see rejections')
# Balance decisions by examples: weighting FINISHED by its tiny token count
# would require enormous repetition and teach premature stopping.
rng=random.Random(9182);rows=[];exposed=set()
fractions=experiment['sampling_fractions']
total_examples=experiment['max_steps']*resolved.data.batch_size
assert math.isclose(sum(fractions.values()),1.0)
quotas={g:total_examples*f for g,f in fractions.items()}
counts={g:math.floor(q) for g,q in quotas.items()}
for group in sorted(fractions,key=lambda g:quotas[g]-counts[g],reverse=True)[:total_examples-sum(counts.values())]:
 counts[group]+=1
assert sum(counts.values())==total_examples
for group,fraction in fractions.items():
 count=counts[group]
 pool=list(accepted[group]);rng.shuffle(pool)
 if group=='repair':pool.sort(key=lambda r:r['source_group'] in exposed)
 chosen=[pool[i%len(pool)] for i in range(count)];rows.extend(chosen);exposed.update(r['source_group'] for r in chosen if r.get('source_group'))
assert len(rows)==total_examples
rng.shuffle(rows)
# Replay intentionally samples with replacement. Each dataset row needs its own
# identity; retain the original example ID separately for provenance.
rows=[dict(row,source_example_id=row.get('source_example_id',row['id']),
           id=f'{name}-draw-{index:06d}') for index,row in enumerate(rows)]
assert len({row['id'] for row in rows})==len(rows)
mixture={'unit':'examples','requested_fractions':fractions,'actual_fractions':{g:sum(r['mixture_group']==g for r in rows)/len(rows) for g in fractions},'unique_training_source_groups':len(exposed), 'assistant_token_fractions':{g:sum(r['supervised_tokens'] for r in rows if r['mixture_group']==g)/sum(r['supervised_tokens'] for r in rows) for g in fractions}}
(dest/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(dest/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in valid))
from scheduled_data import load_exact_split
for split,expected_rows in [('train',rows),('validation',valid)]:
 loaded=load_exact_split(dest,split)
 assert list(loaded['id'])==[row['id'] for row in expected_rows]
write(P/'dataset-receipt.json',dict(train_sha256=hashlib.sha256((dest/'train.jsonl').read_bytes()).hexdigest(),validation_sha256=hashlib.sha256((dest/'validation.jsonl').read_bytes()).hexdigest(),ordered_training_ids=[r['id'] for r in rows],ordered_validation_ids=[r['id'] for r in valid]));write(dest/'mixture.json',mixture);write(P/'token-audit.json',audits);write(P/'training-prefixes.json',prefixes)
write(setup/'resolved.setup.json',dict(initialization_mode='trained_adapter',init_adapter=str(Path((P/'initial-adapter-path.txt').read_text().strip()).resolve()),step_zero_export_required=False,optimizer='fresh',base_model=profile()['model_id'],base_revision=profile()['revision'],heldout_validation=True,max_steps=experiment['max_steps'],counts=dict(train=len(rows),validation=len(valid)),pool_counts={k:len(v) for k,v in accepted.items()}))
write(P/'preflight.json',dict(status='prepared',training_started=False,accepted=len(audits),rejected=len(rejected),pools={k:len(v) for k,v in accepted.items()},max_tokens=max(r['total_tokens'] for r in audits),steps=experiment['max_steps'],lr=experiment['learning_rate'],batch_size=8,mixture=mixture['actual_fractions']))
print(json.dumps(json.loads((P/'preflight.json').read_text())))
