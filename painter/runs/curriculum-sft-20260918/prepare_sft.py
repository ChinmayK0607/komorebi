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
config=dict(max_steps=96,run=dict(name=name),deployment=dict(gpus_per_node=1,num_train_gpus=1),model=dict(name=str(model),impl='custom',compile='None',attn='auto',optimization_dtype='bfloat16',reduce_dtype='bfloat16',vlm=dict(vision_encoder_attr='model.visual',language_model_attr='model.language_model',freeze_vision_encoder=True),lora=dict(rank=16,alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']),ac=dict(freq=1),ac_offloading='None',optim_cpu_offload=False),renderer=dict(name='qwen3.5',enable_thinking=False),data=dict(**common,splits=['train'],shuffle=True),val=dict(interval=24,eval_on_start=True,data=dict(**common,splits=['validation'],shuffle=False)),optim=dict(type='adamw',lr=1e-5,betas1=.9,betas2=.999,weight_decay=.01,max_norm=1.),scheduler=dict(type='constant'),ckpt=dict(interval=24,keep_last=1),monitors=dict(file={}))
resolved=SFTConfig.model_validate(copy.deepcopy(config));(setup/'resolved.toml').write_text(tomli_w.dumps(config))
check_repository(experiment['checkpoint_repo'],name);assert not HfApi().repo_info(experiment['checkpoint_repo']).private
train=read(P/'data/train.jsonl');dev=read(P/'data/dev.jsonl');retention=read(P/'retention.jsonl')
assert not {r['source_group'] for r in train}&{r['source_group'] for r in dev}
# A fixed development example per family, all three stages. Keep the remaining
# development references available for broader evaluation; never train on them.
manifest=read(P/'data/manifest.jsonl');selected={}
for r in manifest:
 if r['split']=='dev':selected.setdefault(r['family'],r)
selected_ids={r['source_group'] for r in selected.values()};validation=[r for r in dev if r['source_group'] in selected_ids]
write(P/'evaluation-references.json',list(selected.values()))
tok=setup_tokenizer(resolved.tokenizer);renderer=create_renderer(tok,resolved.renderer);renderer._processor=setup_processor(resolved.model)
wrapper=StrictSFTDataset(Dataset.from_list([train[0]]),renderer,seq_len=16384,multimodal=True,loss_mask_config=resolved.data.loss_mask)
audits=[];accepted={'current':[],'review':[],'retention':[]};valid=[];rejected=[];prefixes={}
allrows=[(r,'current' if r['level']==1 else 'review',False) for r in train]+[(r,'retention',False) for r in retention]+[(r,'current' if r['level']==1 else 'review',True) for r in validation]
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
if not accepted['current'] or not accepted['review'] or not valid:raise RuntimeError('No usable curriculum or validation examples; see rejections')
rows,mixture=mix(accepted,{'current':.7,'review':.2,'retention':.1},epochs=2,seed=918)
(dest/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(dest/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in valid))
write(P/'dataset-receipt.json',dict(train_sha256=hashlib.sha256((dest/'train.jsonl').read_bytes()).hexdigest(),validation_sha256=hashlib.sha256((dest/'validation.jsonl').read_bytes()).hexdigest(),ordered_training_ids=[r['id'] for r in rows],ordered_validation_ids=[r['id'] for r in valid]));write(dest/'mixture.json',mixture);write(P/'token-audit.json',audits);write(P/'training-prefixes.json',prefixes)
write(setup/'resolved.setup.json',dict(initialization_mode='fresh_base',init_adapter=None,step_zero_export_required=True,optimizer='fresh',base_model=profile()['model_id'],base_revision=profile()['revision'],heldout_validation=True,max_steps=96,counts=dict(train=len(rows),validation=len(valid)),pool_counts={k:len(v) for k,v in accepted.items()}))
write(P/'preflight.json',dict(status='prepared',training_started=False,accepted=len(audits),rejected=len(rejected),pools={k:len(v) for k,v in accepted.items()},max_tokens=max(r['total_tokens'] for r in audits),steps=96,lr=1e-5,batch_size=8,mixture=mixture['actual_fractions']))
print(json.dumps(json.loads((P/'preflight.json').read_text())))
