"""Resolve mixed reference-painting and general retention SFT on pinned PRIME."""
import argparse
import base64
import collections
import copy
import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
import tomli_w
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_shared', HERE.parents[1] / 'prime/watercolour_sft/prepare.py')
shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
spec = importlib.util.spec_from_file_location('mixed_schedule', HERE / 'scheduled_data.py')
schedule = importlib.util.module_from_spec(spec); spec.loader.exec_module(schedule)

@lru_cache(maxsize=1)
def ledger_replay():
    spec = importlib.util.spec_from_file_location('mixed_ledger_replay', HERE.parent/'replay.py')
    replay = importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
    return replay

def validate_data(folder):
    identities, contents, images, source_groups = set(), {}, {}, {}
    counts, kinds = {}, {}
    for split in ('train', 'validation'):
        counts[split] = 0; kinds[split] = collections.Counter()
        # Photo/canvas bytes repeat across trajectories. Do not retain the whole
        # multi-GB JSONL and a second parsed copy just to validate each row.
        def records(path):
            with path.open() as stream:
                for line in stream:
                    if line.strip(): yield json.loads(line)
        for row in records(folder / (split+'.jsonl')):
            counts[split] += 1
            ident = row.get('id')
            if not ident or ident in identities: raise ValueError('Missing/duplicate example id')
            identities.add(ident)
            source_group = row.get('source_group')
            if source_group:
                if source_group in source_groups and source_groups[source_group] != split:
                    raise ValueError('Source group leaks across splits')
                source_groups[source_group] = split
            messages = row.get('messages', [])
            if not messages or messages[-1]['role'] != 'assistant': raise ValueError(f'{ident}: final assistant required')
            if not any(m['role']=='user' for m in messages): raise ValueError(f'{ident}: user required')
            image_count = 0
            for message in messages:
                if message['role'] not in ('system','user','assistant'): raise ValueError('Only system/user/assistant supported')
                if not isinstance(message['content'], list): raise ValueError('Typed content lists required for Arrow')
                feedback = row.get('task_family') == 'painting' and row.get('action_contract') == 'brush-turn-v1' and message['role'] == 'user'
                if feedback:
                    parts = message['content']
                    if (len(parts) not in (5,6,7) or parts[0] != {'type':'text','text':'Reference image:'}
                            or parts[2] != {'type':'text','text':'Canvas observation:'}
                            or parts[1].get('type') != 'image_url' or parts[3].get('type') != 'image_url'
                            or not parts[4].get('text','').startswith('Host state: ')):
                        raise ValueError('Unknown feedback observation layout')
                    if len(parts) == 7:
                        ledger_prefix = 'Complete committed program, in paint order (read-only context; return only NEW marks):\n'
                        ledger_part = parts[6]
                        if ledger_part.get('type') != 'text' or not ledger_part.get('text','').startswith(ledger_prefix):
                            raise ValueError('Unknown committed-program ledger layout')
                        ledger, _ = ledger_replay().validate(ledger_part['text'][len(ledger_prefix):],256,expected_contract='bounded-brush-v2')
                        host_state = json.loads(parts[4]['text'][len('Host state: '):])
                        if ledger['paperRGB'] != host_state.get('paperRGB'):
                            raise ValueError('Committed ledger and host state paper differ')
                for part_index, part in enumerate(message['content']):
                    if part['type'] == 'text':
                        if not isinstance(part.get('text'), str): raise ValueError('Invalid text')
                    elif part['type']=='image_url' and message['role']=='user':
                        url=part['image_url']['url']
                        if not url.startswith(('data:image/png;base64,','data:image/jpeg;base64,')): raise ValueError('Self-contained PNG/JPEG required')
                        digest=hashlib.sha256(base64.b64decode(url.split(',',1)[1],validate=True)).hexdigest()
                        # Blank/partial canvas observations are environment states,
                        # not source identities. Shared blank canvases are expected.
                        # References and every non-feedback image retain leak checks.
                        if not (feedback and part_index == 3):
                            if digest in images and images[digest]!=split: raise ValueError('Image leaks across splits')
                            images[digest]=split
                        image_count+=1
                    else: raise ValueError('Unsupported content part')
            if not any(p['type']=='text' and p['text'].strip() for p in messages[-1]['content']): raise ValueError('Empty assistant')
            # Exclude the answer: changed labels must not hide identical input leakage.
            key=hashlib.sha256(json.dumps(messages[:-1],sort_keys=True).encode()).hexdigest()
            if key in contents and contents[key]!=split: raise ValueError('Prompt leaks across splits')
            contents[key]=split
            kinds[split]['image' if image_count else 'text']+=1
        if not counts[split]: raise ValueError(f'{split} is empty')
    return counts, {k:dict(v) for k,v in kinds.items()}

def build(dataset, model, adapter=None, seq_len=32768, run_name='astra-mixed-sft', epochs=2., lr=1e-5,
          batch_size=4, steps=None, checkpoint_interval=None, validation_interval=None, activation_offload=False, optimizer_offload=False, fresh_base=False,
          warmup_steps=0, min_lr=None):
    from prime_rl.configs.sft import SFTConfig
    if bool(adapter) == bool(fresh_base): raise ValueError('Select exactly one initializer: --init-adapter or --fresh-base')
    dataset,model=[Path(p).resolve(strict=True) for p in (dataset,model)]
    import sys
    sys.path.insert(0,str(HERE.parents[1]/'prime/watercolour_sft'))
    from model_profiles import profile, verify_snapshot
    selected=profile()
    if model.name != selected['revision']: raise ValueError('Use selected pinned Qwen base snapshot')
    if selected['name']!='qwen35_9b':verify_snapshot(model)
    counts,kinds=validate_data(dataset)
    if seq_len not in (8192,16384,32768,65536): raise ValueError('Unsupported context length')
    if not 0<epochs<=4 or not 0<lr<=2e-5: raise ValueError('Pilot epochs (0,4], LR (0,2e-5] required')
    if not 1<=batch_size<=32: raise ValueError('Batch size must be 1..32')
    if adapter:
        adapter=Path(adapter).resolve(strict=True)
        metadata=json.loads((adapter/'adapter_config.json').read_text())
        if not (adapter/'adapter_model.safetensors').is_file(): raise ValueError('Missing adapter weights')
        if metadata.get('r')!=16 or metadata.get('lora_alpha')!=32 or set(metadata['target_modules'])!=set(shared.TARGETS): raise ValueError('Incompatible adapter')
    scheduled = schedule.validate(dataset, batch_size, steps)
    steps = scheduled['max_steps'] if scheduled else (steps if steps is not None else math.ceil(epochs*counts['train']/batch_size))
    if not 1<=steps<=1000: raise ValueError('Pilot steps must be 1..1000')
    if type(warmup_steps) is not int or not 0<=warmup_steps<steps:raise ValueError('Warmup must be smaller than run length')
    if min_lr is not None and not 0<=min_lr<=lr:raise ValueError('Minimum LR must be between zero and peak LR')
    checkpoint_interval=checkpoint_interval or max(1,math.ceil(steps/2))
    validation_interval=validation_interval or checkpoint_interval
    data={'type':'sft','name':str(dataset),'batch_size':batch_size,'micro_batch_size':1,'seq_len':seq_len,'num_workers':1,'seed':42,
          'loss_mask':{'system':False,'user':False,'assistant':True,'tool':False}}
    config={'max_steps':steps,'run':{'name':run_name},'deployment':{'gpus_per_node':1,'num_train_gpus':1},
      'model':{'name':str(model),'impl':'custom','compile':'None','attn':'auto','optimization_dtype':'bfloat16','reduce_dtype':'bfloat16',
        'vlm':{'vision_encoder_attr':'model.visual','language_model_attr':'model.language_model','freeze_vision_encoder':True},
        'lora':{'rank':16,'alpha':32,'target_modules':shared.TARGETS},'ac':{'freq':1},
        'ac_offloading':{} if activation_offload else 'None','optim_cpu_offload':optimizer_offload},
      'renderer':{'name':'qwen3.5','enable_thinking':False},
      'data':{**data,'splits':['train'],'shuffle':not bool(scheduled)},
      'val':{'interval':validation_interval,'eval_on_start':True,'data':{**data,'splits':['validation'],'shuffle':False}},
      'optim':{'type':'adamw','lr':lr,'betas1':.9,'betas2':.999,'weight_decay':.01,'max_norm':1.},
      'scheduler':{'type':'cosine','warmup_steps':warmup_steps,'min_lr':min_lr if min_lr is not None else lr*.5} if warmup_steps or min_lr is not None else {'type':'constant'},
      'ckpt':{'interval':checkpoint_interval,'keep_last':1},'monitors':{'file':{}}}
    resolved = SFTConfig.model_validate(copy.deepcopy(config))
    if scheduled:
        schedule.guard_runtime(resolved, {})
    report={'counts':counts,'modalities':kinds,'steps':steps,'approx_epochs':steps*batch_size/counts['train'],
      'examples_per_optimizer_step':batch_size,'packing':'one_example_per_sequence','init_adapter':str(adapter) if adapter else None,'optimizer':'fresh',
      'initialization_mode':'fresh_base' if fresh_base else 'trained_adapter',
      'base_model':selected['model_id'],'base_revision':selected['revision'],'native_resume':False,'checkpoint_keep_last':1,'checkpoint_peak_native_snapshots':2,
      'initialization_audit_required':True,'step_zero_export_required':bool(fresh_base),
      'seq_len':seq_len,'activation_offload':activation_offload,'optimizer_offload':optimizer_offload,'mask':'assistant_only','length_policy':'fail_closed_full_render_no_truncation',
      'optimizer_config':config['optim'],'scheduler':config['scheduler']}
    if scheduled:
        report['schedule'] = {k: scheduled[k] for k in ('version','max_steps','row_count','step_counts','painting_target_passes','train_sha256')}
        report['mixture_note'] = 'Family-homogeneous optimizer steps; row and token fractions are not exact gradient weights.'
    return config,report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','model-dir','output'): p.add_argument('--'+name,type=Path,required=True)
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--init-adapter',type=Path);init.add_argument('--fresh-base',action='store_true')
    p.add_argument('--seq-len',type=int,default=32768);p.add_argument('--run-name',default='astra-mixed-sft')
    p.add_argument('--epochs',type=float,default=2.);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--batch-size',type=int,default=4)
    for name in ('steps','checkpoint-interval','validation-interval'):p.add_argument('--'+name,type=int)
    p.add_argument('--activation-offload',type=int,choices=(0,1),default=0);p.add_argument('--optimizer-offload',type=int,choices=(0,1),default=0)
    p.add_argument('--warmup-steps',type=int,default=0);p.add_argument('--min-lr',type=float)
    a=p.parse_args();config,report=build(a.dataset,a.model_dir,a.init_adapter,a.seq_len,a.run_name,a.epochs,a.lr,a.batch_size,a.steps,a.checkpoint_interval,a.validation_interval,bool(a.activation_offload),bool(a.optimizer_offload),a.fresh_base,a.warmup_steps,a.min_lr)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(tomli_w.dumps(config));a.output.with_suffix('.setup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
