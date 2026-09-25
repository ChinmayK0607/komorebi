"""Validate self-contained image/code data and resolve the bounded VLM SFT config."""
import argparse
import copy
import base64
import hashlib
import json
import math
from pathlib import Path
import tomli_w

REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
           'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']


def validate_data(folder):
    identities = set()
    image_splits = {}
    counts = {}
    for split in ['train', 'validation']:
        path = folder / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'{split} is empty')
        counts[split] = len(rows)
        for row in rows:
            if not row.get('id') or row['id'] in identities:
                raise ValueError('Missing or duplicate id')
            identities.add(row['id'])
            messages = row['messages']
            if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                raise ValueError('Expected one system/user/assistant conversation')
            if any(not isinstance(message['content'], list) for message in messages):
                raise ValueError('All roles must use typed content-part lists for Arrow compatibility')
            completion = messages[-1]['content']
            if len(completion) != 1 or completion[0].get('type') != 'text' or not completion[0].get('text', '').strip():
                raise ValueError('Assistant code must be one nonempty text part')
            images = [part['image_url']['url'] for part in messages[1]['content'] if part['type'] == 'image_url']
            if len(images) != 1 or not images[0].startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
                raise ValueError('Exactly one self-contained reference image required')
            raw = base64.b64decode(images[0].split(',', 1)[1], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in image_splits and image_splits[digest] != split:
                raise ValueError('Reference image leaks between train and validation')
            image_splits[digest] = split
    return counts


def build(dataset, model, seq_len=8192, run_name='watercolour-sft', smoke=False,
          steps=None, epochs=2., lr=5e-5, validation_interval=None, checkpoint_interval=None, init_adapter=None):
    from prime_rl.configs.sft import SFTConfig
    dataset, model = dataset.resolve(strict=True), model.resolve(strict=True)
    if model.name != REVISION:
        raise ValueError('Use the locked Qwen3.5-9B base snapshot')
    counts = validate_data(dataset)
    if seq_len not in (8192, 16384):
        raise ValueError('Pilot sequence length must be 8192 or 16384')
    if not 0 < epochs <= 16 or not 0 < lr <= 5e-5:
        raise ValueError('Pilot epochs must be (0,16], LR (0,5e-5]')
    steps = 2 if smoke else (steps if steps is not None else min(200, math.ceil(epochs * counts['train'] / 4)))
    if not 1 <= steps <= 200:
        raise ValueError('Pilot steps must be 1..200')
    validation_interval = max(1, steps-1) if validation_interval is None else validation_interval
    checkpoint_interval = steps if checkpoint_interval is None else checkpoint_interval
    if not 1 <= validation_interval <= steps or not 1 <= checkpoint_interval <= steps:
        raise ValueError('Validation/checkpoint intervals must be 1..steps')
    if init_adapter:
        init_adapter = Path(init_adapter).resolve(strict=True)
        if not (init_adapter/'adapter_model.safetensors').is_file():
            raise ValueError('Missing source adapter safetensors')
        metadata = json.loads((init_adapter/'adapter_config.json').read_text())
        if metadata.get('r') != 16 or metadata.get('lora_alpha') != 32 or set(metadata['target_modules']) != set(TARGETS):
            raise ValueError('Incompatible source adapter configuration')
    data = {'type': 'sft', 'name': str(dataset), 'batch_size': 4, 'micro_batch_size': 1,
            'seq_len': seq_len, 'num_workers': 1, 'seed': 42,
            'loss_mask': {'system': False, 'user': False, 'assistant': True, 'tool': False}}
    config = {'max_steps': steps, 'run': {'name': run_name},
        'deployment': {'gpus_per_node': 1, 'num_train_gpus': 1},
        'model': {'name': str(model), 'impl': 'custom', 'compile': 'None', 'attn': 'auto',
                  'optimization_dtype': 'bfloat16', 'reduce_dtype': 'bfloat16',
                  'vlm': {'vision_encoder_attr': 'model.visual', 'language_model_attr': 'model.language_model',
                          'freeze_vision_encoder': True},
                  'lora': {'rank': 16, 'alpha': 32, 'target_modules': TARGETS}, 'ac': {'freq': 1}},
        'renderer': {'name': 'qwen3.5', 'enable_thinking': False},
        'data': {**data, 'splits': ['train'], 'shuffle': True},
        'val': {'interval': validation_interval, 'eval_on_start': True,
                'data': {**data, 'splits': ['validation'], 'shuffle': False}},
        'optim': {'lr': lr}, 'ckpt': {'interval': checkpoint_interval, 'keep_last': 2},
        'monitors': {'file': {}}}
    SFTConfig.model_validate(copy.deepcopy(config))
    return config, {'counts': counts, 'steps': steps, 'examples_per_optimizer_step': 4,
                    'approx_epochs': steps * 4 / counts['train'], 'packing': 'one_example_per_sequence',
                    'init_adapter': str(init_adapter) if init_adapter else None, 'optimizer': 'fresh'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--seq-len', type=int, default=8192)
    p.add_argument('--run-name', default='watercolour-sft')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--steps', type=int)
    p.add_argument('--epochs', type=float, default=2.)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--validation-interval', type=int)
    p.add_argument('--checkpoint-interval', type=int)
    p.add_argument('--init-adapter', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config, report = build(a.dataset, a.model_dir, a.seq_len, a.run_name, a.smoke,
                          a.steps, a.epochs, a.lr, a.validation_interval, a.checkpoint_interval, a.init_adapter)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(tomli_w.dumps(config))
    a.output.with_suffix('.setup.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
