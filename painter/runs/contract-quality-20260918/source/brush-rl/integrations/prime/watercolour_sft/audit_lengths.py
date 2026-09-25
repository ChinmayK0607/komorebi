"""Measure real VLM rendered token lengths and assistant masks before training."""
import argparse
import json
from pathlib import Path
import tomllib
from datasets import load_dataset
from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.sft.data import SFTDataset
from prime_rl.trainer.model import setup_processor, setup_tokenizer
from renderers.base import create_renderer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
config = SFTConfig.model_validate(tomllib.loads(a.config.read_text()))
tokenizer = setup_tokenizer(config.tokenizer)
renderer = create_renderer(tokenizer, config.renderer)
renderer._processor = setup_processor(config.model)
rows = []
for split in ['train', 'validation']:
    dataset = load_dataset(config.data.name, split=split)
    # Large audit window measures full sequences; the training window is checked below.
    wrapper = SFTDataset(dataset, renderer, seq_len=131072, multimodal=True)
    for row in dataset:
        sample = wrapper._process(row)
        if sample is None or not sample['mm_kwargs']:
            raise ValueError(f'{row["id"]}: missing trainable code or image tensors')
        total = len(sample['input_ids'])
        supervised = sum(sample['loss_mask'])
        image_types = sample.get('mm_token_type_ids')
        if image_types is None or not any(image_types):
            raise ValueError(f'{row["id"]}: missing expanded image-token mask')
        if any(mask and kind for mask, kind in zip(sample['loss_mask'], image_types)):
            raise ValueError(f'{row["id"]}: image tokens unexpectedly contribute to loss')
        if supervised <= 0 or supervised >= total:
            raise ValueError(f'{row["id"]}: invalid assistant-only supervision mask')
        rows.append({'id': row['id'], 'split': split, 'total_tokens': total,
                     'assistant_tokens': supervised, 'image_tokens': sum(bool(t) for t in image_types),
                     'image_tokens_masked': True, 'fits': total <= config.data.seq_len})
report = {'seq_len': config.data.seq_len, 'all_fit': all(r['fits'] for r in rows),
          'max_tokens': max(r['total_tokens'] for r in rows), 'examples': rows}
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'examples'}))
if not report['all_fit']:
    raise SystemExit('Some targets would truncate: curate shorter code or raise seq_len')
