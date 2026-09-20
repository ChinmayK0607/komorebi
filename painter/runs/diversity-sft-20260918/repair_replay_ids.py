"""One-time metadata-only repair of sampled replay identities on the node."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
assert sys.platform == 'linux'
path = P / 'sft-data/train.jsonl'
receipt_path = P / 'dataset-receipt.json'
assert not (P / 'replay-id-repair.json').exists(), 'Repair already performed'
assert not (P / 'train-output/diversity-sft-20260918/artifacts/optimizer-update-audit.jsonl').exists()
original = path.read_bytes()
receipt = json.loads(receipt_path.read_text())
assert hashlib.sha256(original).hexdigest() == receipt['train_sha256']
rows = [json.loads(line) for line in original.splitlines() if line.strip()]
duplicates = sum(n - 1 for n in Counter(r['id'] for r in rows).values())
assert len(rows) == 3072 and duplicates > 0
name = json.loads((P / 'EXPERIMENT.json').read_text())['run_name']
corrected = [dict(row, source_example_id=row.get('source_example_id', row['id']),
                  id=f'{name}-draw-{index:06d}') for index, row in enumerate(rows)]
assert len({r['id'] for r in corrected}) == len(corrected)
assert all({k:v for k,v in before.items() if k not in ('id','source_example_id')} ==
           {k:v for k,v in after.items() if k not in ('id','source_example_id')}
           for before, after in zip(rows, corrected))
path.rename(path.with_name('train.before-replay-id-fix.jsonl'))
path.write_text(''.join(json.dumps(row)+'\n' for row in corrected))
receipt_path.rename(receipt_path.with_name('dataset-receipt.before-replay-id-fix.json'))
receipt.update(train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               ordered_training_ids=[r['id'] for r in corrected],
               ordered_training_source_example_ids=[r['source_example_id'] for r in corrected])
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
sys.path.insert(0, str(P / 'source/brush-rl/integrations/speedpainting/mixed_sft'))
from scheduled_data import load_exact_split
train = load_exact_split(P / 'sft-data', 'train')
validation = load_exact_split(P / 'sft-data', 'validation')
assert len(train) == len(rows) and len(validation) == 48
assert list(train['id']) == receipt['ordered_training_ids']
assert hashlib.sha256((P / 'sft-data/validation.jsonl').read_bytes()).hexdigest() == receipt['validation_sha256']
report = dict(actual_loader_passed=True, rows=len(train), validation_rows=len(validation),
              repeated_source_draws=duplicates, unique_draw_ids=True,
              messages_and_learning_targets_unchanged=True,
              old_sha256=hashlib.sha256(original).hexdigest(), new_sha256=receipt['train_sha256'])
(P / 'replay-id-repair.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
