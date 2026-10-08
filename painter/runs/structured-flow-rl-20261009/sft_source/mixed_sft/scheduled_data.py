"""Exact, pre-materialized family-homogeneous optimizer steps for one GPU.

No loss rewrite: a retention update is normalized by its own assistant tokens.
Rows may repeat explicitly; each emission keeps its original example identity.
"""
import collections
import copy
import hashlib
import json
import random
from pathlib import Path

VERSION = 'brush-step-schedule-v1'
RETENTION = ('retention_text', 'retention_chart', 'retention_document')
PATTERN = tuple(group for retained in RETENTION for group in ('painting', 'painting', 'painting', retained))
EMISSION_FIELDS = {'id', 'source_example_id', 'scheduled_step', 'scheduled_slot', 'schedule_group'}


def encoded_rows(rows):
    return ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows).encode()


def original_payload(row):
    result = {k: v for k, v in row.items() if k not in EMISSION_FIELDS}
    result['id'] = row['source_example_id']
    return result


def build(painting, retained, repetitions=4, seed=20260910, *, concentrated=False):
    # An explicit small reconstruction probe can receive enough practice to
    # test learning. Ordinary diverse-curriculum limits remain unchanged.
    if type(concentrated) is not bool:
        raise ValueError('concentrated must be an explicit boolean')
    limit = 256 if concentrated else 8
    if type(repetitions) is not int or not 1 <= repetitions <= limit:
        raise ValueError(f'Use 1..{limit} explicit passes per painting target')
    if concentrated and len(painting) > 8:
        raise ValueError('Concentrated reconstruction is limited to eight painting targets')
    all_sources = painting + retained
    if len({r['id'] for r in all_sources}) != len(all_sources):
        raise ValueError('Duplicate original example ID')
    if not painting or any(r['task_family'] != 'painting' for r in painting):
        raise ValueError('Nonempty painting pool required')
    if any(r.get('split') != 'train' for r in all_sources):
        raise ValueError('Only training sources may enter the schedule')
    if any(r['task_family'] not in RETENTION for r in retained):
        raise ValueError('Unexpected retention family')
    if len(painting) * repetitions % 4:
        raise ValueError('Painting exposure must divide into complete batches of four; no silent repeats/drops')
    painting_steps = len(painting) * repetitions // 4
    if painting_steps + painting_steps // 3 > 1000:
        raise ValueError('Schedule exceeds the 1000-update pilot limit')
    rng = random.Random(seed)
    paint_order = []
    for _ in range(repetitions):
        epoch = list(painting)
        rng.shuffle(epoch)
        paint_order.extend(epoch)
    pools = {g: [r for r in retained if r['task_family'] == g] for g in RETENTION}
    if any(len(pool) < 4 for pool in pools.values()):
        raise ValueError('Each retention family needs at least four distinct source examples')
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = collections.Counter()
    emissions, steps = [], []

    def emit(group, sources):
        step = len(steps) + 1
        if group != PATTERN[(step - 1) % len(PATTERN)]:
            raise ValueError('Schedule construction lost optimizer-step alignment')
        ids = []
        for slot, source in enumerate(sources):
            row = copy.deepcopy(source)
            ident = source['id'] + f'__schedule_{len(emissions):06d}'
            row.update(id=ident, source_example_id=source['id'], scheduled_step=step,
                       scheduled_slot=slot, schedule_group=group)
            emissions.append(row)
            ids.append(ident)
        steps.append({'step': step, 'group': group, 'emission_ids': ids})

    for offset in range(0, len(paint_order), 4):
        emit('painting', paint_order[offset:offset + 4])
        if (offset // 4 + 1) % 3 == 0:
            group = RETENTION[((offset // 4 + 1) // 3 - 1) % len(RETENTION)]
            samples = []
            for _ in range(4):
                if cursors[group] == len(pools[group]):
                    rng.shuffle(pools[group])
                    cursors[group] = 0
                samples.append(pools[group][cursors[group]])
                cursors[group] += 1
            emit(group, samples)
    metadata = {'version': VERSION, 'seed': seed, 'batch_size': 4,
                'max_steps': len(steps), 'row_count': len(emissions),
                'pattern': list(PATTERN), 'steps': steps,
                'painting_target_passes': repetitions,
                'step_counts': dict(collections.Counter(s['group'] for s in steps)),
                'exposures_by_original_id': dict(collections.Counter(r['source_example_id'] for r in emissions)),
                'train_sha256': hashlib.sha256(encoded_rows(emissions)).hexdigest(),
                'note': 'Ordered finite schedule, no PRIME shuffle. Four rows per optimizer step; three painting steps then one retention step rotating text/chart/document. Explicit repetitions are not new reference diversity. Stop at max_steps before dataset cycling.'}
    if concentrated:
        metadata['concentrated_reconstruction'] = True
    return emissions, metadata


def validate(folder, batch_size=None, steps=None):
    folder = Path(folder)
    path = folder / 'schedule.json'
    meta = json.loads(path.read_text()) if path.exists() else None
    if meta is None:
        # A missing manifest must not silently turn scheduled rows into a
        # shuffled legacy dataset. Explicit null clears stale pod metadata.
        with (folder / 'train.jsonl').open() as stream:
            if any('scheduled_step' in json.loads(line) for line in stream if line.strip()):
                raise ValueError('Scheduled rows require their schedule manifest')
        return None
    if meta.get('version') != VERSION or meta.get('pattern') != list(PATTERN):
        raise ValueError('Unknown ordered training schedule')
    concentrated = meta.get('concentrated_reconstruction', False)
    passes = meta.get('painting_target_passes')
    if type(concentrated) is not bool:
        raise ValueError('Concentrated reconstruction flag must be boolean')
    limit = 256 if concentrated else 8
    if type(passes) is not int or not 1 <= passes <= limit:
        raise ValueError('Painting passes require the explicit bounded reconstruction mode')
    if type(meta.get('batch_size')) is not int or meta['batch_size'] != 4:
        raise ValueError('Scheduled optimizer batch must be four')
    if batch_size is not None and batch_size != 4:
        raise ValueError('Scheduled data requires effective batch size four')
    n = meta.get('max_steps')
    if type(n) is not int or not 1 <= n <= 1000 or (steps is not None and steps != n):
        raise ValueError('Training must consume exactly the declared schedule steps')
    raw = (folder / 'train.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['train_sha256']:
        raise ValueError('Scheduled training file hash mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) != n * 4 or meta['row_count'] != len(rows) or len(meta['steps']) != n:
        raise ValueError('Incomplete scheduled optimizer batch')
    seen, originals = set(), {}
    for index, row in enumerate(rows):
        step, slot = divmod(index, 4)
        group = PATTERN[step % len(PATTERN)]
        if (row.get('scheduled_step') != step + 1 or row.get('scheduled_slot') != slot
                or row.get('schedule_group') != group or row.get('task_family') != group
                or row.get('split') != 'train'):
            raise ValueError('Scheduled row order/family differs from optimizer-step contract')
        if not isinstance(row.get('id'), str) or row['id'] in seen:
            raise ValueError('Duplicate or missing emission ID')
        source = row.get('source_example_id')
        if not isinstance(source, str) or not source:
            raise ValueError('Original source identity required')
        seen.add(row['id'])
        payload = json.dumps(original_payload(row), sort_keys=True, separators=(',', ':'))
        if source in originals and originals[source] != payload:
            raise ValueError('A repeated source changed its target or input')
        originals[source] = payload
    for i, record in enumerate(meta['steps']):
        if record != {'step': i + 1, 'group': PATTERN[i % len(PATTERN)],
                      'emission_ids': [r['id'] for r in rows[4 * i:4 * i + 4]]}:
            raise ValueError('Step receipt differs from scheduled data')
    counts = dict(collections.Counter(r['source_example_id'] for r in rows))
    if counts != meta['exposures_by_original_id']:
        raise ValueError('Incorrect exposure accounting')
    actual_groups = dict(collections.Counter(s['group'] for s in meta['steps']))
    if actual_groups != meta['step_counts']:
        raise ValueError('Incorrect optimizer-family accounting')
    paint_sources = {r['source_example_id'] for r in rows if r['task_family'] == 'painting'}
    if concentrated and len(paint_sources) > 8:
        raise ValueError('Concentrated reconstruction is limited to eight painting targets')
    if any(counts[s] != meta['painting_target_passes'] for s in paint_sources):
        raise ValueError('Incorrect painting target exposure')
    return meta


def guard_runtime(config, environ):
    meta = validate(config.data.name, config.data.batch_size, config.max_steps)
    if meta is None:
        return None
    data = config.data
    if data.shuffle or data.micro_batch_size != 1 or data.num_workers != 1 or data.splits != ['train']:
        raise ValueError('Ordered schedule requires no shuffle, one worker, microbatch one and train split only')
    if data.subsets is not None or data.probabilities is not None or config.model.cp != 1:
        raise ValueError('Ordered schedule forbids subsets, interleaving and context parallelism')
    if config.resume is not None:
        raise ValueError('Scheduled pilot does not support partial optimizer/dataloader resume; use a new explicit run')
    if any(int(environ.get(key, '1')) != 1 for key in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE')):
        raise ValueError('Ordered scheduled pilot supports exactly one GPU process')
    if config.deployment.num_train_gpus != 1 or config.deployment.gpus_per_node != 1:
        raise ValueError('Ordered scheduled pilot supports exactly one GPU')
    return meta


def load_exact_split(folder, split):
    """Read one frozen file with its complete schema, ignoring stale configs.

    The JSON streaming builder fixes its Arrow schema from the first chunk,
    which can omit columns introduced by appended painting validation rows.
    Dataset.from_list also takes top-level names only from the first row, so
    explicitly union those names before asking Arrow to infer their types.
    """
    from datasets import Dataset
    if split not in ('train', 'validation'):
        raise ValueError('Mixed SFT supports exact train/validation files only')
    folder = Path(folder)
    meta = validate(folder)
    rows = [json.loads(line) for line in (folder / (split + '.jsonl')).read_bytes().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Exact mixed SFT split requires nonempty JSON object rows')
    expected_ids = [row.get('id') for row in rows]
    if any(not isinstance(ident, str) or not ident for ident in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Exact mixed SFT split requires unique nonempty row IDs')
    columns = dict.fromkeys(key for row in rows for key in row)
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    dataset = Dataset.from_list(normalized)
    if list(dataset['id']) != expected_ids:
        raise ValueError('Actual HF loaded IDs differ from exact file order')
    if meta is not None and split == 'train':
        expected = [ident for step in meta['steps'] for ident in step['emission_ids']]
        if list(dataset['id']) != expected:
            raise ValueError('Actual HF loaded training IDs differ from scheduled file order')
    return dataset


def load_bound_config(config):
    """Replacement for PRIME's directory-discovering load_sft_dataset."""
    if config.subsets is not None or config.probabilities is not None or len(config.splits or []) != 1:
        raise ValueError('Mixed SFT uses exactly one explicit split, without subsets/interleaving')
    split, = config.splits
    dataset = load_exact_split(config.name, split)
    for name, values in (('__subset', [None] * len(dataset)), ('__split', [split] * len(dataset)),
                         ('__index', list(range(len(dataset))))):
        dataset = dataset.add_column(name, values)
    return dataset
