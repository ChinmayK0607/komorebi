import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location('scheduled_under_test', Path(__file__).with_name('scheduled_data.py'))
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


def row(ident, family):
    return {'id': ident, 'split': 'train', 'task_family': family,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': ident}]},
                         {'role': 'assistant', 'content': [{'type': 'text', 'text': 'a'}]}]}


def sources(n=64):
    return ([row(f'photo{i//2}_{i%2}', 'painting') for i in range(n)],
            [row(f'{group}{i}', group) for group in schedule.RETENTION for i in range(12)])


def write(folder, rows, meta):
    (folder / 'train.jsonl').write_bytes(schedule.encoded_rows(rows))
    (folder / 'schedule.json').write_text(json.dumps(meta))


class Tests(unittest.TestCase):
    def test_concentrated_reconstruction_requires_explicit_bounded_mode(self):
        paint, retained = sources(6)
        with self.assertRaisesRegex(ValueError, '1..8'):
            schedule.build(paint, retained, repetitions=256)
        rows, meta = schedule.build(paint, retained, repetitions=256, concentrated=True)
        self.assertEqual(meta['max_steps'], 512)
        self.assertEqual(len(rows), 2048)
        self.assertEqual(meta['step_counts']['painting'], 384)
        self.assertEqual(sum(v for k,v in meta['step_counts'].items() if k!='painting'),128)
        self.assertTrue(meta['concentrated_reconstruction'])
        self.assertTrue(all(meta['exposures_by_original_id'][r['id']]==256 for r in paint))
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);write(folder,rows,meta)
            self.assertEqual(schedule.validate(folder,4,512),meta)
            for flag in [None,False,'true',1]:
                altered=copy.deepcopy(meta)
                if flag is None:altered.pop('concentrated_reconstruction')
                else:altered['concentrated_reconstruction']=flag
                write(folder,rows,altered)
                with self.assertRaises(ValueError):schedule.validate(folder,4,512)
            ordinary_rows,ordinary_meta=schedule.build(*sources(12),repetitions=4)
            ordinary_meta['concentrated_reconstruction']=True
            write(folder,ordinary_rows,ordinary_meta)
            with self.assertRaisesRegex(ValueError,'eight painting targets'):schedule.validate(folder)
        with self.assertRaisesRegex(ValueError,'eight painting targets'):
            schedule.build(*sources(12),repetitions=256,concentrated=True)
        with self.assertRaisesRegex(ValueError,'1..256'):
            schedule.build(paint,retained,repetitions=257,concentrated=True)

    def test_exact_exposure_partial_cycle_and_optimizer_groups(self):
        paint, retained = sources()
        rows, meta = schedule.build(paint, retained)
        # 32 photos, two targets each, four passes: 64 painting updates,
        # 21 retention updates. Never invent examples to fill the last cycle.
        self.assertEqual(len(rows), 340)
        self.assertEqual(meta['max_steps'], 85)
        self.assertEqual(meta['step_counts'], {'painting':64,'retention_text':7,'retention_chart':7,'retention_document':7})
        counts = collections.Counter(r['source_example_id'] for r in rows)
        self.assertTrue(all(counts[p['id']] == 4 for p in paint))
        for offset in range(0, len(rows), 4):
            self.assertEqual(len({r['task_family'] for r in rows[offset:offset+4]}), 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d); write(path, rows, meta)
            self.assertEqual(schedule.validate(path,4,85), meta)
            with self.assertRaisesRegex(ValueError,'exactly'):
                schedule.validate(path,4,84)
            with self.assertRaisesRegex(ValueError,'batch size'):
                schedule.validate(path,8,85)

    def test_reordered_rows_rejected_even_with_rehashed_manifest(self):
        rows, meta = schedule.build(*sources())
        rows[11], rows[12] = rows[12], rows[11]
        meta['train_sha256'] = hashlib.sha256(schedule.encoded_rows(rows)).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path,rows,meta)
            with self.assertRaisesRegex(ValueError,'order/family'):
                schedule.validate(path)

    def test_changed_repeated_answer_rejected(self):
        rows, meta = schedule.build(*sources())
        source = rows[0]['source_example_id']
        duplicate = next(r for r in rows[1:] if r['source_example_id'] == source)
        duplicate['messages'][-1]['content'][0]['text'] = 'changed'
        meta['train_sha256'] = hashlib.sha256(schedule.encoded_rows(rows)).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path,rows,meta)
            with self.assertRaisesRegex(ValueError,'repeated source changed'):
                schedule.validate(path)

    def test_missing_or_cleared_manifest_cannot_shuffle_scheduled_rows(self):
        rows, meta = schedule.build(*sources())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path,rows,meta)
            (path/'schedule.json').unlink()
            with self.assertRaisesRegex(ValueError,'require their schedule'):
                schedule.validate(path)
            (path/'schedule.json').write_text('null\n')
            with self.assertRaisesRegex(ValueError,'require their schedule'):
                schedule.validate(path)
            (path/'train.jsonl').write_bytes(schedule.encoded_rows(sources()[0]))
            self.assertIsNone(schedule.validate(path))

    def test_runtime_guard_refuses_shuffling_parallelism_and_partial_resume(self):
        rows, meta = schedule.build(*sources())
        ns=types.SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path,rows,meta)
            config=ns(data=ns(name=str(path),batch_size=4,micro_batch_size=1,num_workers=1,shuffle=False,splits=['train'],subsets=None,probabilities=None),
                      max_steps=85,resume=None,deployment=ns(num_train_gpus=1,gpus_per_node=1),model=ns(cp=1))
            self.assertEqual(schedule.guard_runtime(config,{}),meta)
            for field,value in [('shuffle',True),('micro_batch_size',2),('num_workers',2)]:
                changed=copy.deepcopy(config);setattr(changed.data,field,value)
                with self.assertRaisesRegex(ValueError,'Ordered schedule'):
                    schedule.guard_runtime(changed,{})
            with self.assertRaisesRegex(ValueError,'one GPU process'):
                schedule.guard_runtime(config,{'WORLD_SIZE':'2'})
            changed=copy.deepcopy(config);changed.data.subsets=['unexpected']
            with self.assertRaisesRegex(ValueError,'forbids subsets'):
                schedule.guard_runtime(changed,{})
            changed=copy.deepcopy(config);changed.model.cp=2
            with self.assertRaisesRegex(ValueError,'context parallelism'):
                schedule.guard_runtime(changed,{})
            config.resume=ns(step=32)
            with self.assertRaisesRegex(ValueError,'partial optimizer'):
                schedule.guard_runtime(config,{})

    def test_real_hf_loader_ignores_unbound_stale_shards_and_readme(self):
        rows,meta=schedule.build(*sources())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path,rows,meta)
            (path/'train-old.jsonl').write_bytes(schedule.encoded_rows([row('stale','painting')]))
            (path/'README.md').write_text('---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train-old.jsonl\n---\n')
            dataset=schedule.load_exact_split(path,'train')
            self.assertEqual(list(dataset['id']),[r['id'] for r in rows])
            config=types.SimpleNamespace(name=str(path),splits=['train'],subsets=None,probabilities=None)
            loaded=schedule.load_bound_config(config)
            self.assertEqual(list(loaded['id']),[r['id'] for r in rows])
            self.assertEqual(list(loaded['__index']),list(range(len(rows))))

    def test_real_hf_loader_unions_late_columns_and_preserves_nested_content(self):
        old = row('old_validation', 'retention_text'); old['split'] = 'validation'
        new = row('new_validation', 'painting'); new.update(split='validation', mixture_group='painting',
            category='flower', reference_id='new_reference', action_contract='bounded-brush-v2', target_level='final')
        new['messages'][0]['content'].append({'type':'image_url','image_url':{'url':'data:image/png;base64,exact-image-bytes','detail':'original'}})
        new['messages'][-1]['content'][0]['text']='{"version":"bounded-brush-v2","actions":[{"type":"STOP"}]}'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d); (path/'train.jsonl').write_bytes(schedule.encoded_rows([row('train','retention_text')]))
            raw=schedule.encoded_rows([old,new]); (path/'validation.jsonl').write_bytes(raw)
            dataset=schedule.load_exact_split(path,'validation')
            self.assertEqual(list(dataset['id']),['old_validation','new_validation'])
            for key in ('mixture_group','category','reference_id','action_contract','target_level'):
                self.assertIn(key,dataset.column_names)
                self.assertIsNone(dataset[0][key]);self.assertEqual(dataset[1][key],new[key])
            self.assertEqual(dataset[0]['messages'][0]['content'][0]['text'],'old_validation')
            self.assertEqual(dataset[1]['messages'][0]['content'][1]['image_url'],new['messages'][0]['content'][1]['image_url'])
            self.assertEqual(dataset[1]['messages'][-1]['content'][0]['text'],new['messages'][-1]['content'][0]['text'])
            self.assertEqual((path/'validation.jsonl').read_bytes(),raw)

    def test_real_hf_loader_actual_package_both_splits_without_file_changes(self):
        folder=Path(__file__).resolve().parents[3]/'data/astra-high-structure-v1'
        if not folder.exists():self.skipTest('Actual frozen package unavailable')
        def preserves(original,loaded):
            # Arrow pads absent nested struct keys with None; every original
            # value and list position must survive exactly, without schema loss.
            if isinstance(original,dict):
                self.assertIsInstance(loaded,dict)
                for key,value in original.items():
                    self.assertIn(key,loaded);preserves(value,loaded[key])
                self.assertTrue(all(value is None for key,value in loaded.items() if key not in original))
            elif isinstance(original,list):
                self.assertEqual(len(original),len(loaded))
                for left,right in zip(original,loaded):preserves(left,right)
            else:self.assertEqual(original,loaded)
        for split,count in [('train',260),('validation',138)]:
            path=folder/(split+'.jsonl');raw=path.read_bytes();original=[json.loads(line) for line in raw.splitlines()]
            dataset=schedule.load_exact_split(folder,split)
            self.assertEqual(len(dataset),count);self.assertEqual(list(dataset['id']),[r['id'] for r in original])
            self.assertEqual(set(dataset.column_names),{key for r in original for key in r})
            for row_index,source in enumerate(original):preserves(source,dataset[row_index])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),hashlib.sha256(raw).hexdigest())

    def test_exact_loader_rejects_duplicate_validation_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'train.jsonl').write_bytes(schedule.encoded_rows([row('train','retention_text')]))
            duplicate=row('duplicate','painting');duplicate['split']='validation'
            (path/'validation.jsonl').write_bytes(schedule.encoded_rows([duplicate,duplicate]))
            with self.assertRaisesRegex(ValueError,'unique nonempty row IDs'):schedule.load_exact_split(path,'validation')


if __name__ == '__main__':
    unittest.main()
