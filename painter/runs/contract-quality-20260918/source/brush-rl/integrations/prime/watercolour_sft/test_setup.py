"""Run locally with datasets and PRIME config packages; no GPU needed."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from prepare import build, validate_data


class SetupTests(unittest.TestCase):
    def test_actual_curated_arrow_load_preserves_images_and_code(self):
        from datasets import load_dataset
        folder = PROJECT / 'data/watercolour/sft'
        counts = validate_data(folder)
        loaded = load_dataset(str(folder))
        self.assertEqual(set(loaded), {'train', 'validation'})
        for split in counts:
            self.assertEqual(len(loaded[split]), counts[split])
            originals = [json.loads(line) for line in (folder/f'{split}.jsonl').read_text().splitlines()]
            for raw, row in zip(originals, loaded[split]):
                self.assertEqual(row['messages'][2]['content'][0]['text'], raw['messages'][2]['content'][0]['text'])
                self.assertEqual(row['messages'][1]['content'][1]['image_url'], raw['messages'][1]['content'][1]['image_url'])

    def test_validated_config_has_bounded_full_language_lora(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            model=Path(t)/'c202236235762e1c871ad0ccb60c8ee5ba337b9a';model.mkdir()
            config, report=build(PROJECT/'data/watercolour/sft',model)
            self.assertLessEqual(config['max_steps'],40)
            self.assertEqual(report['steps'],29)
            self.assertEqual(config['data']['batch_size'],4)
            self.assertEqual(config['data']['loss_mask'],{'system':False,'user':False,'assistant':True,'tool':False})
            self.assertTrue(config['model']['vlm']['freeze_vision_encoder'])
            self.assertIn('in_proj_qkv',config['model']['lora']['target_modules'])
            self.assertIn('out_proj',config['model']['lora']['target_modules'])

    def test_export_hook_writes_only_after_native_success_and_fails_closed(self):
        import tempfile
        import export_adapter
        from types import SimpleNamespace
        events=[]
        class Manager:
            def save(self,*args): events.append('native-save')
        class Sender:
            def __init__(self,*args): pass
            def _broadcast(self,model,step,folder):
                events.append('serialize-adapter')
                (folder/'adapter_config.json').write_text('{}')
        modules={}
        for name in ['prime_rl.configs.trainer','prime_rl.trainer.ckpt',
                     'prime_rl.transports.weights.filesystem','prime_rl.trainer.world','torch','torch.distributed']:
            modules[name]=types.ModuleType(name)
        modules['prime_rl.configs.trainer'].FileSystemWeightBroadcastConfig=lambda:None
        modules['prime_rl.trainer.ckpt'].CheckpointManager=Manager
        modules['prime_rl.transports.weights.filesystem'].FileSystemWeightSender=Sender
        modules['prime_rl.trainer.world'].get_world=lambda:SimpleNamespace(is_master=True)
        modules['torch'].distributed=modules['torch.distributed']
        modules['torch.distributed'].barrier=lambda:None
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules), patch.object(export_adapter, 'cloud_settings', return_value=('owner/repo','run',lambda *a: {'verified':True,'url':'https://huggingface.co/owner/repo'})):
            config=SimpleNamespace(model=SimpleNamespace(lora=object()),run_dir=Path(tmp))
            export_adapter.install_export_hook(config)
            with patch.object(export_adapter,'validate_files',return_value={'sha256':'verified'}):
                Manager().save(29,object(),[],None,None)
            folder=Path(tmp)/'artifacts/adapters/step_29'
            self.assertTrue((folder/'.finished').exists())
            self.assertEqual(events,['native-save','serialize-adapter'])
            with patch.object(export_adapter,'validate_files',return_value={'sha256':'verified'}), patch.object(export_adapter, 'cloud_settings', return_value=('owner/repo','run',lambda *a: (_ for _ in ()).throw(RuntimeError('upload failed')))):
                export_adapter.install_export_hook(config)
                with self.assertRaises(RuntimeError):Manager().save(29,object(),[],None,None)
            self.assertFalse((folder/'.finished').exists())
            with patch.object(export_adapter,'validate_files',side_effect=ValueError('nonfinite')):
                with self.assertRaises(ValueError):Manager().save(29,object(),[],None,None)
            self.assertFalse((folder/'.finished').exists())

    def test_isolated_padding_against_actual_pinned_prime_packer(self):
        # Execute the exact pinned upstream CatDataset class, whose padding path
        # is pure Python. This tests against upstream without importing GPU libs.
        repo = next(p for p in HERE.parents if (p/'work/prime-rl-upstream').exists())/'work/prime-rl-upstream'
        source=ast.parse((repo/'src/prime_rl/trainer/sft/data.py').read_text())
        klass=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='CatDataset')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),klass],type_ignores=[])
        scope={'IterableDataset':object,'StatefulIterableDataset':object,'get_logger':lambda:None}
        exec(compile(ast.fix_missing_locations(module),'<pinned CatDataset>','exec'),scope)
        data=types.ModuleType('data');data.CatDataset=scope['CatDataset']
        sft=types.ModuleType('prime_rl.trainer.sft');sft.data=data
        with patch.dict(sys.modules,{'prime_rl.trainer.sft':sft}):
            spec=importlib.util.spec_from_file_location('watercolour_test_train',HERE/'train.py')
            train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
        samples=[]
        for image in ['first-image','second-image']:
            samples.append({'input_ids':[10,20,30], 'target_ids':[20,30,40],
                            'position_ids':[0,1,2], 'loss_mask':[False,True,True],
                            'seq_lens':[3], 'mm_kwargs':{'pixel_values':image}, 'mm_token_type_ids':[1,0,0]})
        padded=list(train.SingleExampleDataset(samples,8))
        self.assertEqual(len(padded),2)
        for i,row in enumerate(padded):
            self.assertEqual(len(row['input_ids']),8)
            self.assertEqual(row['loss_mask'],[False,True,True]+[False]*5)
            self.assertEqual(row['mm_kwargs'],samples[i]['mm_kwargs'])
            self.assertEqual(row['mm_token_type_ids'],[1,0,0]+[0]*5)
            self.assertEqual(row['target_ids'][:3],samples[i]['target_ids'])

if __name__ == '__main__': unittest.main()
