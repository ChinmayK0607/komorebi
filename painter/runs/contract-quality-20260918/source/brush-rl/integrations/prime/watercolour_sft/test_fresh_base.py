"""CPU-only native-reset/audit/export tests; synthetic tensors, no model download."""
import ast
import asyncio
import functools
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import init_adapter as init
import export_adapter as export

torch.set_num_threads(1)

def fixture():
    # Synthetic rank-16 shapes with the production aggregate size, not Qwen weights.
    return {f'model.layers.{i}.proj.lora_{letter}.weight': torch.full(shape, .125 if letter == 'A' else 0.)
            for i in range(248) for letter, shape in [('A', (16,4096)), ('B', (7018 if i == 0 else 6810,16))]}

class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.state = fixture()

    def test_explicit_selection(self):
        self.assertEqual(init.initialization_mode('/adapter'), 'trained_adapter')
        self.assertEqual(init.initialization_mode(fresh_base=True), 'fresh_base')
        for args in [(None,False),('/adapter',True)]:
            with self.assertRaisesRegex(ValueError, 'exactly one'): init.initialization_mode(*args)

    def test_full_count_finite_and_zero_b(self):
        report = init.audit_fresh_adapter(self.state)
        self.assertEqual(report['parameters'],43278336)
        self.assertEqual(report['zero_b_tensors'],248)
        b = next(k for k in self.state if '.lora_B.' in k)
        bad = dict(self.state); bad[b] = self.state[b].clone(); bad[b][0,0] = 1
        with self.assertRaisesRegex(ValueError,'exactly zero'): init.audit_fresh_adapter(bad)
        bad[b][0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'): init.audit_fresh_adapter(bad)
        bad = dict(self.state); bad[b] = self.state[b].T
        with self.assertRaisesRegex(ValueError,'rank 16'): init.audit_fresh_adapter(bad)
        bad = dict(self.state); bad.pop(b)
        with self.assertRaisesRegex(ValueError,'496'): init.audit_fresh_adapter(bad)
        bad = dict(self.state); bad[b] = torch.zeros(1,16)
        with self.assertRaisesRegex(ValueError,'43,278,336'): init.audit_fresh_adapter(bad)

    def test_native_prime_linear_reset_is_random_a_zero_b(self):
        source=Path.cwd()/'work/prime-rl-upstream/src/prime_rl/trainer/models/layers/lora/multi_linear.py'
        tree=ast.parse(source.read_text())
        method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='reset_parameters')
        scope={'nn':torch.nn,'math':math}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
        layer=types.SimpleNamespace(lora_A=[torch.empty(16,32)],lora_B=[torch.full((32,16),10.)])
        scope['reset_parameters'](layer,0)
        self.assertTrue(torch.isfinite(layer.lora_A[0]).all())
        self.assertTrue(torch.count_nonzero(layer.lora_A[0]).item())
        self.assertEqual(torch.count_nonzero(layer.lora_B[0]).item(),0)

    def test_hook_order_real_model_and_single_reset(self):
        from torch.distributed.elastic.multiprocessing.errors import record
        events=[]; actual_model=object(); state=self.state
        class LoRA:
            def reset_adapter_parameters(self): events.append('native_reset')
            def adapter_state_dict(self): return state
        def setup(*args): events.append('model_setup'); return actual_model
        namespace={'setup_model':setup}
        exec('def train():\n model=setup_model()\n state.reset_adapter_parameters()\n events.append("optimizer")\n events.append("forward")',namespace)
        namespace.update(state=LoRA(),events=events)
        decorated=record(namespace['train'])
        self.assertIsNot(decorated.__globals__,namespace)
        self.assertNotIn('setup_model',decorated.__globals__)
        mods={'prime_rl.trainer.lora':types.SimpleNamespace(LoRAState=LoRA),
              'prime_rl.trainer.world':types.SimpleNamespace(get_world=lambda:types.SimpleNamespace(world_size=1))}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,mods):
            root=Path(d); base=root/init.BASE_REVISION; base.mkdir()
            config=types.SimpleNamespace(resume=None,run_dir=root/'run',model=types.SimpleNamespace(name=str(base),lora=types.SimpleNamespace(rank=16,alpha=32,target_modules=list(init.TARGETS))))
            def callback(model,step,**kwargs):
                self.assertIs(model,actual_model); self.assertEqual(step,0)
                self.assertEqual(kwargs,{'initialization_mode':'fresh_base'})
                self.assertNotIn('optimizer',events)
                self.assertEqual(json.loads((config.run_dir/'artifacts/initial-adapter-audit.json').read_text())['zero_b_tensors'],248)
                events.append('export_verified'); return {'verified':True,'private':False}
            init.install_fresh_base_hook(config,decorated,callback)
            decorated()
            self.assertEqual(events,['model_setup','native_reset','export_verified','optimizer','forward'])
            with self.assertRaisesRegex(RuntimeError,'exactly once'): namespace['state'].reset_adapter_parameters()
            report=json.loads((config.run_dir/'artifacts/initial-adapter-audit.json').read_text())
            self.assertEqual(report['step_zero_export'],'verified')
            config.resume=object()
            with self.assertRaisesRegex(ValueError,'resume'): init.install_fresh_base_hook(config,namespace['train'],callback)

    def test_pinned_decorated_effective_entrypoint_before_model_smoke(self):
        from torch.distributed.elastic.multiprocessing.errors import record
        workspace=Path.cwd()
        upstream=workspace/'work/prime-rl-upstream/src/prime_rl'
        # Execute the exact pinned clean_exit decorator with cleanup dependencies
        # stubbed, then group_validation.install on the full pinned trainer body.
        events=[]
        utils_tree=ast.parse((upstream/'utils/utils.py').read_text())
        node=next(n for n in utils_tree.body if isinstance(n,ast.FunctionDef) and n.name=='clean_exit')
        decorators={'Callable':object,'asyncio':asyncio,'functools':functools,
                    'dist':types.SimpleNamespace(is_initialized=lambda:events.append('cleanup') or False)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_clean_exit','exec'),decorators)
        module=types.ModuleType('pinned_trainer_fixture');module.__file__=str(upstream/'trainer/sft/train.py')
        class BeforeModel(BaseException): pass
        def stop_before_model(): events.append('entry'); raise BeforeModel()
        def no_model(*args,**kwargs): self.fail('Smoke must not load a model')
        module.__dict__.update(clean_exit=decorators['clean_exit'],SFTConfig=object,
                               setup_model=no_model,get_world=stop_before_model)
        path=workspace/'outputs/brush-rl/integrations/speedpainting/mixed_sft/group_validation.py'
        spec=importlib.util.spec_from_file_location('fresh_group_smoke',path)
        group=importlib.util.module_from_spec(spec);spec.loader.exec_module(group)
        class LoRA:
            def reset_adapter_parameters(self): self_test.fail('Smoke must not reset a model')
        self_test=self
        mods={'prime_rl.trainer.lora':types.SimpleNamespace(LoRAState=LoRA),
              'prime_rl.trainer.world':types.SimpleNamespace(get_world=lambda:None)}
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,mods):
            root=Path(d);base=root/init.BASE_REVISION;base.mkdir()
            with patch.object(group.importlib,'import_module',return_value=module):
                trainer=record(group.install(root/'setup'))
            self.assertNotIn('setup_model',trainer.__globals__)
            self.assertIs(inspect.unwrap(trainer).__globals__,module.__dict__)
            config=types.SimpleNamespace(resume=None,run_dir=root/'run',model=types.SimpleNamespace(name=str(base),lora=types.SimpleNamespace(rank=16,alpha=32,target_modules=list(init.TARGETS))))
            init.install_fresh_base_hook(config,trainer,lambda *a,**kw:self.fail('Smoke must not export'))
            self.assertIsNot(module.setup_model,no_model)
            with self.assertRaises(BeforeModel): trainer(config)
            self.assertEqual(events,['entry','cleanup'])
            self.assertFalse((config.run_dir/'artifacts').exists())

    def test_unwrapped_missing_or_noncallable_setup_fails_closed(self):
        from torch.distributed.elastic.multiprocessing.errors import record
        class LoRA:
            def reset_adapter_parameters(self): pass
        original_reset=LoRA.reset_adapter_parameters
        mods={'prime_rl.trainer.lora':types.SimpleNamespace(LoRAState=LoRA),
              'prime_rl.trainer.world':types.SimpleNamespace(get_world=lambda:None)}
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,mods):
            root=Path(d);base=root/init.BASE_REVISION;base.mkdir()
            config=types.SimpleNamespace(resume=None,run_dir=root/'run',model=types.SimpleNamespace(name=str(base),lora=types.SimpleNamespace(rank=16,alpha=32,target_modules=list(init.TARGETS))))
            for namespace in ({},{'setup_model':None},{'setup_model':'not callable'}):
                exec('def train(): pass',namespace)
                with self.assertRaisesRegex(ValueError,'Unwrapped trainer must bind callable setup_model'):
                    init.install_fresh_base_hook(config,record(namespace['train']),lambda *a:None)
                self.assertIs(LoRA.reset_adapter_parameters,original_reset)

    def test_failed_export_stops_before_optimizer_and_cannot_reset_again(self):
        state=self.state; events=[]
        class LoRA:
            def reset_adapter_parameters(self): events.append('native_reset')
            def adapter_state_dict(self): return state
        namespace={'setup_model':lambda:object(),'state':LoRA(),'events':events}
        exec('def train():\n setup_model()\n state.reset_adapter_parameters()\n events.append("optimizer")',namespace)
        mods={'prime_rl.trainer.lora':types.SimpleNamespace(LoRAState=LoRA),
              'prime_rl.trainer.world':types.SimpleNamespace(get_world=lambda:types.SimpleNamespace(world_size=1))}
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,mods):
            root=Path(d);base=root/init.BASE_REVISION;base.mkdir()
            config=types.SimpleNamespace(resume=None,run_dir=root/'run',model=types.SimpleNamespace(name=str(base),lora=types.SimpleNamespace(rank=16,alpha=32,target_modules=list(init.TARGETS))))
            init.install_fresh_base_hook(config,namespace['train'],lambda *a,**kw:{'verified':False,'private':False})
            with self.assertRaisesRegex(ValueError,'verified public'):namespace['train']()
            self.assertEqual(events,['native_reset'])
            with self.assertRaisesRegex(RuntimeError,'exactly once'):namespace['state'].reset_adapter_parameters()
            self.assertEqual(json.loads((config.run_dir/'artifacts/initial-adapter-audit.json').read_text())['step_zero_export'],'pending')

    def test_step_zero_reread_and_public_receipt_before_success(self):
        from safetensors.torch import save_file
        state=self.state; events=[]; actual_model=object()
        class Manager:
            def save(self,*args): events.append('native_checkpoint')
        class Sender:
            def __init__(self,*args): pass
            def _broadcast(self,model,step,folder):
                self_test.assertIs(model,actual_model); events.append('serialized')
                save_file(state,str(folder/'adapter_model.safetensors'))
                (folder/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':16,'lora_alpha':32,'target_modules':sorted(init.TARGETS)}))
        self_test=self
        mods={'prime_rl.configs.trainer':types.SimpleNamespace(FileSystemWeightBroadcastConfig=lambda:None),
              'prime_rl.trainer.ckpt':types.SimpleNamespace(CheckpointManager=Manager),
              'prime_rl.transports.weights.filesystem':types.SimpleNamespace(FileSystemWeightSender=Sender),
              'prime_rl.trainer.world':types.SimpleNamespace(get_world=lambda:types.SimpleNamespace(is_master=True))}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,mods), patch('torch.distributed.barrier'):
            config=types.SimpleNamespace(run_dir=Path(d),model=types.SimpleNamespace(lora=object()))
            def upload(folder,*args):
                report=json.loads((folder/'validation.json').read_text())
                if report['step']==0: self.assertEqual(report['zero_b_tensors'],248)
                self.assertFalse((folder/'.finished').exists())
                events.append('public_verified'); return {'verified':True,'private':False,'url':'https://example.test/receipt'}
            with patch.object(export,'cloud_settings',return_value=('repo','run',upload)):
                callback=export.install_export_hook(config)
                callback(actual_model,0,initialization_mode='fresh_base')
            folder=config.run_dir/'artifacts/adapters/step_0'
            self.assertTrue((folder/'.finished').exists())
            self.assertEqual(events,['serialized','public_verified'])
            events.clear()
            Manager().save(32,actual_model,[],None,None)
            self.assertEqual(events,['native_checkpoint','serialized','public_verified'])
            self.assertEqual(json.loads((folder.parent/'latest.json').read_text())['step'],32)
            with self.assertRaisesRegex(ValueError,'overwrite'): callback(actual_model,0,initialization_mode='fresh_base')
            with self.assertRaisesRegex(ValueError,'genuine'): callback(actual_model,1,initialization_mode='fresh_base')
            config.run_dir=Path(d)/'private'
            with patch.object(export,'cloud_settings',return_value=('repo','run',lambda *a:{'verified':True,'private':True})):
                callback=export.install_export_hook(config)
                with self.assertRaisesRegex(ValueError,'public'): callback(actual_model,0,initialization_mode='fresh_base')
            self.assertFalse((config.run_dir/'artifacts/adapters/step_0/.finished').exists())

if __name__=='__main__': unittest.main()
