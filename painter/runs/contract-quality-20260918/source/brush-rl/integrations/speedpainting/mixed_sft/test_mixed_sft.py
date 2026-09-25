import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('mixed_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def row(ident,image=None):
    parts=[{'type':'text','text':ident+' question'}]
    if image:parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}})
    return {'id':ident,'messages':[{'role':'system','content':[{'type':'text','text':'Answer'}]},{'role':'user','content':parts},{'role':'assistant','content':[{'type':'text','text':'Answer '+ident}]}]}

class Tests(unittest.TestCase):
    def test_feedback_canvas_can_repeat_but_reference_and_source_cannot_leak(self):
        m=load('prepare')
        def feedback(ident,reference,source):
            def image(value):return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(value).decode()}}
            r=row(ident);r.update(task_family='painting',action_contract='brush-turn-v1',source_group=source)
            r['messages'][1]['content']=[{'type':'text','text':'Reference image:'},image(reference),
                {'type':'text','text':'Canvas observation:'},image(b'shared-blank-canvas'),{'type':'text','text':'Host state: {}'}]
            return r
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'train.jsonl').write_text(json.dumps(feedback('train',b'train-photo','source-a')))
            def heldout(r):
                (p/'validation.jsonl').write_text(json.dumps(r));return m.validate_data(p)
            self.assertEqual(heldout(feedback('val',b'val-photo','source-b'))[0],{'train':1,'validation':1})
            with self.assertRaisesRegex(ValueError,'Image leaks'):heldout(feedback('val',b'train-photo','source-b'))
            with self.assertRaisesRegex(ValueError,'Source group leaks'):heldout(feedback('val',b'val-photo','source-a'))
            bad=feedback('val',b'val-photo','source-b');bad['messages'][1]['content'][0]['text']='Canvas observation:'
            with self.assertRaisesRegex(ValueError,'Unknown feedback'):heldout(bad)

    def test_fresh_base_actual_schedule_and_explicit_modes(self):
        m=load('prepare')
        dataset=Path.cwd()/'outputs/brush-rl/data/astra-high-structure-v1'
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/m.shared.REVISION;model.mkdir()
            config,report=m.build(dataset,model,fresh_base=True,steps=65,lr=1e-5,
                                  activation_offload=False,optimizer_offload=False)
            self.assertEqual(config['max_steps'],65)
            self.assertEqual(config['ckpt']['keep_last'],1)
            self.assertEqual(report['checkpoint_peak_native_snapshots'],2)
            self.assertEqual(config['optim']['lr'],1e-5)
            self.assertEqual(report['counts'],{'train':260,'validation':138})
            self.assertEqual(report['initialization_mode'],'fresh_base')
            self.assertIsNone(report['init_adapter'])
            self.assertTrue(report['step_zero_export_required'])
            self.assertEqual(config['model']['lora']['rank'],16)
            self.assertEqual(config['model']['lora']['alpha'],32)
            for adapter,fresh in [(None,False),('/not-consulted',True)]:
                with self.assertRaisesRegex(ValueError,'exactly one'):
                    m.build(dataset,model,adapter,fresh_base=fresh)

    def test_config_and_mixed_validation(self):
        m=load('prepare')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'data';dataset.mkdir();model=root/m.shared.REVISION;model.mkdir();adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
            (adapter/'adapter_config.json').write_text(json.dumps({'r':16,'lora_alpha':32,'target_modules':m.shared.TARGETS}))
            for split in ('train','validation'):
                (dataset/(split+'.jsonl')).write_text('\n'.join(json.dumps(row(split+str(i),split.encode() if i==0 else None)) for i in range(2)))
            config,report=m.build(dataset,model,adapter)
            off_config,_=m.build(dataset,model,adapter,activation_offload=False,optimizer_offload=False)
            self.assertEqual(off_config['model']['ac_offloading'],'None')
            self.assertFalse(off_config['model']['optim_cpu_offload'])
            self.assertEqual(config['data']['seq_len'],32768);self.assertEqual(config['max_steps'],1)
            self.assertEqual(report['modalities']['train'],{'image':1,'text':1})
            self.assertEqual(config['data']['loss_mask'],{'system':False,'user':False,'assistant':True,'tool':False})
            warmed,wr=m.build(dataset,model,adapter,steps=96,lr=1e-5,batch_size=16,warmup_steps=4,min_lr=5e-6)
            self.assertEqual(warmed['scheduler'],{'type':'cosine','warmup_steps':4,'min_lr':5e-6})
            self.assertFalse(warmed['model']['optim_cpu_offload'])
            self.assertEqual(wr['optimizer_config']['max_norm'],1.)
            with self.assertRaisesRegex(ValueError,'Warmup'):m.build(dataset,model,adapter,steps=4,warmup_steps=4)
            (dataset/'validation.jsonl').write_text(json.dumps(row('different',b'train')))
            with self.assertRaisesRegex(ValueError,'Image leaks'):m.validate_data(dataset)
    def test_strict_processing_and_actual_upstream_padding(self):
        class FakeSFT:
            def _process(self,example):
                # Simulate upstream truncation; wrapper must override window first.
                n=min(len(example['input_ids']),self.seq_len)
                return {**example,'input_ids':example['input_ids'][:n]}
        class FakeCat:pass
        source=HERE.parents[4]/'work/prime-rl-upstream/src/prime_rl/trainer/sft/data.py'
        # Resolve from workspace, without importing CUDA trainer dependencies.
        source=Path.cwd()/'work/prime-rl-upstream/src/prime_rl/trainer/sft/data.py'
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CatDataset')
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_finalize_pack')
        code=ast.Module(body=[method],type_ignores=[]);scope={'Any':object};exec(compile(code,str(source),'exec'),scope);FakeCat._finalize_pack=scope['_finalize_pack']
        fake=types.SimpleNamespace(SFTDataset=FakeSFT,CatDataset=FakeCat)
        with patch.dict(sys.modules,{'prime_rl.trainer':types.ModuleType('prime_rl.trainer'),'prime_rl.trainer.sft':types.SimpleNamespace(data=fake)}):m=load('train')
        strict=m.StrictSFTDataset();strict.seq_len=256
        with self.assertRaisesRegex(ValueError,'refusing truncation'):strict._process({'id':'oversize','input_ids':list(range(257))})
        self.assertEqual(strict.seq_len,256)
        samples=[]
        for n,image in ((17,False),(301,True)):
            samples.append({'input_ids':list(range(n)),'target_ids':list(range(n)),'position_ids':list(range(n)),
                'loss_mask':[False]*8+[True]*(n-8),'seq_lens':[n], 'mm_kwargs':{'pixels':'sentinel'} if image else None,
                'mm_token_type_ids':[0,1,1]+[0]*(n-3) if image else None})
        packing=m.SingleExampleDataset();packing.seq_len=32768;packing.pending_sample=None;packing.dataset=samples
        result=list(packing)
        self.assertEqual([len(r['input_ids']) for r in result],[256,512])
        self.assertIsNone(result[0]['mm_kwargs']);self.assertEqual(result[1]['mm_kwargs'],{'pixels':'sentinel'})
        self.assertEqual(sum(result[0]['loss_mask']),9);self.assertEqual(sum(result[1]['loss_mask']),293)
        self.assertEqual(len(result[1]['mm_token_type_ids']),512)
        self.assertFalse(any(result[1]['loss_mask'][301:]))
    def test_group_validation_instrumentation_is_pinned(self):
        m=load('group_validation')
        source=(Path.cwd()/'work/prime-rl-upstream/src/prime_rl/trainer/sft/train.py').read_text()
        effective=m.transform(source);compile(effective,'effective_train.py','exec')
        self.assertIn('assistant_token_mean_nll',effective)
        self.assertIn("phase='after_optimizer'",effective)
        with self.assertRaisesRegex(ValueError,'Unrecognized'):m.transform(source+'#changed')
    def test_reject_unpinned_init(self):
        m=load('download_init')
        with self.assertRaisesRegex(ValueError,'Immutable'):m.download('/tmp/not-created',revision='main')
    def test_ordered_schedule_resolves_real_prime_config(self):
        m=load('prepare');schedule=load('scheduled_data')
        def source(ident,family):
            return {**row(ident),'split':'train','task_family':family}
        paint=[source(f'paint{i}','painting') for i in range(4)]
        retained=[source(f'{g}{i}',g) for g in schedule.RETENTION for i in range(4)]
        rows,meta=schedule.build(paint,retained)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'data';dataset.mkdir();model=root/m.shared.REVISION;model.mkdir();adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
            (adapter/'adapter_config.json').write_text(json.dumps({'r':16,'lora_alpha':32,'target_modules':m.shared.TARGETS}))
            (dataset/'train.jsonl').write_bytes(schedule.encoded_rows(rows))
            (dataset/'validation.jsonl').write_text(json.dumps(row('heldout'))+'\n')
            (dataset/'schedule.json').write_text(json.dumps(meta))
            config,report=m.build(dataset,model,adapter)
            self.assertFalse(config['data']['shuffle'])
            self.assertEqual(config['max_steps'],5)
            self.assertEqual(report['schedule']['step_counts'],{'painting':4,'retention_text':1})
            with self.assertRaisesRegex(ValueError,'exactly'):
                m.build(dataset,model,adapter,steps=6)

if __name__=='__main__':unittest.main()
