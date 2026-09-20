import json
from pathlib import Path
import tempfile
import unittest
from curriculum import pass_at_k,sample,mastery,speed_budget
from demonstrations import build
from hardware import plan
from pairwise import reconcile,group_advantages
from mix_data import mix


class CurriculumTests(unittest.TestCase):
    def test_pass_k(self):
        self.assertAlmostEqual(pass_at_k(2,4,2),5/6)
        self.assertEqual(pass_at_k(0,8,8),0)
    def test_no_evaluation_leakage_or_impossible_level(self):
        rows=[dict(id=str(i),source_group=str(i),split=split,level=level,family='shape') for i,(split,level) in enumerate([('train',0),('train',1),('dev',1),('test',1),('train',4)])]
        chosen=sample(rows,1,200)
        self.assertEqual({r['id'] for r in chosen},{'0','1'})
    def test_missing_and_order_bias_masked(self):
        a=dict(winner='A',acceptable_A=True,acceptable_B=False)
        self.assertTrue(reconcile(a,a)['mask'])
        self.assertTrue(reconcile(a,None)['mask'])
        b=dict(winner='B',acceptable_A=False,acceptable_B=True)
        self.assertEqual(reconcile(a,b)['reward'],1)
    def test_both_bad_can_improve_without_mastery(self):
        a=dict(winner='A',acceptable_A=False,acceptable_B=False)
        b=dict(winner='B',acceptable_A=False,acceptable_B=False)
        r=reconcile(a,b);self.assertEqual(r['reward'],1);self.assertFalse(r['candidate_acceptable'])
    def test_judge_long_reason_is_not_failure(self):
        a=dict(winner='tie',acceptable_A=True,acceptable_B=True,reason='x'*5000)
        self.assertEqual(reconcile(a,a)['reward'],.5)
    def test_unscored_rollouts_never_receive_advantage(self):
        r=group_advantages([dict(reward=x,mask=False) for x in (0.,1.,.5,1.)]+[dict(reward=None,mask=True)])
        self.assertTrue(r['train']);self.assertIsNone(r['advantages'][-1])
    def test_quality_has_no_speed_budget(self):
        config=json.loads((Path(__file__).parent/'campaign.json').read_text())
        self.assertIsNone(speed_budget('quality',0,config,100))
        with self.assertRaises(ValueError):speed_budget('speed',0,config,100)
        with self.assertRaises(ValueError):speed_budget('speed',2,config,100,mastered_level=1)
        self.assertEqual(speed_budget('speed',2,config,100,mastered_level=4),90)
    def test_repeated_reference_does_not_manufacture_mastery(self):
        settings=json.loads((Path(__file__).parent/'campaign.json').read_text())['mastery']
        r=dict(source_group='same',family='shape',split='dev',completed=True,absolute_acceptable=True)
        self.assertFalse(mastery([r]*100,settings)['mastered'])
    def test_software_plan_uses_capacity_not_gpu_name(self):
        def hw(n,m):return dict(cpu_count=16,gpus=[dict(name='arbitrary',memory_gib=m,compute_capability=8.) for _ in range(n)])
        self.assertEqual(plan(hw(1,80))['serving']['tensor_parallel_size'],1)
        self.assertEqual(plan(hw(2,48))['serving']['tensor_parallel_size'],2)
        self.assertIsNone(plan(hw(1,24))['serving']['tensor_parallel_size'])
    def test_demo_split_programs_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=build(folder,4)
            self.assertEqual(len(rows),24)
            self.assertEqual(sum(r['split']=='dev' for r in rows),6)
            self.assertEqual(len({r['source_group'] for r in rows}),24)
            self.assertTrue(all((Path(folder)/r['programs'][-1]).exists() for r in rows))
    def test_mixture_counts_tokens_and_discloses_missing_pools(self):
        pool=[dict(id=str(i),split='train',supervised_tokens=100+i*10) for i in range(4)]
        rows,report=mix({'current':pool,'review':[],'retention':[]},{'current':.7,'review':.2,'retention':.1})
        self.assertEqual(len(rows),8);self.assertEqual(report['missing_pools'],['review','retention'])
        self.assertEqual(report['actual_tokens']['current'],920)
    def test_mixture_rejects_dev_and_unmeasured_tokens(self):
        for r in [dict(id='x',split='dev',supervised_tokens=3),dict(id='x',split='train')]:
            with self.assertRaises(ValueError):mix({'current':[r]},{'current':1})


if __name__=='__main__':unittest.main()
