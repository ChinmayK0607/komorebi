"""Pure unit tests; no rendering, inference or ML locally."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import prepare_async
import run_from_sft

class AsyncContractTest(unittest.TestCase):
    def test_capacity_not_gpu_brand(self):
        for name in ['H100','H200','A100','unfamiliar']:
            info={'gpus':[dict(name=name,memory_gib=80,compute_capability=8)]*2}
            self.assertEqual(prepare_async.topology(info)['num_train_gpus'],1)
            self.assertEqual(prepare_async.topology(info)['num_infer_gpus'],1)
        for devices in [[],[dict(memory_gib=141,compute_capability=9)],
                        [dict(memory_gib=48,compute_capability=9)]*2]:
            with self.assertRaises(ValueError):prepare_async.topology({'gpus':devices})
    def test_only_native_trainer_is_replaced(self):
        wrapper=Path('/node/train_from_sft.py')
        command=['torchrun','--nproc-per-node=1','-m','prime_rl.trainer.rl.train','@','trainer.json']
        modified=run_from_sft.trainer_command(command,wrapper)
        self.assertEqual(modified,['torchrun','--nproc-per-node=1',str(wrapper),'@','trainer.json'])
        for other in [['inference','@','inference.json'],['orchestrator','@','orchestrator.json']]:
            self.assertEqual(run_from_sft.trainer_command(other,wrapper),other)

if __name__=='__main__':unittest.main()
