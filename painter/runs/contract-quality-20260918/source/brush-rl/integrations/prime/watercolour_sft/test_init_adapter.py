"""CPU tests for strict portable adapter initialization; no model weights loaded."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from init_adapter import copy_and_verify


class AdapterInitTests(unittest.TestCase):
    def setUp(self):
        self.source={f'model.layers.{i}.proj.lora_{letter}.weight':torch.full((2,2),.125)
                     for i in range(248) for letter in ('A','B')}
        self.destination={k:torch.zeros_like(v) for k,v in self.source.items()}

    def test_exact_copy_into_registered_tensors(self):
        report=copy_and_verify(self.source,self.destination,lambda:self.destination)
        self.assertEqual(report['exact_tensor_matches'],496)
        self.assertEqual(report['nonzero_b_tensors'],248)
        self.assertTrue(all(torch.equal(self.source[k],v) for k,v in self.destination.items()))

    def test_missing_key_and_dtype_fail_before_copy(self):
        missing=dict(self.source);missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError,'key mismatch'):
            copy_and_verify(missing,self.destination,lambda:self.destination)
        bad=dict(self.source);bad[next(iter(bad))]=next(iter(bad.values())).double()
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            copy_and_verify(bad,self.destination,lambda:self.destination)
        self.assertTrue(all(not v.any() for v in self.destination.values()))

    def test_detached_conversion_copy_cannot_claim_load(self):
        detached={k:v.clone() for k,v in self.destination.items()}
        with self.assertRaisesRegex(ValueError,'Post-load adapter differs'):
            copy_and_verify(self.source,detached,lambda:self.destination)

    def test_zero_b_rejected(self):
        self.source[next(k for k in self.source if '.lora_B.' in k)].zero_()
        with self.assertRaisesRegex(ValueError,'248 trained B'):
            copy_and_verify(self.source,self.destination,lambda:self.destination)

if __name__=='__main__':unittest.main()
