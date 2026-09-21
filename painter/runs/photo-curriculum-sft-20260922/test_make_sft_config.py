import argparse
import importlib.util
import tomllib
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("make_sft_config.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_make_config", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MakeSftConfigTests(unittest.TestCase):
    def test_replicated_two_gpu_layout_round_trips(self):
        args = argparse.Namespace(
            num_train_gpus=2, gpus_per_node=2, dp_replicate=2, dp_shard=1,
            micro_batch_size=1, global_batch_size=8, steps=512, seq_len=16384,
            learning_rate=1e-5, data=Path("/tmp/data"), model_path=Path("/tmp/model"),
            run_name="photo-curriculum-sft-20260922", validation_interval=64,
            checkpoint_interval=64, seed=92022, shuffle=False, resume=None,
        )
        report, text = MODULE.config(args)
        parsed = tomllib.loads(text)
        self.assertEqual(parsed["model"]["dp_replicate"], 2)
        self.assertEqual(report["dp_replicate"], 2)
        self.assertEqual(report["dp_shard"], 1)


if __name__ == "__main__":
    unittest.main()
