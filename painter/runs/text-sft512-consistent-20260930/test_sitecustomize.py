import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("sitecustomize.py")


class SitecustomizeActivationTests(unittest.TestCase):
    def test_prime_wrapper_forwards_only_guard_controls_to_orchestrator(self):
        wrapper_path = Path(__file__).parents[1] / "photo-curriculum-rl-20260922" / "run_from_sft.py"
        spec = importlib.util.spec_from_file_location("photo_run_from_sft_for_guard_test", wrapper_path)
        assert spec and spec.loader
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        with patch.dict(os.environ, {
            "PAINTER_GROUP_RANK_MODULE": "painter.consistent_group_admission",
            "PAINTER_WHOLE_GROUP_ADMISSION": "1",
            "PAINTER_WHOLE_GROUP_SIZE": "8",
            "PAINTER_CONSISTENT_GUARD_RECEIPT": "/tmp/guard-receipt.json",
            "PYTHONPATH": "/reviewed/path",
        }, clear=False):
            forwarded = wrapper._orchestrator_env({"SAFE_ROLE_VALUE": "kept", "SECRET_SHOULD_NOT_BE_ADDED": ""})
        self.assertEqual(forwarded["PAINTER_PRIME_ROLE"], "orchestrator")
        self.assertEqual(forwarded["PAINTER_WHOLE_GROUP_ADMISSION"], "1")
        self.assertEqual(forwarded["PAINTER_WHOLE_GROUP_SIZE"], "8")
        self.assertEqual(forwarded["PYTHONPATH"], "/reviewed/path")

    def test_orchestrator_child_calls_explicit_guard_and_writes_receipt(self):
        class FakeAlgorithm:
            async def finalize_group(self, _group):
                return None

        class FakeTrainSource:
            def on_result(self, _group):
                return True

        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
            "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
            "prime_rl.orchestrator.algo.base": types.ModuleType("prime_rl.orchestrator.algo.base"),
            "prime_rl.orchestrator.train_source": types.ModuleType("prime_rl.orchestrator.train_source"),
        }
        modules["prime_rl.orchestrator.algo.base"].Algorithm = FakeAlgorithm
        modules["prime_rl.orchestrator.train_source"].TrainSource = FakeTrainSource
        receipt = Path(self.id().replace("/", "_") + ".json")
        try:
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {
                "PAINTER_PRIME_ROLE": "orchestrator",
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                "PAINTER_WHOLE_GROUP_ADMISSION": "1",
                "PAINTER_WHOLE_GROUP_SIZE": "8",
                "PAINTER_CONSISTENT_GUARD_RECEIPT": str(receipt),
            }, clear=False):
                spec = importlib.util.spec_from_file_location("run_sitecustomize", MODULE_PATH)
                assert spec and spec.loader
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            data = json.loads(receipt.read_text())
            self.assertTrue(data["installed"])
            self.assertEqual(data["role"], "orchestrator")
            self.assertEqual(data["group_size"], 8)
        finally:
            receipt.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
