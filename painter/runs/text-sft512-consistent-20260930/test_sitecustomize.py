import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("sitecustomize.py")


class SitecustomizeActivationTests(unittest.TestCase):
    @staticmethod
    def _spawn_cache_child(root: Path, *, rank: str | None = None) -> str:
        env = {
            "PYTHONPATH": str(MODULE_PATH.parent),
            "PAINTER_PRIME_ROLE": "inference",
            "PAINTER_COMPILE_CACHE_ROOT": str(root / "inference"),
            "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TRITON_CACHE_DIR": str(root / "inference" / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inference" / "torchinductor"),
        }
        if rank is not None:
            env["LOCAL_RANK"] = rank
        result = subprocess.run(
            [sys.executable, "-c", "import os; assert os.environ['VLLM_WORKER_MULTIPROC_METHOD'] == 'spawn'; print(os.environ['TRITON_CACHE_DIR'])"],
            check=True, env=env, capture_output=True, text=True,
        )
        return result.stdout.strip()

    def test_spawned_inference_children_isolate_rank_and_pid_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rank_zero = self._spawn_cache_child(root, rank="0")
            rank_one = self._spawn_cache_child(root, rank="1")
            self.assertNotEqual(rank_zero, rank_one)
            self.assertRegex(rank_zero, r"/workers/rank-0-pid-\d+-[0-9a-f]{32}/triton$")
            self.assertRegex(rank_one, r"/workers/rank-1-pid-\d+-[0-9a-f]{32}/triton$")

            pid_one = self._spawn_cache_child(root)
            pid_two = self._spawn_cache_child(root)
            self.assertNotEqual(pid_one, pid_two)
            self.assertIn("/workers/pid-", pid_one)
            self.assertIn("/workers/pid-", pid_two)

    def test_inference_workers_get_distinct_compile_cache_namespaces(self):
        spec = importlib.util.spec_from_file_location("run_sitecustomize_cache_test", MODULE_PATH)
        assert spec and spec.loader
        with self.subTest(rank="0"), patch.dict(os.environ, {
            "PAINTER_PRIME_ROLE": "inference",
            "PAINTER_COMPILE_CACHE_ROOT": "/tmp/rl-cache/inference",
            "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
            "LOCAL_RANK": "0",
            "TRITON_CACHE_DIR": "/tmp/rl-cache/inference/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/rl-cache/inference/torchinductor",
        }, clear=False):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            first_triton = os.environ["TRITON_CACHE_DIR"]
            first_inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
            first_vllm = os.environ["VLLM_CACHE_ROOT"]
        spec = importlib.util.spec_from_file_location("run_sitecustomize_cache_test_rank1", MODULE_PATH)
        assert spec and spec.loader
        with self.subTest(rank="1"), patch.dict(os.environ, {
            "PAINTER_PRIME_ROLE": "inference",
            "PAINTER_COMPILE_CACHE_ROOT": "/tmp/rl-cache/inference",
            "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
            "LOCAL_RANK": "1",
            "TRITON_CACHE_DIR": "/tmp/rl-cache/inference/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/rl-cache/inference/torchinductor",
        }, clear=False):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            second_triton = os.environ["TRITON_CACHE_DIR"]
            second_inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
            second_vllm = os.environ["VLLM_CACHE_ROOT"]
        self.assertNotEqual(first_triton, second_triton)
        self.assertNotEqual(first_inductor, second_inductor)
        self.assertNotEqual(first_vllm, second_vllm)
        self.assertRegex(first_triton, r"/workers/rank-0-pid-\d+-[0-9a-f]{32}/triton$")
        self.assertRegex(second_triton, r"/workers/rank-1-pid-\d+-[0-9a-f]{32}/triton$")

    def test_unlabelled_inference_processes_use_pid_fallback(self):
        spec = importlib.util.spec_from_file_location("run_sitecustomize_pid_test", MODULE_PATH)
        assert spec and spec.loader
        with patch.dict(os.environ, {
            "PAINTER_PRIME_ROLE": "inference",
            "PAINTER_COMPILE_CACHE_ROOT": "/tmp/rl-cache/inference",
            "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
        }, clear=False), patch.object(os, "getpid", return_value=1234):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertRegex(os.environ["TRITON_CACHE_DIR"], r"/workers/pid-1234-[0-9a-f]{32}/triton$")

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
