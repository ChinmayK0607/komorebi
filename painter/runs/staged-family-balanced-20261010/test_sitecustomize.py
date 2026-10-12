"""Offline tests for staged inference cache isolation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("sitecustomize.py")


class SitecustomizeCacheTests(unittest.TestCase):
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
            [sys.executable, "-c", "import os; print(os.environ['TRITON_CACHE_DIR'])"],
            check=True, env=env, capture_output=True, text=True,
        )
        return result.stdout.strip()

    def test_spawned_workers_get_distinct_rank_and_pid_caches(self):
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

    def test_torchinductor_compile_worker_keeps_parent_namespace(self):
        spec = importlib.util.spec_from_file_location("staged_sitecustomize", MODULE_PATH)
        assert spec and spec.loader
        prior_argv = list(sys.argv)
        try:
            sys.argv = [
                "/opt/prime/lib/python3.12/site-packages/torch/_inductor/compile_worker/__main__.py",
                "--pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler",
                "--kind=fork", "--parent=1234",
            ]
            expected = "/tmp/rl-cache/inference/workers/rank-3-pid-1234-parent"
            with patch.dict(os.environ, {
                "PAINTER_PRIME_ROLE": "inference",
                "PAINTER_COMPILE_CACHE_ROOT": "/tmp/rl-cache/inference",
                "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
                "LOCAL_RANK": "3",
                "VLLM_CACHE_ROOT": expected + "/vllm",
                "TRITON_CACHE_DIR": expected + "/triton",
                "TORCHINDUCTOR_CACHE_DIR": expected + "/torchinductor",
            }, clear=False):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(module._is_torchinductor_compile_worker())
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], expected + "/triton")
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], expected + "/torchinductor")
        finally:
            sys.argv = prior_argv

    def test_inference_workers_get_distinct_all_cache_namespaces(self):
        values = []
        for rank in ("0", "1"):
            spec = importlib.util.spec_from_file_location(f"staged_sitecustomize_cache_{rank}", MODULE_PATH)
            assert spec and spec.loader
            with self.subTest(rank=rank), patch.dict(os.environ, {
                "PAINTER_PRIME_ROLE": "inference",
                "PAINTER_COMPILE_CACHE_ROOT": "/tmp/rl-cache/inference",
                "PAINTER_COMPILE_CACHE_PROCESS_ISOLATION": "1",
                "LOCAL_RANK": rank,
                "TRITON_CACHE_DIR": "/tmp/rl-cache/inference/triton",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/rl-cache/inference/torchinductor",
            }, clear=False):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                values.append(tuple(os.environ[key] for key in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")))
        self.assertNotEqual(values[0], values[1])
        self.assertRegex(values[0][1], r"/workers/rank-0-pid-\d+-[0-9a-f]{32}/triton$")
        self.assertRegex(values[1][1], r"/workers/rank-1-pid-\d+-[0-9a-f]{32}/triton$")


if __name__ == "__main__":
    unittest.main()
