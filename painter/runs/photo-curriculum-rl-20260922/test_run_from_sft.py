"""Offline checks for adapter preload and pinned Prime command interception."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("photo_rl_runner", ROOT / "run_from_sft.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhotoRLRunnerTests(unittest.TestCase):
    def test_initial_live_and_frozen_aliases_are_distinct(self):
        self.assertEqual(MODULE._initial_aliases("/tmp/base", "sft-step512"), ("/tmp/base", "sft-step512"))
        with self.assertRaises(ValueError):
            MODULE._initial_aliases("/tmp/base", "/tmp/base")

    def test_trainer_command_only_replaces_pinned_prime_child(self):
        command = [
            "torchrun", "--nproc-per-node=1", "-m", "prime_rl.trainer.rl.train", "@", "trainer.toml"
        ]
        result = MODULE.trainer_command(command, Path("train_from_sft.py"))
        self.assertEqual(result[2], "train_from_sft.py")
        self.assertEqual(result[-2:], ["@", "trainer.toml"])
        self.assertEqual(MODULE.trainer_command(["orchestrator", "@", "orch.toml"], Path("x")), ["orchestrator", "@", "orch.toml"])

    def test_policy_verification_preserves_prime_base_name(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "orchestrator.toml"
            path.write_text('[model]\nname = "/tmp/base-revision"\n')
            MODULE._verify_orchestrator_policy(path, "/tmp/base-revision")
            self.assertEqual(tomllib.loads(path.read_text())["model"]["name"], "/tmp/base-revision")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE._verify_orchestrator_policy(path, "wrong")

    def test_policy_verification_accepts_prime_json_child_config(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "orchestrator.json"
            path.write_text(json.dumps({"model": {"name": "/tmp/base-revision"}}))
            MODULE._verify_orchestrator_policy(path, "/tmp/base-revision")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE._verify_orchestrator_policy(path, "wrong")

    def test_cache_paths_are_role_specific(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inference = MODULE._cache_env(root, "inference", {"MODEL_DIR": "/tmp/base"})
            trainer = MODULE._cache_env(root, "trainer", {"MODEL_DIR": "/tmp/base"})
            self.assertNotEqual(inference["TORCHINDUCTOR_CACHE_DIR"], trainer["TORCHINDUCTOR_CACHE_DIR"])
            self.assertTrue(Path(inference["VLLM_CACHE_ROOT"]).is_dir())
            self.assertTrue(Path(trainer["TRITON_CACHE_DIR"]).is_dir())


if __name__ == "__main__":
    unittest.main()
