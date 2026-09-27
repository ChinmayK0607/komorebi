"""Offline invariants for the photo-curriculum RL configuration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("photo_rl_prepare", ROOT / "prepare_rl_config.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhotoRLConfigTests(unittest.TestCase):
    def test_requested_finite_shape_and_reward_policy(self):
        config = MODULE.build_config(
            model="/tmp/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            run_root=Path("/tmp/photo-rl"),
        )
        self.assertEqual(config["max_steps"], 32)
        self.assertEqual(config["trainer"]["max_steps"], 32)
        self.assertEqual(config["orchestrator"]["batch_size"], 32)
        self.assertEqual(config["orchestrator"]["group_size"], 8)
        self.assertEqual(config["orchestrator"]["seq_len"], 32768)
        self.assertEqual(config["trainer"]["optim"]["lr"], 1e-6)
        self.assertEqual(config["ckpt"]["interval"], 8)
        self.assertEqual(config["inference"]["vllm"]["max_loras"], 2)
        self.assertEqual(config["orchestrator"]["concurrency"], {
            "min_inflight": 32,
            "initial_inflight": 32,
            "max_inflight": 64,
        })
        self.assertEqual(config["inference"]["vllm"]["max_num_seqs"], 16)
        self.assertEqual(config["inference"]["vllm"]["tensor_parallel_size"], 1)
        self.assertEqual(config["inference"]["vllm"]["data_parallel_size"], 1)
        self.assertEqual(config["deployment"], {
            "type": "single_node",
            "gpus_per_node": 2,
            "num_train_gpus": 1,
            "num_infer_gpus": 1,
        })
        source = config["orchestrator"]["train"]["source"][0]
        self.assertEqual(source["env"]["agent"]["timeout"], {
            "rollout": 7200,
            "scoring": 6300,
        })
        self.assertEqual(source["env"]["timeout"]["episode"], 13600)
        taskset = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
        self.assertEqual(taskset["manifest"], "source-manifest.json")
        self.assertEqual(taskset["expected_count"], 2486)
        self.assertTrue(taskset["lazy_opponent"])
        self.assertTrue(taskset["preserve_manifest_order"])
        self.assertEqual(taskset["opponent_model_alias"], "sft-step512")
        self.assertEqual(taskset["opponent_generator"], "frozen_opponent:generate_frozen_opponent")
        self.assertEqual(taskset["max_turns"], 6)
        self.assertEqual(taskset["opponent_lock_timeout_seconds"], 4200)
        self.assertEqual(config["orchestrator"]["train"]["sampling"]["max_completion_tokens"], 8192)

    def test_recovered_policy_metadata_is_explicit_and_anchor_is_separate(self):
        config = MODULE.build_config(
            model="/tmp/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            run_root=Path("/tmp/photo-rl"),
            mode="recovered_policy",
            adapter_revision="recovered-revision",
            adapter_prefix="photo-curriculum-rl-20260922/step_8",
            adapter_sha256="a" * 64,
        )
        self.assertEqual(config["deployment"]["num_infer_gpus"], 1)
        self.assertEqual(config["orchestrator"]["group_size"], 8)
        self.assertEqual(MODULE.FROZEN_ALIAS, "sft-step512")
        self.assertNotEqual("recovered-policy", MODULE.FROZEN_ALIAS)
        with self.assertRaisesRegex(ValueError, "adapter_revision"):
            MODULE._adapter_identity(mode="recovered_policy", revision="", prefix="step_8", sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            MODULE._adapter_identity(mode="recovered_policy", revision="r", prefix="step_8", sha256="A" * 64)

    def test_four_gpu_role_split_is_supported_explicitly(self):
        config = MODULE.build_config(
            model="/tmp/model",
            run_root=Path("/tmp/photo-rl"),
            gpus_per_node=4,
            num_train_gpus=1,
            num_infer_gpus=3,
        )
        self.assertEqual(config["deployment"]["gpus_per_node"], 4)
        self.assertEqual(config["deployment"]["num_train_gpus"], 1)
        self.assertEqual(config["deployment"]["num_infer_gpus"], 3)
        self.assertEqual(config["inference"]["vllm"]["data_parallel_size"], 3)
        self.assertEqual(config["orchestrator"]["concurrency"]["initial_inflight"], 64)
        self.assertEqual(config["orchestrator"]["concurrency"]["max_inflight"], 96)
        parsed = tomllib.loads(MODULE.render_toml(config))
        self.assertEqual(parsed["deployment"]["gpus_per_node"], 4)
        self.assertEqual(parsed["inference"]["vllm"]["data_parallel_size"], 3)

    def test_bounded_inflight_can_limit_stale_speculation(self):
        config = MODULE.build_config(
            model="/tmp/model", run_root=Path("/tmp/photo-rl"),
            batch_size=8, group_size=4,
            initial_inflight=8, max_inflight=12,
        )
        self.assertEqual(config["orchestrator"]["concurrency"], {
            "min_inflight": 8, "initial_inflight": 8, "max_inflight": 12,
        })
        self.assertEqual(config["orchestrator"]["max_off_policy_steps"], 1)
        with self.assertRaisesRegex(ValueError, "inflight bounds"):
            MODULE.build_config(model="/tmp/model", run_root=Path("/tmp/photo-rl"),
                                batch_size=8, group_size=4,
                                initial_inflight=8, max_inflight=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MODULE.build_config(model="/tmp/model", run_root=Path("/tmp/photo-rl"),
                                batch_size=8, group_size=4,
                                initial_inflight=8, max_inflight=10)

    def test_recovered_checkpoint_caps_are_explicitly_supported(self):
        for cap in (24, 25):
            config = MODULE.build_config(
                model="/tmp/model",
                run_root=Path("/tmp/photo-rl"),
                max_steps=cap,
                mode="recovered_policy",
                adapter_revision="r",
                adapter_prefix=f"step_{cap}",
                adapter_sha256="b" * 64,
            )
            self.assertEqual(config["max_steps"], cap)

    def test_horizon_is_extensible_but_bounded(self):
        extended = MODULE.build_config(
            model="/tmp/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            run_root=Path("/tmp/photo-rl"),
            max_steps=128,
            checkpoint_interval=8,
        )
        self.assertEqual(extended["max_steps"], 128)
        with self.assertRaisesRegex(ValueError, "<= 128"):
            MODULE.build_config(
                model="/tmp/model",
                run_root=Path("/tmp/photo-rl"),
                max_steps=129,
            )

    def test_prime_validation_cannot_mutate_toml_source(self):
        config = MODULE.build_config(
            model="/tmp/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            run_root=Path("/tmp/photo-rl"),
        )

        def mutating_validator(value, _prime_root):
            value["trainer"]["model"]["compile"] = None
            return True, "stub accepted after normalization"

        with tempfile.TemporaryDirectory() as temp, patch.object(
            MODULE, "validate_with_prime", side_effect=mutating_validator
        ):
            output = Path(temp) / "native-rl.toml"
            message = MODULE.write_config(output, config, validate_prime=True)
            parsed = tomllib.loads(output.read_text())
            limits = json.loads((output.parent / "frozen-opponent-config.json").read_text())

        self.assertEqual(message, "stub accepted after normalization")
        self.assertEqual(config["trainer"]["model"]["compile"], "None")
        self.assertEqual(parsed["trainer"]["model"]["compile"], "None")
        self.assertEqual(limits["operational_limits"]["episode_timeout_seconds"], 3600)


if __name__ == "__main__":
    unittest.main()
