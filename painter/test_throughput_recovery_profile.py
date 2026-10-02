"""Offline contracts for the opt-in throughput recovery profile."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import sys


ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = _load("throughput_recovery_profile", ROOT / "throughput_recovery_profile.py")
config_tool = _load(
    "throughput_recovery_config",
    ROOT / "runs/text-pairwise-rl-20260928/prepare_config.py",
)


class ThroughputRecoveryProfileTests(unittest.TestCase):
    def _prepare(self, root: Path, **kwargs):
        manifest = root / "train-manifest.json"
        manifest.write_text(json.dumps({"rows": [
            {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
            for i in range(40)
        ]}) + "\n")
        output = root / "native.toml"
        checkpoint_interval = kwargs.pop("checkpoint_interval", profile.CHECKPOINT_INTERVAL)
        if checkpoint_interval == profile.CHECKPOINT_INTERVAL:
            receipt = profile.prepare_profile(
                config_tool.prepare,
                output=output,
                model=root / "model",
                run_root=root,
                manifest=manifest,
                run_name="throughput-recovery-test",
                max_steps=8,
                **kwargs,
            )
        else:
            receipt = config_tool.prepare(
                output=output,
                model=root / "model",
                run_root=root,
                manifest=manifest,
                run_name="throughput-recovery-test",
                max_steps=8,
                checkpoint_interval=checkpoint_interval,
                judge_protocol=profile.REWARD_PROTOCOL,
                mode=profile.INITIALIZER_MODE,
                batch_size=profile.BATCH_SIZE,
                group_size=profile.GROUP_SIZE,
                initial_inflight=profile.INITIAL_INFLIGHT,
                max_inflight=profile.MAX_INFLIGHT,
                adapter_revision=profile.INITIALIZER_REVISION,
                adapter_prefix=profile.INITIALIZER_PREFIX,
                adapter_sha256=profile.INITIALIZER_SHA256,
                **kwargs,
            )
        return output, receipt

    def test_generated_native_config_matches_opt_in_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            output, receipt = self._prepare(Path(temp))
            value = tomllib.loads(output.read_text())
            profile.validate_config(value, receipt)
            self.assertEqual(receipt["checkpoint_interval"], 1)
            self.assertEqual(receipt["initial_inflight"], 32)
            self.assertEqual(
                profile.required_environment(),
                {
                    "PAINTER_RL_CHECKPOINT_INTERVAL": "1",
                    "ADAPTER_ONLY_CHECKPOINTS": "1",
                    "PAINTER_ACTIVATION_CPU_OFFLOAD": "1",
                },
            )

    def test_omitted_environment_override_preserves_historical_interval(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=False):
            with patch.dict("os.environ", {"PAINTER_RL_CHECKPOINT_INTERVAL": "1"}, clear=False):
                # Explicit historical value wins over the opt-in environment.
                output, receipt = self._prepare(Path(temp), checkpoint_interval=4)
            self.assertEqual(receipt["checkpoint_interval"], 4)
            self.assertEqual(tomllib.loads(output.read_text())["ckpt"]["interval"], 4)

    def test_environment_override_applies_only_when_value_is_omitted(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"PAINTER_RL_CHECKPOINT_INTERVAL": "1"}, clear=False):
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            receipt = config_tool.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="throughput-recovery-env-test", max_steps=8,
                judge_protocol="text-paint-sft-pairwise-consistent-v1", mode="fresh",
                batch_size=16, group_size=4, initial_inflight=32, max_inflight=32,
                adapter_revision="ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                adapter_prefix="text-pairwise-scaled-20260929/step_16",
                adapter_sha256="e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
            )
            self.assertEqual(receipt["checkpoint_interval"], 1)

    def test_omitted_checkpoint_argument_without_environment_keeps_interval_four(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            receipt = config_tool.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="historical-default-test", max_steps=8,
                judge_protocol=profile.REWARD_PROTOCOL, mode="fresh",
                batch_size=16, group_size=4, initial_inflight=32, max_inflight=32,
                adapter_revision=profile.INITIALIZER_REVISION,
                adapter_prefix=profile.INITIALIZER_PREFIX,
                adapter_sha256=profile.INITIALIZER_SHA256,
            )
            self.assertEqual(receipt["checkpoint_interval"], 4)

    def test_invalid_profile_values_and_receipt_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            config_tool._resolved_checkpoint_interval("zero")
        with self.assertRaises(ValueError):
            config_tool._resolved_checkpoint_interval(True)
        with self.assertRaises(ValueError):
            profile.ObservedTiming(float("nan"), 0, 1, 1)
        with self.assertRaises(ValueError):
            profile.ObservedTiming(0, 0, float("inf"), 1)
        with self.assertRaises(ValueError):
            profile.budget_check(profile.ObservedTiming(0, 0, 1, 1), 1, float("nan"))
        with tempfile.TemporaryDirectory() as temp:
            output, receipt = self._prepare(Path(temp))
            value = tomllib.loads(output.read_text())
            bad = dict(receipt, checkpoint_interval=4)
            with self.assertRaisesRegex(ValueError, "receipt"):
                profile.validate_config(value, bad)
            value["orchestrator"]["concurrency"]["max_inflight"] = 96
            with self.assertRaisesRegex(ValueError, "does not match"):
                profile.validate_config(value)
            value["orchestrator"]["concurrency"]["max_inflight"] = 32
            value["ckpt"]["interval"] = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                profile.validate_config(value)

    def test_capacity_estimate_is_observation_bound_without_throughput_claim(self):
        observed = profile.ObservedTiming(
            initial_warmup=600, first_batch_wait=900, update_seconds=300, publish_budget=30,
            refill_seconds_per_update=120, policy_cycle_seconds=30,
        )
        result = profile.budget_check(observed, 4, 3_400)
        self.assertEqual(result["required_runtime_seconds"], 3_420)
        self.assertFalse(result["fits_observed_bound"])
        self.assertEqual(result["runtime_bound_type"], "optimistic_lower_bound")
        self.assertFalse(result["capacity_proven"])
        self.assertFalse(result["launch_ready"])
        self.assertIsNone(result["throughput_claim"])

    def test_profile_runtime_requires_adapter_only_binding(self):
        profile.validate_runtime_environment(profile.required_environment())
        with self.assertRaisesRegex(ValueError, "ADAPTER_ONLY_CHECKPOINTS"):
            profile.validate_runtime_environment({"PAINTER_RL_CHECKPOINT_INTERVAL": "1"})

    def test_profile_prepare_rejects_critical_override_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            with self.assertRaisesRegex(ValueError, "profile-critical"):
                profile.prepare_profile(
                    config_tool.prepare,
                    output=output,
                    model=root / "model",
                    run_root=root,
                    manifest=manifest,
                    max_steps=8,
                    batch_size=8,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
