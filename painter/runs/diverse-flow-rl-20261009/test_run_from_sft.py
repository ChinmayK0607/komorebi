"""Regression coverage for the amended post-anchor config refresh."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalConfigRefreshTests(unittest.TestCase):
    def test_legacy_32_inflight_receipt_is_rebound_to_amended_16_window(self):
        run_from_sft = load("diverse_flow_run_from_sft_contract", RUN / "run_from_sft.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "native.toml"
            config.write_text(
                "[model]\nname = \"/models/policy\"\n"
                "[orchestrator.train]\n"
                "[[orchestrator.train.source]]\n"
                "[orchestrator.train.source.env.taskset]\n"
                "output_root = \"rl-rollouts\"\n",
                encoding="utf-8",
            )
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"rows": [{} for _ in range(96)]}), encoding="utf-8")
            (root / "rl-config-receipt.json").write_text(json.dumps({
                "hardware_profile": "l40s_4",
                "curriculum": {"active_rows": 96},
                "manifest_sha256": run_from_sft._sha256(manifest),
                "updates": 20,
                "effective_updates": 20,
                "checkpoint_steps": [4, 8, 12, 16, 20],
                "checkpoint_interval": 4,
                "policy_initializer": {"revision": "r", "prefix": "p", "sha256": "a" * 64},
                "deployment": {"gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
                "batch_size": 16,
                "group_size": 4,
                "initial_inflight": 32,
                "max_inflight": 32,
            }), encoding="utf-8")

            prepare_calls = []

            def fake_spec(*_args, **_kwargs):
                return SimpleNamespace(
                    name="diverse_flow_final_config_test",
                    loader=SimpleNamespace(exec_module=lambda _module: None),
                )

            def fake_module(_spec):
                def prepare(*, output, **kwargs):
                    prepare_calls.append(kwargs)
                    output.write_text("[model]\nname = \"/models/policy\"\n", encoding="utf-8")
                    output.with_name("rl-config-receipt.json").write_text(json.dumps({
                        "curriculum": {"active_rows": 96},
                        "manifest_sha256": run_from_sft._sha256(manifest),
                        "updates": 20,
                        "effective_updates": 20,
                        "checkpoint_steps": [4, 8, 12, 16, 20],
                        "initial_inflight": kwargs["initial_inflight"],
                        "max_inflight": kwargs["max_inflight"],
                    }), encoding="utf-8")

                return SimpleNamespace(prepare=prepare)

            with patch.object(run_from_sft.importlib.util, "spec_from_file_location", side_effect=fake_spec), \
                    patch.object(run_from_sft.importlib.util, "module_from_spec", side_effect=fake_module):
                run_from_sft._refresh_final_config(config, manifest, 96)

            receipt = json.loads((root / "rl-config-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(len(prepare_calls), 1)
            self.assertEqual(prepare_calls[0]["initial_inflight"], 16)
            self.assertEqual(prepare_calls[0]["max_inflight"], 16)
            self.assertEqual(receipt["initial_inflight"], 16)
            self.assertEqual(receipt["max_inflight"], 16)

            with patch.object(run_from_sft.importlib.util, "spec_from_file_location", side_effect=AssertionError("unexpected refresh")):
                run_from_sft._refresh_final_config(config, manifest, 96)

            self.assertEqual(len(prepare_calls), 1)


if __name__ == "__main__":
    unittest.main()
