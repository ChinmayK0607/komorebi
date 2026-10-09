"""Regression coverage for receipt-backed remote topology validation."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def representative_receipt() -> dict:
    """Return the prepared receipt shape used by the Diverse-flow run."""
    return {
        "schema": "diverse-flow-rl-config-receipt-v1",
        "run_name": "diverse-flow-rl-20261009",
        "hardware_profile": "h100_2",
        "mode": "recovered_policy",
        "requested_updates": 20,
        "updates": 20,
        "effective_updates": 20,
        "checkpoint_steps": [4, 8, 12, 16, 20],
        "checkpoint_interval": 4,
        "batch_size": 16,
        "group_size": 4,
        "initial_inflight": 16,
        "max_inflight": 16,
        "activation_cpu_offload": False,
        "deployment": {
            "type": "single_node",
            "gpus_per_node": 2,
            "num_train_gpus": 1,
            "num_infer_gpus": 1,
        },
        "inference_parallelism": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
        },
        "reward": "text-paint-sft-pairwise-consistent-v1",
    }


class RemoteSummaryTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the training controller even when a caller's shell has an
        # evaluation-only setting left over from another offline check.
        cls._evaluation_env = patch.dict(os.environ, {"PAINTER_EVALUATION_ONLY": "0"}, clear=False)
        cls._evaluation_env.start()
        cls.addClassCleanup(cls._evaluation_env.stop)
        cls.operation = load("diverse_flow_remote_summary_topology", RUN / "operate_single.py")

    def _summary(self, receipt: dict, receipt_path: Path) -> dict:
        base_summary = {
            "config": {
                "profile": "diverse-flow",
                "checkpoint_interval": 4,
                "batch_size": 16,
                "group_size": 4,
                "initial_inflight": 16,
                "max_inflight": 16,
                "activation_cpu_offload": False,
                "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                "reward": "text-paint-sft-pairwise-consistent-v1",
            },
            "preparation": {"schema": "Diverse-flow-rl-preparation-v1"},
        }

        receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

        def read(_sftp, remote, *, timeout):
            self.assertTrue(remote.endswith("/rl-config-receipt.json"))
            self.assertGreater(timeout, 0)
            return receipt_path.read_bytes()

        with patch.object(self.operation, "_base_summary", return_value=base_summary), \
             patch.object(self.operation.base, "_sftp_read", side_effect=read):
            return self.operation._remote_summary(object(), timeout=5.0)

    def test_h100_receipt_topology_is_overlaid_and_accepted(self):
        receipt = representative_receipt()
        with tempfile.TemporaryDirectory() as directory:
            summary = self._summary(receipt, Path(directory) / "rl-config-receipt.json")

        with patch.object(self.operation, "HARDWARE_PROFILE", "h100_2"):
            self.assertEqual(summary["config"]["deployment"], receipt["deployment"])
            self.assertEqual(summary["config"]["inference_parallelism"], receipt["inference_parallelism"])
            self.operation._validate_config_summary(summary)

    def test_wrong_or_missing_topology_fails_closed(self):
        cases = (
            ("wrong inference tensor parallelism", "inference_parallelism", {"tensor_parallel_size": 2, "data_parallel_size": 1}),
            ("missing inference tensor parallelism", "inference_parallelism", None),
            ("wrong deployment", "deployment", {"type": "single_node", "gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2}),
            ("missing deployment", "deployment", None),
        )
        with patch.object(self.operation, "HARDWARE_PROFILE", "h100_2"):
            for name, field, value in cases:
                with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                    receipt = representative_receipt()
                    if value is None:
                        receipt.pop(field)
                    else:
                        receipt[field] = value
                    summary = self._summary(receipt, Path(directory) / "rl-config-receipt.json")
                    with self.assertRaisesRegex(self.operation.base.OperationError, "remote .*topology"):
                        self.operation._validate_config_summary(summary)


if __name__ == "__main__":
    unittest.main()
