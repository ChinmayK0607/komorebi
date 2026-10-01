"""Contracts for the smaller-group RL16 retry."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from painter.consistent_group_admission import WholeGroupAdmissionGuard, check_group


RUN = Path(__file__).parent


def _episode(index: int):
    class Trace:
        has_error = False
        error = None
        info = {"reward_status": "model_invalid_canvas", "final_canvas_score": 0.0}

    class Episode:
        id = f"episode-{index}"
        ok = True
        error = None
        traces = [Trace()]

    return Episode()


class RetryContractTests(unittest.TestCase):
    def test_initializer_identity_remains_separate_and_retry_shape_is_explicit(self):
        identity = json.loads((RUN / "source-identity.json").read_text())
        self.assertNotEqual(
            identity["policy_initializer"]["adapter_sha256"],
            identity["frozen_sft_initializer"]["adapter_sha256"],
        )
        self.assertEqual(identity["protocol"]["group_size"], 4)
        self.assertEqual(identity["protocol"]["initial_inflight"], 32)
        self.assertEqual(identity["protocol"]["max_inflight"], 32)

    def test_four_member_group_is_checked_without_changing_zero_score_semantics(self):
        group = [_episode(index) for index in range(4)]
        self.assertTrue(check_group(group, group_size=4).valid)
        guard = WholeGroupAdmissionGuard(lambda _group: True, group_size=4)
        self.assertTrue(guard(group))
        self.assertEqual(guard.stats()["groups_admitted"], 1)

    def test_optional_group_telemetry_records_admission_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "groups.jsonl"
            with patch.dict(os.environ, {"PAINTER_CONSISTENT_GROUP_TELEMETRY": str(path)}):
                guard = WholeGroupAdmissionGuard(lambda _group: True, group_size=4)
                self.assertTrue(guard([_episode(index) for index in range(4)]))
            row = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(row["schema"], "consistent-group-admission-telemetry-v1")
            self.assertEqual(row["group_size"], 4)
            self.assertEqual(row["groups_admitted"], 1)

    def test_retry_launchers_keep_final_credit_syntax_and_group_overrides(self):
        for name in ("run.sh", "execute_pipeline.sh", "prepare_node.sh"):
            text = (RUN / name).read_text()
            self.assertIn("PAINTER_GROUP_SIZE=4", text)
            self.assertIn("PAINTER_INITIAL_INFLIGHT=32", text)
            self.assertIn("PAINTER_MAX_INFLIGHT=32", text)
        self.assertIn("PAINTER_REQUIRE_VALID_FINAL_ACTION=1", (RUN / "run.sh").read_text())
        self.assertIn("PAINTER_JS_SYNTAX_PREFLIGHT=1", (RUN / "run.sh").read_text())
        self.assertIn("patch_brush_centroid.py", (RUN / "prepare_node.sh").read_text())
        self.assertIn("renderer-receipt.json", (RUN / "prepare_node.sh").read_text())

    def test_config_builder_emits_batch16_group4_and_32_inflight(self):
        source = RUN.parent / "text-pairwise-rl-20260928" / "prepare_config.py"
        spec = importlib.util.spec_from_file_location("retry_prepare_config", source)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            module.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="text-rl16-retry-20261001", max_steps=50,
                checkpoint_interval=4, mode="recovered_policy", batch_size=16,
                group_size=4, initial_inflight=32, max_inflight=32,
                adapter_revision="ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                adapter_prefix="text-pairwise-scaled-20260929/step_16",
                adapter_sha256="e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
            )
            config = tomllib.loads(output.read_text())
            self.assertEqual(config["orchestrator"]["batch_size"], 16)
            self.assertEqual(config["orchestrator"]["group_size"], 4)
            self.assertEqual(config["orchestrator"]["concurrency"]["max_inflight"], 32)


if __name__ == "__main__":
    unittest.main()
