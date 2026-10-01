"""Offline contracts for the finite MiMo Pro RL16 preparation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from painter import text_pairwise_consistent_side_judge_v2_pro as pro_judge


RUN = Path(__file__).parent


class ProRunContractTests(unittest.TestCase):
    def test_source_identity_binds_initializer_anchor_and_pro_judge(self):
        value = json.loads((RUN / "source-identity.json").read_text())
        self.assertEqual(value["policy_initializer"]["revision"], "ac4d3eda8e5925e2d23992cfdc1a63620ebde573")
        self.assertEqual(value["policy_initializer"]["adapter_sha256"], "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c")
        self.assertEqual(value["frozen_sft_initializer"]["revision"], "adf687e5b321514eda7be6eab050434fdc8e44ff")
        self.assertEqual(value["training_data"]["train_briefs"], 40)
        self.assertEqual(value["training_data"]["heldout_briefs"], 47)
        self.assertEqual(value["judge"]["model"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(value["judge"]["format"], "identity-joint-side-schema-v2-pro-v1")

    def test_all_entrypoints_pin_pro_transport_and_cpu_aware_renderer_limit(self):
        for name in ("run.sh", "execute_pipeline.sh", "prepare_node.sh", "evaluate.sh", "finalize.sh"):
            text = (RUN / name).read_text()
            with self.subTest(name=name):
                self.assertIn("PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-pro", text)
                self.assertIn("PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-pro-v1", text)
        train = (RUN / "run.sh").read_text()
        for flag in (
            "PAINTER_P5_BRUSH_API_REFERENCE=1",
            "PAINTER_INSPECT_CHOICE_FEEDBACK=1",
            "PAINTER_REQUIRE_VALID_FINAL_ACTION=1",
            "PAINTER_JS_SYNTAX_PREFLIGHT=1",
            "PAINTER_RENDER_WORKERS=\"${PAINTER_RENDER_WORKERS:-2}\"",
        ):
            self.assertIn(flag, train)
        self.assertIn("renderer-env/bin/python", train)

    def test_pro_wrapper_preserves_v2_prompt_and_records_requested_returned_model(self):
        async def fake_judge(**kwargs):
            self.assertEqual(kwargs["model"], "xiaomi/mimo-v2.6-pro")
            return {
                "protocol": "text-paint-sft-pairwise-consistent-v2-model-calibration",
                "format_version": "identity-joint-side-schema-v2-model-calibration",
                "requested_model": kwargs["model"],
                "returned_models": [kwargs["model"]],
                "status": "scored",
                "score": 0.5,
            }

        with patch.object(pro_judge.model_judge, "judge_pair", fake_judge):
            result = asyncio.run(pro_judge.judge_pair(
                brief=Path("brief.txt"), anchor=Path("anchor.png"),
                candidate=Path("candidate.png"), key_file=Path("gateway.env"),
                model="xiaomi/mimo-v2.6-pro",
            ))
        self.assertEqual(result["protocol"], "text-paint-sft-pairwise-consistent-v1")
        self.assertEqual(result["format_version"], "identity-joint-side-schema-v2-pro-v1")
        self.assertEqual(result["judge_model"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(result["requested_model"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(result["returned_models"], ["xiaomi/mimo-v2.6-pro"])

    def test_pro_wrapper_rejects_flash_override(self):
        with self.assertRaises(ValueError):
            asyncio.run(pro_judge.judge_pair(
                brief=Path("brief.txt"), anchor=Path("anchor.png"),
                candidate=Path("candidate.png"), key_file=Path("gateway.env"),
                model="xiaomi/mimo-v2.6-flash",
            ))


if __name__ == "__main__":
    unittest.main()
