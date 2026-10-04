"""Offline contract tests for the three-judge fallback."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_three_judge_test_module", RUN / "frozen_three_judge.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PANEL_SPEC = importlib.util.spec_from_file_location("frozen_three_panel_test_module", RUN / "prepare_panel.py")
assert PANEL_SPEC and PANEL_SPEC.loader
panel_builder = importlib.util.module_from_spec(PANEL_SPEC)
PANEL_SPEC.loader.exec_module(panel_builder)


class FrozenThreeJudgeTests(unittest.TestCase):
    def test_one_pair_receipt_resumes_replaces_incomplete_and_rejects_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel_path = root / "pairs.json"
            panel_path.write_text(json.dumps(panel_builder.build_pairs()), encoding="utf-8")
            manifest_path = root / "input-manifest.json"
            manifest = runner.prepare_manifest(panel_path=panel_path, output=manifest_path, repeat_count=2)
            pair = manifest["pairs"][0]
            output = root / "results"
            sem = asyncio.Semaphore(1)
            lock = threading.Lock()
            calls = 0

            def fake_view(*, key, model, brief, left, right, left_id, right_id):
                nonlocal calls
                calls += 1
                return {
                    "schema": runner.CALL_SCHEMA,
                    "protocol": runner.PROTOCOL,
                    "status": "valid",
                    "image_ids": {"LEFT": left_id, "RIGHT": right_id},
                    "requested_model": model,
                    "returned_model": model,
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "reasoning_tokens": 3, "cost": 0.01},
                    "elapsed_seconds": 0.01,
                    "finish_reason": "stop",
                    "assessment": {
                        "winner_id": left_id,
                        "preference": "slight",
                        "reason": "The main form is clearer.",
                        "left_assessment": {"weaknesses": ["A soft edge remains."]},
                        "right_assessment": {"weaknesses": ["The silhouette is less clear."]},
                    },
                }

            async def one_call():
                return await runner._one_call(
                    manifest=manifest, pair=pair, model=runner.MODELS[0], round_index=0,
                    order="canonical", output=output, key="unused-test-key",
                    progress_lock=lock, sem=sem,
                )

            with mock.patch.object(runner, "_view", side_effect=fake_view), \
                 mock.patch.object(runner.urllib.request, "urlopen", side_effect=AssertionError("network called")):
                first = asyncio.run(one_call())
                self.assertEqual(first["status"], "valid")
                self.assertEqual(calls, 1)
                resumed = asyncio.run(one_call())
                self.assertEqual(resumed["identity"], first["identity"])
                self.assertEqual(calls, 1)

                receipt = output / "receipts" / "calls" / runner.MODELS[0].replace("/", "__") / pair["pair_id"] / "round-00-canonical.json"
                receipt.write_text("{\"schema\":", encoding="utf-8")
                replaced = asyncio.run(one_call())
                self.assertEqual(replaced["status"], "valid")
                self.assertEqual(calls, 2)

                stale = json.loads(receipt.read_text(encoding="utf-8"))
                stale["identity"]["pair_id"] = "wrong-pair"
                receipt.write_text(json.dumps(stale), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "stale receipt identity"):
                    asyncio.run(one_call())

    def test_two_repeats_and_credit_group_metadata_are_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel_path = root / "pairs.json"
            panel_path.write_text(json.dumps(panel_builder.build_pairs()), encoding="utf-8")
            manifest = runner.prepare_manifest(panel_path=panel_path, output=root / "manifest.json", repeat_count=2)
            self.assertEqual(len(manifest["pairs"]), 24)
            self.assertEqual(manifest["repeat_count"], 2)
            self.assertEqual(manifest["planned_call_count"], 288)
            self.assertTrue(all(pair.get("group_id") == pair.get("case_id") for pair in manifest["pairs"]))
            self.assertEqual(manifest["human_labels"]["explicit_pair_count"], 3)
            self.assertIn("four-image on-policy rollout groups", manifest["panel_source_evidence"]["credit_assignment_limitation"])
            self.assertNotIn("groups_with_four_distinct_rollouts", manifest["panel_source_evidence"])
            self.assertIn("groups_with_four_distinct_images", manifest["panel_source_evidence"])
            self.assertEqual(len(manifest["human_labels"]["provenance"]), 3)
            self.assertIn("correlated", manifest["human_labels"]["correlation_note"])
            gold_pair = next(pair for pair in manifest["pairs"] if pair.get("human_gold") == "LEFT")
            rows = []
            for round_index in range(2):
                for order in ("canonical", "reverse"):
                    rows.append({
                        "identity": {"pair_id": gold_pair["pair_id"], "model": runner.MODELS[0], "round": round_index},
                        "status": "valid",
                        "call": {"assessment": {"winner_id": gold_pair["left"]["image_id"]}},
                    })
            gold = runner._human_gold_agreement(rows=rows, manifest=manifest)
            self.assertEqual(gold["by_model"][runner.MODELS[0]]["eligible_rounds"], 2)
            self.assertEqual(gold["by_model"][runner.MODELS[0]]["agreeing_rounds"], 2)

    def test_real_side_parser_keeps_reversed_winner_weakness_and_conservative_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel_path = root / "pairs.json"
            panel_path.write_text(json.dumps(panel_builder.build_pairs()), encoding="utf-8")
            manifest = runner.prepare_manifest(panel_path=panel_path, output=root / "manifest.json", repeat_count=2)
            pair = manifest["pairs"][0]
            left_id = pair["left"]["image_id"]
            right_id = pair["right"]["image_id"]

            canonical_raw = json.dumps({
                "left_assessment": {"brief_fidelity": "clear", "visible_evidence": "left", "weaknesses": ["left weak"]},
                "right_assessment": {"brief_fidelity": "clear", "visible_evidence": "right", "weaknesses": ["right weak"]},
                "reason": "The right panel is clearer.", "winner": "RIGHT", "preference": "clear",
            })
            reverse_raw = json.dumps({
                # In the reversed transport order, LEFT is the original RIGHT.
                "left_assessment": {"brief_fidelity": "clear", "visible_evidence": "right", "weaknesses": ["right weak"]},
                "right_assessment": {"brief_fidelity": "clear", "visible_evidence": "left", "weaknesses": ["left weak"]},
                "reason": "The first panel is clearer.", "winner": "LEFT", "preference": "slight",
            })
            canonical = runner.rubric._parse(canonical_raw, (left_id, right_id))
            reverse = runner.rubric._parse(reverse_raw, (right_id, left_id))
            rows = []
            for order, assessment in (("canonical", canonical), ("reverse", reverse)):
                rows.append({
                    "identity": runner._identity(manifest, pair, runner.MODELS[0], 0, order),
                    "_pair": pair,
                    "status": "valid",
                    "call": {"assessment": assessment},
                })
            round_result = runner._round_result(
                manifest=manifest, pair=pair, model=runner.MODELS[0], round_index=0, calls=rows,
            )
            self.assertEqual(round_result["status"], "scored")
            self.assertEqual(round_result["winner_id"], right_id)
            self.assertEqual(round_result["preference"], "slight")
            self.assertAlmostEqual(round_result["right_image_reward"], 0.5 + 1 / 6)
            gallery = runner._gallery_rounds(rows)
            rendered = gallery[(pair["pair_id"], runner.MODELS[0], 0)]
            self.assertEqual(rendered["winner"], "RIGHT")
            self.assertEqual(rendered["winner_weakness"], "right weak")
            self.assertAlmostEqual(rendered["right_image_reward"], 0.5 + 1 / 6)


if __name__ == "__main__":
    unittest.main()
