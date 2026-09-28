"""Offline checks for prompt-bound, order-reversed pairwise rewards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from painter import text_pairwise_judge as judge


class TextPairwiseJudgeTests(unittest.TestCase):
    def test_reversed_views_agree_on_candidate_and_bind_same_brief(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint the pear to the left of the mug.\n")
            anchor, candidate = root / "sft.png", root / "rl.png"
            Image.new("RGB", (20, 20), "red").save(anchor)
            Image.new("RGB", (20, 20), "blue").save(candidate)
            key_file = root / "gateway.env"
            key_file.write_text("AI_GATEWAY_API_KEY=dummy-for-offline-test\n")
            key_file.chmod(0o600)

            def fake_view(*, key, brief, a, b):
                self.assertEqual(key, "dummy-for-offline-test")
                self.assertIn("pear to the left", brief)
                return {"label": {"winner": "B" if a == anchor else "A"},
                        "image_sha256": {"A": judge.digest(a), "B": judge.digest(b)}}

            with patch.object(judge, "_view", side_effect=fake_view) as mocked:
                result = asyncio.run(judge.judge_pair(
                    brief=brief, anchor=anchor, candidate=candidate,
                    key_file=key_file,
                ))
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["brief_sha256"], judge.digest(brief))

    def test_label_rejects_missing_weaknesses(self):
        with self.assertRaisesRegex(ValueError, "weaknesses_B"):
            judge._label('{"winner":"A","reason":"clearer pear","weaknesses_A":["flat"]}')

    def test_malformed_provider_view_gets_one_retry_without_becoming_a_loss(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a pear.\n")
            anchor, candidate = root / "sft.png", root / "rl.png"
            Image.new("RGB", (20, 20), "red").save(anchor)
            Image.new("RGB", (20, 20), "blue").save(candidate)
            key_file = root / "gateway.env"
            key_file.write_text("AI_GATEWAY_API_KEY=dummy-for-offline-test\n")
            key_file.chmod(0o600)
            calls = {"forward": 0}

            def fake_view(*, key, brief, a, b):
                if a == anchor:
                    calls["forward"] += 1
                    if calls["forward"] == 1:
                        raise ValueError("provider returned malformed JSON")
                    winner = "B"
                else:
                    winner = "A"
                return {"label": {"winner": winner}}

            with patch.object(judge, "_view", side_effect=fake_view) as mocked:
                result = asyncio.run(judge.judge_pair(
                    brief=brief, anchor=anchor, candidate=candidate,
                    key_file=key_file,
                ))
            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
