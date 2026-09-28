"""Offline checks for the paid visual judge's resume boundary."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_resumable as scorer  # noqa: E402


class ResumableJudgeTest(unittest.TestCase):
    def test_completed_orientations_are_not_repurchased(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = Path(directory) / "gallery"
            state = Path(directory) / "state"
            gallery.mkdir()
            paths = {}
            for label, color in (("reference", "white"), ("sft", "red"), ("rl", "blue")):
                name = f"test-case-{label}.png"
                path = gallery / name
                Image.new("RGB", (8, 8), color).save(path)
                paths[label] = path
            row = {
                "id": "test-case", "split": "heldout",
                "reference_sha256": scorer.digest(paths["reference"]),
                "sft-step512": {"final": paths["sft"].name,
                                "final_sha256": scorer.digest(paths["sft"])},
                "rl-step-4": {"final": paths["rl"].name,
                              "final_sha256": scorer.digest(paths["rl"])},
            }
            calls = []

            def fake_view(*, reference, a, b, **_kwargs):
                calls.append((a.name, b.name))
                return {
                    "label": {"winner": "A", "reason": "fixture",
                              "weaknesses_A": [], "weaknesses_B": []},
                    "image_sha256": {"reference": scorer.digest(reference),
                                     "A": scorer.digest(a), "B": scorer.digest(b)},
                    "prompt_sha256": hashlib.sha256(scorer.PROMPT.encode()).hexdigest(),
                    "elapsed_seconds": 0.01, "usage": {"cost": 0},
                }

            with patch.object(scorer, "judge_view", side_effect=fake_view):
                first = scorer.score_row(row, gallery=gallery, state=state,
                                         key="unused", timeout=360)
            self.assertEqual(len(calls), 2)
            self.assertEqual(first["score_sft"], 0.5)
            with patch.object(scorer, "judge_view", side_effect=AssertionError("repurchased")):
                second = scorer.score_row(row, gallery=gallery, state=state,
                                          key="unused", timeout=360)
            self.assertEqual(first, second)
            self.assertEqual(len(list(state.rglob("view-*.json"))), 2)
            receipt = state / "test-case/view-1.json"
            tampered = json.loads(receipt.read_text())
            tampered["requested_model"] = "wrong-model"
            receipt.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                scorer.score_row(row, gallery=gallery, state=state,
                                 key="unused", timeout=360)


if __name__ == "__main__":
    unittest.main()
