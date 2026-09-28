"""Invalid painting output must count as a policy loss, not a judge timeout."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("score_overfit_gallery", HERE / "score_overfit_gallery.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MissingCanvasScoringTests(unittest.TestCase):
    def test_one_valid_canvas_wins_without_api_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            gallery = Path(temp)
            (gallery / "scene-reference.png").write_bytes(b"reference")
            (gallery / "sft.png").write_bytes(b"valid")
            row = {"id": "scene", "reference_sha256": module.digest(gallery / "scene-reference.png"),
                   "sft-step512": {"final": "sft.png", "final_sha256": module.digest(gallery / "sft.png")},
                   "rl-step-2": {"final": None, "final_sha256": None}}
            with patch.object(module, "judge_pair") as judge:
                result = module.score_row(row, gallery=gallery, step=2, key="unused")
            self.assertEqual(result["status"], "rl_invalid")
            self.assertEqual((result["score_sft"], result["score_rl"]), (1.0, 0.0))
            judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
