import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "text_pairwise_score_validation", ROOT / "painter/runs/text-pairwise-rl-20260928/score_validation.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ScoreValidationTests(unittest.TestCase):
    def test_uncertain_consistent_judgment_is_not_comparable(self):
        self.assertFalse(MODULE._resolved_judgment({"status": "uncertain", "score": None}))

    def test_scored_and_identical_results_require_finite_unit_score(self):
        self.assertTrue(MODULE._resolved_judgment({"status": "scored", "score": 0.75}))
        self.assertTrue(MODULE._resolved_judgment({"status": "identical_canvases", "score": 0.5}))
        self.assertFalse(MODULE._resolved_judgment({"status": "scored", "score": None}))
        self.assertFalse(MODULE._resolved_judgment({"status": "scored", "score": float("nan")}))


if __name__ == "__main__":
    unittest.main()
