import json
import unittest

from painter.text_pairwise_margin_judge import _label


class MarginJudgeTest(unittest.TestCase):
    def test_strength_and_winner_must_agree(self):
        payload = {"winner": "A", "strength": 3, "reason": "Better silhouette",
                   "weaknesses_A": ["Flat color"],
                   "weaknesses_B": ["Wrong silhouette"]}
        self.assertEqual(_label(json.dumps(payload))["strength"], 3)
        payload["winner"] = "B"
        with self.assertRaises(ValueError):
            _label(json.dumps(payload))
        payload["winner"] = "A"
        payload["strength"] = 3.5
        with self.assertRaises(ValueError):
            _label(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
