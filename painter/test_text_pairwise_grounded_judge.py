import unittest

from painter.text_pairwise_grounded_judge import PROTOCOL, cap_candidate_margin


def receipt(candidate_first: str, candidate_second: str, score: float) -> dict:
    return {
        "status": "scored", "score": score,
        "views": [
            {"label": {"brief_fidelity_A": "missing",
                       "brief_fidelity_B": candidate_first,
                       "visible_evidence_A": "No subject",
                       "visible_evidence_B": "Visible marks"}},
            {"label": {"brief_fidelity_A": candidate_second,
                       "brief_fidelity_B": "partial",
                       "visible_evidence_A": "Visible marks",
                       "visible_evidence_B": "Incomplete subject"}},
        ],
    }


class GroundedJudgeTest(unittest.TestCase):
    def test_missing_subject_cannot_win_large_margin(self):
        result = cap_candidate_margin(receipt("missing", "partial", 1.0))
        self.assertEqual(result["protocol"], PROTOCOL)
        self.assertEqual(result["raw_pairwise_score"], 1.0)
        self.assertEqual(result["score"], 0.5)

    def test_partial_subject_has_moderate_cap(self):
        result = cap_candidate_margin(receipt("partial", "clear", 1.0))
        self.assertAlmostEqual(result["score"], 2 / 3)

    def test_clear_subject_preserves_pairwise_score(self):
        result = cap_candidate_margin(receipt("clear", "clear", 0.75))
        self.assertEqual(result["score"], 0.75)

    def test_judge_must_report_visible_evidence(self):
        row = receipt("clear", "clear", 0.75)
        del row["views"][0]["label"]["visible_evidence_B"]
        with self.assertRaises(ValueError):
            cap_candidate_margin(row)


if __name__ == "__main__":
    unittest.main()
