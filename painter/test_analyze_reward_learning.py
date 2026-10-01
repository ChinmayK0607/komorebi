import json
import tempfile
import unittest
from pathlib import Path

from painter.analyze_reward_learning import (
    _advantage_audit,
    _completed_steps,
    _receipt_summary,
    _repeated_prompt_steps,
    _retained_invalid,
)


class RewardLearningAuditTest(unittest.TestCase):
    def test_retained_canvas_after_invalid_last_turn(self):
        row = {"info": {"final_canvas_valid": True, "turns": [{"valid": True}, {"valid": False}]}, "trace": {}}
        self.assertTrue(_retained_invalid(row))

    def test_advantage_scope_and_unaligned_are_reported(self):
        row = {"info": {"final_canvas_valid": True, "turns": [{"valid": True}]},
               "trace": {"nodes": [
                   {"message": {"role": "user"}, "token_ids": [1], "mask": [False]},
                   {"message": {"role": "assistant"}, "token_ids": [2, 3], "mask": [True, True]},
               ]}}
        ann = {"info": {"advantage": 1.0}, "branches": [{"advantages": [0.0, 1.0, 1.0]}]}
        got = _advantage_audit(row, ann)
        self.assertTrue(got["aligned"])
        self.assertEqual(got["prompt_nonzero"], 0)
        self.assertEqual(got["assistant_nonzero"], 2)
        ann["branches"][0]["advantages"] = [1.0]
        self.assertFalse(_advantage_audit(row, ann)["aligned"])

    def test_unaligned_advantages_do_not_emit_token_claims(self):
        row = {"info": {"final_canvas_valid": True, "turns": [{"valid": False}]},
               "trace": {"nodes": [
                   {"message": {"role": "user"}, "token_ids": [1, 2], "mask": [False, False]},
                   {"message": {"role": "assistant"}, "token_ids": [3, 4], "mask": [True, True]},
               ]}}
        ann = {"info": {"advantage": 0.5}, "branches": [{"advantages": [9.0]}]}
        got = _advantage_audit(row, ann)
        self.assertEqual(got["alignment"], "unaligned")
        self.assertIsNone(got["prompt_nonzero"])
        self.assertIsNone(got["assistant_nonzero"])
        self.assertEqual(got["retained_last_invalid_positive_scalar"], 1)

    def test_samples_in_one_ship_step_are_not_repeated_prompts(self):
        rows = [{"trace_id": f"t{i}", "task_id": "fixed"} for i in range(8)]
        orch = {row["trace_id"]: {"info": {"ship": {"step": 8}}} for row in rows}
        self.assertEqual(_repeated_prompt_steps(rows, orch), {})

    def test_multiple_branches_are_unknown_not_first_branch(self):
        row = {"info": {"final_canvas_valid": True, "turns": [{"valid": True}]},
               "trace": {"nodes": [{"message": {"role": "assistant"},
                                      "token_ids": [1], "mask": [True]}]}}
        ann = {"branches": [{"index": 0, "advantages": [1.0]},
                             {"index": 1, "advantages": [2.0]}]}
        got = _advantage_audit(row, ann)
        self.assertIsNone(got["aligned"])
        self.assertEqual(got["alignment"], "unknown")
        self.assertIsNone(got["assistant_nonzero"])

    def test_completed_steps_tolerate_ansi_log(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trainer.log"
            p.write_text("SUCCESS\x1b[0m Step 19 | 1m\nSUCCESS Step 20 | 1m\n")
            self.assertEqual(_completed_steps(p), [19, 20])

    def test_receipt_conflict_recheck_and_fidelity(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "a"; d.mkdir()
            (d / "text-judge.json").write_text(json.dumps({
                "status": "scored", "score": 1/3, "initial_conflict": True,
                "recheck_votes": ["A", "B"], "views": [{"assessment": {"assessments": {
                    "a": {"brief_fidelity": "partial"}}}}]}))
            got = _receipt_summary(Path(td))
            self.assertEqual(got["receipt_count"], 1)
            self.assertEqual(got["initial_conflicts"], 1)
            self.assertEqual(got["recheck_conflicts"], 1)
            self.assertEqual(got["fidelity_labels"]["partial"], 1)


if __name__ == "__main__":
    unittest.main()
