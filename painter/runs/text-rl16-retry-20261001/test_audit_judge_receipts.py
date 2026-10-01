"""Pure tests for the offline RL16 receipt audit."""

from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

from audit_judge_receipts import audit


def _receipt(anchor: str, candidate: str, *, score: float, conflict: bool = False) -> dict:
    first = {
        "assessment": {"winner_id": candidate, "preference": "moderate", "reason": "visible"},
        "calls": [{
            "assessment": {"winner_id": candidate, "preference": "moderate"},
            "image_ids": {"A": anchor, "B": candidate},
            "image_sha256": {"A": "a" * 64, "B": "b" * 64},
        }],
    }
    second_winner = anchor if conflict else candidate
    second = {
        "assessment": {"winner_id": second_winner, "preference": "slight", "reason": "visible"},
        "calls": [{
            "assessment": {"winner_id": second_winner, "preference": "slight"},
            "image_ids": {"A": candidate, "B": anchor},
            "image_sha256": {"A": "b" * 64, "B": "a" * 64},
        }],
    }
    return {
        "status": "scored",
        "protocol": "text-paint-sft-pairwise-consistent-v1",
        "image_ids": {"anchor": anchor, "candidate": candidate},
        "image_sha256": {"anchor": "a" * 64, "candidate": "b" * 64},
        "winner_id": candidate,
        "preference": "moderate",
        "score": score,
        "views": [first, second],
        "recheck_views": [],
    }


class AuditTests(unittest.TestCase):
    def test_join_recomputes_score_and_detects_swapped_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "skinny-traces.json").write_text(json.dumps([{
                "id": "trace-a", "trace_id": "trace-a", "task_id": "task-a", "family": "simple",
                "group": {"id": "group-a"}, "info": {
                    "arrival": {"step": 2}, "dispatch": {"step": 1},
                    "final_canvas_score": 0.8333333333333333, "reward_status": "scored",
                }, "rewards": {"final_canvas_aesthetic": {"score": 0.8333333333333333}},
            }]) + "\n")
            receipt_dir = root / "rl-rollouts" / "trace-a"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "text-judge.json").write_text(json.dumps(
                _receipt("img_" + "a" * 20, "img_" + "b" * 20, score=0.8333333333333333, conflict=True)
            ) + "\n")
            (root / "metrics.jsonl").write_text(json.dumps({
                "step": 1, "train/agg/effective/agent/reward/mean": 0.25,
                "train/agg/all/agent/reward/mean": 0.125,
            }) + "\n")
            report = audit(root)
            self.assertEqual(report["trace_records"], 1)
            self.assertEqual(report["receipt_records"], 1)
            self.assertEqual(report["receipt_issue_counts"]["initial_swapped_winner_conflict"], 1)
            self.assertNotIn("score_recomputation_mismatch", report["receipt_issue_counts"])
            self.assertEqual(report["trainer_metrics"]["rows"][0]["effective_mean"], 0.25)
            self.assertEqual(report["arrival_step_summary"][0]["visual_conditional_mean"], 0.8333333333333333)

    def test_selected_repeat_stays_out_of_rollout_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "skinny-traces.json").write_text(json.dumps([{
                "id": "trace-a", "trace_id": "trace-a", "task_id": "task-a", "family": "simple",
                "info": {"final_canvas_score": 0.8333333333333333, "reward_status": "scored"},
            }]) + "\n")
            original = _receipt("img_" + "a" * 20, "img_" + "b" * 20, score=0.8333333333333333)
            receipt_dir = root / "rl-rollouts" / "trace-a"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "text-judge.json").write_text(json.dumps(original) + "\n")
            repeat = copy.deepcopy(original)
            repeat["preference"] = "slight"
            repeat["score"] = 2 / 3
            for view in repeat["views"]:
                view["assessment"]["preference"] = "slight"
                view["calls"][0]["assessment"]["preference"] = "slight"
            repeat_dir = root / "repeat"
            repeat_dir.mkdir()
            (repeat_dir / "trace-a.json").write_text(json.dumps(repeat) + "\n")
            report = audit(root)
            self.assertEqual(report["receipt_records"], 1)
            self.assertEqual(report["repeat_receipt_records"], 1)
            diagnostic = report["frozen_repeat_diagnostics"]
            self.assertEqual(diagnostic["same_image_identity_count"], 1)
            self.assertEqual(diagnostic["preference_changed_count"], 1)
            self.assertEqual(diagnostic["rows"][0]["original_score"], 0.8333333333333333)
            self.assertAlmostEqual(diagnostic["rows"][0]["repeat_score"], 2 / 3)

    def test_audit_detects_final_selection_that_disagrees_with_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = _receipt("img_" + "a" * 20, "img_" + "b" * 20, score=1.0)
            receipt["preference"] = "clear"
            receipt["score"] = 1.0
            receipt_dir = root / "rl-rollouts" / "trace-a"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "text-judge.json").write_text(json.dumps(receipt) + "\n")
            report = audit(root)
            self.assertIn("consensus_selection_mismatch", report["receipt_issue_counts"])


if __name__ == "__main__":
    unittest.main()
