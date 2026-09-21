"""Offline tests for matched-evaluation plumbing; no model or renderer calls."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import evaluate_matched


class EvaluateMatchedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_root = evaluate_matched.P
        evaluate_matched.P = self.root
        self.addCleanup(setattr, evaluate_matched, "P", self.old_root)
        tasks = [
            {
                "id": f"{group}-paint",
                "reference": f"data/{group}/reference.png",
                "source_group": group,
                "kind": "paint",
            }
            for group in evaluate_matched.HELDOUT_GROUPS
        ]
        (self.root / "evaluation-tasks.json").write_text(json.dumps(tasks))
        (self.root / "campaign.json").write_text(json.dumps({
            "operational_limits": {"max_turns": 8, "max_tokens_per_response": 16384},
        }))

    def test_subset_is_exact_and_ordered(self):
        rows = evaluate_matched._task_subset(json.loads(
            (self.root / "evaluation-tasks.json").read_text()
        ))
        self.assertEqual(
            [row["source_group"] for row in rows], list(evaluate_matched.HELDOUT_GROUPS)
        )
        with self.assertRaises(ValueError):
            evaluate_matched._task_subset(rows[:3])

    def test_model_name_does_not_become_an_output_path_and_metadata_is_matched(self):
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return {"valid_canvas": True, "finished": True, "turns": []}

        old_run = evaluate_matched.run
        evaluate_matched.run = fake_run
        self.addCleanup(setattr, evaluate_matched, "run", old_run)
        with mock.patch.object(evaluate_matched.sys, "platform", "linux"):
            summary = evaluate_matched.main([
                "--model", "served/model:candidate",
                "--label", "candidate",
                "--policy-version", "adapter-step-0",
                "--output", str(self.root / "evaluations"),
                "--workers", "2",
            ])
        self.assertEqual(len(calls), 4)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["api_version"], "v1")
        self.assertEqual(summary["max_turns"], 3)
        self.assertEqual(summary["max_tokens"], 8192)
        self.assertEqual(summary["temperature"], 1.0)
        self.assertEqual({kwargs["seed"] for _, kwargs in calls}, {918200})
        self.assertEqual({args[3] for args, _ in calls}, {"served/model:candidate"})
        metadata = json.loads((self.root / "evaluations/candidate/metadata.json").read_text())
        self.assertEqual(metadata["policy_version"], "adapter-step-0")
        self.assertFalse((self.root / "evaluations/served").exists())

    def test_failures_are_preserved_in_summary(self):
        def fake_run(*args, **kwargs):
            if "bus" in str(args[0]):
                raise RuntimeError("mock renderer unavailable")
            return {"valid_canvas": True, "finished": False}

        old_run = evaluate_matched.run
        evaluate_matched.run = fake_run
        self.addCleanup(setattr, evaluate_matched, "run", old_run)
        with mock.patch.object(evaluate_matched.sys, "platform", "linux"):
            summary = evaluate_matched.main([
                "--model", "base",
                "--label", "simplebaseline",
                "--policy-version", "step24",
                "--output", str(self.root / "evaluations"),
            ])
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(sum("failure" in row for row in summary["results"]), 1)

    def test_non_linux_is_rejected_before_runner_load(self):
        with mock.patch.object(evaluate_matched.sys, "platform", "darwin"):
            with self.assertRaises(SystemExit):
                evaluate_matched.main([
                    "--model", "base", "--label", "candidate", "--policy-version", "x"
                ])


if __name__ == "__main__":
    unittest.main()
