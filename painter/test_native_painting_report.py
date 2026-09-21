"""Focused parser/gallery checks for the native painting evidence report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from painter.native_painting_report import build_report
except ImportError:
    from native_painting_report import build_report


class NativePaintingReportTests(unittest.TestCase):
    def test_maps_node_paths_preserves_partial_states_and_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "collected"
            evidence.mkdir()
            reference = root / "reference.png"
            reference.write_bytes(b"png")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "controlled-comparison-eval-v1",
                        "cases": [
                            {
                                "id": "v2-box-007",
                                "family": "box",
                                "source_group": "v2-box-007",
                                "reference_png": str(reference),
                            },
                            {
                                "id": "v2-cup-007",
                                "family": "cup",
                                "source_group": "v2-cup-007",
                                "reference_png": str(reference),
                            },
                        ],
                    }
                )
            )
            canvas_dir = evidence / "evaluation-rollouts/sft/v2-box-007/episode/turn-1"
            canvas_dir.mkdir(parents=True)
            canvas = canvas_dir / "painting.png"
            canvas.write_bytes(b"png")
            (canvas_dir / "program.js").write_text("function draw() {}\n")
            rl_dir = evidence / "evaluation-rollouts/rl/v2-box-007/episode"
            rl_dir.mkdir(parents=True)
            trace_dir = evidence / "sft-out"
            trace_dir.mkdir()
            trace = {
                "version": 1,
                "id": "trace-sft",
                "task": {
                    "data": {
                        "task_id": "v2-box-007",
                        "source_group": "v2-box-007",
                        "family": "box",
                        "policy_label": "sft",
                    }
                },
                "calls": [
                    {"usage": {"prompt_tokens": 12, "completion_tokens": 5}}
                ],
                "info": {
                    "task_id": "v2-box-007",
                    "policy_label": "sft",
                    "turns": [
                        {
                            "turn": 1,
                            "action": "paint",
                            "valid": True,
                            "canvas": "/root/painter/evaluation-rollouts/sft/v2-box-007/episode/turn-1/painting.png",
                        }
                    ],
                    "finished": True,
                    "final_canvas_valid": True,
                    "final_canvas": "/root/painter/evaluation-rollouts/sft/v2-box-007/episode/turn-1/painting.png",
                },
            }
            trace_rl = {
                "version": 1,
                "id": "trace-rl",
                "task": {
                    "data": {
                        "task_id": "v2-box-007",
                        "source_group": "v2-box-007",
                        "family": "box",
                        "policy_label": "rl",
                    }
                },
                "info": {
                    "task_id": "v2-box-007",
                    "policy_label": "rl",
                    "turns": [{"turn": 1, "action": "paint", "valid": False, "failure_kind": "program"}],
                    "operational_cutoff": True,
                    "final_canvas_valid": False,
                },
                "calls": [{"usage": {"prompt_tokens": 9}}],
            }
            (trace_dir / "traces.jsonl").write_text(
                json.dumps({"traces": [trace, trace_rl]}) + "\n"
            )

            report = build_report(evidence, manifest, root / "report")
            self.assertTrue(report["partial"])
            self.assertEqual(report["discovered"]["matched_episodes"], 2)
            self.assertEqual(report["cases"][0]["policies"]["sft"][0]["status"], "complete")
            self.assertEqual(report["cases"][0]["policies"]["rl"][0]["status"], "censored")
            self.assertEqual(report["cases"][0]["policies"]["sft"][0]["tokens"]["total_tokens"], 17)
            self.assertEqual(report["cases"][0]["policies"]["sft"][0]["turns"][0]["code"], str((canvas.parent / "program.js").resolve()))
            self.assertEqual(report["cases"][1]["policies"]["sft"], [])
            page = (root / "report/index.html").read_text()
            self.assertIn("No native trace for this policy/case", page)
            self.assertIn("painting.png", page)
            serialized = json.dumps(report).lower()
            self.assertNotIn('"reward"', serialized)
            self.assertNotIn('"visual_quality"', serialized)

    def test_missing_recorded_final_never_uses_reference_or_previous_canvas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "collected"
            reference_copy = evidence / "eval-assets/v2-box-007.png"
            turn_one = evidence / "evaluation-rollouts/sft/v2-box-007/episode/turn-1"
            turn_two = turn_one.parent / "turn-2"
            turn_one.mkdir(parents=True)
            turn_two.mkdir()
            reference_copy.parent.mkdir(parents=True, exist_ok=True)
            reference_copy.write_bytes(b"reference")
            (turn_one / "painting.png").write_bytes(b"turn-one")
            (turn_one / "program.js").write_text("good();\n")
            (turn_two / "program.js").write_text("bad();\n")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "v2-box-007",
                                "family": "box",
                                "source_group": "v2-box-007",
                                "reference_png": "source-does-not-exist.png",
                            }
                        ]
                    }
                )
            )
            trace_dir = evidence / "sft-out"
            trace_dir.mkdir()
            trace_dir.joinpath("traces.jsonl").write_text(
                json.dumps(
                    {
                        "id": "episode",
                        "traces": [
                            {
                                "id": "trace",
                                "task": {"data": {"task_id": "v2-box-007", "policy_label": "sft"}},
                                "calls": [],
                                "info": {
                                    "task_id": "v2-box-007",
                                    "finished": True,
                                    "final_canvas_valid": True,
                                    "final_canvas": "/root/painter/evaluation-rollouts/sft/v2-box-007/episode/turn-2/painting.png",
                                    "turns": [
                                        {
                                            "turn": 1,
                                            "action": "paint",
                                            "valid": True,
                                            "canvas": "/root/painter/evaluation-rollouts/sft/v2-box-007/episode/turn-1/painting.png",
                                            "reply": "good();",
                                        },
                                        {
                                            "turn": 2,
                                            "action": "paint",
                                            "valid": False,
                                            "failure_kind": "program",
                                            "canvas": "/root/painter/evaluation-rollouts/sft/v2-box-007/episode/turn-1/painting.png",
                                            "reply": "bad();",
                                        },
                                    ],
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )

            report = build_report(evidence, manifest, root / "report")
            episode = report["cases"][0]["policies"]["sft"][0]
            self.assertEqual(episode["status"], "artifact_missing")
            self.assertIsNone(episode["final_canvas"])
            self.assertTrue(episode["final_canvas_artifact_missing"])
            self.assertIn("final_canvas_artifact_missing", episode["flags"])
            self.assertEqual(episode["turns"][0]["canvas"], str((turn_one / "painting.png").resolve()))
            self.assertEqual(episode["turns"][1]["code"], str((turn_two / "program.js").resolve()))
            self.assertNotEqual(episode["turns"][1]["code"], str((turn_one / "program.js").resolve()))


if __name__ == "__main__":
    unittest.main()
