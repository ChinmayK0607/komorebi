"""Focused checks for the offline COCO evidence summarizer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("summarize_evidence", RUN_DIR / "summarize_evidence.py")
assert SPEC is not None and SPEC.loader is not None
summarize_evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_evidence)


def _trace(case_id: str, policy: str, *, cutoff: bool = False) -> dict:
    turns = [
        {"turn": 1, "action": "paint", "valid": True, "canvas": "/node/turn-1/painting.png", "reply": "```javascript\nfunction draw(){}\n```", "no_change": False},
        {"turn": 2, "action": "paint", "valid": True, "reply": "```javascript\nfunction draw(){}\n```", "no_change": True},
        {"turn": 3, "action": "paint", "valid": False, "error": "renderer failed", "reply": "A short plan without code."},
        {"turn": 4, "action": "finish", "valid": True, "reply": "FINISHED"},
    ]
    info = {
        "task_id": case_id,
        "family": "box",
        "turns": turns,
        "finished": not cutoff,
        "final_canvas_valid": not cutoff,
        "final_canvas": "/node/turn-1/painting.png" if not cutoff else None,
        "operational_cutoff": cutoff,
    }
    if cutoff:
        info["cutoff_reason"] = "context_exhausted"
    calls = [
        {"finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 3}, "time": {"start": 1790000001, "end": 1790000002}},
        {"usage": {"prompt_tokens": 20, "completion_tokens": 4}, "time": {"start": 1790000003, "end": 1790000004}},
    ]
    return {
        "id": f"trace-{policy}-{case_id}",
        "task": {"data": {"task_id": case_id, "source_group": case_id, "family": "box", "policy_label": policy}},
        "info": info,
        "calls": calls,
        "timing": {"start": 1790000000, "finalize": {"end": 1790000010}},
        "ok": True,
    }


class SummarizeEvidenceTests(unittest.TestCase):
    def test_only_complete_javascript_fences_count_as_code(self):
        self.assertTrue(summarize_evidence._has_code({"reply": "```javascript\nfunction draw(){}\n```"}))
        self.assertFalse(summarize_evidence._has_code({"reply": "```html\n<div>diagnostic</div>\n```"}))
        self.assertFalse(summarize_evidence._has_code({"reply": "No complete JavaScript code block."}))

    def test_episode_and_policy_counts_keep_missing_finish_reason_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "evidence"
            (root / "eval/base/job").mkdir(parents=True)
            (root / "eval/sft/job").mkdir(parents=True)
            (root / "eval/base/job/traces.jsonl").write_text(json.dumps({"traces": [_trace("case-1", "base")]}) + "\n")
            (root / "eval/sft/job/traces.jsonl").write_text(json.dumps({"traces": [_trace("case-1", "sft", cutoff=True)]}) + "\n")
            (root / "gpu.csv").write_text(
                "timestamp,index,utilization.gpu [%],memory.used [MiB]\n"
                "1790000005,0,95 %,100 MiB\n"
                "1790000006,0,85 %,100 MiB\n"
                "1790000100,0,100 %,100 MiB\n"
            )

            result = summarize_evidence.summarize(root)
            base = result["policies"]["base"]
            sft = result["policies"]["sft"]
            self.assertEqual(base["episodes"], 1)
            self.assertEqual(base["statuses"], {"complete": 1})
            self.assertEqual(base["episodes_with_turn_errors"], 1)
            self.assertEqual(base["valid_final_canvases"], 1)
            self.assertEqual(base["paint_attempts"], 3)
            self.assertEqual(base["valid_attempts"], 2)
            self.assertEqual(base["retained_valid_canvases"], 1)
            self.assertEqual(base["valid_canvas_turns"], 1)
            self.assertEqual(base["no_code_count"], 1)
            self.assertEqual(base["unchanged_valid_count"], 1)
            self.assertEqual(base["finish_reason_counts"], {"missing": 1, "stop": 1})
            self.assertEqual(base["tokens"]["total_tokens"], 37)
            self.assertEqual(sft["cutoff_reasons"], {"context_exhausted": 1})
            self.assertEqual(result["gpu"]["sample_count"], 3)
            self.assertAlmostEqual(result["gpu"]["mean_utilization_percent"], 280 / 3)
            self.assertEqual(result["gpu"]["fraction_at_least_90"], 2 / 3)
            self.assertEqual(result["gpu"]["selection"], "all_parseable_samples")
            self.assertTrue(any("aesthetic" in note for note in result["notes"]))

    def test_headerless_nvidia_csv_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "gpu-telemetry.csv").write_text(
                "2026/09/21 19:44:24.868, 0, 99, 94000, 250.00\n"
                "2026/09/21 19:44:34.868, 0, 89, 94000, 250.00\n"
            )
            summary = summarize_evidence._gpu_summary(root, (None, None))
            self.assertEqual(summary["sample_count"], 2)
            self.assertAlmostEqual(summary["mean_utilization_percent"], 94.0)
            self.assertEqual(summary["at_least_90_count"], 1)
            self.assertEqual(summary["fraction_at_least_90"], 0.5)


if __name__ == "__main__":
    unittest.main()
