from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from painter.renderer_latency_profile import _has_no_loop, profile_receipts


class RendererLatencyProfileTests(unittest.TestCase):
    def _receipt(self, root: Path, name: str, *, elapsed: float, valid: bool, timeout: bool) -> None:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "program.js").write_text(
            "function setup(){createCanvas(600,600);}\n"
            + ("function draw(){background('#fff'); noLoop();}\n" if name != "missing-loop" else "function draw(){background('#fff');}\n")
        )
        (folder / "painting.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": elapsed,
                    "valid": valid,
                    "timed_out": timeout,
                    "error_code": "render_timeout" if timeout else None,
                }
            )
        )

    def test_profiles_latency_tail_and_timeout_causes_without_launching_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._receipt(root, "fast", elapsed=2.0, valid=True, timeout=False)
            self._receipt(root, "slow-valid", elapsed=130.0, valid=True, timeout=False)
            self._receipt(root, "missing-loop", elapsed=180.0, valid=False, timeout=True)
            self._receipt(root, "bounded-loop", elapsed=180.0, valid=False, timeout=True)

            report = profile_receipts(root)

        self.assertEqual(report["receipt_count"], 4)
        self.assertEqual(report["receipts_with_elapsed_seconds"], 4)
        self.assertEqual(report["elapsed_seconds"]["p50"], 155.0)
        self.assertEqual(report["elapsed_seconds"]["p95"], 180.0)
        self.assertEqual(report["elapsed_seconds"]["max"], 180.0)
        self.assertEqual(report["causes"]["render_timeout_count"], 2)
        self.assertEqual(report["causes"]["render_timeout_without_no_loop_count"], 1)
        self.assertEqual(report["causes"]["render_timeout_with_no_loop_count"], 1)
        self.assertEqual(report["causes"]["valid_over_tail_threshold_count"], 1)
        self.assertEqual(report["tail"]["receipts_at_or_above_threshold_count"], 3)
        self.assertEqual(report["tail"]["at_or_above_threshold_without_no_loop_count"], 1)
        self.assertEqual(report["tail"]["at_or_above_threshold_with_no_loop_count"], 2)
        self.assertEqual(
            report["cap_tradeoffs"],
            [
                {
                    "cap_seconds": 60.0,
                    "receipts_at_or_above_cap": 3,
                    "historically_valid_censored": 1,
                    "historically_invalid_shortened": 2,
                },
                {
                    "cap_seconds": 90.0,
                    "receipts_at_or_above_cap": 3,
                    "historically_valid_censored": 1,
                    "historically_invalid_shortened": 2,
                },
                {
                    "cap_seconds": 120.0,
                    "receipts_at_or_above_cap": 3,
                    "historically_valid_censored": 1,
                    "historically_invalid_shortened": 2,
                },
            ],
        )

    def test_invalid_tail_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                profile_receipts(Path(temp), tail_threshold_seconds=0)

    def test_no_loop_classifier_ignores_comments_and_strings(self):
        self.assertFalse(_has_no_loop("function draw(){/* noLoop(); */}"))
        self.assertFalse(_has_no_loop("function draw(){const text='noLoop()';}"))
        self.assertTrue(_has_no_loop("function draw(){noLoop();}"))


if __name__ == "__main__":
    unittest.main()
