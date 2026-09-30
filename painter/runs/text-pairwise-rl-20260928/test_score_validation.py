"""Focused offline checks for held-out judge-result validation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("score_validation.py")
SPEC = importlib.util.spec_from_file_location("score_validation_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
score_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_validation)


CONSISTENT = "text-paint-sft-pairwise-consistent-v1"


class ScoreValidationTests(unittest.TestCase):
    def _valid(self, *, protocol: str = CONSISTENT, score: object = 0.75) -> dict:
        return {"protocol": protocol, "status": "scored", "score": score}

    def test_legacy_protocols_and_identical_canvas_remain_usable(self):
        for protocol in (
            "text-paint-sft-pairwise-v1",
            "text-paint-sft-pairwise-margin-v1",
            "text-paint-sft-pairwise-grounded-v1",
            CONSISTENT,
        ):
            with self.subTest(protocol=protocol):
                self.assertIsNone(
                    score_validation._judgment_error(self._valid(protocol=protocol), protocol)
                )
        self.assertIsNone(
            score_validation._judgment_error(
                {"protocol": CONSISTENT, "status": "identical_canvases", "score": 0.5},
                CONSISTENT,
            )
        )

    def test_uncertain_or_missing_score_is_not_resolved(self):
        self.assertEqual(
            score_validation._judgment_error(
                {"protocol": CONSISTENT, "status": "scored", "score": None}, CONSISTENT
            ),
            "score_missing",
        )
        self.assertEqual(
            score_validation._judgment_error(
                {"protocol": CONSISTENT, "status": "uncertain", "score": None}, CONSISTENT
            ),
            "judge_status_uncertain",
        )
        self.assertFalse(score_validation._resolved_judgment(None, CONSISTENT))

    def test_boolean_nonfinite_and_protocol_mismatch_are_rejected(self):
        self.assertEqual(
            score_validation._judgment_error(self._valid(score=True), CONSISTENT),
            "score_boolean",
        )
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertEqual(
                    score_validation._judgment_error(self._valid(score=value), CONSISTENT),
                    "score_nonfinite",
                )
        self.assertEqual(
            score_validation._judgment_error(
                self._valid(protocol="text-paint-sft-pairwise-grounded-v1"), CONSISTENT
            ),
            "protocol_mismatch",
        )

    def test_score_censors_and_preserves_uncertain_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            anchor = root / "anchor.png"
            canvas = root / "canvas.png"
            for path, content in ((brief, "paint a tree"), (anchor, b"anchor"), (canvas, b"canvas")):
                path.write_bytes(content if isinstance(content, bytes) else content.encode())
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "case-1", "family": "test", "reference": str(brief),
                "opponent": str(anchor),
            }]}))
            output = root / "out"
            original_read_traces = score_validation.read_traces
            original_trace_data = score_validation._trace_data
            original_judge = score_validation.consistent_judge_pair

            async def fake_judge(**_kwargs):
                return {"protocol": CONSISTENT, "status": "uncertain", "score": None,
                        "calls": [{"usage": {"prompt_tokens": 3}}]}

            try:
                score_validation.read_traces = lambda _root: ([object()], [], [Path("trace")])
                score_validation._trace_data = lambda _trace: (
                    {"task_id": "case-1"},
                    {"final_canvas_valid": True, "final_canvas": str(canvas)},
                )
                score_validation.consistent_judge_pair = fake_judge
                summary = asyncio.run(
                    score_validation.score(
                        manifest=manifest,
                        eval_root=root / "eval",
                        output=output,
                        key_file=root / "unused-key-file",
                        protocol=CONSISTENT,
                    )
                )
            finally:
                score_validation.read_traces = original_read_traces
                score_validation._trace_data = original_trace_data
                score_validation.consistent_judge_pair = original_judge

            self.assertEqual(summary["censored"], 1)
            self.assertEqual(summary["comparable"], 0)
            row = summary["rows"][0]
            self.assertEqual(row["reason"], "judge_status_uncertain")
            receipt = json.loads((output / "case-1.json").read_text())
            self.assertEqual(receipt["calls"][0]["usage"]["prompt_tokens"], 3)
            self.assertIsNone(row.get("score"))


if __name__ == "__main__":
    unittest.main()
