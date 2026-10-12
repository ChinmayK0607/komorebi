"""Regression test for the staged terminal-action credit contract."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from painter.test_native_painting_rl import _load_rl_module


class FinalActionGateTests(unittest.TestCase):
    def test_invalid_last_paint_gets_zero_even_with_retained_valid_canvas(self) -> None:
        module = _load_rl_module()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n", encoding="utf-8")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            data = module.NativePaintingRLData(
                root=str(root), task_id="staged-final-action", source_group="staged-final-action",
                family="simple", reference=str(brief), reference_sha256=digest(brief),
                reference_kind="text", opponent=str(anchor), opponent_sha256=digest(anchor),
                output_root="rl-rollouts", mode="fresh",
            )
            trace = type("Trace", (), {"info": {
                "final_canvas": str(canvas), "final_canvas_valid": True,
                "turns": [{"turn": 2, "action": "paint", "valid": False, "failure_kind": "program"}],
            }})()
            with patch.dict("os.environ", {"PAINTER_REQUIRE_VALID_FINAL_ACTION": "1"}):
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.0)
            self.assertEqual(trace.info["reward_status"], "model_invalid_final_action")
            self.assertTrue(trace.info["final_canvas_valid"])
            self.assertEqual(trace.info["final_action_validation"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
