from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_cpu_replay import build_replay_manifest


def code(comment: str) -> str:
    return f"```javascript\nfunction setup() {{ createCanvas(600, 600, WEBGL); }}\nfunction draw() {{ translate(-300, -300); background('{comment}'); noLoop(); }}\n```"


def trace_row(task_id: str, turns: list[dict], final_canvas: str | None, *, valid_final: bool = True) -> dict:
    return {
        "id": f"outer-{task_id}",
        "traces": [{
            "info": {
                "task_id": task_id,
                "source_group": task_id,
                "family": "simple",
                "protocol_version": "native-painting-eval-v3",
                "final_canvas": final_canvas,
                "final_canvas_valid": valid_final,
                "turns": turns,
            }
        }],
    }


class CpuReplayPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.renderer = self.root / "renderer.py"
        self.renderer.write_text("# pinned fixture renderer\n")
        (self.root / "assets").mkdir()
        (self.root / "assets" / "p5.min.js").write_text("p5 fixture")
        (self.root / "assets" / "p5.brush.js").write_text("brush fixture")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_resolves_no_change_final_to_last_code_bearing_canvas(self) -> None:
        final = "/node/evaluation/case-a/turn-2/painting.png"
        h1 = "1" * 64
        h2 = "2" * 64
        traces = self.root / "traces.jsonl"
        traces.write_text(json.dumps(trace_row("case-a", [
            {"turn": 1, "action": "paint", "valid": True, "canvas": "/node/evaluation/case-a/turn-1/painting.png", "canvas_sha256": h1, "reply": code("red")},
            {"turn": 2, "action": "paint", "valid": True, "canvas": final, "canvas_sha256": h2, "reply": code("green")},
            {"turn": 3, "action": "paint", "valid": True, "canvas": final, "no_change": True, "reply": "FINISHED"},
        ], final)) + "\n")

        output = self.root / "replay-inputs"
        manifest = build_replay_manifest(traces, output, self.renderer)

        self.assertEqual(manifest["summary"], {"trace_rows": 1, "replay_cases": 1, "skipped_rows": 0})
        case = manifest["cases"][0]
        self.assertEqual(case["target_png_sha256"], h2)
        self.assertEqual(case["source_turn"], 2)
        source = (output / case["source_file"]).read_text()
        self.assertIn("background('green')", source)
        self.assertNotIn("/node/", json.dumps(manifest))

        repeated = build_replay_manifest(traces, output, self.renderer)
        self.assertEqual(repeated["cases"], manifest["cases"])

        traces.write_text(json.dumps(trace_row("case-a", [
            {"turn": 1, "action": "paint", "valid": True, "canvas": final, "canvas_sha256": h2, "reply": code("purple")},
        ], final)) + "\n")
        with self.assertRaisesRegex(ValueError, "existing program differs"):
            build_replay_manifest(traces, output, self.renderer)

    def test_failed_last_turn_uses_prior_valid_source_and_recorded_target(self) -> None:
        final = "/node/evaluation/case-b/turn-1/painting.png"
        target = "b" * 64
        traces = self.root / "traces.jsonl"
        traces.write_text(json.dumps(trace_row("case-b", [
            {"turn": 1, "action": "paint", "valid": True, "canvas": final, "canvas_sha256": target, "reply": code("blue")},
            {"turn": 2, "action": "paint", "valid": False, "canvas": final, "reply": code("bad")},
        ], final)) + "\n")

        manifest = build_replay_manifest(traces, self.root / "replay-inputs", self.renderer)

        case = manifest["cases"][0]
        self.assertEqual(case["source_turn"], 1)
        self.assertEqual(case["target_png_sha256"], target)
        self.assertIn("background('blue')", (self.root / "replay-inputs" / case["source_file"]).read_text())

    def test_missing_final_canvas_is_skipped_without_program_or_target(self) -> None:
        traces = self.root / "traces.jsonl"
        traces.write_text(json.dumps(trace_row("case-c", [
            {"turn": 1, "action": "paint", "valid": False, "canvas": None, "reply": code("bad")},
        ], None, valid_final=False)) + "\n")

        manifest = build_replay_manifest(traces, self.root / "replay-inputs", self.renderer)

        self.assertEqual(manifest["summary"]["replay_cases"], 0)
        self.assertEqual(manifest["skipped"][0]["reason"], "no_recorded_valid_final_canvas")
        self.assertFalse((self.root / "replay-inputs" / "programs" / "case-c.js").exists())


if __name__ == "__main__":
    unittest.main()
