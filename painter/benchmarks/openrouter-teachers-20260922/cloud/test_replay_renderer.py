"""Focused coverage for parallel, isolated provider-free replay."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_renderer


class ReplayConcurrencyTests(unittest.TestCase):
    def test_two_isolated_renders_overlap_without_model_calls(self):
        barrier = threading.Barrier(2)
        items = [
            (f"job-{n}", 1, f"program-{n}".encode(),
             {"model": "test/model", "reference_id": f"ref-{n}",
              "original_episode_sha256": "abc", "original_status": "renderer_error"})
            for n in (1, 2)
        ]

        def fake_render(**kwargs):
            barrier.wait(timeout=5)
            kwargs["output"].write_bytes(b"png")
            return {"valid": True, "elapsed_seconds": 0.1}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "replay"
            argv = ["replay_renderer.py", "--source-run-id", "source-run",
                    "--output", str(output), "--renderer-python", sys.executable,
                    "--workers", "2"]
            with patch.object(sys, "argv", argv), \
                 patch.object(replay_renderer, "public_bytes", return_value=(b"archive", {"bundle_sha256": "abc"})), \
                 patch.object(replay_renderer, "timed_out_programs", return_value=items), \
                 patch.object(replay_renderer, "render_program", side_effect=fake_render):
                self.assertEqual(replay_renderer.main(), 0)
            summary = json.loads((output / "run-summary.json").read_text())
            self.assertEqual(summary["render_workers"], 2)
            self.assertEqual(summary["status_counts"], {"valid": 2})
            self.assertEqual(summary["paid_model_calls"], 0)
            for job, _, program, _ in items:
                episode = output / "episodes" / job
                self.assertEqual((episode / "turn-01.program.js").read_bytes(), program)
                self.assertEqual(json.loads((episode / "replay.json").read_text())["result"]["valid"], True)


if __name__ == "__main__":
    unittest.main()
