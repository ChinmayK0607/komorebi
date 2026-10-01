from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_failed_eval_replay import build_manifest


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FailedEvalReplayPreparationTests(unittest.TestCase):
    def test_packages_first_turn_programs_without_target_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evaluation = root / "evaluation-rollouts"
            renderer = root / "runtime" / "renderer.py"
            renderer.parent.mkdir()
            renderer.write_text("# renderer fixture\n")
            assets_root = renderer.parent / "assets"
            assets_root.mkdir()
            (assets_root / "p5.brush.js").write_text("brush fixture\n")
            (assets_root / "p5.min.js").write_text("p5 fixture\n")
            renderer_sha = digest(renderer.read_bytes())
            assets = {
                name: digest((assets_root / name).read_bytes())
                for name in ("p5.brush.js", "p5.min.js")
            }
            trace = root / "traces.jsonl"
            trace.write_text("trace evidence\n")
            for task_id in ("case-a", "case-b"):
                turn = evaluation / "policy" / task_id / "episode" / "turn-1"
                turn.mkdir(parents=True)
                source = f"function setup(){{/* {task_id} */}}\n".encode()
                (turn / "program.js").write_bytes(source)
                (turn / "painting.json").write_text(json.dumps({
                    "valid": False,
                    "error_code": "worker_failure",
                    "seed": 0,
                    "source_sha256": digest(source),
                    "renderer_sha256": renderer_sha,
                    "assets": assets,
                    "requested_angle_backend": "swiftshader",
                    "source": "/private/node/program.js",
                }))

            output = root / "inputs"
            manifest = build_manifest(
                evaluation_root=evaluation,
                output_dir=output,
                trace=trace,
                renderer=renderer,
            )

            self.assertEqual(manifest["schema"], "text-paint-first-turn-recovery-inputs-v1")
            self.assertEqual(manifest["replay_mode"], "first_turn_recovery")
            self.assertEqual(manifest["summary"]["replay_cases"], 2)
            self.assertTrue(all("target_png_sha256" not in case for case in manifest["cases"]))
            self.assertEqual(manifest["renderer"]["renderer_sha256"], renderer_sha)
            self.assertNotIn("/private/node", (output / "receipts/case-a.json").read_text())
            self.assertEqual(
                (output / "programs/case-a.js").read_bytes(),
                b"function setup(){/* case-a */}\n",
            )

    def test_rejects_saved_program_hash_mismatch_before_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            turn = root / "evaluation" / "policy" / "case-a" / "episode" / "turn-1"
            turn.mkdir(parents=True)
            (turn / "program.js").write_text("function setup(){}\n")
            (turn / "painting.json").write_text(json.dumps({
                "valid": False,
                "error_code": "worker_failure",
                "seed": 0,
                "source_sha256": "0" * 64,
                "renderer_sha256": "1" * 64,
                "assets": {"p5.brush.js": "2" * 64},
            }))
            with self.assertRaisesRegex(ValueError, "saved program hash mismatch"):
                build_manifest(evaluation_root=root / "evaluation", output_dir=root / "inputs")


if __name__ == "__main__":
    unittest.main()
