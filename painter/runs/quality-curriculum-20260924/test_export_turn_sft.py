"""Check that a failed sketch can be context without becoming an SFT target."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_turn_sft import make_rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportTurnSFTTests(unittest.TestCase):
    def test_failed_attempt_is_user_context_and_only_repair_is_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            episode_dir = evidence / "episodes" / "sample"
            episode_dir.mkdir(parents=True)
            reference = root / "reference.png"
            canvas = episode_dir / "turn-02.png"
            Image.new("RGB", (32, 32), "white").save(reference)
            Image.new("RGB", (32, 32), "orange").save(canvas)
            bad = episode_dir / "turn-01.program.js"
            good = episode_dir / "turn-02.program.js"
            bad.write_text("function setup(){createCanvas(600,600,WEBGL)}\nfunction draw(){translate(-300,-300);Brush.init();noLoop()}")
            good.write_text("function setup(){createCanvas(600,600,WEBGL)}\nfunction draw(){translate(-300,-300);background(240);noLoop()}")
            ref_hash = digest(reference)
            for number, program, valid in ((1, bad, False), (2, good, True)):
                record = {
                    "turn": number, "request_binding": {"reference_sha256": ref_hash},
                    "program": str(program.relative_to(evidence)),
                    "canvas": str(canvas.relative_to(evidence)) if valid else None,
                    "plan": "Repair the sketch", "render": {"valid": valid,
                        "receipt": {"source_sha256": digest(good), "png_sha256": digest(canvas)} if valid else None},
                    "request_render_feedback": "Brush is not defined" if number == 2 else "",
                    "render_feedback": "Brush is not defined" if number == 1 else "rendered",
                }
                (episode_dir / f"turn-{number:02d}.json").write_text(json.dumps(record))
            episode = {"reference_id": "sample", "reference_sha256": ref_hash,
                       "model": "teacher", "job_id": "sample", "turns": [{"turn": 1}, {"turn": 2}]}
            (episode_dir / "episode.json").write_text(json.dumps(episode))
            manifest = root / "refs.json"
            manifest.write_text(json.dumps({"references": [{"id": "sample", "split": "train",
                "sha256": ref_hash, "reference_path": str(reference)}]}))
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({"episodes": [{"evidence_root": str(evidence),
                "reference_id": "sample", "supervise_turns": [2], "reason": "visible repair"}]}))

            rows, _ = make_rows(annotations, manifest, include_last_attempt_program=True)
            self.assertEqual(len(rows), 1)
            user_text = "\n".join(part["text"] for part in rows[0]["messages"][1]["content"] if part["type"] == "text")
            target = rows[0]["messages"][2]["content"][0]["text"]
            self.assertIn("Brush.init()", user_text)
            self.assertIn("Brush is not defined", user_text)
            self.assertIn("background(240)", target)
            self.assertNotIn("Brush.init()", target)
            self.assertEqual(rows[0]["example_kind"], "initial_paint")

            plain, _ = make_rows(annotations, manifest)
            plain_text = "\n".join(part["text"] for part in plain[0]["messages"][1]["content"] if part["type"] == "text")
            self.assertNotIn("Brush.init()", plain_text)


if __name__ == "__main__":
    unittest.main()
