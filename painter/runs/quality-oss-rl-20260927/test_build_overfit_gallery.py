"""Keep failed matched-evaluation programs available for diagnosis."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("overfit_gallery", HERE / "build_overfit_gallery.py")
gallery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gallery)


class GalleryEvidenceTests(unittest.TestCase):
    def test_failed_turn_keeps_program_and_renderer_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout = root / "rollouts"
            cases = []
            for index in range(4):
                case_id = f"case-{index}"
                reference = root / f"{case_id}.png"
                reference.write_bytes(f"reference-{index}".encode())
                cases.append({"id": case_id, "reference": str(reference),
                              "reference_sha256": gallery.sha(reference)})
                for policy in ("sft-step512", "rl-step-2"):
                    turn = rollout / policy / case_id / "episode-1" / "turn-1"
                    turn.mkdir(parents=True)
                    (turn / "program.js").write_text("brush.polygon(7);\n")
                    (turn / "painting.json").write_text(json.dumps({
                        "valid": False, "errors": ["a is not iterable"],
                    }))
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({"cases": cases}))
            outdir = root / "review"
            report = gallery.build(manifest=manifest, rollout_root=rollout,
                                   outdir=outdir, step=2)
            attempted = report["rows"][0]["rl-step-2"]["attempted_turns"]
            self.assertEqual(len(attempted), 1)
            self.assertFalse(attempted[0]["valid"])
            self.assertEqual(attempted[0]["errors"], ["a is not iterable"])
            self.assertEqual((outdir / attempted[0]["program"]).read_text(),
                             "brush.polygon(7);\n")
            self.assertIn("turn 1 error", (outdir / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
