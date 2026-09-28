"""Calibration resumes with image-bound receipts and no duplicate calls."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("calibrate_subject_judge", HERE / "calibrate_subject_judge.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SubjectCalibrationTests(unittest.TestCase):
    def test_resume_uses_image_identity_and_no_second_api_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gallery = root / "gallery"
            gallery.mkdir()
            for name, data in (("scene-reference.png", b"ref"),
                               ("scene-sft.png", b"sft"), ("scene-rl.png", b"rl")):
                (gallery / name).write_bytes(data)
            hashes = {role: module.digest(gallery / name) for role, name in (
                ("reference", "scene-reference.png"), ("A", "scene-sft.png"),
                ("B", "scene-rl.png"))}
            (gallery / "report.json").write_text(json.dumps({"step": 4, "rows": [{
                "id": "scene", "sft-step512": {"final": "scene-sft.png"},
                "rl-step-4": {"final": "scene-rl.png"},
            }]}))
            existing = root / "judge.json"
            existing.write_text(json.dumps({"rows": [{"id": "scene", "score_sft": 0.0,
                                                       "judge": {"image_sha256": hashes}}]}))
            with patch.object(module, "load_key", return_value="not-a-real-key"), patch.object(
                module, "judge_pair", return_value={"score_A": 1.0}
            ) as judge:
                first = module.calibrate(gallery=gallery, existing=existing,
                                         outdir=root / "out", key_file=root / "unused")
                second = module.calibrate(gallery=gallery, existing=existing,
                                          outdir=root / "out", key_file=root / "unused")
            self.assertEqual(judge.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first["rows"][0]["subject_score_sft"], 1.0)


if __name__ == "__main__":
    unittest.main()
