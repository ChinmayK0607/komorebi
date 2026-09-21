"""Fixture-driven tests for the review-only matched comparison helper."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("make_matched_review", ROOT / "make_matched_review.py")
assert _SPEC and _SPEC.loader
make_matched_review = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = make_matched_review
_SPEC.loader.exec_module(make_matched_review)


class MakeMatchedReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.node_root = Path("/root/painter")
        self.local_root = self.root / "collected"
        self.reference_root = self.root / "references"
        for path in (
            self.local_root / "matched-evaluation/simplebaseline/sample-paint/painting.png",
            self.local_root / "matched-evaluation/candidate/sample-paint/painting.png",
            self.reference_root / "data/sample/reference.png",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-png")

    def _summary(self, label: str, canvas: str | None, valid: bool = True) -> Path:
        row = {
            "task": {
                "id": "sample-paint",
                "source_group": "sample",
                "family": "box",
            },
            "reference": "/root/painter/data/sample/reference.png",
            "canvas": canvas,
            "valid_canvas": valid,
            "finished": False,
            "operational_cutoff": True,
            "turns": [{"tokens": 11}, {"tokens": 13}],
            "generated_tokens": 24,
        }
        path = self.root / f"{label}.summary.json"
        path.write_text(json.dumps({
            "schema": "painter.matched-evaluation.v1",
            "label": label,
            "policy_version": f"{label}-policy-v1",
            "results": [row],
        }))
        return path

    def test_matching_copies_reference_and_both_canvases(self):
        baseline = self._summary(
            "simplebaseline",
            "/root/painter/matched-evaluation/simplebaseline/sample-paint/painting.png",
        )
        candidate = self._summary(
            "candidate",
            "/root/painter/matched-evaluation/candidate/sample-paint/painting.png",
        )
        output = self.root / "review"
        manifest = make_matched_review.build_review(
            baseline,
            candidate,
            output,
            node_root=self.node_root,
            local_root=self.local_root,
            reference_roots=(self.reference_root,),
        )
        sample = manifest["samples"][0]
        self.assertEqual(manifest["metadata"]["policy_version"], {
            "sft_baseline": "simplebaseline-policy-v1",
            "rl_candidate": "candidate-policy-v1",
        })
        self.assertEqual(sample["reference"]["status"], "present")
        self.assertEqual(sample["sft_baseline"]["status"], "valid")
        self.assertEqual(sample["rl_candidate"]["status"], "valid")
        self.assertEqual(sample["sft_baseline"]["turns"], 2)
        self.assertEqual(sample["rl_candidate"]["generated_tokens"], 24)
        self.assertEqual(len(list((output / "assets").glob("*.png"))), 3)
        html = (output / "index.html").read_text()
        self.assertIn("No automatic scoring or improvement claim", html)
        self.assertIn("sample", html)

    def test_missing_and_invalid_canvas_are_recorded_without_placeholder(self):
        baseline = self._summary("simplebaseline", "/root/painter/missing/baseline.png", valid=True)
        candidate = self._summary("candidate", None, valid=False)
        output = self.root / "review-missing"
        manifest = make_matched_review.build_review(
            baseline,
            candidate,
            output,
            node_root=self.node_root,
            local_root=self.local_root,
            reference_roots=(self.reference_root,),
        )
        sample = manifest["samples"][0]
        self.assertEqual(sample["sft_baseline"]["status"], "missing")
        self.assertEqual(sample["rl_candidate"]["status"], "missing")
        self.assertIsNone(sample["sft_baseline"]["canvas"]["asset"])
        self.assertIsNone(sample["rl_candidate"]["canvas"]["asset"])
        self.assertEqual(len(list((output / "assets").glob("*.png"))), 1)
        html = (output / "index.html").read_text()
        self.assertIn("No image copied", html)


if __name__ == "__main__":
    unittest.main()
