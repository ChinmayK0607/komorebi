from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("calibration_report", HERE / "build_calibration_report.py")
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class CalibrationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.anchor = self.images / "anchor.png"
        self.candidate = self.images / "candidate.png"
        Image.new("RGB", (8, 6), "#d35f4f").save(self.anchor)
        Image.new("RGB", (8, 6), "#4f75d3").save(self.candidate)
        self.brief = "Paint one connected subject with a readable opening."
        self.pair_id = "fixture-pair"
        self.anchor_sha = report.sha256_file(self.anchor)
        self.candidate_sha = report.sha256_file(self.candidate)
        self.anchor_id = "img_" + self.anchor_sha[:20]
        self.candidate_id = "img_" + self.candidate_sha[:20]
        self.manifest = {
            "schema": "text-rl16-frozen-pair-repeat-input-v1",
            "protocols": {"old": "old-protocol", "draft": "draft-protocol"},
            "models": {"old": "model", "draft": "model"},
            "prompts": {"old_sha256": "old-prompt", "draft_sha256": "draft-prompt"},
            "format_versions": {"old": "old-format", "draft": "draft-format"},
            "source_format_sha256": {"old": "old-source", "draft": "draft-source"},
            "repeat_count": 2,
            "selection_note": "Selected diagnostic panel; no blind human gold labels.",
            "scope_note": "Calibration/confirmation are selection notes, not accuracy estimates.",
            "pairs": [{
                "pair_id": self.pair_id, "split": "confirmation", "brief": self.brief,
                "brief_path": None, "brief_sha256": hashlib.sha256(self.brief.encode()).hexdigest(),
                "anchor_path": str(self.anchor), "candidate_path": str(self.candidate),
                "image_ids": {"anchor": self.anchor_id, "candidate": self.candidate_id},
                "image_sha256": {"anchor": self.anchor_sha, "candidate": self.candidate_sha},
            }],
        }
        self.manifest["manifest_content_sha256"] = report._manifest_digest(self.manifest)
        self.manifest_path = self.root / "input-manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest, sort_keys=True), encoding="utf-8")
        self.receipts = self.root / "receipts"
        self.summary_path = self.root / "summary.json"
        self._write_receipts()
        self.summary = {
            "schema": "text-rl16-frozen-pair-repeat-summary-v1",
            "manifest_content_sha256": self.manifest["manifest_content_sha256"],
            "pair_count": 1, "repeat_count": 2, "rows": [],
        }
        self.summary_path.write_text(json.dumps(self.summary, sort_keys=True), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _identity(self, method: str, repeat: int) -> dict:
        return report._expected_identity(self.manifest, self.manifest["pairs"][0], method, repeat)

    def _write_receipts(self):
        assessment = {
            "assessments": {
                self.anchor_id: {
                    "brief_fidelity": "clear", "visible_evidence": "SFT has a connected outline.",
                    "weaknesses": ["The opening is faint."],
                },
                self.candidate_id: {
                    "brief_fidelity": "partial", "visible_evidence": "RL has a visible opening.",
                    "weaknesses": ["A small part is detached."],
                },
            },
            "reason": "RL shows the requested opening more clearly.",
            "winner_id": self.candidate_id, "preference": "slight",
        }
        for method in report.METHODS:
            for repeat in (1, 2):
                result = {
                    "protocol": self.manifest["protocols"][method],
                    "format_version": self.manifest["format_versions"][method],
                    "brief_sha256": self.manifest["pairs"][0]["brief_sha256"],
                    "image_ids": self.manifest["pairs"][0]["image_ids"],
                    "image_sha256": self.manifest["pairs"][0]["image_sha256"],
                    "status": "scored" if not (method == "old" and repeat == 2) else "uncertain",
                    "winner_id": self.candidate_id if not (method == "old" and repeat == 2) else None,
                    "preference": "slight" if not (method == "old" and repeat == 2) else None,
                    "score": 2 / 3 if not (method == "old" and repeat == 2) else None,
                    "initial_votes": [self.candidate_id, self.candidate_id],
                    "recheck_votes": [], "views": [{"assessment": assessment}],
                    "recheck_views": [],
                }
                if method == "old" and repeat == 2:
                    result["views"] = []
                receipt = {
                    "schema": "text-rl16-frozen-pair-repeat-receipt-v1",
                    "identity": self._identity(method, repeat),
                    "status": "censored" if result["status"] == "uncertain" else "scored",
                    "reason": "no_three_of_four" if result["status"] == "uncertain" else None,
                    "judge": result,
                }
                path = self.receipts / method / f"{self.pair_id}--repeat-{repeat}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    def test_report_preserves_identity_and_presents_both_methods(self):
        output = self.root / "report"
        path = report.build_report(manifest_path=self.manifest_path, summary_path=self.summary_path, output=output)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(path.is_file())
        self.assertIn("Previous judge", text)
        self.assertIn("Revised judge", text)
        self.assertIn("fresh repeated calls", text)
        self.assertIn("0 means SFT wins clearly, 0.5 is a tie, and 1 means RL wins clearly", text)
        self.assertIn("not an absolute painting-quality grade", text)
        self.assertIn("SFT · LEFT", text)
        self.assertIn("RL rollout · RIGHT", text)
        self.assertIn("Paint one connected subject", text)
        self.assertIn("A small part is detached.", text)
        self.assertIn("Uncertain — no accepted consensus; no score", text)
        self.assertNotIn("aggregate reward improvement claims", text)
        self.assertIn("grid-template-columns:repeat(2,minmax(0,1fr))", text)
        self.assertIn("@media(max-width:800px){.methods,.assessments", text)
        self.assertEqual((output / "images" / f"{self.pair_id}-sft.png").read_bytes(), self.anchor.read_bytes())
        raw = output / "receipts" / "draft" / f"{self.pair_id}--repeat-1.json"
        self.assertEqual(raw.read_bytes(), (self.receipts / "draft" / raw.name).read_bytes())
        self.assertIn(self.anchor_sha, text)
        self.assertIn(self.candidate_sha, text)

    def test_receipt_identity_mismatch_is_rejected(self):
        path = self.receipts / "draft" / f"{self.pair_id}--repeat-1.json"
        value = json.loads(path.read_text())
        value["identity"]["image_ids"]["candidate"] = "img_wrong"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt identity mismatch"):
            report.build_report(manifest_path=self.manifest_path, summary_path=self.summary_path,
                                output=self.root / "bad-report")


if __name__ == "__main__":
    unittest.main()
