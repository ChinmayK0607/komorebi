import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).with_name("build_consistent_report.py")
spec = importlib.util.spec_from_file_location("consistent_report", MODULE_PATH)
assert spec and spec.loader
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ConsistentReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.gallery_root = root / "gallery"
        (self.gallery_root / "images").mkdir(parents=True)
        (self.gallery_root / "index.html").write_text("gallery", encoding="utf-8")
        Image.new("RGB", (8, 6), (220, 40, 40)).save(self.gallery_root / "images" / "anchor.png")
        Image.new("RGB", (8, 6), (40, 60, 220)).save(self.gallery_root / "images" / "candidate.png")
        self.gallery = self.gallery_root / "index.html"
        self.anchor_sha = report._sha256(self.gallery_root / "images" / "anchor.png")
        self.candidate_sha = report._sha256(self.gallery_root / "images" / "candidate.png")
        self.pair = {
            "pair_id": "case__frozen-sft__rl-step-16", "case_id": "case", "family": "simple",
            "brief": "Paint a readable subject.", "anchor_role": "frozen SFT", "candidate_role": "RL-step-16",
            "images": {
                "frozen SFT": {"path": "images/anchor.png", "sha256": self.anchor_sha},
                "RL-step-16": {"path": "images/candidate.png", "sha256": self.candidate_sha},
            },
        }
        self.manifest = {
            "manifest_content_sha256": "manifest-hash", "pairs": [self.pair],
            "human_feedback": {"case": {"explicit_preferences": [{"preferred_role": "RL-step-16", "other_role": "frozen SFT"}]}},
        }
        self.manifest["manifest_content_sha256"] = report._manifest_digest(self.manifest)
        self.manifest_path = root / "input-manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.receipts = root / "receipts"
        self.receipts.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _receipt(self, *, consensus="initial_agreement", status="scored", winner="img_candidate", initial_conflict=False):
        assessment = {
            "assessments": {
                "img_anchor": {"brief_fidelity": "clear", "visible_evidence": "A connected red form is visible.", "weaknesses": ["Edges are simple."]},
                "img_candidate": {"brief_fidelity": "clear", "visible_evidence": "A connected blue form is visible.", "weaknesses": ["Color is simplified."]},
            },
            "reason": "The right form has clearer visible separation.", "winner_id": winner, "preference": "moderate",
        }
        view = {"assessment": assessment, "calls": [{"assessment": assessment, "elapsed_seconds": 1.25, "usage": {"total_tokens": 10}}]}
        judge = {
            "status": status, "image_ids": {"anchor": "img_anchor", "candidate": "img_candidate"},
            "winner_id": winner if status == "scored" else None, "preference": "moderate" if status == "scored" else None,
            "score": 0.666 if status == "scored" else None, "consensus": consensus, "acceptance": "two_of_two" if consensus == "initial_agreement" else "recheck_consensus",
            "initial_conflict": initial_conflict, "initial_votes": ["img_anchor", "img_candidate"] if initial_conflict else [winner, winner],
            "recheck_votes": [winner, winner] if consensus != "initial_agreement" else [], "views": [view, view],
            "recheck_views": [view, view] if consensus != "initial_agreement" else [],
        }
        return {"schema": report.RECEIPT_SCHEMA, "status": "scored" if status == "scored" else "censored", "identity": {"manifest_content_sha256": self.manifest["manifest_content_sha256"], "pair_id": self.pair["pair_id"], "image_sha256": {"frozen SFT": self.anchor_sha, "RL-step-16": self.candidate_sha}}, "judge": judge}

    def test_build_fixed_left_right_report_and_counts(self):
        receipt = self._receipt()
        receipt["judge"]["views"][0]["assessment"]["reason"] = "img_candidate is clearer than img_anchor."
        receipt["judge"]["views"][0]["assessment"]["assessments"]["img_anchor"]["visible_evidence"] = "img_anchor is pale."
        (self.receipts / "pair.json").write_text(json.dumps(receipt), encoding="utf-8")
        output = Path(self.tmp.name) / "report"
        summary = report.build(manifest_path=self.manifest_path, receipts_path=self.receipts, gallery=self.gallery, output=output)
        self.assertEqual(summary["metrics"]["accepted_2_of_2"], 1)
        self.assertEqual(summary["metrics"]["initial_valid"], 1)
        self.assertEqual(summary["metrics"]["human_selected"], 1)
        self.assertIn("LEFT · Frozen SFT", (output / "report.html").read_text(encoding="utf-8"))
        self.assertIn("RIGHT · RL step 16", (output / "report.html").read_text(encoding="utf-8"))
        text = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Judge agrees: RIGHT", text)
        self.assertIn("RIGHT painting is clearer than LEFT painting", text)
        self.assertIn("LEFT painting is pale", text)
        self.assertNotIn("img_anchor", text)

    def test_recheck_and_flattened_usage_are_reported(self):
        receipt = self._receipt(consensus="three_of_four", initial_conflict=True)
        (self.receipts / "pair.json").write_text(json.dumps(receipt), encoding="utf-8")
        summary = report.build_summary(manifest=self.manifest, receipts={self.pair["pair_id"]: receipt})
        self.assertEqual(summary["metrics"]["initial_conflicts"], 1)
        self.assertEqual(summary["metrics"]["accepted_3_or_4_of_4"], 1)
        self.assertEqual(summary["metrics"]["api_calls"], 4)
        self.assertEqual(summary["metrics"]["usage"]["total_tokens"], 40)
        self.assertEqual(summary["human_selected"][0]["preferred_role"], "RL-step-16")

    def test_receipt_binding_rejects_changed_manifest_identity(self):
        receipt = self._receipt()
        receipt["identity"]["manifest_content_sha256"] = "stale"
        (self.receipts / "pair.json").write_text(json.dumps(receipt), encoding="utf-8")
        # Binding is checked by _load_inputs, which is exercised by build().
        with self.assertRaises(ValueError):
            report.build(manifest_path=self.manifest_path, receipts_path=self.receipts, gallery=self.gallery, output=Path(self.tmp.name) / "out")

    def test_previous_receipt_is_converted_and_layout_stays_two_columns(self):
        receipt = self._receipt()
        (self.receipts / "pair.json").write_text(json.dumps(receipt), encoding="utf-8")
        previous_dir = Path(self.tmp.name) / "previous"
        previous_dir.mkdir()
        previous = {"identity": {"pair_id": self.pair["pair_id"]}, "versions": {"new": {"judge": {"status": "scored", "views": [{"label": {"winner": "A", "reason": "first"}}, {"label": {"winner": "B", "reason": "second"}}]}}}}
        (previous_dir / "previous.json").write_text(json.dumps(previous), encoding="utf-8")
        output = Path(self.tmp.name) / "report-previous"
        report.build(manifest_path=self.manifest_path, receipts_path=self.receipts, gallery=self.gallery, output=output, previous_receipts=previous_dir)
        text = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Previous judge:</b> Judge agrees: LEFT", text)
        self.assertNotIn("grid-template-columns:1fr", text)
        self.assertIn("first painting on the LEFT", text)

    def test_failed_empty_judge_cannot_count_as_vote_or_human_agreement(self):
        receipt = {"schema": report.RECEIPT_SCHEMA, "status": "censored", "identity": {"manifest_content_sha256": self.manifest["manifest_content_sha256"], "pair_id": self.pair["pair_id"], "image_sha256": {"frozen SFT": self.anchor_sha, "RL-step-16": self.candidate_sha}}, "judge": {}}
        summary = report.build_summary(manifest=self.manifest, receipts={self.pair["pair_id"]: receipt})
        self.assertEqual(summary["metrics"]["initial_valid"], 0)
        self.assertEqual(summary["metrics"]["initial_conflicts"], 0)
        self.assertEqual(summary["metrics"]["human_uncertain"], 1)
        self.assertEqual(summary["metrics"]["human_agree"], 0)


if __name__ == "__main__":
    unittest.main()
