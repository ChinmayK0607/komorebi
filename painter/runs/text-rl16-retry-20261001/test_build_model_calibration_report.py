from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = _load("model_calibration_report", HERE / "build_model_calibration_report.py")
runner = _load("model_calibration_runner", HERE / "calibrate_frozen_model_pair_repeats.py")
from painter import text_pairwise_consistent_side_judge_v2_model as transport  # noqa: E402


class ModelCalibrationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.anchor = self.root / "anchor.png"
        self.candidate = self.root / "candidate.png"
        Image.new("RGB", (12, 8), "#d35f4f").save(self.anchor)
        Image.new("RGB", (8, 12), "#4f75d3").save(self.candidate)
        self.pairs_file = self.root / "pairs.json"
        self.pairs_file.write_text(json.dumps({
            "selection_note": "Two selected diagnostic pairs; no gold labels.",
            "pairs": [
                {"pair_id": "cal-1", "split": "calibration", "brief": "Paint a connected subject.",
                 "anchor_path": str(self.anchor), "candidate_path": str(self.candidate)},
                {"pair_id": "conf-1", "split": "confirmation", "brief": "Paint another connected subject.",
                 "anchor_path": str(self.anchor), "candidate_path": str(self.candidate)},
            ],
        }), encoding="utf-8")
        self.evidence = self.root / "evidence"
        runner.prepare_manifest(
            pairs_file=self.pairs_file, output=self.evidence, repeat_count=2,
            baseline_model="xiaomi/mimo-v2.6-flash",
            candidate_model="xiaomi/mimo-v2.6-pro",
        )
        self.manifest_path = self.evidence / "input-manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.summary_path = self.evidence / "summary.json"
        self._write_receipts_and_summary()

    def tearDown(self):
        self.temp.cleanup()

    def _result(self, pair: dict, model: str, *, swapped: bool, uncertain: bool) -> dict:
        ids = pair["image_ids"]
        left_id, right_id = (ids["candidate"], ids["anchor"]) if swapped else (ids["anchor"], ids["candidate"])
        assessment = {
            "assessments": {
                ids["anchor"]: {
                    "brief_fidelity": "clear", "visible_evidence": "SFT subject is readable.",
                    "weaknesses": ["SFT weakness is soft boundary detail."],
                },
                ids["candidate"]: {
                    "brief_fidelity": "partial", "visible_evidence": "RL subject has a visible connected form.",
                    "weaknesses": ["RL weakness is a faint opening."],
                },
            },
            "reason": "The LEFT panel preserves the connected form more clearly.",
            "winner_id": ids["anchor"], "preference": "slight",
        }
        call = {
            "assessment": assessment,
            "image_ids": {"A": left_id, "B": right_id},
            "requested_model": model, "returned_model": model,
            "request_config_sha256": transport._model_config_sha256(model),
            "usage": {"total_tokens": 12}, "elapsed_seconds": 0.1,
        }
        result = {
            "protocol": self.manifest["protocol"], "format_version": self.manifest["format_version"],
            "source_format_sha256": self.manifest["source_format_sha256"],
            "rubric_source_format_sha256": self.manifest["rubric_source_format_sha256"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "requested_model": model, "model_config_sha256": transport._model_config_sha256(model),
            "brief_sha256": pair["brief_sha256"], "image_ids": pair["image_ids"],
            "image_sha256": pair["image_sha256"], "returned_models": [model],
            "views": [{"assessment": assessment, "calls": [call]}],
            "recheck_views": [], "initial_votes": [ids["anchor"], ids["anchor"]],
            "recheck_votes": [],
        }
        if uncertain:
            result.update({"status": "uncertain", "winner_id": None, "preference": None,
                           "score": None, "consensus": "no_three_of_four"})
        else:
            result.update({"status": "scored", "winner_id": ids["anchor"],
                           "preference": "slight", "score": 1 / 3,
                           "consensus": "initial_agreement"})
        return result

    def _write_receipts_and_summary(self):
        for pair in self.manifest["pairs"]:
            for method in ("baseline", "candidate"):
                model = self.manifest["models"][method]
                for repeat in (1, 2):
                    uncertain = method == "candidate" and repeat == 2
                    result = self._result(pair, model, swapped=(repeat == 2), uncertain=uncertain)
                    receipt = {
                        "schema": reporter.RECEIPT_SCHEMA,
                        "identity": reporter._expected_identity(self.manifest, pair, method, repeat),
                        "status": "censored" if uncertain else "scored",
                        "reason": "no_three_of_four" if uncertain else None,
                        "judge": result,
                    }
                    path = self.evidence / "receipts" / method / f"{pair['pair_id']}--repeat-{repeat}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps({
            "schema": reporter.SUMMARY_SCHEMA,
            "manifest_content_sha256": self.manifest["manifest_content_sha256"],
            "pair_count": len(self.manifest["pairs"]), "repeat_count": self.manifest["repeat_count"],
            "models": self.manifest["models"], "protocol": self.manifest["protocol"],
            "format_version": self.manifest["format_version"],
            "source_format_sha256": self.manifest["source_format_sha256"],
            "prompt_sha256": self.manifest["prompt_sha256"], "gold_labels": False,
        }, indent=2), encoding="utf-8")

    def test_report_is_portable_pairwise_and_preserves_orientation(self):
        output = self.root / "report"
        report = reporter.build_report(
            manifest_path=self.manifest_path, summary_path=self.summary_path, output=output,
        )
        text = report.read_text(encoding="utf-8")
        self.assertIn("MiMo Flash", text)
        self.assertIn("MiMo Pro", text)
        self.assertIn("same frozen painting brief", text)
        self.assertIn("0 means SFT wins clearly", text)
        self.assertIn("Relative RL score", text)
        self.assertIn("Uncertain — no_three_of_four; no score", text)
        self.assertIn("SFT weakness is soft boundary detail.", text)
        self.assertIn("RL weakness is a faint opening.", text)
        self.assertIn("LEFT = SFT, RIGHT = RL", text)
        self.assertIn("LEFT = RL, RIGHT = SFT", text)
        self.assertIn("grid-template-columns:repeat(2,minmax(0,1fr))", text)
        self.assertEqual(text.count("SFT · LEFT"), 2)
        self.assertEqual(text.count("RL · RIGHT"), 2)
        self.assertEqual((output / "images" / "cal-1-sft.png").read_bytes(), self.anchor.read_bytes())
        self.assertEqual((output / "images" / "cal-1-rl.png").read_bytes(), self.candidate.read_bytes())
        self.assertEqual(len(list((output / "receipts").rglob("*.json"))), 8)

    def test_source_hash_is_verified_before_copying_images(self):
        self.candidate.write_bytes(b"changed after manifest")
        output = self.root / "rejected-report"
        with self.assertRaisesRegex(ValueError, "candidate image hash mismatch"):
            reporter.build_report(
                manifest_path=self.manifest_path, summary_path=self.summary_path, output=output,
            )
        self.assertFalse((output / "images").exists())


if __name__ == "__main__":
    unittest.main()
