import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from prepare import prepare  # noqa: E402
from report import followup_audit  # noqa: E402
from validate import validate  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfixture"


class JudgeValidationTest(unittest.TestCase):
    def make_run(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        source = root / "source"
        source.mkdir()
        files = {}
        for name in ("reference", "a", "b"):
            files[name] = source / f"{name}.png"
            files[name].write_bytes(PNG + name.encode())
        selection = root / "selection.json"
        selection.write_text(json.dumps({
            "controls": [{"id": "known", "reference": str(files["reference"]),
                           "A": str(files["a"]), "B": str(files["b"]), "expected_winner": "A"}],
            "real_pairs": [{"id": "real", "reference": str(files["reference"]),
                            "A": str(files["a"]), "B": str(files["b"])}]}))
        prepare(selection, root)
        return tmp, root

    def write_labels(self, root, bad_control=False, omit=None, bad_images=False):
        private = json.loads((root / "artifacts" / "private.json").read_text())
        by_public = {public: (pair_id, pair, view) for pair_id, pair in private["pairs"].items()
                     for public, view in pair["views"].items()}
        for judge in ("judge-01", "judge-02", "judge-03"):
            manifest = json.loads((root / f"{judge}.json").read_text())
            rows = []
            for item in manifest:
                if item["id"] == omit:
                    continue
                pair_id, _, view = by_public[item["id"]]
                winner = "A" if view["orientation"] == "ab" else "B"
                if bad_control and pair_id == "known":
                    winner = "B" if winner == "A" else "A"
                reviewed = [item[x] for x in ("reference", "A", "B")]
                if bad_images and rows == []:
                    reviewed[0] = reviewed[1]
                rows.append({"id": item["id"], "winner": winner, "confidence": "high",
                             "acceptable_A": "yes" if view["orientation"] == "ab" else "no",
                             "acceptable_B": "no" if view["orientation"] == "ab" else "yes",
                             "reason": "visible comparison",
                             "observations": {"reference": "reference visible", "A": "A visible", "B": "B visible"},
                             "reviewed_images": reviewed})
            (root / "labels").mkdir(exist_ok=True)
            (root / "labels" / f"{judge}.json").write_text(json.dumps({
                "judge_model": "gpt-5.6-luna", "reasoning_effort": "low", "judgments": rows}))

    def test_reversed_views_agree_after_canonical_mapping(self):
        tmp, root = self.make_run()
        try:
            self.write_labels(root)
            result = validate(root)
            self.assertTrue(result["overall_pass"])
            self.assertEqual(result["orientation_agreement"]["agree"], 2)
            self.assertEqual(result["acceptability_agreement"]["agree"], 2)
            self.assertEqual(result["agreement_by_kind"]["controls"]["orientation"]["agree"], 1)
            self.assertEqual(result["agreement_by_kind"]["real_pairs"]["orientation"]["agree"], 1)
        finally:
            tmp.cleanup()

    def test_control_failure_is_failure_and_never_tie(self):
        tmp, root = self.make_run()
        try:
            self.write_labels(root, bad_control=True)
            result = validate(root)
            self.assertFalse(result["overall_pass"])
            self.assertFalse(result["all_controls_correct"])
            self.assertNotIn("tie", result["real_pair_vote_descriptions"])
        finally:
            tmp.cleanup()

    def test_missing_and_mismatched_labels_are_incomplete_or_failed(self):
        tmp, root = self.make_run()
        try:
            first = json.loads((root / "judge-01.json").read_text())[0]["id"]
            self.write_labels(root, omit=first)
            result = validate(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["missing"])
            self.write_labels(root, bad_images=True)
            result = validate(root)
            self.assertFalse(result["overall_pass"])
            self.assertTrue(any("reviewed_images" in error for error in result["errors"]))
        finally:
            tmp.cleanup()

    def test_wrong_separate_call_control_fails_independently(self):
        tmp, root = self.make_run()
        try:
            self.write_labels(root)
            main = validate(root)
            self.assertTrue(main["overall_pass"])
            private = json.loads((root / "artifacts" / "private.json").read_text())
            control = private["pairs"]["known"]
            views = list(control["views"].items())
            for filename, (public_id, view) in zip(
                    ("separate-calls-manifest.json", "separate-calls-reversed-manifest.json"), views):
                item = {"id": public_id, **view["files"]}
                (root / filename).write_text(json.dumps([item]))
                # Deliberately reverse the canonical expected winner for both
                # follow-up views. The main result above must remain unchanged.
                observed = "B" if view["orientation"] == "ab" else "A"
                label = {"id": public_id, "winner": observed, "confidence": "high",
                         "acceptable_A": "yes", "acceptable_B": "no", "reason": "wrong fixture",
                         "observations": {"reference": "seen", "A": "seen", "B": "seen"},
                         "reviewed_images": [view["files"][x] for x in ("reference", "A", "B")]}
                out = "separate-calls-labels.json" if "manifest.json" in filename and "reversed" not in filename else "separate-calls-reversed-labels.json"
                (root / out).write_text(json.dumps({"judge_model": "gpt-5.6-luna",
                    "reasoning_effort": "low", "judgments": [label]}))
            followup = followup_audit(root)
            self.assertFalse(followup["overall_pass"])
            self.assertEqual(followup["correct_views"], 0)
            self.assertTrue(validate(root, write=False)["overall_pass"])
            duplicate = json.loads((root / "separate-calls-manifest.json").read_text())
            (root / "separate-calls-reversed-manifest.json").write_text(json.dumps(duplicate))
            coverage = followup_audit(root)
            self.assertTrue(any("cover every private control view" in error
                                for error in coverage["errors"]))
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
