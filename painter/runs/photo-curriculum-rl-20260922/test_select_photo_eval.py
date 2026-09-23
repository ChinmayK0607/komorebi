"""Tests for the bounded, family-diverse photo eval subset."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("select_photo_eval", ROOT / "select_photo_eval.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectPhotoEvalTests(unittest.TestCase):
    def test_reviewed_manifest_selects_first_twelve_distinct_families(self):
        # Resolve from the repository's painter/runs directory rather than
        # assuming the test is launched from the workspace root.
        source = ROOT.parents[0] / "photo-curriculum-sft-20260922" / "eval-prep" / "eval-manifest.json"
        raw = json.loads(source.read_text())
        selected = MODULE.select_cases(raw["cases"])
        self.assertEqual(len(selected), 12)
        self.assertEqual([row["selection_rank"] for row in selected], list(range(1, 13)))
        self.assertEqual(len({row["primary_family"] for row in selected}), 12)

    def test_selection_is_independent_of_input_order_and_has_bound(self):
        rows = []
        for index in range(28):
            rows.append({
                "id": f"case-{index:02d}",
                "primary_family": f"family-{index:02d}",
                "selection_rank": index + 1,
            })
        a = MODULE.select_cases(rows, count=12)
        b = MODULE.select_cases(list(reversed(rows)), count=12)
        self.assertEqual([row["id"] for row in a], [row["id"] for row in b])
        with self.assertRaisesRegex(ValueError, "source cases"):
            MODULE.select_cases(rows[:27])

    def test_manifest_records_source_hash_and_selection_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.json"
            source.write_text(json.dumps({
                "cases": [
                    {"id": f"case-{index:02d}", "primary_family": f"family-{index:02d}", "selection_rank": index + 1}
                    for index in range(28)
                ]
            }))
            output = MODULE.build_manifest(source)
            self.assertEqual(output["selection"]["count"], 12)
            self.assertEqual(output["selection"]["source_case_count"], 28)
            self.assertIn("lowest selection_rank", output["selection"]["rule"])
            self.assertEqual(len(output["cases"]), 12)

    def test_materialized_manifest_uses_absolute_reference_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            references = root / "references"
            references.mkdir()
            rows = []
            for index in range(28):
                image = references / f"{index:02d}.png"
                image.write_bytes(str(index).encode())
                rows.append({
                    "id": f"case-{index:02d}", "primary_family": f"family-{index:02d}",
                    "selection_rank": index + 1, "reference": f"references/{image.name}",
                })
            source = root / "source.json"
            source.write_text(json.dumps({"cases": rows}))
            output = MODULE.build_manifest(source)
            self.assertTrue(all(Path(row["reference"]).is_absolute() for row in output["cases"]))
            self.assertTrue(all(Path(row["reference"]).is_file() for row in output["cases"]))


if __name__ == "__main__":
    unittest.main()
