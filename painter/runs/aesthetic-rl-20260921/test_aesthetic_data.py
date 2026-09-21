"""Pure manifest checks for the aesthetic data preparation step."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
BASE_INPUTS = ROOT.parent / "revision-rl-20260918" / "inputs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AestheticDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((ROOT / "aesthetic-states.json").read_text())
        cls.runtime = json.loads((ROOT / "rl-states.json").read_text())
        cls.inputs = json.loads((ROOT.parent / "revision-rl-20260918" / "inputs" / "rl-states.json").read_text())
        cls.heldout = json.loads((ROOT.parent / "revision-rl-20260918" / "inputs" / "evaluation-tasks.json").read_text())

    def test_state_count_family_balance_and_disjointness(self) -> None:
        rows = self.manifest["training_states"]
        self.assertEqual(len(self.runtime), 24)
        self.assertEqual(self.manifest["counts"]["anchors"], 41)
        self.assertEqual(self.manifest["counts"]["archived_teacher_anchors"], 2)
        self.assertEqual(self.manifest["counts"]["archived_teacher_states"], 2)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["source_group"] for row in rows}), 24)
        self.assertEqual(sorted({row["family"] for row in rows}), sorted({row["family"] for row in self.inputs}))
        self.assertEqual({row["family"] for row in rows}, {row["family"] for row in self.inputs})
        heldout = {row["source_group"] for row in self.heldout}
        self.assertFalse(heldout & {row["source_group"] for row in rows})

    def test_runtime_loader_shape_uses_relative_anchor_strings(self) -> None:
        for row in self.runtime:
            self.assertIsInstance(row.get("anchors"), list)
            self.assertGreaterEqual(len(row["anchors"]), 1)
            self.assertTrue(all(isinstance(anchor, str) for anchor in row["anchors"]))
            self.assertTrue(all(not Path(anchor).is_absolute() for anchor in row["anchors"]))

    def test_anchor_pool_is_same_reference_and_has_no_fake_reference(self) -> None:
        for row in self.manifest["training_states"]:
            # Overlay mode relies on the already bootstrapped historical
            # rl-states tree; full bundles flatten it at the same root.
            reference = ROOT / row["reference"]
            if not reference.is_file():
                reference = BASE_INPUTS / row["reference"]
            self.assertTrue(reference.is_file(), row["id"])
            reference_hash = sha(reference)
            self.assertGreaterEqual(len(row["anchors"]), 1, row["id"])
            self.assertLessEqual(len(row["anchors"]), 3, row["id"])
            for anchor in row["anchors"]:
                canvas = ROOT / anchor["canvas"]
                program = ROOT / anchor["program"]
                self.assertTrue(canvas.is_file(), anchor["id"])
                self.assertTrue(program.is_file(), anchor["id"])
                self.assertEqual(anchor["canvas_sha256"], sha(canvas))
                self.assertEqual(anchor["program_sha256"], sha(program))
                self.assertNotEqual(anchor["canvas_sha256"], reference_hash, anchor["id"])
                if anchor.get("source_kind") == "archived_teacher_render":
                    self.assertIn("render_receipt_artifact", anchor)
                    self.assertIn("selection_reason", anchor)
                    self.assertEqual(anchor["verified_source_sha256"], anchor["program_sha256"])
                    self.assertEqual(anchor["verified_png_sha256"], anchor["canvas_sha256"])
                    self.assertTrue(anchor["renderer_sha256"])

    def test_manifest_paths_are_portable(self) -> None:
        for row in self.manifest["training_states"]:
            for key in ("reference", "canvas", "program", "initial_canvas", "initial_program"):
                value = row[key]
                self.assertFalse(Path(value).is_absolute(), value)
                self.assertNotIn("..", Path(value).parts)
            for anchor in row["anchors"]:
                for key in ("canvas", "program", "source_artifact", "source_program_artifact"):
                    value = anchor[key]
                    self.assertFalse(Path(value).is_absolute(), value)
                    self.assertNotIn("..", Path(value).parts)


if __name__ == "__main__":
    unittest.main()
