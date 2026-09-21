"""Focused tests for manifest-derived native evaluation counts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from painter.native_painting_eval import config_text, validate_manifest


class NativePaintingEvalTests(unittest.TestCase):
    def test_count_is_inferred_for_scaled_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index in range(3):
                image = root / f"reference-{index}.png"
                image.write_bytes(f"reference-{index}".encode())
                rows.append({
                    "id": f"case-{index}",
                    "source_group": f"group-{index}",
                    "reference": image.name,
                    "reference_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                })
            manifest = root / "dev.json"
            manifest.write_text(json.dumps({"cases": rows}))
            audit = validate_manifest(root, manifest)
            self.assertEqual(audit["count"], 3)
            text = config_text(
                taskset_id="native_painting",
                root=root,
                manifest=manifest,
                policy_label="dev",
                model="model",
                output_dir=root / "out",
                count=audit["count"],
            )
            self.assertIn("num_tasks = 3", text)
            self.assertIn("expected_count = 3", text)
            self.assertNotIn("num_tasks = 12", text)

    def test_controlled_comparison_keeps_default_count_argument_compatibility(self):
        text = config_text(
            taskset_id="native_painting",
            root=Path("/tmp/root"),
            manifest=Path("/tmp/eval.json"),
            policy_label="baseline",
            model="model",
            output_dir=Path("/tmp/output"),
        )
        self.assertIn("num_tasks = 12", text)
        self.assertIn("expected_count = 12", text)

    def test_expected_count_can_still_guard_a_controlled_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(2):
                (root / f"{index}.png").write_bytes(str(index).encode())
            manifest = root / "eval.jsonl"
            manifest.write_text("\n".join(json.dumps({
                "id": str(index),
                "source_group": str(index),
                "reference": f"{index}.png",
            }) for index in range(2)) + "\n")
            with self.assertRaisesRegex(ValueError, "expected 12"):
                validate_manifest(root, manifest, expected_count=12)


if __name__ == "__main__":
    unittest.main()
