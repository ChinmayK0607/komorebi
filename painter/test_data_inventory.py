"""Metadata-only tests for the scalable curriculum inventory."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import data_inventory


class DataInventoryTests(unittest.TestCase):
    def test_hashes_actual_files_and_preserves_declared_unrendered_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_root = root / "data"
            image_root.mkdir()
            image = image_root / "ref.png"
            image.write_bytes(b"rendered-reference")
            actual = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            rows = [
                {"id": "rendered", "source_group": "g-rendered", "split": "train", "family": "sphere", "reference_kind": "procedural_known_program", "reference": "ref.png", "reference_sha256": actual},
                {"id": "pending", "source_group": "g-pending", "split": "dev", "family": "sphere", "reference_kind": "procedural_known_program", "reference": "missing.png", "target_sha256": "a" * 64},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = data_inventory.inventory([manifest], [image_root])
            self.assertEqual(result["counts"]["task_rows"], 2)
            self.assertEqual(result["counts"]["actual_unique_image_hashes"], 1)
            self.assertEqual(result["counts"]["declared_unrendered_hashes"], 1)
            self.assertEqual(result["counts"]["unique_reference_hashes"], 2)
            self.assertEqual(result["counts"]["hash_mismatches"], 0)
            self.assertFalse(result["records"][1]["rendered"])

    def test_photo_inference_and_split_overlap_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "photos.json"
            manifest.write_text(json.dumps({"references": [
                {"id": "p1", "source_group": "photo/1", "family": "vehicles", "split": "train", "source_url": "https://commons.wikimedia.org/wiki/File:x.jpg", "photo_evidence": {}, "reference_sha256": "1" * 64},
                {"id": "p2", "source_group": "photo/1", "family": "vehicles", "split": "dev", "source_url": "https://commons.wikimedia.org/wiki/File:y.jpg", "photo_evidence": {}, "reference_sha256": "2" * 64},
            ]}))
            result = data_inventory.inventory([manifest])
            self.assertEqual(result["by_source_type"], {"real_photo": 2})
            self.assertTrue(result["train_eval_source_group_overlap"]["dev"]["train"])


if __name__ == "__main__":
    unittest.main()
