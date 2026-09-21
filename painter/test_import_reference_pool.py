from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import import_reference_pool as importer


def _classes_yaml() -> str:
    lines = ["names:"]
    lines.extend(f"  {index}: class-{index}" for index in range(80))
    return "\n".join(lines) + "\n"


class ImportReferencePoolTests(unittest.TestCase):
    def test_parse_labels_preserves_multiclass_and_distinguishes_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.txt"
            labels.write_text("2 0.5 0.5 0.25 0.5\n7 0.2 0.3 0.1 0.2\n")
            classes = {index: f"class-{index}" for index in range(80)}
            annotations, names, status = importer._parse_labels(labels, classes)
            self.assertEqual(status, "present")
            self.assertEqual(names, ["class-2", "class-7"])
            self.assertEqual([row["class_id"] for row in annotations or []], [2, 7])
            missing, missing_names, missing_status = importer._parse_labels(root / "absent.txt", classes)
            self.assertIsNone(missing)
            self.assertIsNone(missing_names)
            self.assertEqual(missing_status, "missing_label_file")

    def test_manifest_is_deterministic_and_excludes_historical_exact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run" / "real-pool"
            images = output / "raw" / "coco128" / "images" / "train2017"
            labels = output / "raw" / "coco128" / "labels" / "train2017"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (output / "raw" / "coco128.yaml").parent.mkdir(parents=True, exist_ok=True)
            (output / "raw" / "coco128.yaml").write_text(_classes_yaml())
            archive = output / "raw" / "coco128.zip"
            archive.write_bytes(b"test-archive")
            for index in range(4):
                stem = f"{index:012d}"
                (images / f"{stem}.jpg").write_bytes(f"image-{index}".encode())
                (labels / f"{stem}.txt").write_text(f"{index} 0.5 0.5 0.2 0.2\n")
            historical = Path(temp) / "history.jsonl"
            digest = hashlib.sha256(b"image-0").hexdigest()
            historical.write_text(json.dumps({"source_group": "old", "reference_sha256": digest}) + "\n")
            first = importer.build_manifest(output=output, dev_count=1, historical_paths=[historical], download=False, expected_archive_sha256=None, expected_classes_sha256=None)
            second = importer.build_manifest(output=output, dev_count=1, historical_paths=[historical], download=False, expected_archive_sha256=None, expected_classes_sha256=None)
            self.assertEqual(first["image_counts"], {"available": 4, "admitted": 3, "train": 2, "dev": 1})
            self.assertEqual(first["image_counts"], second["image_counts"])
            self.assertEqual(first["rejections"][0]["reason"], "historical_image_hash_overlap")
            manifest = json.loads((output / "reference-manifest.json").read_text())
            rows = manifest["references"]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["image_sha256"] for row in rows}), 3)
            self.assertEqual(len({row["source_group"] for row in rows if row["split"] == "train"} & {row["source_group"] for row in rows if row["split"] == "dev"}), 0)
            self.assertTrue(all(row["mode"] == "fresh" for row in rows))
            self.assertTrue(all(row["teacher_claim"] is False for row in rows))
            self.assertTrue(all(row["per_image_license"] is None for row in rows))
            self.assertTrue(all(row["per_image_license_status"] == "unverified_not_provided_in_archive" for row in rows))
            self.assertIn("dataset_license_documentation", rows[0])


if __name__ == "__main__":
    unittest.main()
