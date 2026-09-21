import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scaled_curriculum as curriculum


class ScaledCurriculumTests(unittest.TestCase):
    def test_target_counts_are_family_balanced_and_source_disjoint(self):
        rows = curriculum.generate_procedural_records(4, 2, seed=17)
        self.assertEqual(len(rows), 12 * 6)
        self.assertEqual(sum(row["split"] == "train" for row in rows), 48)
        self.assertEqual(sum(row["split"] == "dev" for row in rows), 24)
        self.assertEqual(
            {family: sum(row["family"] == family and row["split"] == "train" for row in rows)
             for family in curriculum.FAMILIES},
            {family: 4 for family in curriculum.FAMILIES},
        )
        train_groups = {row["source_group"] for row in rows if row["split"] == "train"}
        dev_groups = {row["source_group"] for row in rows if row["split"] == "dev"}
        self.assertFalse(train_groups & dev_groups)
        self.assertEqual(len({row["source_sha256"] for row in rows}), len(rows))
        self.assertEqual(len({row["geometry_sha256"] for row in rows}), len(rows))

    def test_compositions_use_one_canvas_translation_and_safe_direct_programs(self):
        rows = curriculum.generate_procedural_records(2, 0, seed=23)
        self.assertEqual({row["composition"] for row in rows}, set(curriculum.COMPOSITIONS))
        for row in rows:
            for source in row["programs"]:
                self.assertEqual(source.count("translate(-300,-300)"), 1)
                self.assertEqual(source.count("function draw()"), 1)
                self.assertNotIn("frame=", source)
                self.assertNotIn("source_id:", source)
                self.assertTrue(source.rstrip().endswith("}"))

    def test_schedule_is_planned_and_balanced_without_implicit_repeats(self):
        rows = curriculum.generate_procedural_records(3, 1, seed=31)
        schedule = curriculum.build_exposure_schedule(rows, epochs=2, seed=31)
        train = [row for row in rows if row["split"] == "train"]
        self.assertEqual(len(schedule), len(train) * 2)
        for epoch in range(2):
            ids = [row["id"] for row in schedule if row["epoch"] == epoch]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {row["id"] for row in train})
        self.assertTrue(any(schedule[i]["family"] != schedule[i + 1]["family"] for i in range(len(schedule) - 1)))
        self.assertTrue(all(row["admitted_training_exposure"] == 0 for row in rows))

    def test_inventory_exact_hash_dedup_preserves_external_taxonomy_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "ref.bin"
            image.write_bytes(b"reference image bytes")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            rows, stats = curriculum.import_inventory_rows([
                {"id": "real-a", "path": str(image), "image_sha256": digest, "family": "animals", "split": "train"},
                {"id": "real-b", "path": "remote://b", "image_sha256": digest, "family": "animals", "split": "train"},
                {"id": "real-c", "path": "remote://c", "family": "food", "split": "holdout"},
                {"id": "real-d", "path": str(image), "image_sha256": "0" * 64, "family": "food", "split": "train"},
            ])
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["skipped_duplicate_image_hash"], 1)
        self.assertEqual(stats["skipped_image_hash_mismatch"], 1)
        self.assertEqual(rows[0]["family"], "animals")
        self.assertEqual(rows[0]["image_hash_status"], "verified_local")
        self.assertEqual(rows[1]["split"], "holdout")
        self.assertEqual(rows[1]["family"], "food")

    def test_materialization_is_resumable_and_reuses_existing_program_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = curriculum.generate_procedural_records(1, 0, seed=3)
            first = curriculum.materialize_programs(rows, root / "programs", shard_count=3)
            second = curriculum.materialize_programs(rows, root / "programs", shard_count=3)
            self.assertEqual(first["files_written"], 36)
            self.assertEqual(second["files_written"], 0)
            self.assertEqual(second["files_reused"], 36)
            manifest_rows = [json.loads(line) for line in (root / "programs" / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(manifest_rows), 12)
            self.assertTrue(all(len(row["programs"]) == 3 for row in manifest_rows))

    def test_reference_manifest_keeps_imported_images_outside_renderer_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            imported = {
                "id": "photo-1",
                "path": "https://example.invalid/photo-1.png",
                "family": "animals",
                "split": "train",
            }
            rows = curriculum.generate_procedural_records(1, 0, seed=5)
            imported_rows, _ = curriculum.import_inventory_rows([imported])
            receipt = curriculum.materialize_programs(rows + imported_rows, root / "programs", shard_count=2)
            self.assertEqual(receipt["procedural_rows"], 12)
            refs = [json.loads(line) for line in (root / "programs" / "reference-manifest.jsonl").read_text().splitlines()]
            photo = next(row for row in refs if row["id"] == "inv-v1-photo-1")
            self.assertEqual(photo["reference"], imported["path"])
            self.assertEqual(photo["programs"], [])
            self.assertEqual(sum("reference_png" in row for row in refs), 12)

    def test_node_interpreter_path_keeps_virtualenv_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "system-python"
            target.write_text("python")
            link = root / "renderer-env-python"
            link.symlink_to(target)
            resolved = curriculum._absolute_preserving_symlink(link)
            self.assertEqual(resolved, link.absolute())
            self.assertEqual(resolved.name, "renderer-env-python")


if __name__ == "__main__":
    unittest.main()
