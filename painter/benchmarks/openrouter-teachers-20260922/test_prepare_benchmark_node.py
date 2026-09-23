import importlib.util
from pathlib import Path
import tarfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_benchmark_node", HERE / "prepare_benchmark_node.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PrepareBenchmarkNodeTests(unittest.TestCase):
    def test_reference_manifest_is_exact_and_hash_checked(self):
        rows = module.references()
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({name for name, _, _ in rows}), 40)

    def test_reference_archive_contains_only_images_and_manifest(self):
        rows = module.references()
        with __import__("tempfile").TemporaryDirectory() as temp:
            archive_path = Path(temp) / "references.tar.gz"
            module.create_references(archive_path, rows)
            with tarfile.open(archive_path, "r:gz") as archive:
                names = archive.getnames()
            self.assertEqual(sum(name.startswith("references/") for name in names), 40)
            self.assertIn("references-manifest.json", names)
            self.assertFalse(any(".env" in name or "key" in name.lower() for name in names))

    def test_dry_run_does_not_upload_or_create_bundle(self):
        self.assertEqual(module.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
