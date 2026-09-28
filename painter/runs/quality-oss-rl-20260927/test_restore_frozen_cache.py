"""Restoring a cached opponent requires archive and image hashes."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("restore_frozen_cache", HERE / "restore_frozen_cache.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class RestoreCacheTests(unittest.TestCase):
    def test_restores_only_verified_cache_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "evidence.tar.gz"
            key = "a" * 64
            image = b"a fake PNG for hash testing"
            receipt = json.dumps({"status": "valid", "sha256": module.sha_bytes(image)}).encode()
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in (
                    (f"frozen-opponents/images/{key}.png", image),
                    (f"frozen-opponents/receipts/{key}.json", receipt),
                    ("unrelated/secret.txt", b"must not extract"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            expected = module.sha_bytes(archive.read_bytes())
            result = module.restore(archive=archive, expected_sha256=expected,
                                    root=root / "target")
            self.assertEqual(result["valid_images"], 1)
            self.assertEqual((root / "target/frozen-opponents/images" / f"{key}.png").read_bytes(), image)
            self.assertFalse((root / "target/unrelated/secret.txt").exists())
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                module.restore(archive=archive, expected_sha256="0" * 64,
                               root=root / "elsewhere")


if __name__ == "__main__":
    unittest.main()
