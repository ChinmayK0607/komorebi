import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "cloud"))
import publish_results


class SplitPublicationTests(unittest.TestCase):
    def test_text_parts_round_trip_through_public_verification(self):
        stored = {}

        class Api:
            def create_commit(self, *, operations, **_kwargs):
                for operation in operations:
                    stored[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
                return SimpleNamespace(oid="test-revision")

        def public_get(url, *, timeout):
            self.assertEqual(timeout, 180)
            path = url.split("/resolve/test-revision/", 1)[1].split("?", 1)[0]
            return io.BytesIO(stored[path])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bundle.tar.gz"
            raw = b"recovery-data-across-several-parts"
            archive.write_bytes(raw)
            with patch.object(publish_results, "PART_RAW_BYTES", 5), \
                 patch.object(publish_results, "PARTS_PER_COMMIT", 2), \
                 patch.object(publish_results, "urlopen", side_effect=public_get):
                result = publish_results.publish_split(
                    Api(), archive, "test-run", hashlib.sha256(raw).hexdigest(), root,
                )
            self.assertEqual(result["representation"], "split-base64-text")
            self.assertEqual(result["encoding"], "base64-per-part")
            self.assertGreater(len(result["part_paths"]), 2)
            self.assertEqual(set(result["part_paths"]), set(stored))


if __name__ == "__main__":
    unittest.main()
