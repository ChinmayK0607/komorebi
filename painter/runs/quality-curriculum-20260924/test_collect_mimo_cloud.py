import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from collect_mimo_cloud import unpack_checked


class CollectorArchiveTests(unittest.TestCase):
    def test_extracts_episode_receipt_and_rejects_escape_or_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "safe.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                data = b'{"status":"complete"}'
                member = tarfile.TarInfo("episodes/example/episode.json")
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
            self.assertEqual(unpack_checked(archive, root / "out"), 1)
            self.assertEqual((root / "out/episodes/example/episode.json").read_bytes(), data)

            for name, kind in (("../escape.json", "file"), ("episodes/link", "symlink")):
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as output:
                    member = tarfile.TarInfo(name)
                    if kind == "file":
                        member.size = 1
                        output.addfile(member, io.BytesIO(b"x"))
                    else:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../escape.json"
                        output.addfile(member)
                with self.assertRaises(ValueError):
                    unpack_checked(archive, root / "out")
            self.assertFalse((root / "escape.json").exists())


if __name__ == "__main__":
    unittest.main()
