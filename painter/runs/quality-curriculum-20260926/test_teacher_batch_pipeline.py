"""Verify that packaged teacher data can be staged by the cloud renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
COLLECTED = HERE.parents[1] / "collected/quality-curriculum-20260926"
SPEC = importlib.util.spec_from_file_location("render_teacher_batch_cloud", HERE / "render_teacher_batch_cloud.py")
assert SPEC and SPEC.loader
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


class BatchPipelineTest(unittest.TestCase):
    def test_text_and_reference_bundles_stage_with_verified_hashes(self) -> None:
        for batch in ("sol-text-seed", "astra-val2017-wave1", "astra-openverse-wave1",
                      "sol-val2017-wave5", "astra-openverse-wave5"):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as temporary:
                source = COLLECTED / batch
                receipt = json.loads((source / "source-receipt.json").read_text())
                archive = (source / "source.tar.gz").read_bytes()
                public = {"schema": "painter.teacher500-public-source.v1", "batch": batch,
                          "anonymous_hash_verified": True, "dataset_commit": "fixture",
                          "path": "source.tar.gz", "count": receipt["count"],
                          "archive_bytes": len(archive), "archive_sha256": receipt["archive_sha256"]}

                def fetch(_commit: str, path: str) -> bytes:
                    return json.dumps(public).encode() if path.endswith("source-public.json") else archive

                with patch.object(RENDERER, "public_bytes", side_effect=fetch):
                    manifest, staged_receipt = RENDERER.stage(batch, Path(temporary) / "stage")
                self.assertEqual(manifest["count"], receipt["count"])
                self.assertEqual(staged_receipt["archive_sha256"], receipt["archive_sha256"])
                self.assertEqual({row["mode"] for row in manifest["rows"]},
                                 {"text_to_image"} if "text" in batch else {"image_to_image"})
                if batch == "sol-val2017-wave5":
                    self.assertTrue(all(row["prompt"] and row["prompt_sha256"]
                                        and row["source_metadata"]["source_id"] for row in manifest["rows"]))
                if batch == "astra-openverse-wave5":
                    self.assertTrue(all(row["prompt"] and row["prompt_sha256"]
                                        and row["source_metadata"]["provider"] for row in manifest["rows"]))

    def test_archive_hash_mismatch_blocks_staging(self) -> None:
        batch = "sol-text-seed"
        source = COLLECTED / batch
        receipt = json.loads((source / "source-receipt.json").read_text())
        public = {"schema": "painter.teacher500-public-source.v1", "batch": batch,
                  "anonymous_hash_verified": True, "dataset_commit": "fixture", "path": "source.tar.gz",
                  "count": receipt["count"], "archive_bytes": receipt["archive_bytes"],
                  "archive_sha256": receipt["archive_sha256"]}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(RENDERER, "public_bytes", side_effect=lambda _c, p: (
                json.dumps(public).encode() if p.endswith("source-public.json") else b"bad")):
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    RENDERER.stage(batch, Path(temporary) / "stage")


if __name__ == "__main__":
    unittest.main()
