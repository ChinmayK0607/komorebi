"""Verify that packaged teacher data can be staged by the cloud renderer."""

from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import tarfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
COLLECTED = HERE.parents[1] / "collected/quality-curriculum-20260926"
SPEC = importlib.util.spec_from_file_location("render_teacher_batch_cloud", HERE / "render_teacher_batch_cloud.py")
assert SPEC and SPEC.loader
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
PUBLISH_SPEC = importlib.util.spec_from_file_location("publish_teacher_batch", HERE / "publish_teacher_batch.py")
assert PUBLISH_SPEC and PUBLISH_SPEC.loader
PUBLISHER = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(PUBLISHER)
COLLECT_SPEC = importlib.util.spec_from_file_location("collect_teacher_render", HERE / "collect_teacher_render.py")
assert COLLECT_SPEC and COLLECT_SPEC.loader
COLLECTOR = importlib.util.module_from_spec(COLLECT_SPEC)
COLLECT_SPEC.loader.exec_module(COLLECTOR)


class BatchPipelineTest(unittest.TestCase):
    def test_text_and_reference_bundles_stage_with_verified_hashes(self) -> None:
        for batch in ("sol-text-seed", "astra-val2017-wave1", "astra-openverse-wave1",
                      "sol-val2017-wave5", "astra-openverse-wave5", "astra-text-wave6",
                      "sol-text-curated-v1",
                      "sol-photo-static-repair-v1", "astra-sol-text-static-repair-v1",
                      "astra-photo-static-repair-v1", "astra-sol-text-static-repair-v2",
                      "sol-astra-photo-static-repair-v1", "astra-simple-cc0-static-repair-v1",
                      "astra-render-conditioned-correction-v1", "sol-render-conditioned-correction-v1",
                      "astra-shell-render-conditioned-turn3-v1", "sol-can-render-conditioned-turn3-v1"):
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
                if batch == "astra-text-wave6":
                    self.assertEqual({row["difficulty"] for row in manifest["rows"]},
                                     {"simple", "intermediate"})
                if batch == "sol-photo-static-repair-v1":
                    self.assertEqual(manifest["count"], 12)
                    self.assertTrue(all(row["role"] == "unrendered_alternative_candidate"
                                        and row["baseline_batch"] in {"sol-val2017-wave5", "sol-val2017-wave6"}
                                        and row["baseline_program_sha256"] for row in manifest["rows"]))
                if batch == "astra-sol-text-static-repair-v1":
                    self.assertEqual(manifest["count"], 12)
                    self.assertTrue(all(row["role"] == "unrendered_alternative_candidate"
                                        and row["baseline_batch"].startswith("sol-text-")
                                        and row["baseline_program_sha256"] for row in manifest["rows"]))
                if batch == "astra-photo-static-repair-v1":
                    self.assertEqual(manifest["count"], 3)
                    self.assertTrue(all(row["role"] == "unrendered_alternative_candidate"
                                        and row["baseline_batch"].startswith("astra-")
                                        and row["baseline_program_sha256"] for row in manifest["rows"]))
                if batch in {"astra-sol-text-static-repair-v2", "sol-astra-photo-static-repair-v1",
                             "astra-simple-cc0-static-repair-v1"}:
                    self.assertTrue(all(row["role"] == "unrendered_alternative_candidate"
                                        and row["baseline_program_sha256"] for row in manifest["rows"]))
                if "render-conditioned" in batch:
                    self.assertTrue(all(row["role"] == "render_conditioned_correction_candidate"
                                        and row["turn_count"] == (3 if "turn3" in batch else 2)
                                        and row["prior_run_id"] == (
                                            "teacher500-simple-cc0-pilot-20260926" if "turn3" not in batch else
                                            ("teacher500-astra-turn2-20260926" if batch.startswith("astra") else
                                             "teacher500-sol-turn2-20260926"))
                                        and (Path(temporary) / "stage" / row["prior_canvas"]).is_file()
                                        and (Path(temporary) / "stage" / row["baseline_program"]).is_file()
                                        for row in manifest["rows"]))

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

    def test_publication_requires_image_source_and_license_metadata(self) -> None:
        PUBLISHER.validate_publication_metadata(COLLECTED / "astra-openverse-wave1/source.tar.gz")
        with self.assertRaisesRegex(ValueError, "image source/rights metadata incomplete for 12 rows"):
            PUBLISHER.validate_publication_metadata(COLLECTED / "astra-reference-seed/source.tar.gz")

    def test_parallel_render_keeps_manifest_order(self) -> None:
        barrier = threading.Barrier(2)
        rows = [{"id": name} for name in ("first", "second", "third")]

        def fake_render(_output, row, _renderer, _python, _browser, _timeout):
            if row["id"] in {"first", "second"}:
                barrier.wait(timeout=2)
            return {"id": row["id"], "valid": True}

        with patch.object(RENDERER, "render_one", side_effect=fake_render), redirect_stdout(io.StringIO()):
            statuses = RENDERER.render_rows(Path("/tmp"), rows, Path("renderer"), Path("python"),
                                             Path("browser"), 30, workers=2)
        self.assertEqual([status["id"] for status in statuses], ["first", "second", "third"])

    def test_public_render_collection_binds_input_program_and_canvas(self) -> None:
        digest = lambda raw: hashlib.sha256(raw).hexdigest()
        prompt, program, canvas = b"A blue flower", b"function setup() {}", b"fake-canvas"
        source_sha, revision = "a" * 64, "b" * 40
        status = {"id": "flower", "mode": "text_to_image", "valid": True,
                  "input_sha256": digest(prompt), "program_sha256": digest(program),
                  "canvas_sha256": digest(canvas)}
        summary = {"schema": "painter.teacher500-render.v1", "run_id": "fixture-run",
                   "batch": "fixture-batch", "count": 1, "valid": 1,
                   "statuses": [status], "source_bundle_sha256": source_sha,
                   "source_dataset_commit": revision}
        members = {"run-summary.json": json.dumps(summary).encode(),
                   "episodes/flower/input.txt": prompt,
                   "episodes/flower/program.js": program,
                   "episodes/flower/canvas.png": canvas,
                   "episodes/flower/render-status.json": json.dumps(status).encode()}
        memory = io.BytesIO()
        with tarfile.open(fileobj=memory, mode="w:gz") as archive:
            for name, raw in members.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(raw)
                archive.addfile(entry, io.BytesIO(raw))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "rendered-cloud"
            source = out.parent / "fixture-batch"
            source.mkdir()
            (source / "source-public.json").write_text(json.dumps({"archive_sha256": source_sha,
                                                                     "dataset_commit": revision}))

            def fetch(_run_id, path):
                path.write_bytes(memory.getvalue())
                return {"sha256": digest(memory.getvalue()), "public_hash_verified": True}

            with patch.object(COLLECTOR, "OUT", out), patch.object(COLLECTOR, "download", side_effect=fetch):
                target, loaded = COLLECTOR.collect("fixture-run")
                page = COLLECTOR.gallery(target, loaded)
            self.assertIn("A blue flower", page.read_text())
            self.assertIn("renderer-valid", page.read_text())


if __name__ == "__main__":
    unittest.main()
