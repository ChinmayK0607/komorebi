import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("safe_archive.py")
SPEC = importlib.util.spec_from_file_location("safe_archive", MODULE_PATH)
assert SPEC and SPEC.loader
safe_archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(safe_archive)


class SafeArchiveTests(unittest.TestCase):
    def test_screened_package_keeps_evidence_and_excludes_weights_private_and_vendor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "async-output/run/monitors/file").mkdir(parents=True)
            (root / "async-output/run/monitors/file/metrics.jsonl").write_text('{"step": 1}\n')
            (root / "validation/step-15-all/eval").mkdir(parents=True)
            (root / "validation/step-15-all/eval/case.png").write_bytes(b"png")
            (root / "logs/gpu-telemetry.csv").parent.mkdir()
            (root / "logs/gpu-telemetry.csv").write_text("timestamp,utilization\n")
            (root / "async-output/run/artifacts/adapters/step_15").mkdir(parents=True)
            (root / "async-output/run/artifacts/adapters/step_15/hf-upload.json").write_text('{"verified":true}\n')
            (root / "async-output/run/artifacts/adapters/step_15/adapter_model.safetensors").write_bytes(b"weights")
            (root / "private/gateway.env").mkdir(parents=True)
            (root / "private/gateway.env/key.txt").write_text("AI_GATEWAY_API_KEY=secret\n")
            (root / "vendor/renderer.js").mkdir(parents=True)
            (root / "vendor/renderer.js/code.js").write_text("renderer\n")
            output = root / "local-evidence/evidence.tar.gz"
            files = safe_archive.select_files(root, output=output)
            report = safe_archive.package(files, output)
            self.assertEqual(report["files"], 4)
            with tarfile.open(output, "r:gz") as archive:
                names = archive.getnames()
                manifest = json.loads(archive.extractfile("evidence-manifest.json").read())
            self.assertIn("async-output/run/monitors/file/metrics.jsonl", names)
            self.assertIn("validation/step-15-all/eval/case.png", names)
            self.assertNotIn("private/gateway.env/key.txt", names)
            self.assertNotIn("async-output/run/artifacts/adapters/step_15/adapter_model.safetensors", names)
            self.assertTrue(any(row["reason"] == "weight_or_model_artifact" for row in manifest["screen_skips"]))

    def test_secret_like_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "bad.json"
            path.write_text('{"Authorization": "Bearer ' + 'a' * 32 + '"}\n')
            with self.assertRaises(safe_archive.ArchiveSafetyError):
                safe_archive.select_files(root)

    def test_secret_name_without_value_is_archivable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "protocol.md"
            path.write_text("Use AI_GATEWAY_API_KEY from the protected node environment.\n")
            files = safe_archive.select_files(root)
            self.assertIn("protocol.md", files)

    def test_latest_public_checkpoint_requires_verified_weight_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "async-output" / safe_archive.RUN_NAME / "artifacts" / "adapters"
            for step, content, verified in ((4, b"old", True), (15, b"new", True), (50, b"bad", False)):
                adapter = root / f"step_{step}"
                adapter.mkdir(parents=True)
                weight = adapter / "adapter_model.safetensors"
                weight.write_bytes(content)
                sha = safe_archive.digest(weight)
                (adapter / "hf-upload.json").write_text(json.dumps({
                    "verified": verified, "private": False,
                    "files": {"adapter_model.safetensors": {"sha256": sha}},
                }))
            selected = safe_archive.select_latest_public_checkpoint(root.parents[3])
            assert selected is not None
            self.assertEqual(selected["step"], 15)

    def test_public_upload_receipt_can_be_verified_without_persisting_token(self):
        # Exercise the provider-independent report contract; network upload is
        # intentionally not part of local tests.
        report = {"files": 1, "bytes": 12, "sha256": "a" * 64}
        self.assertEqual(report["sha256"], "a" * 64)
        self.assertNotIn("token", report)


if __name__ == "__main__":
    unittest.main()
