from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("staged_bootstrap", RUN / "bootstrap.py")
assert SPEC is not None and SPEC.loader is not None
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class BootstrapUnitTests(unittest.TestCase):
    def test_supported_topologies_are_explicit_and_hardware_agnostic(self) -> None:
        BOOTSTRAP.validate_topology(gpus_per_node=2, num_train_gpus=1, num_infer_gpus=1, visible_gpus=2)
        BOOTSTRAP.validate_topology(gpus_per_node=4, num_train_gpus=2, num_infer_gpus=2, visible_gpus=4)
        with self.assertRaises(ValueError):
            BOOTSTRAP.validate_topology(gpus_per_node=3, num_train_gpus=1, num_infer_gpus=2, visible_gpus=3)
        with self.assertRaises(ValueError):
            BOOTSTRAP.validate_topology(gpus_per_node=4, num_train_gpus=2, num_infer_gpus=2, visible_gpus=2)

    def test_gpu_probe_is_foreground_and_requires_linux(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="0\n1\n", stderr="")
        with mock.patch.object(BOOTSTRAP.platform, "system", return_value="Linux"), mock.patch.object(BOOTSTRAP.shutil, "which", return_value="/usr/bin/nvidia-smi"), mock.patch.object(BOOTSTRAP.subprocess, "run", return_value=completed) as run:
            self.assertEqual(BOOTSTRAP.visible_gpu_count(), 2)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["timeout"], 20)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_archive_extracts_only_regular_allowlisted_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "evidence.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, data in {
                    "curriculum/pilot-train-manifest.json": b'{"rows": []}\n',
                    "curriculum/matched32-manifest.json": b'{"rows": []}\n',
                    "curriculum/prompts/train/example.txt": b"prompt\n",
                    "unrelated/raw.log": b"ignored\n",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    bundle.addfile(info, io.BytesIO(data))
            out = root / "extracted"
            receipt = BOOTSTRAP.safe_extract_archive(archive, out)
            self.assertEqual(receipt["files"], 3)
            self.assertEqual((out / "curriculum/prompts/train/example.txt").read_text(), "prompt\n")
            self.assertFalse((out / "unrelated").exists())
            self.assertFalse((out / "curriculum/prompts/train/example.txt").is_symlink())

    def test_archive_rejects_symlink_in_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("curriculum/anchors/escape.png")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside.png"
                bundle.addfile(info)
                for name in ("curriculum/pilot-train-manifest.json", "curriculum/matched32-manifest.json"):
                    data = b"{}\n"
                    item = tarfile.TarInfo(name)
                    item.size = len(data)
                    bundle.addfile(item, io.BytesIO(data))
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.safe_extract_archive(archive, root / "extracted")

    def test_archive_rejects_path_traversal_even_outside_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("curriculum/../escape.txt")
                data = b"escape"
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.safe_extract_archive(archive, root / "extracted")

    def test_download_helper_hashes_and_reuses_without_network(self) -> None:
        payload = b"pinned evidence\n"
        expected = hashlib.sha256(payload).hexdigest()

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "asset.bin"
            with mock.patch.object(BOOTSTRAP.urllib.request, "urlopen", return_value=FakeResponse(payload)) as urlopen:
                BOOTSTRAP.download_file("https://public.invalid/asset", destination, expected)
            urlopen.assert_called_once()
            self.assertEqual(destination.read_bytes(), payload)
            # Exact local bytes are enough for the idempotent second call.
            with mock.patch.object(BOOTSTRAP.urllib.request, "urlopen", side_effect=AssertionError("network")):
                BOOTSTRAP.download_file("https://public.invalid/asset", destination, expected)
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.download_file("https://public.invalid/asset", destination, "0" * 64)

    def test_ready_receipt_is_atomic_and_declares_no_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap-ready.json"
            BOOTSTRAP.atomic_json(path, {
                "schema": "staged-family-balanced-bootstrap-ready-v1",
                "status": "ready_no_launch",
                "run_id": BOOTSTRAP.RUN_ID,
                "training_started": False,
            })
            self.assertEqual(json.loads(path.read_text())["status"], "ready_no_launch")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_shell_and_source_are_foreground_and_pinned(self) -> None:
        shell = (RUN / "bootstrap.sh").read_text()
        source = (RUN / "bootstrap.py").read_text()
        subprocess.run(["bash", "-n", str(RUN / "bootstrap.sh")], check=True)
        self.assertIn("exec python3", shell)
        for unsafe in ("nohup", "setsid", "subprocess.Popen", "shell=True", "&> "):
            self.assertNotIn(unsafe, source + shell)
        for pinned in (BOOTSTRAP.PRIME_COMMIT, BOOTSTRAP.MODEL_REVISION, BOOTSTRAP.POLICY_SHA256, BOOTSTRAP.FROZEN_SHA256, BOOTSTRAP.ARCHIVE_SHA256, BOOTSTRAP.RENDERER_SHA256, BOOTSTRAP.P5_BRUSH_SHA256):
            self.assertIn(pinned, source)
        for required in ("materialize.py", "prepare_config.py", "prepare_node.py", "canary.py", "prepare_renderer_runtime.py", "sandbox_smoke", "ready_no_launch"):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
