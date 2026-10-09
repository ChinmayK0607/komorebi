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

    def test_runtime_verifier_preserves_normal_venv_symlink_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter_home = root / "trusted-python"
            interpreter_home.mkdir()
            target = interpreter_home / "python3.12"
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text(f"home = {interpreter_home}\n", encoding="utf-8")
            link = venv / "bin/python"
            link.symlink_to(target)
            self.assertEqual(BOOTSTRAP.ensure_runtime_executable(link, "fixture venv Python", venv), link.absolute())
            self.assertTrue(link.is_symlink())

            escape = root / "escape-python"
            escape.write_text("#!/bin/sh\n", encoding="utf-8")
            escape.chmod(0o755)
            link.unlink()
            link.symlink_to(escape)
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.ensure_runtime_executable(link, "fixture venv Python", venv)

    def test_explicit_renderer_root_is_used_and_old_default_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node"
            node.mkdir()
            old = node / "renderer"
            old.mkdir()
            (old / "partial.marker").write_text("preserve\n", encoding="utf-8")
            explicit = root / "workspace" / "painter-renderer"
            selected = BOOTSTRAP.ensure_renderer_root(node, explicit)
            self.assertEqual(selected, explicit.resolve())
            self.assertTrue((old / "partial.marker").is_file())
            self.assertFalse(selected.is_symlink())
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.ensure_renderer_root(node, node / "models")
            link = root / "renderer-link"
            link.symlink_to(explicit, target_is_directory=True)
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.ensure_renderer_root(node, link)

    def test_ready_receipt_binds_renderer_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "bootstrap-ready.json"
            renderer_root = root / "renderer"
            renderer_root.mkdir()
            value = {
                "schema": "staged-family-balanced-bootstrap-ready-v1",
                "status": "ready_no_launch",
                "run_id": BOOTSTRAP.RUN_ID,
                "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                "renderer_root": str(renderer_root.resolve()),
            }
            receipt.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(BOOTSTRAP.validate_ready(receipt, topology=value["topology"], renderer_root=renderer_root.resolve()), value)
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.validate_ready(receipt, topology=value["topology"], renderer_root=root / "other-renderer")

    def test_renderer_helper_allowlists_existing_prior_run_root_only_when_relocated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node"
            prior = node / "renderer"
            current = root / "workspace" / "renderer"
            prior.mkdir(parents=True)
            current.mkdir(parents=True)
            command = BOOTSTRAP._renderer_helper_command(
                bootstrap_python=root / "bootstrap/bin/python",
                helper=root / "helper.py",
                render_root=current,
                renderer_python=current / "renderer-env/bin/python",
                browser_root=current / "browsers",
                prime_root=root / "prime",
                node_root=node,
            )
            marker = command.index("--allowed-old-renderer-root")
            self.assertEqual(Path(command[marker + 1]), prior)

            same_root_command = BOOTSTRAP._renderer_helper_command(
                bootstrap_python=root / "bootstrap/bin/python",
                helper=root / "helper.py",
                render_root=prior,
                renderer_python=prior / "renderer-env/bin/python",
                browser_root=prior / "browsers",
                prime_root=root / "prime",
                node_root=node,
            )
            self.assertNotIn("--allowed-old-renderer-root", same_root_command)
            prior.rename(node / "old-renderer")
            missing_old_command = BOOTSTRAP._renderer_helper_command(
                bootstrap_python=root / "bootstrap/bin/python",
                helper=root / "helper.py",
                render_root=current,
                renderer_python=current / "renderer-env/bin/python",
                browser_root=current / "browsers",
                prime_root=root / "prime",
                node_root=node,
            )
            self.assertNotIn("--allowed-old-renderer-root", missing_old_command)

    def test_prepare_receipt_renderer_root_accepts_legacy_key_but_rejects_ambiguity(self) -> None:
        expected = "/node/renderer"
        self.assertEqual(BOOTSTRAP._renderer_receipt_root({"root": expected}), expected)
        self.assertEqual(BOOTSTRAP._renderer_receipt_root({"renderer_root": expected}), expected)
        self.assertEqual(
            BOOTSTRAP._renderer_receipt_root({"root": expected, "renderer_root": expected}),
            expected,
        )
        with self.assertRaises(BOOTSTRAP.BootstrapError):
            BOOTSTRAP._renderer_receipt_root({})
        with self.assertRaises(BOOTSTRAP.BootstrapError):
            BOOTSTRAP._renderer_receipt_root({"root": expected, "renderer_root": "/other"})
        with self.assertRaises(BOOTSTRAP.BootstrapError):
            BOOTSTRAP._renderer_receipt_root({"root": None})

    def test_hf_snapshot_blob_symlink_is_allowed_but_cache_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "models--fixture--adapter"
            snapshot = cache / "snapshots" / "pinned"
            blobs = cache / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / ("a" * 64)
            blob.write_bytes(b"adapter bytes")
            weights = snapshot / "adapter_model.safetensors"
            weights.symlink_to(Path("../../blobs") / blob.name)
            self.assertEqual(BOOTSTRAP.ensure_hf_snapshot_file(weights, "fixture adapter"), weights.absolute())
            self.assertEqual(BOOTSTRAP.sha256_file(weights), hashlib.sha256(b"adapter bytes").hexdigest())

            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            weights.unlink()
            weights.symlink_to(outside)
            with self.assertRaises(BOOTSTRAP.BootstrapError):
                BOOTSTRAP.ensure_hf_snapshot_file(weights, "fixture adapter")

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
