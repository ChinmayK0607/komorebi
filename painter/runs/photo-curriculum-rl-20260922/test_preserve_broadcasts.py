from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("preserve_broadcasts", ROOT / "preserve_broadcasts.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreserveBroadcastTests(unittest.TestCase):
    def test_waits_for_both_markers_and_preserves_latest_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broadcasts = root / "broadcasts"
            backups = root / "backups"
            manifest = root / "preservation.json"
            for step in range(1, 4):
                current = broadcasts / f"step_{step}"
                current.mkdir(parents=True)
                (current / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
                (current / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
                (current / ".finished").touch()
                if step != 3:
                    (current / ".receiver_ready").touch()

            def validate(folder: Path, step: int) -> dict[str, object]:
                return {"step": step, "sha256": (folder / "adapter_model.safetensors").read_bytes().hex()}

            def upload(folder: Path, repo: str, run: str) -> dict[str, object]:
                self.assertRegex(folder.name, r"^step_[0-9]+$")
                return {"verified": True, "private": False, "repo_id": repo, "path": f"{run}/{folder.name}"}

            state = MODULE.run_worker(
                broadcast_root=broadcasts, backup_root=backups, manifest=manifest,
                repo="owner/repo", run="run", first_step=1, max_step=2,
                retention=2, deadline_seconds=0, poll_seconds=0,
                validate_files=validate, upload_checkpoint=upload,
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual([item["step"] for item in state["records"]], [1, 2])
            self.assertTrue((backups / "step_1").is_dir())
            self.assertTrue((backups / "step_2").is_dir())
            self.assertFalse((backups / "step_3").exists())
            self.assertFalse((backups / "step_1" / ".receiver_ready").exists())

    def test_transient_upload_failure_is_retried_and_public_receipt_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "broadcasts" / "step_1"
            source.mkdir(parents=True)
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            (source / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
            (source / ".finished").touch()
            (source / ".receiver_ready").touch()
            attempts = 0

            def upload(folder: Path, repo: str, run: str) -> dict[str, object]:
                nonlocal attempts
                self.assertEqual(folder.name, "step_1")
                attempts += 1
                if attempts == 1:
                    self.assertTrue((root / "backups" / "step_1").is_dir())
                    shutil.rmtree(source)
                    raise OSError("temporary provider failure")
                return {"verified": True, "private": False}

            state = MODULE.run_worker(
                broadcast_root=root / "broadcasts", backup_root=root / "backups", manifest=root / "state.json",
                repo="owner/repo", run="run", first_step=1, max_step=1, retention=2,
                deadline_seconds=1, poll_seconds=0, sleep=lambda _: None,
                validate_files=lambda folder, value: {"sha256": "hash"}, upload_checkpoint=upload,
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(attempts, 2)

            source.mkdir(parents=True)
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            (source / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
            (source / ".finished").touch()
            (source / ".receiver_ready").touch()

            def private_upload(folder: Path, repo: str, run: str) -> dict[str, object]:
                return {"verified": True, "private": True}

            private_state = MODULE.run_worker(
                broadcast_root=root / "broadcasts", backup_root=root / "private-backups", manifest=root / "private.json",
                repo="owner/repo", run="run", first_step=1, max_step=1, retention=2,
                deadline_seconds=0, poll_seconds=0, validate_files=lambda folder, value: {"sha256": "hash"},
                upload_checkpoint=private_upload,
            )
            self.assertEqual(private_state["status"], "deadline")
            self.assertEqual(private_state["last_error"]["type"], "PreservationError")

    def test_existing_final_backup_is_revalidated_after_manifest_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "broadcasts" / "step_1"
            source.mkdir(parents=True)
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            (source / "adapter_config.json").write_text('{}')
            (source / ".finished").touch()
            (source / ".receiver_ready").touch()
            final = root / "backups" / "step_1"
            final.mkdir(parents=True)
            (final / "adapter_model.safetensors").write_bytes(b"weights")
            (final / "adapter_config.json").write_text('{}')
            seen: list[str] = []

            def upload(folder: Path, repo: str, run: str) -> dict[str, object]:
                seen.append(folder.name)
                return {"verified": True, "private": False}

            state = MODULE.run_worker(
                broadcast_root=root / "broadcasts", backup_root=root / "backups", manifest=root / "state.json",
                repo="owner/repo", run="run", first_step=1, max_step=1, retention=2,
                deadline_seconds=0, poll_seconds=0, validate_files=lambda folder, value: {"sha256": "hash"},
                upload_checkpoint=upload,
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(seen, ["step_1"])
            self.assertTrue((final / "adapter_model.safetensors").is_file())

    def test_deadline_records_missing_steps_without_uploading_partial_broadcast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broadcasts = root / "broadcasts"
            broadcasts.mkdir()
            step = broadcasts / "step_1"
            step.mkdir()
            (step / "adapter_model.safetensors").write_bytes(b"partial")
            (step / "adapter_config.json").write_text("{}")
            (step / ".finished").touch()
            calls: list[int] = []

            state = MODULE.run_worker(
                broadcast_root=broadcasts, backup_root=root / "backups", manifest=root / "state.json",
                repo="owner/repo", run="run", first_step=1, max_step=1, retention=2,
                deadline_seconds=0, poll_seconds=0,
                validate_files=lambda folder, value: {"sha256": "unused"},
                upload_checkpoint=lambda folder, repo, run: calls.append(1) or {"verified": True},
            )
            self.assertEqual(state["status"], "deadline")
            self.assertEqual(state["missing_steps"], [1])
            self.assertEqual(calls, [])
            self.assertFalse((root / "backups").exists())


if __name__ == "__main__":
    unittest.main()
