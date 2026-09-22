"""Controller-only tests for the finite native RL entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from painter import rl_entrypoint as launch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RLEntrypointTests(unittest.TestCase):
    def make_spec(self, root: Path, *, mode: str = "fresh", command=None, resume=None) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.toml").write_text("[run]\nname = 'test'\n")
        (root / "manifest.json").write_text('{"rows": []}\n')
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        identity = {
            "profile": "qwen38_27b",
            "base_model": "Qwen/Qwen3.8-27B",
            "base_revision": "base-revision",
            "prime_commit": "prime-commit",
            "config": {"path": "config.toml", "sha256": digest(root / "config.toml")},
            "manifest": {"path": "manifest.json", "sha256": digest(root / "manifest.json")},
            "adapter": {"path": "adapter", "file": "adapter_model.safetensors", "sha256": digest(adapter / "adapter_model.safetensors")},
        }
        document = {
            "schema": launch.SCHEMA,
            "run_id": "test-run",
            "run_root": ".",
            "mode": mode,
            "identity": identity,
            "commands": {"fresh": ["python3", "{run_root}/launch.py"]},
        }
        if command is not None:
            document["commands"][mode] = command
        if resume is not None:
            document["resume"] = resume
        spec = root / "launch.json"
        spec.write_text(json.dumps(document, indent=2) + "\n")
        return spec

    def test_manifest_hash_binds_full_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root)
            spec = launch.validate_spec(spec_path)
            self.assertEqual(spec.paths["manifest"], (root / "manifest.json").resolve())
            (root / "manifest.json").write_text('{"rows": ["changed"]}\n')
            with self.assertRaisesRegex(launch.LaunchError, "manifest SHA-256 mismatch"):
                launch.validate_spec(spec_path)

    def test_start_attaches_to_active_identity_without_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = launch.validate_spec(self.make_spec(Path(directory)))
            fake = type("Process", (), {"pid": 12345})()
            with patch.object(launch.subprocess, "Popen", return_value=fake) as popen, patch.object(
                launch, "_pid_alive", return_value=True
            ):
                first = launch.start(spec, setup=False)
                second = launch.start(spec, setup=False)
            self.assertEqual(first["status"], "started")
            self.assertEqual(second["status"], "already_running")
            self.assertEqual(popen.call_count, 1)

    def test_prior_artifacts_without_state_refuse_fresh_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = launch.validate_spec(self.make_spec(root))
            (root / "async-output").mkdir()
            with self.assertRaisesRegex(launch.LaunchError, "prior artifacts"):
                launch.start(spec, setup=False)

    def test_native_resume_is_explicitly_unsupported_until_full_state_wrapper_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root, mode="native_resume", command=["python3", "--resume", "{checkpoint_dir}"], resume={})
            with self.assertRaisesRegex(launch.LaunchError, "native optimizer resume is not implemented"):
                launch.validate_spec(spec_path)

    def test_photo_prepare_builds_hash_bound_spec_from_staged_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            source_manifest = root / "manifest.jsonl"
            source_manifest.write_text('{"id": 1}\n{"id": 2}\n')
            model = root / "model"
            model.mkdir()
            prime = root / "prime"
            (prime / ".git").mkdir(parents=True)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            result = launch._prepare_photo(type("Args", (), {
                "profile": "photo", "run_root": run, "data": source_manifest,
                "run_id": "photo-test", "model": model, "adapter": adapter,
                "prime_root": prime, "python": None, "max_steps": 8,
                "base_revision": "base-revision", "prime_commit": "prime-commit",
                "render_root": None, "validate_prime": False,
            })())
            self.assertEqual(result["manifest_rows"], 2)
            prepared = launch.validate_spec(run / "launch.json")
            self.assertEqual(prepared.paths["adapter"], adapter.resolve())
            self.assertEqual(prepared.paths["manifest"], (run / "source-manifest.jsonl").resolve())
            self.assertEqual(prepared.raw["identity"]["manifest"]["sha256"], digest(run / "source-manifest.jsonl"))
            self.assertEqual(prepared.raw["environment"]["set"]["PAINTER_RL_RUN_NAME"], "photo-test")
            self.assertEqual(prepared.raw["environment"]["set"]["PAINTER_RL_MANIFEST"], str((run / "source-manifest.jsonl").resolve()))
            self.assertEqual(prepared.raw["environment"]["set"]["PAINTER_RL_MAX_STEPS"], "8")

    def test_preflight_runs_fresh_launcher_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = launch.validate_spec(self.make_spec(root, command=["bash", "{run_root}/launch.sh"]))
            completed = subprocess.CompletedProcess([], 0, "preflight output", "")
            with patch.object(launch.subprocess, "run", return_value=completed) as run:
                result = launch.preflight(spec)
            self.assertEqual(result["status"], "preflight_passed")
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["env"]["PAINTER_RL_PREFLIGHT_ONLY"], "1")
            self.assertEqual(kwargs["env"]["PAINTER_RL_SKIP_EDITABLE_INSTALL"], "1")
            self.assertEqual(kwargs["cwd"], spec.run_root)

    def test_huggingface_snapshot_adapter_symlink_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root)
            blob = root / "blobs" / "adapter-model"
            blob.parent.mkdir()
            blob.write_bytes(b"adapter")
            adapter = root / "adapter"
            (adapter / "adapter_model.safetensors").unlink()
            (adapter / "adapter_model.safetensors").symlink_to(blob)
            document = json.loads(spec_path.read_text())
            document["identity"]["adapter"]["sha256"] = digest(blob)
            spec_path.write_text(json.dumps(document))
            validated = launch.validate_spec(spec_path)
            self.assertEqual(validated.digests["adapter"], digest(blob))

    def test_run_preserves_failed_terminal_status_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root)
            prepared = {"status": "prepared", "spec": str(spec_path), "manifest_rows": 2}
            args = type("Args", (), {"judge_command": [sys.executable, "-c", "pass"], "no_setup": True,
                                      "wait_seconds": 1.0, "poll_seconds": 0.01})()
            with patch.object(launch, "_prepare_photo", return_value=prepared), patch.object(
                launch, "start", return_value={"status": "started"}
            ), patch.object(launch, "status", return_value={"status": "failed", "phase": "runner", "exit_code": 1}):
                with self.assertRaisesRegex(launch.LaunchError, "finite RL run ended failed"):
                    launch._run_photo(args)

    def test_external_manifest_jsonl_preserves_format_and_resolves_missing_relative_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "input"
            source_dir.mkdir()
            asset = source_dir / "assets" / "reference.png"
            asset.parent.mkdir()
            asset.write_bytes(b"png")
            source = source_dir / "manifest.jsonl"
            source.write_text(json.dumps({"id": "one", "reference": "assets/reference.png"}) + "\n")
            destination = root / "run" / "source-manifest.jsonl"
            staged_asset = destination.parent / "assets" / "reference.png"
            staged_asset.parent.mkdir(parents=True)
            staged_asset.write_bytes(b"different-content")
            launch._stage_manifest(source, destination)
            staged = json.loads(destination.read_text().strip())
            self.assertEqual(staged["reference"], str(asset.resolve()))

    def test_launch_receipt_uses_runtime_run_name(self):
        launcher = (Path(__file__).parent / "runs" / "photo-curriculum-rl-20260922" / "launch_rl.sh").read_text()
        self.assertIn('"$ADAPTER_SHA256" "$RUN_NAME" <<\'PY\'', launcher)
        self.assertIn('"run": run_name', launcher)
        self.assertLess(launcher.index("PAINTER_RL_PREFLIGHT_ONLY"), launcher.index('TELEMETRY_PID=""'))
        self.assertIn("PAINTER_RL_SKIP_EDITABLE_INSTALL", launcher)

    def test_runtime_judge_is_started_without_persisting_its_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root, command=[
                sys.executable, "-c", "import time; time.sleep(0.5)",
            ])
            judge_marker = root / "judge-started"
            runtime_judge = [
                sys.executable, "-c",
                f"from pathlib import Path; import time; Path({str(judge_marker)!r}).write_text('ok'); time.sleep(10)",
            ]
            spec = launch.validate_spec(spec_path)
            with patch.dict(launch.os.environ, {"PAINTER_RL_RUNTIME_JUDGE_JSON": json.dumps(runtime_judge)}, clear=False):
                self.assertEqual(launch._supervise(spec, attempt=1, setup_requested=False), 0)
            self.assertTrue(judge_marker.is_file())
            self.assertNotIn("judge-started", spec_path.read_text())

    def test_start_short_run_does_not_overwrite_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = launch.validate_spec(self.make_spec(root, command=[
                sys.executable, "-c", "from pathlib import Path; Path('{run_root}/done').write_text('ok')",
            ]))
            started = launch.start(spec, setup=False)
            self.assertEqual(started["status"], "started")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = launch.status(spec)
                if current.get("status") in launch.TERMINAL_STATES:
                    break
                time.sleep(0.05)
            self.assertEqual(current.get("status"), "complete")
            self.assertTrue((root / "done").is_file())
            while time.monotonic() < deadline and launch._pid_alive(started["pid"]):
                time.sleep(0.05)

    def test_stop_terminates_only_the_owned_controller_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = launch.validate_spec(self.make_spec(root, command=[
                sys.executable, "-c", "import time; time.sleep(30)",
            ]))
            started = launch.start(spec, setup=False)
            stopped = launch.stop(spec, wait_seconds=5)
            self.assertIn(stopped["status"], {"failed", "stopped"})
            self.assertFalse(launch._pid_alive(started["pid"], stopped.get("supervisor_start_token")))

    def test_supervisor_records_real_finite_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root, command=[
                sys.executable, "-c", "from pathlib import Path; Path('{run_root}/done').write_text('ok')",
            ])
            spec = launch.validate_spec(spec_path)
            self.assertEqual(launch._supervise(spec, attempt=1, setup_requested=False), 0)
            self.assertTrue((root / "done").is_file())
            state = json.loads(spec.state_path.read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["supervisor_start_token"], launch._pid_start_token(os.getpid()))

    def test_required_judge_exit_is_a_failure_and_stops_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self.make_spec(root, command=[
                sys.executable, "-c", "import time; time.sleep(10)",
            ])
            document = json.loads(spec_path.read_text())
            document["judge"] = {
                "required": True,
                "command": [sys.executable, "-c", "pass"],
            }
            spec_path.write_text(json.dumps(document))
            spec = launch.validate_spec(spec_path)
            started = time.monotonic()
            self.assertEqual(launch._supervise(spec, attempt=1, setup_requested=False), 1)
            self.assertLess(time.monotonic() - started, 5)
            state = json.loads(spec.state_path.read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "judge")


if __name__ == "__main__":
    unittest.main()
