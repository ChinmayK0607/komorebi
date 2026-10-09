from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("staged_launch_prepared", RUN / "launch_prepared.py")
assert SPEC is not None and SPEC.loader is not None
HANDOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDOFF)


class PreparedLaunchTests(unittest.TestCase):
    def _verified(self, root: Path) -> dict[str, object]:
        paths = {name: root / name for name in ("repo", "prime", "prime-python", "materialized", "model", "policy", "frozen", "renderer", "renderer-python", "browsers", "brush", "prepared")}
        for name, path in paths.items():
            if name.endswith("python") or name in {"renderer", "brush"}:
                path.write_text("fixture\n", encoding="utf-8")
                path.chmod(0o755)
            else:
                path.mkdir()
        gateway = root / "gateway.env"
        gateway.write_text("SECRET=fixture-value\n", encoding="utf-8")
        gateway.chmod(0o600)
        return {
            "repo_root": paths["repo"], "prime_root": paths["prime"], "prime_python": paths["prime-python"],
            "materialized": paths["materialized"], "model_dir": paths["model"], "policy": paths["policy"],
            "frozen": paths["frozen"], "renderer": paths["renderer"], "renderer_python": paths["renderer-python"],
            "browser_root": paths["browsers"], "brush_root": paths["brush"], "prepared": paths["prepared"],
            "gateway": gateway,
        }

    def test_environment_derives_all_paths_without_reading_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._verified(Path(directory))
            gateway = verified.pop("gateway")
            with patch.dict(HANDOFF.os.environ, {"OPENAI_API_KEY": "must-not-forward", "PAINTER_PRIME_ROLE": "orchestrator"}, clear=False):
                env = HANDOFF.build_launch_environment(verified=verified, gateway_env=gateway, timeout_seconds=17)
            self.assertEqual(env["PAINTER_REPO_ROOT"], str(verified["repo_root"]))
            self.assertEqual(env["PRIME_ROOT"], str(verified["prime_root"]))
            self.assertEqual(env["PAINTER_MATERIALIZED_ROOT"], str(verified["materialized"]))
            self.assertEqual(env["MODEL_DIR"], str(verified["model_dir"]))
            self.assertEqual(env["WATERCOLOUR_INIT_ADAPTER"], str(verified["policy"]))
            self.assertEqual(env["PAINTER_FROZEN_SFT_ADAPTER"], str(verified["frozen"]))
            self.assertEqual(env["PAINTER_RENDERER_PATH"], str(verified["renderer"]))
            self.assertEqual(env["PAINTER_RENDERER_PYTHON_PATH"], str(verified["renderer_python"]))
            self.assertEqual(env["PAINTER_BROWSER_ROOT"], str(verified["browser_root"]))
            self.assertEqual(env["BRUSH_ROOT"], str(verified["brush_root"]))
            self.assertEqual(env["PAINTER_GATEWAY_ENV_FILE"], str(gateway.resolve()))
            self.assertEqual(env["PAINTER_RUN_TIMEOUT_SECONDS"], "17")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("PAINTER_PRIME_ROLE", env)

    def test_invoke_is_foreground_and_does_not_print_gateway_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._verified(Path(directory))
            gateway = verified.pop("gateway")
            calls: list[tuple[list[str], dict[str, object]]] = []

            class Result:
                returncode = 7

            def fake_runner(command, **kwargs):
                calls.append((command, kwargs))
                return Result()

            status = HANDOFF.invoke(verified=verified, gateway_env=gateway, timeout_seconds=23, runner=fake_runner)
            self.assertEqual(status, 7)
            self.assertEqual(len(calls), 1)
            command, kwargs = calls[0]
            self.assertEqual(command[:2], ["bash", str(RUN / "launch.sh")])
            self.assertEqual(command[2], str(verified["prepared"]))
            self.assertFalse(kwargs["check"])
            self.assertEqual(kwargs["env"]["PAINTER_RUN_TIMEOUT_SECONDS"], "23")
            self.assertEqual(kwargs["env"]["PAINTER_GATEWAY_ENV_FILE"], str(gateway.resolve()))
            self.assertNotIn("SECRET=fixture-value", repr(kwargs["env"]))

    def test_gateway_mode_and_timeout_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = root / "gateway.env"
            gateway.write_text("SECRET=fixture\n", encoding="utf-8")
            gateway.chmod(0o644)
            with self.assertRaises(HANDOFF.PreparedLaunchError):
                HANDOFF.validate_gateway_file(gateway)
            gateway.chmod(0o600)
            with self.assertRaises(HANDOFF.PreparedLaunchError):
                HANDOFF.build_launch_environment(verified=self._verified(root), gateway_env=gateway, timeout_seconds=0)

    def test_receipt_runtime_path_accepts_venv_link_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "python-home"
            home.mkdir()
            target = home / "python3.12"
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text(f"home = {home}\n", encoding="utf-8")
            link = venv / "bin/python"
            link.symlink_to(target)
            self.assertEqual(HANDOFF._runtime_executable(str(link), "receipt Python", venv), link.absolute())
            link.unlink()
            outside = root / "outside"
            outside.write_text("#!/bin/sh\n", encoding="utf-8")
            outside.chmod(0o755)
            link.symlink_to(outside)
            with self.assertRaises(HANDOFF.PreparedLaunchError):
                HANDOFF._runtime_executable(str(link), "receipt Python", venv)

    def test_adapter_weight_hash_accepts_hf_blob_link_but_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "models--fixture--adapter"
            snapshot = cache / "snapshots" / "pinned"
            blobs = cache / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / ("c" * 64)
            blob.write_bytes(b"adapter")
            adapter = snapshot
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.symlink_to(Path("../../blobs") / blob.name)
            expected = hashlib.sha256(b"adapter").hexdigest()
            HANDOFF._weights(adapter, expected, "fixture adapter")
            outside = root / "outside"
            outside.write_bytes(b"escape")
            weights.unlink()
            weights.symlink_to(outside)
            with self.assertRaises(HANDOFF.PreparedLaunchError):
                HANDOFF._weights(adapter, hashlib.sha256(b"escape").hexdigest(), "fixture adapter")

    def test_topology_shape_is_checked_before_launch(self) -> None:
        ready = {"topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1}}
        self.assertEqual(HANDOFF._validate_topology(ready, visible_gpus=2)["gpus_per_node"], 2)
        with self.assertRaises(HANDOFF.PreparedLaunchError):
            HANDOFF._validate_topology({"topology": {"gpus_per_node": 3, "num_train_gpus": 1, "num_infer_gpus": 2}}, visible_gpus=3)

    def test_wrapper_is_foreground_and_does_not_shell_expand_environment(self) -> None:
        source = (RUN / "launch_prepared.py").read_text(encoding="utf-8")
        for forbidden in ("subprocess.Popen", "shell=True", "nohup", "setsid", "&>"):
            self.assertNotIn(forbidden, source)
        self.assertIn('runner(["bash", str(RUN_ROOT / "launch.sh")', source)
        self.assertIn("gateway env file must have owner-only mode 600", source)

    def test_fake_ready_receipt_with_wrong_prime_fails_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = {
                "schema": "staged-family-balanced-bootstrap-ready-v1",
                "status": "ready_no_launch",
                "run_id": HANDOFF.RUN_ID,
                "node_root": str(root),
                "repo_root": str(HANDOFF.REPO_ROOT),
                "source_root": str(RUN),
                "brush_root": str(HANDOFF.REPO_ROOT / "painter"),
                "training_started": False,
                "model_generation": False,
                "judge_calls": False,
                "background_processes_left_running": False,
                "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                "prime": {"revision": "wrong-prime", "root": str(root), "python": str(root / "python")},
            }
            (root / "bootstrap-ready.json").write_text(json.dumps(ready), encoding="utf-8")
            with self.assertRaises(HANDOFF.PreparedLaunchError):
                HANDOFF.validate_ready(ready_path=root / "bootstrap-ready.json", node_root=root, visible_gpus=2)


if __name__ == "__main__":
    unittest.main()
