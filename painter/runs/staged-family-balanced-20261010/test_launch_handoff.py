from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = load("staged_launch_prepare_config", "prepare_config.py")
WRAPPER = load("staged_launch_wrapper", "run_from_sft.py")
CANARY = load("staged_launch_canary", "canary.py")
CANARY_SOURCE = (RUN / "canary.py").read_text(encoding="utf-8")
LAUNCH_SOURCE = (RUN / "launch.sh").read_text(encoding="utf-8")
WRAPPER_SOURCE = (RUN / "run_from_sft.py").read_text(encoding="utf-8")


class LaunchHandoffTests(unittest.TestCase):
    def test_config_is_finite_and_requires_constant_trainer_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / CONFIG.BASE_MODEL_REVISION
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            materialized = root / "materialized"
            materialized.mkdir()
            repo = root / "repo"
            repo.mkdir()
            config = CONFIG.build_config(
                model=model,
                node_root=root / "node",
                materialized_root=materialized,
                repo_root=repo,
            )
            CONFIG.validate_config(config, node_root=root / "node", materialized_root=materialized)
            self.assertEqual(config["orchestrator"]["batch_size"], 24)
            self.assertEqual(config["orchestrator"]["group_size"], 4)
            self.assertIs(config["orchestrator"]["constant_trainer_batch_size"], True)
            self.assertEqual(config["max_steps"], 20)

    def test_config_receipt_is_atomic_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / CONFIG.BASE_MODEL_REVISION
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            materialized = root / "materialized"
            materialized.mkdir()
            repo = root / "repo"
            repo.mkdir()
            node = root / "node"
            config = CONFIG.build_config(
                model=model,
                node_root=node,
                materialized_root=materialized,
                repo_root=repo,
            )
            receipt = CONFIG.write_config(
                output=node / "native.toml",
                receipt_output=node / "config-receipt.json",
                config=config,
                node_root=node,
                materialized_root=materialized,
            )
            self.assertEqual(receipt["config_sha256"], hashlib.sha256((node / "native.toml").read_bytes()).hexdigest())
            parsed = tomllib.loads((node / "native.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["orchestrator"]["batch_size"], 24)
            self.assertIs(parsed["orchestrator"]["constant_trainer_batch_size"], True)

    def test_config_preserves_explicit_one_plus_one_and_two_plus_two_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / CONFIG.BASE_MODEL_REVISION
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            materialized = root / "materialized"
            materialized.mkdir()
            repo = root / "repo"
            repo.mkdir()
            one_plus_one = CONFIG.build_config(
                model=model, node_root=root / "one", materialized_root=materialized,
                repo_root=repo, gpus_per_node=2, num_train_gpus=1, num_infer_gpus=1,
            )
            two_plus_two = CONFIG.build_config(
                model=model, node_root=root / "two", materialized_root=materialized,
                repo_root=repo, gpus_per_node=4, num_train_gpus=2, num_infer_gpus=2,
            )
            self.assertEqual((one_plus_one["deployment"]["num_train_gpus"], one_plus_one["deployment"]["num_infer_gpus"]), (1, 1))
            self.assertEqual((two_plus_two["deployment"]["num_train_gpus"], two_plus_two["deployment"]["num_infer_gpus"]), (2, 2))

    def test_staged_wrapper_uses_photo_runner_and_forwards_activation(self) -> None:
        self.assertIn('SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "run_from_sft.py"', WRAPPER_SOURCE)
        self.assertIn("base._orchestrator_env = _staged_orchestrator_env", WRAPPER_SOURCE)
        self.assertIn("base.main()", WRAPPER_SOURCE)
        self.assertIn("ORIGINAL_ORCHESTRATOR_ENV = base._orchestrator_env", WRAPPER_SOURCE)
        for key in (
            "PAINTER_STAGED_CURRICULUM_ENABLE",
            "PAINTER_STAGED_CURRICULUM_RUN_ID",
            "PAINTER_STAGED_CURRICULUM_PROTOCOL",
            "PAINTER_STAGED_CURRICULUM_ADMISSION",
            "PAINTER_STAGED_CURRICULUM_ROOT",
            "PAINTER_STAGED_CURRICULUM_STATE",
            "WATERCOLOUR_INIT_ADAPTER",
            "PAINTER_FROZEN_SFT_ADAPTER",
        ):
            self.assertIn(key, WRAPPER_SOURCE)

    def test_staged_wrapper_builds_child_environment_without_recursion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy"
            frozen = root / "frozen"
            policy.mkdir()
            frozen.mkdir()
            (policy / "adapter_model.safetensors").write_bytes(b"policy-fixture")
            (frozen / "adapter_model.safetensors").write_bytes(b"frozen-fixture")
            old_policy_sha = WRAPPER.POLICY_SHA256
            old_frozen_sha = WRAPPER.FROZEN_SFT_SHA256
            WRAPPER.POLICY_SHA256 = hashlib.sha256(b"policy-fixture").hexdigest()
            WRAPPER.FROZEN_SFT_SHA256 = hashlib.sha256(b"frozen-fixture").hexdigest()
            environment = {
                "WATERCOLOUR_INIT_ADAPTER": str(policy),
                "PAINTER_FROZEN_SFT_ADAPTER": str(frozen),
                "PAINTER_FROZEN_SFT_ALIAS": "sft-step512",
                "PAINTER_RL_ADAPTER_REVISION": WRAPPER.POLICY_REVISION,
                "PAINTER_RL_ADAPTER_PREFIX": WRAPPER.POLICY_PREFIX,
                "PAINTER_RL_ADAPTER_SHA256": WRAPPER.POLICY_SHA256,
                "PAINTER_FROZEN_SFT_REVISION": WRAPPER.FROZEN_SFT_REVISION,
                "PAINTER_FROZEN_SFT_PREFIX": WRAPPER.FROZEN_SFT_PREFIX,
                "PAINTER_FROZEN_SFT_SHA256": WRAPPER.FROZEN_SFT_SHA256,
                "PAINTER_STAGED_CURRICULUM_ENABLE": "1",
                "PAINTER_STAGED_CURRICULUM_RUN_ID": CONFIG.RUN_ID,
                "PAINTER_STAGED_CURRICULUM_PROTOCOL": "staged-family-balanced-protocol-v1",
                "PAINTER_STAGED_CURRICULUM_ADMISSION": "1",
                "PAINTER_STAGED_CURRICULUM_ROOT": str(root),
                "PAINTER_STAGED_CURRICULUM_STATE": str(root / "runtime-state.json"),
            }
            with patch.dict(os.environ, environment, clear=False):
                child = WRAPPER._staged_orchestrator_env({"SENTINEL": "kept"})
            WRAPPER.POLICY_SHA256 = old_policy_sha
            WRAPPER.FROZEN_SFT_SHA256 = old_frozen_sha
            self.assertEqual(child["SENTINEL"], "kept")
            self.assertEqual(child["WATERCOLOUR_INIT_ADAPTER"], str(policy.resolve()))
            self.assertEqual(child["PAINTER_FROZEN_SFT_ADAPTER"], str(frozen.resolve()))
            self.assertEqual(child["PAINTER_STAGED_CURRICULUM_RUN_ID"], CONFIG.RUN_ID)
            self.assertEqual(child["PAINTER_RL_ADAPTER_SHA256"], environment["PAINTER_RL_ADAPTER_SHA256"])

    def test_canary_is_import_hook_only_and_uses_no_cuda(self) -> None:
        self.assertIn("CUDA_VISIBLE_DEVICES", CANARY_SOURCE)
        self.assertIn("NVIDIA_VISIBLE_DEVICES", CANARY_SOURCE)
        self.assertIn("install_from_environment", CANARY_SOURCE)
        self.assertIn("orchestrator_setup_called", CANARY_SOURCE)
        self.assertIn("passed_import_hook_and_renderer", CANARY_SOURCE)
        self.assertIn("staged-family-balanced-renderer-canary-v1", CANARY_SOURCE)
        self.assertLessEqual(CANARY.RENDER_TIMEOUT_SECONDS, 90)
        self.assertNotIn("Orchestrator()", CANARY_SOURCE)
        self.assertNotIn("Orchestrator.setup(", CANARY_SOURCE)

    def test_renderer_canary_uses_fake_interpreter_without_chromium(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            renderer = root / "renderer.py"
            renderer.write_text("# fake renderer boundary\n", encoding="utf-8")
            renderer_python = root / "renderer-python"
            renderer_python.write_text(
                "#!/usr/bin/env python3\n"
                "import hashlib, json, sys\n"
                "from pathlib import Path\n"
                "renderer, source = map(Path, sys.argv[1:3])\n"
                "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "output.write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')\n"
                "png = hashlib.sha256(output.read_bytes()).hexdigest()\n"
                "metadata = {'valid': True, 'renderer_sha256': hashlib.sha256(renderer.read_bytes()).hexdigest(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'png_sha256': png}\n"
                "output.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')\n",
                encoding="utf-8",
            )
            renderer_python.chmod(0o755)
            expected = hashlib.sha256(renderer.read_bytes()).hexdigest()
            with patch.object(CANARY, "RENDERER_SHA256", expected):
                receipt = CANARY._renderer_probe(
                    renderer=renderer,
                    renderer_python=renderer_python,
                    node_root=root / "node",
                )
            self.assertEqual(receipt["status"], "verified")
            self.assertTrue(receipt["valid"])
            self.assertLessEqual(receipt["timeout_seconds"], 90)
            output = root / "node/renderer-canary/probe.png"
            self.assertTrue(output.is_file())
            self.assertEqual(receipt["png_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_root_renderer_canary_uses_painter_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            probe.mkdir()
            renderer = root / "renderer.py"
            renderer_python = root / "renderer-python"
            renderer.write_text("# renderer\n", encoding="utf-8")
            renderer_python.write_text("#!/bin/sh\n", encoding="utf-8")
            renderer_python.chmod(0o755)
            with patch.object(CANARY.os, "geteuid", return_value=0), patch.object(CANARY.shutil, "which", return_value="/usr/sbin/runuser"), patch.object(CANARY.os, "chown") as chown, patch.object(CANARY.subprocess, "run", side_effect=[
                type("Result", (), {"returncode": 0, "stdout": "1001\n"})(),
                type("Result", (), {"returncode": 0, "stdout": "1001\n"})(),
            ]):
                command, environment, unprivileged = CANARY._renderer_invocation(
                    renderer=renderer, renderer_python=renderer_python,
                    source=root / "program.js", output=root / "output.png",
                    browser_root=root / "browsers", probe_root=probe,
                )
            self.assertTrue(unprivileged)
            self.assertEqual(command[:6], ["runuser", "-u", "painter", "--", "env", "-i"])
            self.assertIn("PAINTER_RENDER_BACKEND=swiftshader", command)
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(chown.call_count, 2)

    def test_launcher_is_foreground_bounded_and_wires_both_adapters(self) -> None:
        self.assertIn('RUN_WRAPPER="${PAINTER_PRIME_RUN_WRAPPER:-$RUN_SOURCE_ROOT/run_from_sft.py}"', LAUNCH_SOURCE)
        self.assertIn('TRAIN_WRAPPER="${PAINTER_PRIME_TRAIN_WRAPPER:-$RUN_SOURCE_ROOT/../photo-curriculum-rl-20260922/train_from_sft.py}"', LAUNCH_SOURCE)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=60", LAUNCH_SOURCE)
        self.assertNotIn("setsid", LAUNCH_SOURCE)
        self.assertNotIn("nohup", LAUNCH_SOURCE)
        self.assertNotIn("export PAINTER_PRIME_ROLE=orchestrator", LAUNCH_SOURCE)
        self.assertIn('export WATERCOLOUR_INIT_ADAPTER="$POLICY_ADAPTER" PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_ADAPTER"', LAUNCH_SOURCE)
        self.assertIn('"constant_trainer_batch_size") is not True', LAUNCH_SOURCE)
        self.assertIn('"policy_initializer":', LAUNCH_SOURCE)
        self.assertIn('"frozen_sft":', LAUNCH_SOURCE)
        self.assertIn('canary.get("status") != "passed_import_hook_and_renderer"', LAUNCH_SOURCE)
        self.assertIn('renderer_probe.get("status") != "verified"', LAUNCH_SOURCE)
        self.assertIn('TRAIN_GPUS=', LAUNCH_SOURCE)
        self.assertNotIn('PAINTER_NUM_TRAIN_GPUS=2', LAUNCH_SOURCE)

    def test_run_surface_records_hash_all_preflight_and_reviewed_wrappers(self) -> None:
        config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
        surface = config["run_surface"]
        expected = {
            "bootstrap", "bootstrap_shell", "materializer", "prepare_config", "prepare_node", "canary", "launch", "launch_prepared",
            "run_from_sft", "runtime_binding", "sitecustomize", "prime_tail_purge",
            "photo_runner", "photo_trainer",
        }
        self.assertEqual(set(surface["files"]), expected)
        for details in surface["files"].values():
            path = (RUN / details["path"]).resolve()
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), details["sha256"])
        self.assertEqual(surface["canary_scope"]["status"], "prepared_not_run")
        self.assertFalse(surface["canary_scope"]["orchestrator_setup_called"])
        self.assertFalse(surface["canary_scope"]["gpu_or_cuda"])
        self.assertFalse(surface["canary_scope"]["network"])
        self.assertFalse(surface["canary_scope"]["model"])
        self.assertFalse(surface["canary_scope"]["judge"])
        bootstrap = surface["bootstrap_scope"]
        self.assertEqual(bootstrap["status"], "prepared_not_run")
        self.assertEqual(bootstrap["receipt_status"], "ready_no_launch")
        self.assertEqual(
            bootstrap["topologies"],
            [
                {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                {"gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
            ],
        )
        self.assertFalse(bootstrap["launch_called"])
        self.assertFalse(bootstrap["training_started"])
        self.assertFalse(bootstrap["network_or_hardware_used"])
        launch_scope = surface["launch_scope"]
        self.assertEqual(launch_scope["receipt_derived_launcher"], "launch_prepared.py")
        self.assertEqual(launch_scope["requires_owner_only_gateway_mode"], "0600")
        self.assertTrue(launch_scope["requires_explicit_timeout_seconds"])
        self.assertFalse(launch_scope["gateway_contents_read_or_logged"])
        self.assertFalse(launch_scope["scheduling"])
        protocol = json.loads((RUN / "PROTOCOL.json").read_text(encoding="utf-8"))
        receipt = json.loads((RUN / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["run_surface"], surface)
        self.assertEqual(receipt["run_surface"], surface)


if __name__ == "__main__":
    unittest.main()
