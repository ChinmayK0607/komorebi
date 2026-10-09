from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from pydantic import BaseModel, ConfigDict, ValidationError


RUN = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = load("staged_launch_prepare_config", "prepare_config.py")
PREPARE_NODE = load("staged_launch_prepare_node", "prepare_node.py")
WRAPPER = load("staged_launch_wrapper", "run_from_sft.py")
TRAIN_WRAPPER = load_path("reviewed_photo_train_wrapper", RUN.parent / "photo-curriculum-rl-20260922/train_from_sft.py")
CANARY = load("staged_launch_canary", "canary.py")
CANARY_SOURCE = (RUN / "canary.py").read_text(encoding="utf-8")
LAUNCH_SOURCE = (RUN / "launch.sh").read_text(encoding="utf-8")
WRAPPER_SOURCE = (RUN / "run_from_sft.py").read_text(encoding="utf-8")
TRAIN_WRAPPER_SOURCE = (RUN.parent / "photo-curriculum-rl-20260922/train_from_sft.py").read_text(encoding="utf-8")


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

    def test_config_places_scoring_timeout_under_prime_agent_schema(self) -> None:
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
            env = config["orchestrator"]["train"]["source"][0]["env"]
            self.assertEqual(env["timeout"], {"episode": 3600})
            self.assertEqual(env["agent"]["timeout"]["scoring"], 600)

            class PrimeTimeout(BaseModel):
                episode: int
                model_config = ConfigDict(extra="forbid")

            class PrimeAgentTimeout(BaseModel):
                rollout: int
                scoring: int
                model_config = ConfigDict(extra="forbid")

            class PrimeAgent(BaseModel):
                timeout: PrimeAgentTimeout
                model_config = ConfigDict(extra="forbid")

            class PrimeEnv(BaseModel):
                timeout: PrimeTimeout
                agent: PrimeAgent
                model_config = ConfigDict(extra="forbid")

            corrected = {
                "timeout": env["timeout"],
                "agent": {"timeout": env["agent"]["timeout"]},
            }
            PrimeEnv.model_validate(corrected)
            misplaced = {
                "timeout": {**env["timeout"], "scoring": 600},
                "agent": {"timeout": {"rollout": 1200}},
            }
            with self.assertRaises(ValidationError):
                PrimeEnv.model_validate(misplaced)

    def test_prepare_helpers_accept_venv_links_but_reject_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertEqual(PREPARE_NODE._runtime_executable(link, "Prime Python", venv), link.absolute())
            self.assertEqual(CONFIG._runtime_executable(link, "renderer Python", venv), link.absolute())
            link.unlink()
            outside = root / "outside"
            outside.write_text("#!/bin/sh\n", encoding="utf-8")
            outside.chmod(0o755)
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                PREPARE_NODE._runtime_executable(link, "Prime Python", venv)

    def test_prepare_adapter_accepts_hf_blob_link_but_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "models--fixture--adapter"
            snapshot = cache / "snapshots" / "pinned"
            blobs = cache / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / ("b" * 64)
            blob.write_bytes(b"adapter")
            weights = snapshot / "adapter_model.safetensors"
            weights.symlink_to(Path("../../blobs") / blob.name)
            self.assertEqual(PREPARE_NODE._hf_snapshot_file(weights, "adapter weights"), weights.absolute())
            outside = root / "outside"
            outside.write_bytes(b"escape")
            weights.unlink()
            weights.symlink_to(outside)
            with self.assertRaises(ValueError):
                PREPARE_NODE._hf_snapshot_file(weights, "adapter weights")

    def test_staged_wrapper_uses_photo_runner_and_forwards_activation(self) -> None:
        self.assertIn('SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "run_from_sft.py"', WRAPPER_SOURCE)
        self.assertIn("ensure_hf_snapshot_file", WRAPPER_SOURCE)
        self.assertIn("def _trainer_dependency_paths", TRAIN_WRAPPER_SOURCE)
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

    def test_staged_wrapper_accepts_trusted_snapshot_links_but_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def adapter(name: str, payload: bytes) -> tuple[Path, Path]:
                cache = root / name
                snapshot = cache / "snapshots" / "pinned"
                blobs = cache / "blobs"
                snapshot.mkdir(parents=True)
                blobs.mkdir()
                weight_blob = blobs / (hashlib.sha256(payload).hexdigest() * 2)
                weight_blob.write_bytes(payload)
                config_blob = blobs / ("config-" + name)
                config_blob.write_bytes(b"{}\n")
                (snapshot / "adapter_model.safetensors").symlink_to(Path("../../blobs") / weight_blob.name)
                (snapshot / "adapter_config.json").symlink_to(Path("../../blobs") / config_blob.name)
                return snapshot, snapshot / "adapter_model.safetensors"

            policy, policy_weights = adapter("policy-cache", b"policy-fixture")
            frozen, frozen_weights = adapter("frozen-cache", b"frozen-fixture")
            environment = {
                "WATERCOLOUR_INIT_ADAPTER": str(policy),
                "PAINTER_FROZEN_SFT_ADAPTER": str(frozen),
                "PAINTER_FROZEN_SFT_ALIAS": "sft-step512",
            }
            with patch.object(WRAPPER, "POLICY_SHA256", hashlib.sha256(policy_weights.read_bytes()).hexdigest()), patch.object(
                WRAPPER, "FROZEN_SFT_SHA256", hashlib.sha256(frozen_weights.read_bytes()).hexdigest()
            ), patch.dict(os.environ, environment, clear=False):
                values = WRAPPER._adapter_environment()
            self.assertEqual(values["WATERCOLOUR_INIT_ADAPTER"], str(policy.resolve()))
            self.assertEqual(values["PAINTER_FROZEN_SFT_ADAPTER"], str(frozen.resolve()))

            outside = root / "outside.safetensors"
            outside.write_bytes(b"escape")
            policy_weights.unlink()
            policy_weights.symlink_to(outside)
            with patch.object(WRAPPER, "POLICY_SHA256", hashlib.sha256(b"escape").hexdigest()), patch.object(
                WRAPPER, "FROZEN_SFT_SHA256", hashlib.sha256(frozen_weights.read_bytes()).hexdigest()
            ), patch.dict(os.environ, environment, clear=False):
                with self.assertRaises(ValueError):
                    WRAPPER._adapter_environment()

    def test_trainer_dependency_paths_require_exact_helper_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush = root / "brush"
            helpers = brush / "integrations/prime/watercolour_sft"
            helpers.mkdir(parents=True)
            for name in ("init_adapter.py", "export_adapter.py", "model_profiles.py"):
                (helpers / name).write_text("# fixture\n", encoding="utf-8")
            with patch.dict(os.environ, {"BRUSH_ROOT": str(brush)}, clear=False):
                resolved = TRAIN_WRAPPER._trainer_dependency_paths()
            self.assertEqual(resolved[1], brush.resolve())
            self.assertEqual(resolved[2], helpers.resolve())
            (helpers / "model_profiles.py").unlink()
            with patch.dict(os.environ, {"BRUSH_ROOT": str(brush)}, clear=False):
                with self.assertRaises(FileNotFoundError):
                    TRAIN_WRAPPER._trainer_dependency_paths()

    def test_staged_wrapper_builds_child_environment_without_recursion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy"
            frozen = root / "frozen"
            policy.mkdir()
            frozen.mkdir()
            (policy / "adapter_model.safetensors").write_bytes(b"policy-fixture")
            (policy / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (frozen / "adapter_model.safetensors").write_bytes(b"frozen-fixture")
            (frozen / "adapter_config.json").write_text("{}\n", encoding="utf-8")
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
        self.assertIn('"renderer_root": str(renderer_root)', CANARY_SOURCE)
        self.assertIn("from prime_rl.configs.rl import RLConfig", CANARY_SOURCE)
        self.assertIn("RLConfig.model_validate(config)", CANARY_SOURCE)
        self.assertIn('"prime_config_validation":', CANARY_SOURCE)
        self.assertIn('env["PATH"] = os.pathsep.join((str(prime_bin), "/usr/local/bin", "/usr/bin", "/bin"))', CANARY_SOURCE)
        self.assertIn("_hf_snapshot_file", CANARY_SOURCE)
        self.assertIn("_trainer_dependency_smoke", CANARY_SOURCE)
        self.assertIn('importlib.import_module("prime_rl.trainer.rl.train")', CANARY_SOURCE)

    def test_trainer_dependency_smoke_executes_wrapper_and_prime_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush = root / "brush"
            helpers = brush / "integrations/prime/watercolour_sft"
            helpers.mkdir(parents=True)
            (helpers / "model_profiles.py").write_text(
                "def profile(): return {'name': 'qwen38_27b'}\n", encoding="utf-8"
            )
            (helpers / "init_adapter.py").write_text("", encoding="utf-8")
            (helpers / "export_adapter.py").write_text("", encoding="utf-8")
            wrapper = root / "train_from_sft.py"
            wrapper.write_text(TRAIN_WRAPPER_SOURCE, encoding="utf-8")
            prime_train = root / "prime_rl/trainer/rl"
            prime_train.mkdir(parents=True)
            (root / "prime_rl/__init__.py").write_text("", encoding="utf-8")
            (root / "prime_rl/trainer/__init__.py").write_text("", encoding="utf-8")
            (prime_train / "__init__.py").write_text("", encoding="utf-8")
            (prime_train / "train.py").write_text("", encoding="utf-8")
            env = dict(os.environ)
            env.update({
                "BRUSH_ROOT": str(brush),
                "BRUSH_MODEL_PROFILE": "qwen38_27b",
                "PYTHONPATH": str(root),
            })
            result = CANARY._trainer_dependency_smoke(
                prime_python=Path(sys.executable),
                trainer_wrapper=wrapper,
                env=env,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["model_profile"], "qwen38_27b")
            (helpers / "model_profiles.py").unlink()
            with self.assertRaises(RuntimeError):
                CANARY._trainer_dependency_smoke(
                    prime_python=Path(sys.executable),
                    trainer_wrapper=wrapper,
                    env=env,
                )

    def test_canary_validates_trusted_snapshot_adapter_files_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def adapter(name: str, payload: bytes) -> tuple[Path, Path]:
                cache = root / name
                snapshot = cache / "snapshots" / "pinned"
                blobs = cache / "blobs"
                snapshot.mkdir(parents=True)
                blobs.mkdir()
                weight_blob = blobs / (hashlib.sha256(payload).hexdigest() * 2)
                weight_blob.write_bytes(payload)
                config_blob = blobs / ("config-" + name)
                config_blob.write_bytes(b"{}\n")
                (snapshot / "adapter_model.safetensors").symlink_to(Path("../../blobs") / weight_blob.name)
                (snapshot / "adapter_config.json").symlink_to(Path("../../blobs") / config_blob.name)
                return snapshot, snapshot / "adapter_model.safetensors"

            policy, policy_weights = adapter("policy-cache", b"policy-fixture")
            frozen, frozen_weights = adapter("frozen-cache", b"frozen-fixture")
            policy_sha = hashlib.sha256(policy_weights.read_bytes()).hexdigest()
            frozen_sha = hashlib.sha256(frozen_weights.read_bytes()).hexdigest()
            receipt = {
                "policy_initializer": {"path": str(policy), "weights_sha256": policy_sha},
                "frozen_sft": {"path": str(frozen), "weights_sha256": frozen_sha},
            }
            (root / "source").mkdir()
            (root / "node").mkdir()
            with patch.dict(CANARY.CONFIG.POLICY, {"sha256": policy_sha}), patch.dict(
                CANARY.CONFIG.FROZEN_SFT, {"sha256": frozen_sha}
            ):
                environment = CANARY._runtime_environment(
                    prepare_receipt=receipt,
                    source_root=root / "source",
                    node_root=root / "node",
                    repo_root=root,
                )
            self.assertEqual(environment["WATERCOLOUR_INIT_ADAPTER"], str(policy.resolve()))
            self.assertEqual(environment["PAINTER_FROZEN_SFT_ADAPTER"], str(frozen.resolve()))

            outside = root / "outside.safetensors"
            outside.write_bytes(b"escape")
            policy_weights.unlink()
            policy_weights.symlink_to(outside)
            with patch.dict(CANARY.CONFIG.POLICY, {"sha256": hashlib.sha256(b"escape").hexdigest()}), patch.dict(
                CANARY.CONFIG.FROZEN_SFT, {"sha256": frozen_sha}
            ):
                with self.assertRaises(ValueError):
                    CANARY._runtime_environment(
                        prepare_receipt=receipt,
                        source_root=root / "source",
                        node_root=root / "node",
                        repo_root=root,
                    )

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

    def test_canary_requires_prime_scripts_in_exact_venv_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prime_root = root / "prime"
            prime_bin = prime_root / ".venv/bin"
            prime_bin.mkdir(parents=True)
            python = prime_bin / "python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            for name in ("inference", "trainer", "orchestrator"):
                path = prime_bin / name
                path.write_text("#!/bin/sh\n", encoding="utf-8")
                path.chmod(0o755)
            resolved, scripts = CANARY._prime_console_scripts(
                prime_root=prime_root,
                prime_python=python,
            )
            self.assertEqual(resolved, prime_bin.resolve())
            self.assertEqual(set(scripts), {"inference", "trainer", "orchestrator"})
            (prime_bin / "trainer").unlink()
            with self.assertRaises(ValueError):
                CANARY._prime_console_scripts(prime_root=prime_root, prime_python=python)

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
            self.assertEqual(command[:6], ["/usr/sbin/runuser", "-u", "painter", "--", "env", "-i"])
            self.assertIn("PAINTER_RENDER_BACKEND=swiftshader", command)
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(chown.call_count, 2)

    def test_renderer_canary_preserves_prior_attempt_before_retry(self) -> None:
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
            prior = root / "node/renderer-canary"
            prior.mkdir(parents=True)
            (prior / "failed-attempt.txt").write_text("preserve\n", encoding="utf-8")
            expected = hashlib.sha256(renderer.read_bytes()).hexdigest()
            with patch.object(CANARY, "RENDERER_SHA256", expected):
                receipt = CANARY._renderer_probe(
                    renderer=renderer,
                    renderer_python=renderer_python,
                    node_root=root / "node",
                )
            preserved = Path(receipt["previous_evidence"])
            self.assertTrue(preserved.is_dir())
            self.assertEqual((preserved / "failed-attempt.txt").read_text(), "preserve\n")
            self.assertTrue((root / "node/renderer-canary/probe.png").is_file())

    def test_launcher_is_foreground_bounded_and_wires_both_adapters(self) -> None:
        self.assertIn('RUN_WRAPPER="${PAINTER_PRIME_RUN_WRAPPER:-$RUN_SOURCE_ROOT/run_from_sft.py}"', LAUNCH_SOURCE)
        self.assertIn('TRAIN_WRAPPER="${PAINTER_PRIME_TRAIN_WRAPPER:-$RUN_SOURCE_ROOT/../photo-curriculum-rl-20260922/train_from_sft.py}"', LAUNCH_SOURCE)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=60", LAUNCH_SOURCE)
        self.assertNotIn("setsid", LAUNCH_SOURCE)
        self.assertNotIn("nohup", LAUNCH_SOURCE)
        self.assertNotIn("export PAINTER_PRIME_ROLE=orchestrator", LAUNCH_SOURCE)
        self.assertIn('export WATERCOLOUR_INIT_ADAPTER="$POLICY_ADAPTER" PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_ADAPTER"', LAUNCH_SOURCE)
        self.assertIn('PAINTER_RENDERER_ROOT', LAUNCH_SOURCE)
        self.assertIn('"constant_trainer_batch_size") is not True', LAUNCH_SOURCE)
        self.assertIn('"policy_initializer":', LAUNCH_SOURCE)
        self.assertIn('"frozen_sft":', LAUNCH_SOURCE)
        self.assertIn('canary.get("status") != "passed_import_hook_and_renderer"', LAUNCH_SOURCE)
        self.assertIn('renderer_probe.get("status") != "verified"', LAUNCH_SOURCE)
        self.assertIn('TRAIN_GPUS=', LAUNCH_SOURCE)
        self.assertNotIn('PAINTER_NUM_TRAIN_GPUS=2', LAUNCH_SOURCE)
        self.assertIn('PRIME_BIN="$PRIME_ROOT/.venv/bin"', LAUNCH_SOURCE)
        self.assertIn('export PATH="$PRIME_BIN${PATH:+:$PATH}"', LAUNCH_SOURCE)
        self.assertIn('for executable in inference trainer orchestrator; do', LAUNCH_SOURCE)
        self.assertIn('EXPECTED_BRUSH_ROOT="$REPO_ROOT/painter/rl_helpers"', LAUNCH_SOURCE)
        self.assertIn('export BRUSH_MODEL_PROFILE=qwen38_27b', LAUNCH_SOURCE)
        self.assertIn('integrations/prime/watercolour_sft/model_profiles.py', LAUNCH_SOURCE)
        self.assertIn('integrations/prime/watercolour_sft/model_profiles.py', LAUNCH_SOURCE)

    def test_launcher_prime_path_precedes_sanitized_ambient_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prime_bin = root / "prime/.venv/bin"
            ambient = root / "ambient"
            prime_bin.mkdir(parents=True)
            ambient.mkdir()
            for name in ("python", "inference", "trainer", "orchestrator"):
                path = prime_bin / name
                path.write_text("#!/bin/sh\n", encoding="utf-8")
                path.chmod(0o755)
            (ambient / "inference").write_text("#!/bin/sh\n", encoding="utf-8")
            (ambient / "inference").chmod(0o755)
            shell = f'''\
set -Eeuo pipefail
PRIME_ROOT="{root / "prime"}"
PYTHON="$PRIME_ROOT/.venv/bin/python"
PRIME_BIN="$PRIME_ROOT/.venv/bin"
[[ "$PYTHON" == "$PRIME_BIN/python" && -x "$PYTHON" ]]
for executable in inference trainer orchestrator; do
    [[ -f "$PRIME_BIN/$executable" && ! -L "$PRIME_BIN/$executable" && -x "$PRIME_BIN/$executable" ]]
done
PATH="{ambient}"
export PATH="$PRIME_BIN${{PATH:+:$PATH}}"
[[ "$(command -v inference)" == "$PRIME_BIN/inference" ]]
'''
            completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_canonicalizes_photo_trainer_root_before_comparison(self) -> None:
        self.assertIn(
            'EXPECTED_TRAIN_WRAPPER_ROOT="$(cd "$RUN_SOURCE_ROOT/../photo-curriculum-rl-20260922" && pwd -P)"',
            LAUNCH_SOURCE,
        )
        self.assertIn(
            'TRAIN_WRAPPER_ROOT="$(cd "$(dirname "$TRAIN_WRAPPER")" && pwd -P)"',
            LAUNCH_SOURCE,
        )
        self.assertIn(
            '[[ "$TRAIN_WRAPPER_ROOT" == "$EXPECTED_TRAIN_WRAPPER_ROOT" ]]',
            LAUNCH_SOURCE,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "staged-family-balanced-20261010"
            trainer_root = root / "photo-curriculum-rl-20260922"
            run_root.mkdir()
            trainer_root.mkdir()
            (trainer_root / "train_from_sft.py").write_text("# fixture\n", encoding="utf-8")
            shell = f'''\
set -Eeuo pipefail
RUN_SOURCE_ROOT="$(cd "{run_root}" && pwd -P)"
TRAIN_WRAPPER="$RUN_SOURCE_ROOT/../photo-curriculum-rl-20260922/train_from_sft.py"
EXPECTED_TRAIN_WRAPPER_ROOT="$(cd "$RUN_SOURCE_ROOT/../photo-curriculum-rl-20260922" && pwd -P)"
TRAIN_WRAPPER_ROOT="$(cd "$(dirname "$TRAIN_WRAPPER")" && pwd -P)"
[[ "$TRAIN_WRAPPER_ROOT" == "$EXPECTED_TRAIN_WRAPPER_ROOT" ]]
'''
            completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)

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
