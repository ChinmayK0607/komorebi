"""CPU-only contracts for the single-arm recovery launcher and controller."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_operator():
    spec = importlib.util.spec_from_file_location("single_operator", RUN / "operate_single.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operator = load_operator()


def load_profile_preparer():
    spec = importlib.util.spec_from_file_location(
        "single_profile_preparer", RUN / "prepare_profile_single.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_preparer = load_profile_preparer()


class FakeSSHFailure:
    def __enter__(self):
        raise RuntimeError("transport unavailable")

    def __exit__(self, *_args):
        return False


class FakePod:
    id = "pod-test"


class FakeAPI:
    def __init__(self):
        self.removed = []

    def ps(self):
        return [FakePod()]

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        return FakeSSHFailure()

    def rm(self, pod):
        self.removed.append(pod)


class InventoryFailureAPI(FakeAPI):
    def ps(self):
        raise RuntimeError("inventory unavailable")


class SingleArmTests(unittest.TestCase):
    def test_plan_only_uses_fake_interpreter_without_pydantic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "fake-python"
            log = root / "interpreter.log"
            blocked = root / "pydantic.py"
            blocked.write_text("raise RuntimeError('pydantic imported before bootstrap')\n")
            fake.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {str(log)!r}\n"
                f"exec {str(Path(sys.executable))!r} \"$@\"\n"
            )
            fake.chmod(0o755)
            env = {**os.environ, "PAINTER_PREPARE_PYTHON": str(fake), "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}"}
            result = subprocess.run(
                [str(RUN / "prepare_single.sh"), "--plan-only"],
                cwd=RUN,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("THROUGHPUT_RECOVERY_SINGLE_ARM_PLAN_VALIDATED", result.stdout)
            self.assertIn("-", log.read_text())

    def test_plan_binds_four_gpu_topology_as_one_trainer_plus_three_inference(self):
        env = {**os.environ, "PAINTER_GPUS_PER_NODE": "4", "PAINTER_NUM_TRAIN_GPUS": "1", "PAINTER_NUM_INFER_GPUS": "3"}
        result = subprocess.run(
            [str(RUN / "prepare_single.sh"), "--plan-only"],
            cwd=RUN,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"PAINTER_NUM_INFER_GPUS": 3', result.stdout)

    def test_plan_rejects_two_trainer_topology(self):
        env = {**os.environ, "PAINTER_GPUS_PER_NODE": "4", "PAINTER_NUM_TRAIN_GPUS": "2", "PAINTER_NUM_INFER_GPUS": "2"}
        result = subprocess.run(
            [str(RUN / "prepare_single.sh"), "--plan-only"],
            cwd=RUN,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one trainer", result.stderr)

    def test_controller_releases_exact_pod_after_transport_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("AI_GATEWAY_API_KEY=opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = FakeAPI()
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            with self.assertRaises(RuntimeError):
                operator.operate(
                    window=window,
                    source_bundle=bundle,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "ops",
                    api=api,
                    now=lambda: 1000.0,
                )
            self.assertEqual(len(api.removed), 1)
            receipt = (root / "ops/operations-single.jsonl").read_text()
            self.assertNotIn("pod-test", receipt)
            self.assertNotIn("AI_GATEWAY_API_KEY", receipt)

    def test_controller_budget_rejects_cap_above_twenty(self):
        with self.assertRaises(ValueError):
            operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.01,
            )

    def test_controller_replays_bundle_without_remote_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Single Arm Test"], check=True)
            (source / "marker.txt").write_text("bundle replay\n")
            subprocess.run(["git", "-C", str(source), "add", "marker.txt"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "bundle"], check=True)
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(source), "branch", "-M", "codex/single-arm-bundle"], check=True)
            bundle = root / "source.bundle"
            subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "refs/heads/codex/single-arm-bundle"], check=True, capture_output=True, text=True)
            destination = root / "repo"
            command = operator._source_checkout_command(str(bundle), str(destination), commit)
            subprocess.run(["bash", "-Eeuo", "pipefail", "-c", command], check=True)
            self.assertEqual((destination / "marker.txt").read_text(), "bundle replay\n")
            self.assertEqual(subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip(), commit)

    def test_controller_releases_requested_id_when_inventory_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = InventoryFailureAPI()
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            with self.assertRaises(RuntimeError):
                operator.operate(
                    window=window,
                    source_bundle=bundle,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "ops",
                    api=api,
                    now=lambda: 1000.0,
                )
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])
            self.assertNotIn("pod-test", (root / "ops/operations-single.jsonl").read_text())

    def test_controller_waits_for_absent_config_and_rejects_malformed_config(self):
        valid = {
            "profile": "rl-throughput-recovery-v1",
            "checkpoint_interval": 1,
            "batch_size": 16,
            "group_size": 4,
            "initial_inflight": 32,
            "max_inflight": 32,
            "activation_cpu_offload": True,
            "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
        }

        def summary_for(config_raw):
            values = [b"{}", b"{}", config_raw, None]
            with patch.object(operator, "_sftp_read", side_effect=values):
                return operator._remote_summary(object(), timeout=1)

        self.assertEqual(summary_for(None)["config"], {})
        valid_summary = summary_for(json.dumps(valid).encode())
        self.assertEqual(valid_summary["config"]["profile"], "rl-throughput-recovery-v1")
        operator._validate_config_summary(valid_summary)
        malformed = summary_for(b"{not-json")
        self.assertEqual(malformed["config"], {"_malformed": True})
        with self.assertRaises(operator.OperationError):
            operator._validate_config_summary(malformed)

    def test_wrappers_keep_matched_panel_renderer_and_four_update_bindings(self):
        execute = (RUN / "execute_single.sh").read_text()
        evaluate = (RUN / "evaluate.sh").read_text()
        telemetry = (RUN / "run_single.sh").read_text()
        self.assertIn("PAINTER_RL_MAX_STEPS=4", execute)
        self.assertIn("PAINTER_EVAL_PREFERRED_STEP=4", execute)
        self.assertIn("balanced16", evaluate)
        self.assertIn("PAINTER_EVAL_EXPECTED_COUNT=47", evaluate)
        self.assertIn("cancellable_watercolour_renderer.py", evaluate)
        self.assertIn("ADAPTER_ONLY_CHECKPOINTS=1", execute)
        self.assertIn('"average_utilization_percent"', telemetry)
        self.assertIn('"sampled_busy_fraction"', telemetry)
        self.assertIn('"role": "inference" if index < inference_gpu_count else "trainer"', telemetry)

    def test_remote_failure_diagnostics_are_scrubbed(self):
        diagnostics = operator._sanitized_diagnostics(
            "pod-test bootstrap failed API_GATEWAY_KEY=secret vck_live sk-or-v1-secret 10.1.2.3",
            forbidden=("pod-test",),
        )
        joined = " ".join(diagnostics)
        self.assertNotIn("pod-test", joined)
        self.assertNotIn("secret", joined)
        self.assertNotIn("vck_live", joined)
        self.assertNotIn("sk-or-v1-secret", joined)
        self.assertNotIn("10.1.2.3", joined)
        self.assertIn("bootstrap failed", joined)

    def test_remote_diagnostics_keep_stderr_tail_when_stdout_is_verbose(self):
        stdout = "\n".join(f"progress-{index}" for index in range(32))
        diagnostics = operator._sanitized_diagnostics(
            "fatal profile bootstrap error",
            stdout,
            labels=("stderr", "stdout"),
        )
        self.assertIn("stderr: fatal profile bootstrap error", diagnostics)
        self.assertTrue(any(line.startswith("stdout: progress-") for line in diagnostics))

    def test_profile_stage_replaces_existing_bootstrap_config_only_after_validation(self):
        from painter import throughput_recovery_profile as profile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]}) + "\n")
            source = profile_preparer.load_config_tool(REPO)
            native = root / "native.toml"
            receipt = root / "rl-config-receipt.json"
            native.write_text("bootstrap config\n")
            receipt.write_text(json.dumps({"stage": "base"}) + "\n")
            (root / "frozen-opponent-config.json").write_text(json.dumps({
                "schema": "photo-curriculum-rl-frozen-limits-v1",
                "operational_limits": {
                    "episode_timeout_seconds": 3600,
                    "render_timeout_seconds": 180,
                },
            }, indent=2) + "\n")
            frozen_before = (root / "frozen-opponent-config.json").read_bytes()
            kwargs = {
                "model": root / "model",
                "manifest": manifest,
                "prime_root": None,
                "run_name": "throughput-recovery-20261003",
                "output_root": "rl-rollouts",
                "expected_count": 40,
                "max_steps": 4,
                "learning_rate": 1e-6,
                "renderer": str(root / "renderer.py"),
                "renderer_python": str(root / "renderer.py"),
                "gpus_per_node": 2,
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            }
            with self.assertRaises(FileExistsError):
                profile.prepare_profile(
                    source.prepare,
                    output=native,
                    run_root=root,
                    **kwargs,
                )
            self.assertEqual(native.read_text(), "bootstrap config\n")
            profile_preparer.prepare_staged_profile(profile, source, run_root=root, **kwargs)
            self.assertIn("batch_size = 16", native.read_text())
            self.assertEqual(json.loads(receipt.read_text())["batch_size"], 16)
            self.assertEqual((root / "frozen-opponent-config.json").read_bytes(), frozen_before)
            guarded = root / "guarded"
            guarded.mkdir()
            guarded_native = guarded / "native.toml"
            guarded_receipt = guarded / "rl-config-receipt.json"
            guarded_native.write_text("bootstrap config\n")
            guarded_receipt.write_text(json.dumps({"stage": "base"}) + "\n")
            (guarded / "async-output").mkdir()
            with self.assertRaisesRegex(RuntimeError, "pretraining-only"):
                profile_preparer.prepare_staged_profile(profile, source, run_root=guarded, **kwargs)
            self.assertEqual(guarded_native.read_text(), "bootstrap config\n")
            self.assertEqual(json.loads(guarded_receipt.read_text())["stage"], "base")

    def test_profile_main_replays_existing_bootstrap_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prime = root / "prime"
            (prime / "src/prime_rl/configs").mkdir(parents=True)
            (prime / "src/prime_rl/__init__.py").write_text("")
            (prime / "src/prime_rl/configs/__init__.py").write_text("")
            (prime / "src/prime_rl/configs/rl.py").write_text(
                "class RLConfig:\n"
                "    @classmethod\n"
                "    def model_validate(cls, config):\n"
                "        return config\n"
            )
            (root / "runtime/watercolour-cancellable-centroid-v1").mkdir(parents=True)
            (root / "native.toml").write_text("bootstrap config\n")
            (root / "rl-config-receipt.json").write_text(json.dumps({"stage": "base"}) + "\n")
            (root / "frozen-opponent-config.json").write_text(json.dumps({
                "schema": "photo-curriculum-rl-frozen-limits-v1",
                "operational_limits": {
                    "episode_timeout_seconds": 3600,
                    "render_timeout_seconds": 180,
                },
            }, indent=2) + "\n")
            (root / "preparation-receipt.json").write_text(
                json.dumps({"schema": "text-paint-consistent-preparation-v2"}) + "\n"
            )
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]}) + "\n")
            (root / "runtime/watercolour-cancellable-centroid-v1/PATCH-MANIFEST.json").write_text(
                json.dumps({
                    "runtime_id": "test-runtime",
                    "selected_renderer": {"sha256": "b" * 64},
                    "frozen_baseline": {"sha256": "c" * 64},
                    "centroid_patch": {"after_sha256": "d" * 64},
                }) + "\n"
            )
            env = {
                **os.environ,
                "PAINTER_RUN_NAME": "throughput-recovery-20261003",
                "PAINTER_EXPERIMENT_NAME": "throughput-recovery-flash-20261003",
                "PAINTER_JUDGE_MODEL": "xiaomi/mimo-v2.6-flash",
                "PAINTER_JUDGE_EFFORT": "medium",
                "PAINTER_JUDGE_FORMAT": "identity-joint-side-schema-v2-generic-v1",
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                "PAINTER_BASE_MODEL_REVISION": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "PAINTER_POLICY_REVISION": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                "PAINTER_POLICY_PREFIX": "text-pairwise-scaled-20260929/step_16",
                "PAINTER_POLICY_SHA256": "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
                "PAINTER_FROZEN_SFT_MODEL_REVISION": "adf687e5b321514eda7be6eab050434fdc8e44ff",
                "PAINTER_FROZEN_SFT_ADAPTER_SHA256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
                "PAINTER_RL_CHECKPOINT_INTERVAL": "1",
                "ADAPTER_ONLY_CHECKPOINTS": "1",
                "PAINTER_ACTIVATION_CPU_OFFLOAD": "1",
                "PAINTER_GPUS_PER_NODE": "2",
                "PAINTER_NUM_TRAIN_GPUS": "1",
                "PAINTER_NUM_INFER_GPUS": "1",
                "PYTHONPATH": f"{REPO}:{REPO / 'painter'}",
            }
            command = [
                sys.executable,
                str(RUN / "prepare_profile_single.py"),
                "--repo", str(REPO),
                "--run-root", str(root),
                "--model", str(root / "model"),
                "--prime-root", str(prime),
                "--manifest", str(manifest),
                "--renderer", str(root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"),
                "--renderer-python", str(root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"),
            ]
            result = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / "rl-config-receipt.json").read_text())["batch_size"], 16)
            self.assertEqual(
                json.loads((root / "preparation-receipt.json").read_text())["schema"],
                "throughput-recovery-single-arm-preparation-v1",
            )
            self.assertTrue((root / "source-identity.json").is_file())
            self.assertTrue((root / "renderer-receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
