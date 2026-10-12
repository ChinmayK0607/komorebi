from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace


RUN = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, RUN / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load("launch_guard")
resource = _load("resource_preflight")
canary = _load("real_backward_canary")
run_wrapper = _load("run_from_sft")
train_wrapper = _load("train_from_sft")


export_spec = importlib.util.spec_from_file_location(
    "staged_export_adapter", RUN.parents[1] / "rl_helpers/integrations/prime/watercolour_sft/export_adapter.py"
)
assert export_spec is not None and export_spec.loader is not None
export_adapter = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(export_adapter)


class _FakeProcess:
    pid = 1234

    def poll(self):
        return None


class ReliabilityPassTests(unittest.TestCase):
    def test_reported_rank_oom_terminates_the_whole_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = root / "distributed-failure.json"
            def spawn(*args, **kwargs):
                failure.write_text(json.dumps({
                    "schema": "staged-family-balanced-distributed-failure-v1",
                    "status": "rank_oom",
                    "rank": "3",
                    "reason": "CUDA out of memory",
                }), encoding="utf-8")
                return _FakeProcess()
            with patch.object(guard, "_terminate") as terminate:
                status = guard.supervise(
                    ["not-started"], failure_file=failure, rank_dir=root / "ranks",
                    expected_world_size=4, timeout_seconds=10, popen=spawn,
                )
            self.assertEqual(status, guard.FAILURE_EXIT)
            terminate.assert_called_once()

    def test_stale_rank_guard_state_is_rejected_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_dir = root / "ranks"
            rank_dir.mkdir()
            (rank_dir / "rank-0.started.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                guard.supervise(
                    ["not-started"], failure_file=root / "distributed-failure.json", rank_dir=rank_dir,
                    expected_world_size=4, timeout_seconds=10, popen=lambda *args, **kwargs: _FakeProcess(),
                )

    def test_missing_rank_after_startup_grace_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rank_dir = Path(directory) / "ranks"
            rank_dir.mkdir()
            (rank_dir / "rank-0.started.json").write_text("{}", encoding="utf-8")
            with patch.object(guard.time, "monotonic", return_value=guard.STARTUP_GRACE_SECONDS + 1):
                reason = guard._stale_rank(rank_dir, expected_world_size=4, started_at=0)
            self.assertIn("only 1 of 4", reason or "")

    def test_no_rank_after_startup_grace_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rank_dir = Path(directory) / "ranks"
            rank_dir.mkdir()
            with patch.object(guard.time, "monotonic", return_value=guard.STARTUP_GRACE_SECONDS + 1):
                reason = guard._stale_rank(rank_dir, expected_world_size=4, started_at=0)
            self.assertIn("only 0 of 4", reason or "")

    def test_inference_readiness_phase_has_bounded_pretrainer_grace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_dir = root / "ranks"
            phase = root / "launch-phase.json"
            rank_dir.mkdir()
            phase.write_text(json.dumps({
                "phase": "inference_readiness_wait",
                "recorded_at_unix": 1000.0,
            }), encoding="utf-8")
            with patch.object(guard.time, "time", return_value=1200.0), patch.object(
                guard.time, "monotonic", return_value=guard.STARTUP_GRACE_SECONDS + 1
            ):
                self.assertIsNone(guard._stale_rank(
                    rank_dir, expected_world_size=4, started_at=0, phase_file=phase,
                ))
            with patch.object(guard.time, "time", return_value=1601.0), patch.object(
                guard.time, "monotonic", return_value=guard.STARTUP_GRACE_SECONDS + 1
            ):
                reason = guard._stale_rank(
                    rank_dir, expected_world_size=4, started_at=0, phase_file=phase,
                )
            self.assertIn("only 0 of 4", reason or "")
            self.assertIn("inference_readiness_wait", reason or "")

    def test_trainer_process_phase_resets_to_short_rank_grace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_dir = root / "ranks"
            phase = root / "launch-phase.json"
            rank_dir.mkdir()
            phase.write_text(json.dumps({
                "phase": "trainer_process_started",
                "recorded_at_unix": 1000.0,
            }), encoding="utf-8")
            with patch.object(guard.time, "time", return_value=1100.0), patch.object(
                guard.time, "monotonic", return_value=guard.PRETRAINER_STARTUP_GRACE_SECONDS + 1
            ):
                self.assertIsNone(guard._stale_rank(
                    rank_dir, expected_world_size=4, started_at=0, phase_file=phase,
                ))
            with patch.object(guard.time, "time", return_value=1121.0), patch.object(
                guard.time, "monotonic", return_value=guard.PRETRAINER_STARTUP_GRACE_SECONDS + 1
            ):
                reason = guard._stale_rank(
                    rank_dir, expected_world_size=4, started_at=0, phase_file=phase,
                )
            self.assertIn("only 0 of 4", reason or "")
            self.assertIn("trainer_process_started", reason or "")

    def test_trainer_process_exit_writes_prompt_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_dir = root / "ranks"
            failure = root / "distributed-failure.json"
            process = type("Exited", (), {"pid": 42, "wait": lambda self: 17})()
            with patch.dict(os.environ, {
                "PAINTER_RANK_GUARD_DIR": str(rank_dir),
                "PAINTER_DISTRIBUTED_FAILURE_FILE": str(failure),
                "PAINTER_NUM_TRAIN_GPUS": "4",
            }, clear=False):
                paths = run_wrapper._record_trainer_process_started(["torchrun", str(run_wrapper.TRAIN_WRAPPER)], process)
                self.assertIsNotNone(paths)
                run_wrapper._watch_trainer_process(process, ["torchrun", str(run_wrapper.TRAIN_WRAPPER)], paths)
            value = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "rank_exited")
            self.assertIn("exit code 17", value["reason"])
            self.assertTrue((rank_dir / "trainer-process.started.json").is_file())

    def test_hidden_rank_fatal_log_writes_failure_before_torchrun_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_dir = root / "ranks"
            failure = root / "distributed-failure.json"
            log_root = root / "torchrun"
            rank_log = log_root / "session/attempt_0/2/stdout.log"
            rank_log.parent.mkdir(parents=True)
            rank_log.write_text("Fatal error in train\nFileNotFoundError: kernel.cubin\n", encoding="utf-8")

            class Running:
                pid = 42
                def poll(self):
                    return None

            command = ["torchrun", "--log-dir", str(log_root), str(run_wrapper.TRAIN_WRAPPER)]
            with patch.dict(os.environ, {
                "PAINTER_NUM_TRAIN_GPUS": "4",
            }, clear=False):
                run_wrapper._watch_trainer_process(Running(), command, (rank_dir, failure))
            value = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "rank_failed")
            self.assertEqual(value["rank"], "2")
            self.assertIn("Fatal error in train", value["reason"])

    def test_trainer_cache_environment_requests_process_isolation(self) -> None:
        values = run_wrapper._staged_cache_env(Path("/tmp/run"), "trainer", {})
        self.assertEqual(values["PAINTER_COMPILE_CACHE_PROCESS_ISOLATION"], "1")

    def test_missing_prime_hooks_are_reported_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = resource.inspect_resource_hooks(Path(directory), expected_world_size=4)
        self.assertFalse(report["launch_permitted"])
        self.assertEqual(report["token_length_balance"]["status"], "unsupported_no_safe_hook")
        self.assertEqual(report["text_only_vision_memory"]["status"], "unsupported_no_safe_hook")
        self.assertIn("no verified public hook", report["token_length_balance"]["reason"])

    def test_pinned_prime_token_balancing_is_supported_and_vision_requires_memory_contract(self) -> None:
        prime_root = RUN.parents[2] / "work" / "prime-rl-upstream"
        report = resource.inspect_resource_hooks(prime_root, expected_world_size=4)
        self.assertTrue(report["token_length_balance"]["launch_permitted"])
        self.assertEqual(report["token_length_balance"]["measurement"], "prepare_batch.sequence_lengths")
        self.assertFalse(report["text_only_vision_memory"]["launch_permitted"])
        self.assertFalse(report["launch_permitted"])
        high_memory = resource.inspect_resource_hooks(prime_root, expected_world_size=4, trainer_memory_gib=80)
        self.assertTrue(high_memory["launch_permitted"])
        self.assertEqual(high_memory["trainer_memory_contract"]["status"], "supported")

    def test_real_backward_canary_requires_one_update_checkpoint_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "step_1" / "trainer"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").write_text("metadata", encoding="utf-8")
            (checkpoint / "rank_0.pt").write_text("state", encoding="utf-8")
            adapter = root / "artifacts" / "adapters" / "step_1"
            adapter.mkdir(parents=True)
            adapter_bytes = {
                "adapter_config.json": b'{"peft_type":"LORA"}\n',
                "adapter_model.safetensors": b"adapter-weights",
                "validation.json": b'{"finite":true}\n',
            }
            for name, value in adapter_bytes.items():
                (adapter / name).write_bytes(value)
            digest = lambda value: hashlib.sha256(value).hexdigest()
            remote_files = {
                name: {"sha256": digest(value), "bytes": len(value)}
                for name, value in adapter_bytes.items()
            }
            remote_files["upload-manifest.json"] = {"sha256": "a" * 64, "bytes": 1}
            (adapter / "hf-upload.json").write_text(json.dumps({
                "repo_id": "fixture/repo", "path": "fixture-run/step_1", "commit": "fixture-commit",
                "private": False, "verified": True, "files": remote_files,
            }), encoding="utf-8")
            (adapter / "checkpoint-contract.json").write_text(json.dumps({
                "schema": "staged-family-balanced-checkpoint-contract-v1", "step": 1,
                "native_dcp": {"path": str(checkpoint), "resumable": True},
                "portable_adapter": {
                    "path": str(adapter), "receipt_path": str(adapter / "hf-upload.json"),
                    "repo_id": "fixture/repo", "remote_path": "fixture-run/step_1",
                    "verified": True, "private": False,
                },
            }), encoding="utf-8")
            state = root / "runtime-state.json"
            state.write_text("{}", encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "schema": "staged-family-balanced-real-backward-canary-v1",
                "status": "passed", "optimizer_updates": 1, "checkpoint_produced": True,
                "update_receipt": {"controller_recorded": True, "optimizer_update": 1},
                "update_receipt_path": str(state),
                "checkpoint_path": str(checkpoint), "checkpoint_metadata": str(checkpoint / ".metadata"),
                "checkpoint_file_count": 1, "world_size": 4,
                "elapsed_seconds": 120, "cost_usd": 2.5, "node_hourly_price_usd": 75.0,
                "time_bound_seconds": 2400, "cost_bound_usd": 9.0,
                "cost_source": "node_hourly_price_input",
                "processes_left_running": False,
            }), encoding="utf-8")
            self.assertEqual(canary.validate_receipt(receipt)["optimizer_updates"], 1)
            value = json.loads(receipt.read_text())
            value["cost_usd"] = None
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(canary.RealBackwardCanaryError):
                canary.validate_receipt(receipt)
            value["cost_usd"] = 2.5
            value["time_bound_seconds"] = 2401
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(canary.RealBackwardCanaryError):
                canary.validate_receipt(receipt)

    def test_real_backward_canary_mutates_actual_top_level_trainer_config(self) -> None:
        config = SimpleNamespace(max_steps=20, ckpt=SimpleNamespace(interval=4))
        with patch.dict(os.environ, {"PAINTER_CANARY_MODE": "real_backward"}, clear=False):
            started = train_wrapper._configure_real_backward_canary(config)
        self.assertIsInstance(started, float)
        self.assertEqual(config.max_steps, 1)
        self.assertEqual(config.ckpt.interval, 1)
        self.assertFalse(hasattr(config, "trainer"))

    def test_adapter_only_save_keeps_native_dcp_and_uploads_portable_folder(self) -> None:
        calls: list[tuple] = []

        class CheckpointManager:
            def save(self, step, model, optimizers, scheduler, progress, dataloader=None):
                calls.append((step, model, optimizers, scheduler, progress, dataloader))
                native = export_adapter.native_checkpoint_path(config.run_dir, step)
                native.mkdir(parents=True)
                (native / ".metadata").write_text("dcp", encoding="utf-8")
                (native / "state.pt").write_text("state", encoding="utf-8")

        class Sender:
            def __init__(self, *args):
                pass

            def _broadcast(self, model, step, folder):
                folder.mkdir(parents=True, exist_ok=True)
                for name in ("adapter_config.json", "adapter_model.safetensors", "validation.json"):
                    (folder / name).write_text(name, encoding="utf-8")

        modules = {
            "prime_rl.configs.trainer": types.SimpleNamespace(FileSystemWeightBroadcastConfig=lambda: object()),
            "prime_rl.trainer.ckpt": types.SimpleNamespace(CheckpointManager=CheckpointManager),
            "prime_rl.transports.weights.filesystem": types.SimpleNamespace(FileSystemWeightSender=Sender),
            "prime_rl.trainer.world": types.SimpleNamespace(get_world=lambda: SimpleNamespace(is_master=True)),
            "torch.distributed": types.SimpleNamespace(barrier=lambda: None),
        }
        fake_torch = types.ModuleType("torch")
        fake_torch.__path__ = []
        fake_torch.distributed = modules["torch.distributed"]
        modules["torch"] = fake_torch
        config = SimpleNamespace(
            run_dir=Path(tempfile.mkdtemp()), model=SimpleNamespace(lora=object()), resume=object()
        )
        uploaded: list[Path] = []

        def upload(folder, repo, run):
            uploaded.append(Path(folder))
            receipt = {
                "repo_id": repo, "path": f"{run}/step_1", "url": "https://example.invalid/step_1",
                "verified": True, "private": False,
            }
            (Path(folder) / "hf-upload.json").write_text(json.dumps(receipt), encoding="utf-8")
            return receipt

        with patch.dict(sys.modules, modules), patch.object(
            export_adapter, "cloud_settings", return_value=("fixture/repo", "fixture-run", upload)
        ), patch.object(export_adapter, "validate_files", return_value={"sha256": "adapter"}), patch.dict(
            export_adapter.os.environ, {"ADAPTER_ONLY_CHECKPOINTS": "1"}, clear=False
        ):
            export_adapter.install_export_hook(config)
            CheckpointManager().save(1, "model", [], "scheduler", "progress")

        self.assertEqual(len(calls), 1)
        self.assertEqual(uploaded, [export_adapter.adapter_checkpoint_path(config.run_dir, 1)])
        native = export_adapter.native_checkpoint_path(config.run_dir, 1)
        self.assertTrue((native / ".metadata").is_file())
        contract = json.loads((uploaded[0] / "checkpoint-contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["native_dcp"]["path"], str(native))
        self.assertNotIn("state.pt", json.dumps(contract))


if __name__ == "__main__":
    unittest.main()
