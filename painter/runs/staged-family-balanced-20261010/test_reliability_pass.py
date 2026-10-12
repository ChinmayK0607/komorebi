from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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
            self.assertIn("before rank readiness", value["reason"])
            self.assertTrue((rank_dir / "trainer-process.started.json").is_file())

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
                "cost_source": "node_hourly_price_input",
                "processes_left_running": False,
            }), encoding="utf-8")
            self.assertEqual(canary.validate_receipt(receipt)["optimizer_updates"], 1)
            value = json.loads(receipt.read_text())
            value["cost_usd"] = None
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(canary.RealBackwardCanaryError):
                canary.validate_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
