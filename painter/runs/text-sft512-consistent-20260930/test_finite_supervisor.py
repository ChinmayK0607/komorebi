import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("finite_supervisor.py")
SPEC = importlib.util.spec_from_file_location("finite_text_supervisor", MODULE_PATH)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)


def _budget(*, seconds=120.0, reserve=0.0):
    return supervisor.make_budget(
        price_usd_per_hour=3600.0,
        rental_start_epoch=time.time(),
        max_gpu_usd=seconds,
        reserve_seconds=reserve,
    )


def _writer_command(path: Path, rewards: list[float], *, pause: float = 0.0) -> list[str]:
    code = """
import json, sys, time
path, pause = sys.argv[1], float(sys.argv[2])
with open(path, 'w', encoding='utf-8') as stream:
    for step, reward in enumerate(json.loads(sys.argv[3]), 1):
        stream.write(json.dumps({'step': step, 'train/agg/effective/agent/reward/mean': reward}) + '\\n')
        stream.write(json.dumps({'step': step, 'optim/grad_norm': 1.0, 'time/step': 1.0}) + '\\n')
        stream.flush()
        if pause:
            time.sleep(pause)
time.sleep(30)
"""
    return [sys.executable, "-c", code, str(path), str(pause), json.dumps(rewards)]


class FiniteSupervisorTests(unittest.TestCase):
    @staticmethod
    def _clocked_supervision(
        root: Path,
        *,
        initial_metric_timeout_seconds=None,
        metric_timeout_seconds=1800.0,
        writer=None,
    ):
        clock = [1000.0]
        stopped = []

        class Process:
            pid = 321

            def poll(self):
                return None

        process = Process()

        class Owner:
            pid = process.pid
            pgid = process.pid

            def stop(self, *, grace_seconds):
                stopped.append(clock[0])

        owner = Owner()

        def now():
            return clock[0]

        def sleep(seconds):
            clock[0] += seconds
            if writer is not None:
                writer(clock[0])

        with mock.patch.object(supervisor, "OwnedProcessGroup", lambda _process: owner):
            code = supervisor.supervise(
                command=["fake-training"],
                metrics_path=root / "metrics.jsonl",
                state_path=root / "state.json",
                trend_path=root / "trend.jsonl",
                budget=supervisor.make_budget(
                    price_usd_per_hour=3600.0,
                    rental_start_epoch=1000.0,
                    max_gpu_usd=10000.0,
                    reserve_seconds=0.0,
                ),
                poll_seconds=100.0,
                metric_timeout_seconds=metric_timeout_seconds,
                initial_metric_timeout_seconds=initial_metric_timeout_seconds,
                shutdown_grace_seconds=0.0,
                now=now,
                sleep=sleep,
                popen=lambda *_args, **_kwargs: process,
            )
        return code, stopped, json.loads((root / "state.json").read_text())

    def test_admitted_batch_is_distinct_from_completed_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            rows = []
            for step in range(1, 16):
                rows.append(json.dumps({
                    "step": step,
                    "train/agg/effective/agent/reward/mean": 0.25,
                }))
                rows.append(json.dumps({
                    "step": step, "producer": "orch", "time/step": 99.0,
                }))
                if step < 15:
                    rows.append(json.dumps({
                        "step": step, "producer": "trainer",
                        "optim/grad_norm": 1.0, "time/step": 1.0,
                    }))
            path.write_text("\n".join(rows) + "\n")
            self.assertEqual(supervisor.contiguous_step(supervisor.read_optimizer_rewards(path)), 15)
            self.assertEqual(
                max(supervisor.read_completed_optimizer_steps(path)), 14,
                "admitted batch 15 must not be reported as completed optimizer step 15",
            )

    def test_partial_jsonl_tail_is_ignored_until_newline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            complete = json.dumps({
                "step": 1,
                "train/agg/effective/agent/reward/mean": 0.25,
            })
            path.write_text(complete + "\n{\"step\": 2, \"train/agg/effective/agent/reward/mean\":")
            self.assertEqual(supervisor.read_optimizer_rewards(path), {1: 0.25})
            path.write_text(path.read_text() + " 0.5}\n")
            self.assertEqual(supervisor.read_optimizer_rewards(path), {1: 0.25, 2: 0.5})

    def test_complete_malformed_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text("{not-json}\n")
            with self.assertRaises(supervisor.SupervisorError):
                supervisor.read_optimizer_rewards(path)

    def test_gpu_dollar_cap_is_separate_from_update_limit(self):
        budget = supervisor.make_budget(
            price_usd_per_hour=2.5,
            rental_start_epoch=1000,
            max_gpu_usd=20,
            reserve_seconds=60,
        )
        self.assertEqual(budget.as_dict()["gpu_rental_cap_usd"], 20)
        self.assertNotIn("max_total_usd", budget.as_dict())
        self.assertAlmostEqual(budget.rental_deadline_epoch, 29800)
        with self.assertRaises(ValueError):
            supervisor.supervise(
                command=[sys.executable, "-c", "pass"],
                metrics_path=Path("missing"),
                state_path=Path("state"),
                trend_path=Path("trend"),
                budget=budget,
                max_updates=51,
            )

    def test_trend_sampler_accepts_rising_monotonic_rewards(self):
        rewards = {step: step / 100 for step in range(1, 16)}
        trend = supervisor.reward_trend(rewards, 15)
        self.assertEqual(trend["prior_steps"], [6, 10])
        self.assertEqual(trend["latest_steps"], [11, 15])
        self.assertTrue(trend["rising"])
        self.assertGreater(trend["delta"], 0)

    def test_falling_rewards_stop_at_first_preregistered_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = root / "metrics.jsonl"
            state = root / "state.json"
            trend = root / "trend.jsonl"
            rewards = [1.0 - step / 100 for step in range(1, 16)]
            code = supervisor.supervise(
                command=_writer_command(metrics, rewards),
                metrics_path=metrics,
                state_path=state,
                trend_path=trend,
                budget=_budget(),
                max_updates=50,
                poll_seconds=0.01,
                metric_timeout_seconds=2,
                shutdown_grace_seconds=0,
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(state.read_text())["reason"], "reward_not_rising_at_15")
            self.assertEqual(len(trend.read_text().splitlines()), 1)

    def test_oscillating_rewards_are_a_plateau_and_stop(self):
        pattern = [0.2, 0.8, 0.2, 0.8, 0.2]
        rewards = {step: pattern[(step - 1) % len(pattern)] for step in range(1, 16)}
        trend = supervisor.reward_trend(rewards, 15)
        self.assertAlmostEqual(trend["delta"], 0.0)
        self.assertFalse(trend["rising"])
        self.assertEqual(trend["decision"], "stop")

    def test_missing_metric_fails_with_a_bounded_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            started = time.monotonic()
            state = root / "state.json"
            code = supervisor.supervise(
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
                metrics_path=root / "missing.jsonl",
                state_path=state,
                trend_path=root / "trend.jsonl",
                budget=_budget(),
                poll_seconds=0.01,
                metric_timeout_seconds=0.08,
                shutdown_grace_seconds=0,
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(state.read_text())["reason"], "missing_metric_timeout")
            self.assertLess(time.monotonic() - started, 2.0)

    def test_initial_metric_window_allows_slow_first_update_without_heartbeat_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            step = [0]
            observed_at_1900 = []

            def rollout_heartbeat(_now):
                step[0] += 1
                if _now == 1900.0:
                    observed_at_1900.append(True)
                with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "step": step[0],
                        "train/agg/effective/agent/reward/mean": 0.5,
                    }) + "\n")

            code, stopped, state = self._clocked_supervision(
                root,
                initial_metric_timeout_seconds=3600.0,
                writer=rollout_heartbeat,
            )
            self.assertEqual(code, 1)
            self.assertEqual(state["reason"], "missing_metric_timeout")
            self.assertEqual(len(stopped), 1)
            self.assertEqual(observed_at_1900, [True])
            self.assertEqual(stopped[0] - state["started_epoch"], 3600.0)

    def test_default_initial_metric_window_remains_1800_seconds(self):
        with tempfile.TemporaryDirectory() as temp:
            code, stopped, state = self._clocked_supervision(Path(temp))
            self.assertEqual(code, 1)
            self.assertEqual(state["reason"], "missing_metric_timeout")
            self.assertEqual(stopped[0] - state["started_epoch"], 1800.0)

    def test_optimizer_completion_switches_to_existing_1800_second_stall_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            written = [False]

            def first_optimizer_update(_now):
                if not written[0]:
                    written[0] = True
                    with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "step": 1,
                            "producer": "trainer",
                            "optim/grad_norm": 1.0,
                        }) + "\n")

            code, stopped, state = self._clocked_supervision(
                root,
                initial_metric_timeout_seconds=3600.0,
                metric_timeout_seconds=1800.0,
                writer=first_optimizer_update,
            )
            self.assertEqual(code, 1)
            self.assertEqual(state["reason"], "missing_metric_timeout")
            self.assertEqual(len(stopped), 1)
            self.assertEqual(stopped[0], 2900.0)

    def test_initial_metric_timeout_rejects_nonpositive_or_overbound_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kwargs = {
                "command": ["fake-training"],
                "metrics_path": root / "metrics.jsonl",
                "state_path": root / "state.json",
                "trend_path": root / "trend.jsonl",
                "budget": _budget(seconds=10000),
                "popen": lambda *_args, **_kwargs: self.fail("invalid timeout launched training"),
            }
            for value in (0.0, -1.0, 3600.1, float("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    supervisor.supervise(**kwargs, initial_metric_timeout_seconds=value)

    def test_finite_elapsed_budget_stops_before_rental_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            started = time.time()
            budget = supervisor.make_budget(
                price_usd_per_hour=3600.0,
                rental_start_epoch=started,
                max_gpu_usd=1.0,
                reserve_seconds=0.85,
            )
            state = root / "state.json"
            code = supervisor.supervise(
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
                metrics_path=root / "missing.jsonl",
                state_path=state,
                trend_path=root / "trend.jsonl",
                budget=budget,
                poll_seconds=0.01,
                metric_timeout_seconds=30,
                shutdown_grace_seconds=0,
            )
            self.assertEqual(code, 0)
            result = json.loads(state.read_text())
            self.assertEqual(result["reason"], "finite_cost_deadline")
            self.assertLess(result["finished_epoch"], budget.rental_deadline_epoch)

    def test_process_ownership_refuses_foreign_group(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        owner = supervisor.OwnedProcessGroup(process)
        original = supervisor.os.getpgid
        try:
            supervisor.os.getpgid = lambda pid: owner.pgid + 1  # type: ignore[assignment]
            with self.assertRaises(supervisor.SupervisorError):
                owner.stop(grace_seconds=0)
            self.assertIsNone(process.poll())
        finally:
            supervisor.os.getpgid = original  # type: ignore[assignment]
            os.killpg(owner.pgid, signal.SIGKILL)
            process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
