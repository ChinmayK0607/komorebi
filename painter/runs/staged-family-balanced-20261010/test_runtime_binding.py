"""No-GPU fake-Prime tests for the staged runtime binding."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


curriculum = _load("staged_curriculum")
_load("prime_tail_purge")
binding = _load("runtime_binding")


class FakeData:
    def __init__(self, row_id: str, family: str) -> None:
        self.source_row_id = row_id
        self.family = family


class FakeTask:
    def __init__(self, row_id: str, family: str) -> None:
        self.data = FakeData(row_id, family)
        self.key = f"native:{row_id}"


class FakeTrace:
    def __init__(self, trace_id: str) -> None:
        self.id = trace_id
        self.reward = 0.25
        self.info = {
            "code_valid": True,
            "canvas_valid": True,
            "current_canvas_valid": True,
            "terminal_action_valid": True,
            "reward_status": "scored",
        }


class FakeEpisode:
    def __init__(self, task: FakeTask, trace_id: str) -> None:
        self.task = task
        self.traces = [FakeTrace(trace_id)]
        self.ok = True
        self.error = None


class FakeSample:
    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.token_ids = [1, 2, 3]


class FakeCohort:
    def __init__(self, trace_ids: list[str]) -> None:
        self.sampled_trace_ids = set(trace_ids)


class FakeBatch:
    def __init__(self, trace_ids: list[str]) -> None:
        self.cohort = FakeCohort(trace_ids)
        self.samples = [FakeSample(trace_id) for trace_id in trace_ids]


class FakeEnv:
    name = "paint"

    class Config:
        group_size = 4

    config = Config()


class FakeSampler:
    def __init__(self, tasks: list[FakeTask]) -> None:
        self.tasks = tasks


class FakeCurriculum:
    def __init__(self, tasks: list[FakeTask]) -> None:
        self.sampler = FakeSampler(tasks)


class FakeSource:
    def __init__(self, tasks: list[FakeTask]) -> None:
        self.envs = [FakeEnv()]
        self.curricula = {"paint": FakeCurriculum(tasks)}

    def next_task(self, *, step: int):
        return None


class FakeDispatcher:
    def __init__(self) -> None:
        self.disabled = False
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()
        self.groups = {}
        self.available_permits = 100
        self.scheduling_lock = asyncio.Lock()
        self.policy_update_pending = False
        self.mode = types.SimpleNamespace(name="PREFER_TRAIN")
        self.try_schedule_calls = []

    def admission_budget(self):
        return 100

    async def fill_inflight(self):
        return None

    async def try_schedule(self, kind):
        self.try_schedule_calls.append(kind)
        return False

    async def schedule_group_episode(self, group_id, group):
        if hasattr(group, "episodes_to_schedule") and group.episodes_to_schedule > 0:
            group.episodes_to_schedule -= 1
        return True

    def disable_train_scheduling(self) -> None:
        self.disabled = True


class FakeSink:
    def __init__(self) -> None:
        self.pending_groups = {}
        self.pending_group_failures = {}
        self.pending_group_cancellations = {}
        self.pending_batch = {}
        self.episode_by_trace = {}
        self.batch_size = 24
        self.token_batch_size = None
        self.pending_tokens = 0
        self.internal_batch_results = []

    async def process_group(self, group_id):
        group = self.pending_groups.pop(group_id)
        cancellation = self.pending_group_cancellations.pop(group_id, None)
        self.pending_group_failures.pop(group_id, None)
        if cancellation is not None and cancellation.reason == "stale":
            self.internal_batch_results.append(self._maybe_batch())
            return
        for episode in group:
            trace = episode.traces[0]
            if episode.traces[0].info.get("reward_status") in {"scored", "model_invalid"}:
                self.pending_batch[trace.id] = [FakeSample(trace.id)]
                self.episode_by_trace[trace.id] = episode
        # Pinned TrainSink can reach a batch boundary from inside the group
        # finalizer.  The binding must keep this attempt closed until the
        # complete six-group cycle has a pass receipt.
        self.internal_batch_results.append(self._maybe_batch())

    def process_batch(self):
        ids = list(self.pending_batch)[:24]
        for trace_id in ids:
            del self.pending_batch[trace_id]
            del self.episode_by_trace[trace_id]
        return FakeBatch(ids)

    def _maybe_batch(self):
        return self.process_batch() if len(self.pending_batch) >= 24 else None


class FakeOrchestrator:
    def __init__(self, tasks: list[FakeTask]) -> None:
        self.train_source = FakeSource(tasks)
        self.dispatcher = FakeDispatcher()
        self.train_sink = FakeSink()
        self.progress = types.SimpleNamespace(step=1)
        self.policy = types.SimpleNamespace(version=0)
        self.watcher = FakeWatcher(self)

    def update_dispatch_gate(self):
        self.dispatcher.dispatch_allowed.set()

    async def on_policy_update(self, step):
        self.update_dispatch_gate()


class FakeWatcher:
    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator
        self.hooks = []

    def on_update(self, hook):
        self.hooks.append(hook)

    async def emit(self, step: int):
        self.orchestrator.policy.version = step
        for hook in self.hooks:
            await hook(step)


def _prime_task_request_stub() -> None:
    prime = types.ModuleType("prime_rl")
    orchestrator = types.ModuleType("prime_rl.orchestrator")
    types_module = types.ModuleType("prime_rl.orchestrator.types")

    class TaskRequest:
        def __init__(self, *, env_name, task, step, source_index=None):
            self.env_name = env_name
            self.task = task
            self.step = step
            self.source_index = source_index

    types_module.TaskRequest = TaskRequest
    prime.orchestrator = orchestrator
    orchestrator.types = types_module
    sys.modules.update({
        "prime_rl": prime,
        "prime_rl.orchestrator": orchestrator,
        "prime_rl.orchestrator.types": types_module,
    })


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        _prime_task_request_stub()
        self.plan = curriculum.build_curriculum()
        task_rows = {
            (slot["prompt_row_id"], slot["family"]): slot
            for stage in self.plan["stages"]
            for slot in stage["schedule"]
        }
        tasks = [FakeTask(row_id, family) for row_id, family in sorted(task_rows)]
        self.orchestrator = FakeOrchestrator(tasks)
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = binding.RuntimeBinding(
            orchestrator=self.orchestrator,
            root=ROOT,
            state_path=Path(self.temp.name) / "runtime-state.json",
            manifest=self.plan,
            protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_async_out_of_order_groups_bind_native_uuid_to_exact_cycle(self) -> None:
        requests = [self.orchestrator.train_source.next_task(step=1) for _ in range(6)]
        self.assertEqual([request.task.data.source_row_id for request in requests], [
            slot["prompt_row_id"] for slot in self.plan["stages"][0]["schedule"] if slot["cycle"] == 1
        ])
        self.assertIsNone(self.orchestrator.train_source.next_task(step=1))

        async def run_groups() -> None:
            for index, request in reversed(list(enumerate(requests))):
                native_group_id = f"native-{index}"
                group = [FakeEpisode(request.task, f"trace-{index}-{rollout}") for rollout in range(4)]
                self.orchestrator.train_sink.pending_groups[native_group_id] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    native_group_id, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(native_group_id)

        asyncio.run(run_groups())
        self.assertFalse(self.orchestrator.dispatcher.dispatch_allowed.is_set())
        self.assertEqual(self.runtime.controller.next_cycle, 2)
        self.assertEqual(self.runtime.controller.trace_inventory_by_cycle, {"s1-c01": 24})
        self.assertEqual(set(self.runtime.controller.trace_ids_by_cycle["s1-c01"]), {
            f"trace-{index}-{rollout}" for index in range(6) for rollout in range(4)
        })
        batch = self.orchestrator.train_sink.process_batch()
        self.assertEqual(len(batch.cohort.sampled_trace_ids), 24)
        self.assertEqual(self.runtime.controller.stage_optimizer_updates_observed, 0)
        asyncio.run(self.orchestrator.watcher.emit(0))
        self.assertEqual(self.runtime.controller.stage_optimizer_updates_observed, 0)
        asyncio.run(self.orchestrator.watcher.emit(1))
        self.assertEqual(self.runtime.controller.stage_optimizer_updates_observed, 1)
        saved = json.loads((Path(self.temp.name) / "runtime-state.json").read_text())
        self.assertEqual(saved["schema"], binding.STATE_SCHEMA)
        self.assertEqual(saved["batch_receipts"][0]["trace_count"], 24)
        self.assertEqual(len(saved["update_confirmations"]), 2)

    def test_duplicate_native_trace_id_fails_before_group_is_accounted(self) -> None:
        request = self.orchestrator.train_source.next_task(step=1)

        async def run() -> None:
            native_group_id = "native-duplicate"
            group = [FakeEpisode(request.task, "duplicate") for _ in range(4)]
            self.orchestrator.train_sink.pending_groups[native_group_id] = group
            await self.orchestrator.dispatcher.schedule_group_episode(
                native_group_id, types.SimpleNamespace(task=request.task)
            )
            with self.assertRaises(binding.RuntimeBindingError):
                await self.orchestrator.train_sink.process_group(native_group_id)

        asyncio.run(run())
        self.assertEqual(self.runtime.controller.trace_ids_by_cycle, {})
        self.assertEqual(self.orchestrator.train_sink.pending_batch, {})

    def test_atomic_state_resumes_cycle_and_optimizer_inventory(self) -> None:
        requests = [self.orchestrator.train_source.next_task(step=1) for _ in range(6)]

        async def run_groups() -> None:
            for index, request in enumerate(requests):
                native_group_id = f"native-resume-{index}"
                group = [FakeEpisode(request.task, f"resume-{index}-{rollout}") for rollout in range(4)]
                self.orchestrator.train_sink.pending_groups[native_group_id] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    native_group_id, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(native_group_id)

        asyncio.run(run_groups())
        self.orchestrator.train_sink.process_batch()
        state_path = Path(self.temp.name) / "runtime-state.json"
        task_rows = {
            (slot["prompt_row_id"], slot["family"]): slot
            for stage in self.plan["stages"]
            for slot in stage["schedule"]
        }
        resumed_orchestrator = FakeOrchestrator([FakeTask(row_id, family) for row_id, family in sorted(task_rows)])
        resumed_orchestrator.progress.step = 2
        resumed = binding.RuntimeBinding(
            orchestrator=resumed_orchestrator,
            root=ROOT,
            state_path=state_path,
            manifest=self.plan,
            protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
        )
        self.assertEqual(resumed.controller.next_cycle, 2)
        self.assertEqual(resumed.controller.optimizer_updates_observed, 0)
        self.assertEqual(resumed.dispatch_slots_sent, 0)
        self.assertIsNotNone(resumed.pending_batch_receipt)
        applied_orchestrator = FakeOrchestrator([FakeTask(row_id, family) for row_id, family in sorted(task_rows)])
        applied_orchestrator.policy.version = 1
        applied_orchestrator.progress.step = 2
        applied = binding.RuntimeBinding(
            orchestrator=applied_orchestrator,
            root=ROOT,
            state_path=state_path,
            manifest=self.plan,
            protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
        )
        self.assertEqual(applied.controller.optimizer_updates_observed, 1)
        self.assertIsNone(applied.pending_batch_receipt)

    def _fresh_orchestrator(self) -> FakeOrchestrator:
        task_rows = {
            (slot["prompt_row_id"], slot["family"]): slot
            for stage in self.plan["stages"]
            for slot in stage["schedule"]
        }
        return FakeOrchestrator([FakeTask(row_id, family) for row_id, family in sorted(task_rows)])

    def test_restart_rejects_persisted_partial_cycle_or_dispatch_cursor(self) -> None:
        state_path = Path(self.temp.name) / "runtime-state.json"
        self.runtime.journal.write(
            controller=self.runtime._controller_state(),
            event={"type": "test_partial_cycle"},
            cycle_records={"s1-c01": {"s1-c01-g01": {"group_id": "s1-c01-g01"}}},
            dispatch_slots_sent=0,
        )
        with self.assertRaises(binding.RuntimeBindingError):
            binding.RuntimeBinding(
                orchestrator=self._fresh_orchestrator(), root=ROOT, state_path=state_path,
                manifest=self.plan, protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
            )

        # A clean cycle journal is still unsafe when Prime had issued only a
        # prefix of the native task slots: those buffers are outside the JSON
        # journal and cannot be reconstructed on restart.
        self.runtime.journal.write(
            controller=self.runtime._controller_state(),
            event={"type": "test_partial_dispatch"},
            cycle_records={},
            dispatch_slots_sent=1,
        )
        with self.assertRaises(binding.RuntimeBindingError):
            binding.RuntimeBinding(
                orchestrator=self._fresh_orchestrator(), root=ROOT, state_path=state_path,
                manifest=self.plan, protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
            )

    def test_restart_rejects_admitted_inventory_without_reconciliation(self) -> None:
        state_path = Path(self.temp.name) / "runtime-state.json"
        controller = self.runtime._controller_state()
        controller["trace_ids_by_cycle"] = {"s1-c01": ["orphan-trace"]}
        self.runtime.journal.write(
            controller=controller,
            event={"type": "test_orphaned_inventory"},
            cycle_records={},
            dispatch_slots_sent=0,
        )
        with self.assertRaises(binding.RuntimeBindingError):
            binding.RuntimeBinding(
                orchestrator=self._fresh_orchestrator(), root=ROOT, state_path=state_path,
                manifest=self.plan, protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
            )

    def test_restart_accepts_clean_optimizer_boundary(self) -> None:
        resumed = binding.RuntimeBinding(
            orchestrator=self._fresh_orchestrator(),
            root=ROOT,
            state_path=Path(self.temp.name) / "runtime-state.json",
            manifest=self.plan,
            protocol=json.loads((ROOT / "PROTOCOL.json").read_text()),
        )
        self.assertEqual(resumed.controller.active_stage, 1)
        self.assertEqual(resumed.controller.next_cycle, 1)
        self.assertEqual(resumed.controller.trace_ids_by_cycle, {})
        self.assertEqual(resumed.dispatch_slots_sent, 0)

    def test_pinned_prime_source_shape_is_explicitly_checked(self) -> None:
        prime_root = ROOT.parents[2] / "work" / "prime-rl-upstream"
        sink_source = (prime_root / "src/prime_rl/orchestrator/train_sink.py").read_text()
        source_source = (prime_root / "src/prime_rl/orchestrator/train_source.py").read_text()
        watcher_source = (prime_root / "src/prime_rl/orchestrator/watcher.py").read_text()
        orchestrator_source = (prime_root / "src/prime_rl/orchestrator/orchestrator.py").read_text()
        dispatcher_source = (prime_root / "src/prime_rl/orchestrator/dispatcher.py").read_text()
        self.assertIn("self.pending_batch: dict[str, list[TrainingSample]]", sink_source)
        self.assertIn("self.episode_by_trace: dict[str, vf.Episode]", sink_source)
        self.assertIn("def process_batch(self) -> TrainBatch", sink_source)
        self.assertIn("def _maybe_batch(self) -> TrainBatch | None", sink_source)
        self.assertIn("def next_task(self, *, step: int) -> TaskRequest", source_source)
        self.assertIn("return TaskRequest(env_name=env_name, task=", source_source)
        self.assertIn("def on_update(self, hook", watcher_source)
        self.assertIn("await self._notify_update(step)", watcher_source)
        self.assertIn("def update_dispatch_gate(self)", orchestrator_source)
        self.assertIn("async def on_policy_update(self, _step: int)", orchestrator_source)
        self.assertIn("if not self.dispatch_allowed.is_set():", dispatcher_source)
        self.assertIn("self.pending_episodes = TrainEpisodes()", sink_source)

    def test_failure_ownership_is_explicit_and_model_invalid_is_trainable_zero(self) -> None:
        slot = self.plan["stages"][0]["schedule"][0]
        service = FakeEpisode(FakeTask(slot["prompt_row_id"], slot["family"]), "service-trace")
        service.ok = False
        service.traces[0].info["service_failure"] = "renderer unavailable"
        service_record = binding._native_rollout(service, slot=slot, index=0)
        self.assertEqual(service_record["status"], "service_failure")
        self.assertEqual(service_record["ownership"], "service")
        self.assertIsNone(service_record["reward"])

        invalid = FakeEpisode(FakeTask(slot["prompt_row_id"], slot["family"]), "invalid-trace")
        invalid.traces[0].info["reward_status"] = "model_invalid_canvas"
        invalid.traces[0].info["canvas_valid"] = False
        invalid.traces[0].reward = 0.0
        invalid_record = binding._native_rollout(invalid, slot=slot, index=0)
        self.assertEqual(invalid_record["status"], "model_invalid")
        self.assertEqual(invalid_record["reward"], 0.0)

        ambiguous = FakeEpisode(FakeTask(slot["prompt_row_id"], slot["family"]), "ambiguous-trace")
        ambiguous.ok = False
        with self.assertRaises(binding.RuntimeBindingError):
            binding._native_rollout(ambiguous, slot=slot, index=0)

    def test_terminal_invalid_and_corrected_intermediate_use_native_final_semantics(self) -> None:
        slot = self.plan["stages"][0]["schedule"][0]
        corrected = FakeEpisode(FakeTask(slot["prompt_row_id"], slot["family"]), "corrected-trace")
        corrected.traces[0].info.update({
            "turns": [
                {"turn": 1, "action": "paint", "valid": False},
                {"turn": 2, "action": "paint", "valid": True},
                {"turn": 3, "action": "finish", "valid": True},
            ],
            "final_action_validation": {"valid": True, "status": "valid"},
        })
        record = binding._native_rollout(corrected, slot=slot, index=0)
        self.assertEqual(record["status"], "scored")
        self.assertTrue(record["code_valid"])
        self.assertEqual(record["evidence"]["intermediate_invalid_paint_turns"], 1)

        invalid_final = FakeEpisode(FakeTask(slot["prompt_row_id"], slot["family"]), "invalid-final-action")
        invalid_final.traces[0].info.update({
            "reward_status": "model_invalid_final_action",
            "final_canvas_valid": True,
            "final_action_validation": {"valid": False, "status": "invalid"},
            "turns": [{"turn": 1, "action": "paint", "valid": True}],
        })
        invalid_final.traces[0].reward = 0.0
        record = binding._native_rollout(invalid_final, slot=slot, index=0)
        self.assertEqual(record["status"], "model_invalid")
        self.assertFalse(record["terminal_action_valid"])

    def test_dispatch_failure_is_service_placeholder_and_partial_group_can_pass(self) -> None:
        plan = curriculum.build_curriculum()
        controller = curriculum.StageController(plan)
        slots = [slot for slot in plan["stages"][0]["schedule"] if slot["cycle"] == 1]
        records = []
        for index, slot in enumerate(slots):
            task = FakeTask(slot["prompt_row_id"], slot["family"])
            if index == 0:
                group = [FakeEpisode(task, f"partial-{rollout}") for rollout in range(3)]
                failure = types.SimpleNamespace(error="inference unavailable")
                record = binding._group_receipt(group, slot=slot, failures=[failure])
                record["trainer_candidate_trace_ids"] = [row["trace_id"] for row in record["rollouts"][:3]]
                self.assertEqual(record["rollouts"][3]["ownership"], "service")
                self.assertIsNone(record["rollouts"][3]["reward"])
            else:
                record = binding._group_receipt(
                    [FakeEpisode(task, f"partial-{index}-{rollout}") for rollout in range(4)], slot=slot,
                )
            records.append(record)
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "cycle_complete")
        self.assertEqual(receipt["scored_rollouts"], 23)

    def test_two_episodes_plus_cancellation_and_stale_cancellation_censor(self) -> None:
        plan = curriculum.build_curriculum()
        for cancellation_reason in ("overload", "stale"):
            controller = curriculum.StageController(plan)
            slots = [slot for slot in plan["stages"][0]["schedule"] if slot["cycle"] == 1]
            records = []
            for index, slot in enumerate(slots):
                task = FakeTask(slot["prompt_row_id"], slot["family"])
                if index == 0:
                    cancellation = types.SimpleNamespace(count=2, reason=cancellation_reason)
                    record = binding._group_receipt(
                        [FakeEpisode(task, f"cancel-{cancellation_reason}-{rollout}") for rollout in range(2)],
                        slot=slot, cancellation=cancellation,
                    )
                    record["trainer_candidate_trace_ids"] = [row["trace_id"] for row in record["rollouts"][:2]]
                else:
                    record = binding._group_receipt(
                        [FakeEpisode(task, f"cancel-{cancellation_reason}-{index}-{rollout}") for rollout in range(4)],
                        slot=slot,
                    )
                records.append(record)
            receipt = controller.observe_cycle(records)
            self.assertEqual(receipt["status"], "censored")
            self.assertEqual(receipt["decision"], "retry_cycle")
            self.assertTrue(all(row["status"] == "service_failure" for row in records[0]["rollouts"][2:]))

    def test_stale_cancellation_voids_arrived_rollouts_even_with_zero_count(self) -> None:
        plan = curriculum.build_curriculum()
        controller = curriculum.StageController(plan)
        slots = [slot for slot in plan["stages"][0]["schedule"] if slot["cycle"] == 1]
        records = []
        for index, slot in enumerate(slots):
            task = FakeTask(slot["prompt_row_id"], slot["family"])
            cancellation = types.SimpleNamespace(count=0, reason="stale") if index == 0 else None
            record = binding._group_receipt(
                [FakeEpisode(task, f"stale-zero-{index}-{rollout}") for rollout in range(4)],
                slot=slot,
                cancellation=cancellation,
            )
            if index == 0:
                self.assertTrue(record["pipeline_censored"])
                self.assertTrue(all(row["status"] == "service_failure" for row in record["rollouts"]))
                self.assertTrue(all(row["reward"] is None for row in record["rollouts"]))
            records.append(record)
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "censored")
        self.assertEqual(receipt["decision"], "retry_cycle")

    def test_malformed_group_cancellation_fails_closed(self) -> None:
        slot = self.plan["stages"][0]["schedule"][0]
        task = FakeTask(slot["prompt_row_id"], slot["family"])
        group = [FakeEpisode(task, f"bad-cancel-{index}") for index in range(3)]
        with self.assertRaises(binding.RuntimeBindingError):
            binding._group_receipt(group, slot=slot, cancellation=types.SimpleNamespace(count=1, reason="unknown"))

    def test_candidate_inventory_uses_only_train_sink_ids_for_tail_purge(self) -> None:
        controller = curriculum.StageController(self.plan)
        first_cycle = [slot for slot in self.plan["stages"][0]["schedule"] if slot["cycle"] == 1]
        records = []
        first_candidate_ids = []
        for index, slot in enumerate(first_cycle):
            task = FakeTask(slot["prompt_row_id"], slot["family"])
            group = [FakeEpisode(task, f"candidate-1-{index}-{rollout}") for rollout in range(4)]
            record = binding._group_receipt(group, slot=slot)
            if index == 0:
                first_candidate_ids = [row["trace_id"] for row in record["rollouts"][:3]]
                record["trainer_candidate_trace_ids"] = list(first_candidate_ids)
            records.append(record)
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["scored_rollouts"], 24)
        self.assertEqual(receipt["trainer_traces"], 23)
        self.assertEqual(controller.trace_inventory_by_cycle, {"s1-c01": 23})

        second_records = []
        second_slots = [slot for slot in self.plan["stages"][0]["schedule"] if slot["cycle"] == 2]
        second_ids = []
        for index, slot in enumerate(second_slots):
            task = FakeTask(slot["prompt_row_id"], slot["family"])
            group = [FakeEpisode(task, f"candidate-2-{index}-{rollout}") for rollout in range(4)]
            record = binding._group_receipt(group, slot=slot)
            second_ids.extend(row["trace_id"] for row in record["rollouts"])
            second_records.append(record)
        controller.observe_cycle(second_records)
        controller.stage_optimizer_updates_observed = 3
        controller.optimizer_updates_observed = 3
        update = controller.record_optimizer_update(trace_ids_by_cycle={"s1-c02": second_ids})
        self.assertEqual(update["decision"], "drain_stage_tail")

        sink = FakeSink()
        sink.pending_tokens = len(first_candidate_ids) * 3
        for trace_id in controller.trace_ids_by_cycle["s1-c01"]:
            episode = FakeEpisode(FakeTask("row", "single_form"), trace_id)
            sink.pending_batch[trace_id] = [FakeSample(trace_id)]
            sink.episode_by_trace[trace_id] = episode
        purge = _load("prime_tail_purge").purge_pending_stage_tail(
            sink,
            controller,
            {"s1-c01": list(controller.trace_ids_by_cycle["s1-c01"])},
            prime_revision=curriculum.PRIME_REVISION,
            receipt_id="candidate-tail",
        )
        self.assertEqual(purge["removed_trace_count"], 23)
        self.assertEqual(sink.pending_batch, {})

    def test_stage_transition_waits_for_confirmed_update_and_group_drain(self) -> None:
        batch_ids = [f"confirmed-batch-{index}" for index in range(24)]
        tail_ids = [f"confirmed-tail-{index}" for index in range(6)]
        all_ids = batch_ids + tail_ids
        self.runtime.controller.trace_ids_by_cycle = {"s1-c01": list(all_ids)}
        self.runtime.controller.completed_cycle_ids = ["s1-c01"]
        self.runtime.controller.stage_optimizer_updates_observed = 3
        self.runtime.controller.optimizer_updates_observed = 3
        sink = FakeSink()
        sink.pending_tokens = len(tail_ids) * 3
        for trace_id in tail_ids:
            episode = FakeEpisode(FakeTask("row", "single_form"), trace_id)
            sink.pending_batch[trace_id] = [FakeSample(trace_id)]
            sink.episode_by_trace[trace_id] = episode
        self.runtime.confirmed_policy_update = {
            "policy_version": 4,
            "trace_ids_by_cycle": {"s1-c01": list(batch_ids)},
        }
        self.runtime.active_group_ids.add("old-native-group")
        self.runtime._run_stage_transition_if_ready(sink)
        self.assertEqual(self.runtime.controller.stage_optimizer_updates_observed, 3)
        self.assertEqual(self.runtime.controller.active_stage, 1)
        self.runtime.active_group_ids.clear()
        self.runtime._run_stage_transition_if_ready(sink)
        self.assertEqual(self.runtime.controller.active_stage, 2)
        self.assertEqual(sink.pending_batch, {})
        self.assertFalse(self.runtime.controller.halted)
        state = json.loads((Path(self.temp.name) / "runtime-state.json").read_text())
        self.assertEqual(len(state["purge_receipts"]), 1)
        self.assertIsNone(state["dispatch_group_limit"])

    def test_binding_gate_drains_existing_group_without_fresh_dispatch(self) -> None:
        dispatcher = self.orchestrator.dispatcher
        self.runtime.binding_gate_open = False
        self.runtime.prime_lead_permit = True
        dispatcher.dispatch_allowed.clear()
        dispatcher.groups["existing"] = types.SimpleNamespace(kind="train", episodes_to_schedule=1)
        asyncio.run(self.runtime._fill_inflight(dispatcher))
        self.assertEqual(dispatcher.try_schedule_calls, ["train"])
        dispatcher.try_schedule_calls.clear()
        dispatcher.groups.clear()
        asyncio.run(self.runtime._fill_inflight(dispatcher))
        self.assertEqual(dispatcher.try_schedule_calls, [])

    def test_sixth_group_closes_gate_on_its_fourth_native_rollout(self) -> None:
        requests = [self.orchestrator.train_source.next_task(step=1) for _ in range(6)]
        sixth = requests[-1]
        group = types.SimpleNamespace(task=sixth.task, episodes_to_schedule=4)

        async def schedule_four() -> None:
            for index in range(4):
                await self.orchestrator.dispatcher.schedule_group_episode("native-sixth", group)
                if index < 3:
                    self.assertTrue(self.orchestrator.dispatcher.dispatch_allowed.is_set())

        asyncio.run(schedule_four())
        self.assertEqual(group.episodes_to_schedule, 0)
        self.assertFalse(self.orchestrator.dispatcher.dispatch_allowed.is_set())

    def test_partial_carryover_refills_only_the_needed_group_prefix(self) -> None:
        issued_rows = []
        first_cycle_pending_ids = []
        second_cycle_first_group_ids = []

        async def run_cycle(cycle: int, partial_groups = set()):
            emitted_batch = None
            requests = []
            for _ in range(6):
                request = self.orchestrator.train_source.next_task(step=1)
                if request is None:
                    break
                requests.append(request)
            issued_rows.extend(request.task.data.source_row_id for request in requests)
            for index, request in enumerate(requests):
                native_group_id = f"carry-{cycle}-{index}"
                group = [FakeEpisode(request.task, f"carry-{cycle}-{index}-{rollout}") for rollout in range(4)]
                if cycle == 2 and index == 0:
                    second_cycle_first_group_ids.extend(episode.traces[0].id for episode in group)
                uncertain_count = partial_groups.get(index, 1) if isinstance(partial_groups, dict) and index in partial_groups else 1 if index in partial_groups else 0
                for episode in group[:uncertain_count]:
                    episode.traces[0].info["reward_status"] = "judge_uncertain"
                self.orchestrator.train_sink.pending_groups[native_group_id] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    native_group_id, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(native_group_id)
                maybe = self.orchestrator.train_sink._maybe_batch()
                if maybe is not None:
                    emitted_batch = maybe
                if cycle == 2 and index == 0:
                    self.assertIsNotNone(maybe)
            return emitted_batch

        asyncio.run(run_cycle(1, {0, 1, 2, 3}))
        self.assertEqual(len(self.orchestrator.train_sink.pending_batch), 20)
        first_cycle_pending_ids.extend(self.orchestrator.train_sink.pending_batch)
        batch = asyncio.run(run_cycle(2))
        self.assertEqual(len(issued_rows), len(set(issued_rows)))
        self.assertEqual(len(issued_rows), 7)
        rotated = curriculum.ordered_cycle_slots(self.plan["stages"][0], 2)
        self.assertEqual(issued_rows[-1], rotated[0]["prompt_row_id"])
        plans = [event for event in self.runtime.journal.state["events"] if event["type"] == "cycle_dispatch_plan"]
        self.assertEqual([event["group_limit"] for event in plans], [6, 1])
        self.assertIsNone(self.runtime.journal.state["dispatch_group_limit"])
        self.assertTrue(any(
            receipt.get("status") == "partial_refill_complete"
            for receipt in self.runtime.journal.state.get("cycle_receipts", [])
        ))
        self.assertEqual(len(self.runtime.controller.completed_cycle_ids), 2)
        self.assertIsNotNone(batch)
        self.assertEqual(
            set(batch.cohort.sampled_trace_ids),
            set(first_cycle_pending_ids + second_cycle_first_group_ids),
        )
        self.assertTrue(all(result is None for result in self.orchestrator.train_sink.internal_batch_results[:-1]))
        self.assertEqual(self.runtime.controller.optimizer_updates_observed, 0)
        asyncio.run(self.orchestrator.watcher.emit(1))
        self.assertEqual(self.runtime.controller.optimizer_updates_observed, 1)

    def test_stale_group_cancellation_never_enters_pending_batch(self) -> None:
        requests = [self.orchestrator.train_source.next_task(step=1) for _ in range(6)]

        async def run_groups() -> None:
            for index, request in enumerate(requests):
                native_group_id = f"stale-runtime-{index}"
                group = [FakeEpisode(request.task, f"stale-runtime-{index}-{rollout}") for rollout in range(4)]
                cancellation = None
                if index == 0:
                    group = group[:3]
                    cancellation = types.SimpleNamespace(count=1, reason="stale")
                    self.orchestrator.train_sink.pending_group_cancellations[native_group_id] = cancellation
                self.orchestrator.train_sink.pending_groups[native_group_id] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    native_group_id, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(native_group_id)

        asyncio.run(run_groups())
        self.assertEqual(self.runtime.controller.next_cycle, 1)
        self.assertEqual(self.runtime.controller.completed_cycle_ids, [])
        self.assertEqual(self.orchestrator.train_sink.pending_batch, {})
        self.assertTrue(all(result is None for result in self.orchestrator.train_sink.internal_batch_results))

    def test_early_carryover_is_not_shipped_when_later_cycle_is_censored(self) -> None:
        async def run_cycle(cycle: int, uncertain_by_group: dict[int, int]):
            emitted = None
            requests = []
            for _ in range(6):
                request = self.orchestrator.train_source.next_task(step=1)
                if request is None:
                    break
                requests.append(request)
            for index, request in enumerate(requests):
                group = [FakeEpisode(request.task, f"censor-carry-{cycle}-{index}-{rollout}") for rollout in range(4)]
                for episode in group[:uncertain_by_group.get(index, 0)]:
                    episode.traces[0].info["reward_status"] = "judge_uncertain"
                gid = f"censor-carry-{cycle}-{index}"
                self.orchestrator.train_sink.pending_groups[gid] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    gid, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(gid)
                maybe = self.orchestrator.train_sink._maybe_batch()
                if maybe is not None:
                    emitted = maybe
            return emitted

        asyncio.run(run_cycle(1, {0: 1, 1: 1, 2: 1, 3: 1}))
        self.assertEqual(len(self.orchestrator.train_sink.pending_batch), 20)
        emitted = asyncio.run(run_cycle(2, {0: 2}))
        self.assertIsNone(emitted)
        self.assertEqual(self.runtime.controller.next_cycle, 2)
        self.assertEqual(len(self.runtime.controller.completed_cycle_ids), 1)
        self.assertEqual(len(self.orchestrator.train_sink.pending_batch), 20)
        self.assertIsNone(self.runtime.pending_batch_receipt)

    def test_censored_cycle_purges_other_groups_before_retry(self) -> None:
        requests = [self.orchestrator.train_source.next_task(step=1) for _ in range(6)]

        async def run_groups() -> None:
            for index, request in enumerate(requests):
                native_group_id = f"censored-{index}"
                group = [FakeEpisode(request.task, f"censored-{index}-{rollout}") for rollout in range(4)]
                if index == 0:
                    for episode in group[:2]:
                        episode.traces[0].info["reward_status"] = "judge_uncertain"
                self.orchestrator.train_sink.pending_groups[native_group_id] = group
                await self.orchestrator.dispatcher.schedule_group_episode(
                    native_group_id, types.SimpleNamespace(task=request.task, episodes_to_schedule=0)
                )
                await self.orchestrator.train_sink.process_group(native_group_id)

        asyncio.run(run_groups())
        self.assertEqual(self.runtime.controller.next_cycle, 1)
        self.assertEqual(self.runtime.cycle_records, {})
        self.assertEqual(self.orchestrator.train_sink.pending_batch, {})
        state = json.loads((Path(self.temp.name) / "runtime-state.json").read_text())
        self.assertTrue(any(item["type"] == "censored_cycle_pending_purged" for item in state["events"]))


if __name__ == "__main__":
    unittest.main()
