"""No-Prime tests for the whole-group consistent-judge admission guard."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from painter.consistent_group_admission import (
    CONSISTENT_PROTOCOL,
    WholeGroupAdmissionGuard,
    check_group,
    install_from_environment,
)


def _group(size: int = 8, *, trace_info: dict | None = None, bad_index: int | None = None):
    rows = []
    for index in range(size):
        trace = SimpleNamespace(
            id=f"trace-{index}",
            agent=SimpleNamespace(trainable=True),
            has_error=index == bad_index,
            error=None,
            info=dict(trace_info or {}),
        )
        rows.append(SimpleNamespace(id=f"episode-{index}", ok=True, error=None, traces=[trace]))
    return rows


def _group_with_uncertain(size: int = 4, uncertain: tuple[int, ...] = (3,)):
    group = _group(size)
    for index in uncertain:
        group[index].traces[0].info.update(
            reward_status="judge_uncertain", final_canvas_score=None
        )
    return group


class ConsistentGroupAdmissionTests(unittest.TestCase):
    def test_partial_group_lengths_are_bounded_and_observed(self):
        observed = []

        def original(group):
            observed.append(group)
            return True

        guard = WholeGroupAdmissionGuard(original, group_size=8)
        self.assertTrue(guard(_group(7)))
        self.assertEqual(guard.last_check.scored_size, 7)
        self.assertFalse(guard(_group(9)))
        self.assertEqual(guard.last_check.reason, "wrong_group_size")
        self.assertEqual(len(observed), 2)
        self.assertEqual(guard.stats()["groups_observed"], 2)
        self.assertEqual(guard.stats()["groups_rejected"], 1)
        self.assertEqual(guard.stats()["rejections_by_reason"], {"wrong_group_size": 1})

    def test_three_scored_members_are_admitted_as_a_partial_group(self):
        observed = []
        guard = WholeGroupAdmissionGuard(
            lambda group: observed.append(group) or True, group_size=4
        )
        group = _group(3)
        self.assertTrue(guard(group))
        self.assertEqual(guard.last_check.scored_size, 3)
        self.assertEqual(len(observed[0]), 3)

    def test_scored_siblings_are_retained_and_uncertain_trace_is_excluded(self):
        observed = []
        guard = WholeGroupAdmissionGuard(
            lambda group: observed.append(group) or True, group_size=4
        )
        group = _group_with_uncertain()
        self.assertTrue(guard(group))
        self.assertEqual(guard.last_check.scored_size, 3)
        self.assertIn("trace-3:uncertain_reward", guard.last_check.excluded_members)
        self.assertEqual(len(observed[0]), 4)
        self.assertFalse(group[3].traces[0].agent.trainable)
        self.assertEqual(
            group[3].traces[0].info["consistent_group_admission"],
            {"status": "excluded_uncertain_trace", "reason": "uncertain_reward"},
        )

    def test_fewer_than_three_scored_members_are_rejected(self):
        observed = []
        guard = WholeGroupAdmissionGuard(
            lambda group: observed.append(group) or True, group_size=4
        )
        self.assertFalse(check_group(_group(2), group_size=4).valid)
        group = _group_with_uncertain(uncertain=(2, 3))
        self.assertFalse(guard(group))
        self.assertEqual(guard.last_check.reason, "insufficient_scored_members")
        self.assertEqual(guard.last_check.scored_size, 2)
        self.assertEqual(len(observed), 1)

    def test_telemetry_receipt_lists_scored_and_excluded_trace_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = f"{temp}/groups.jsonl"
            with patch.dict(
                "os.environ", {"PAINTER_CONSISTENT_GROUP_TELEMETRY": path}
            ):
                guard = WholeGroupAdmissionGuard(lambda _group: True, group_size=4)
                self.assertTrue(guard(_group_with_uncertain()))
            with open(path, encoding="utf-8") as stream:
                record = json.loads(stream.readline())
        self.assertEqual(record["scored_size"], 3)
        self.assertEqual(record["minimum_scored_size"], 3)
        self.assertEqual(len(record["scored_members"]), 3)
        self.assertEqual(record["excluded_members"], ["trace-3:uncertain_reward"])

    def test_one_error_in_eight_keeps_scored_siblings(self):
        observed = []
        guard = WholeGroupAdmissionGuard(lambda group: observed.append(group) or True, group_size=8)
        result = guard(_group(8, bad_index=3))
        self.assertTrue(result)
        self.assertIsNone(guard.last_check.reason)
        self.assertIn("trace-3:trace_error", guard.last_check.excluded_members)
        self.assertEqual(guard.last_check.scored_size, 7)
        self.assertEqual(len(observed), 1)
        self.assertEqual(guard.stats()["groups_admitted"], 1)

    def test_uncertain_or_null_reward_rejects_without_zeroing(self):
        for info in (
            {"reward_status": "judge_uncertain", "final_canvas_score": None},
            {"score_evidence": {"status": "uncertain"}, "final_canvas_score": None},
        ):
            guard = WholeGroupAdmissionGuard(lambda _group: True, group_size=8)
            self.assertFalse(guard(_group(8, trace_info=info)))
            self.assertEqual(guard.last_check.reason, "insufficient_scored_members")

    def test_valid_group_with_genuine_zero_and_model_invalid_canvas_is_delegated(self):
        observed = []

        def original(group):
            observed.append(group)
            return True

        guard = WholeGroupAdmissionGuard(original, group_size=8)
        group = _group(8, trace_info={
            "reward_status": "model_invalid_canvas",
            "final_canvas_score": 0.0,
        })
        self.assertTrue(guard(group))
        self.assertTrue(guard.last_check.valid)
        self.assertEqual(len(observed), 1)

    def test_original_admission_decision_is_preserved_for_valid_group(self):
        observed = []
        guard = WholeGroupAdmissionGuard(
            lambda group: observed.append(group) or False,
            group_size=8,
        )
        self.assertFalse(guard(_group(8)))
        self.assertEqual(len(observed), 1)

    def test_episode_error_and_missing_trace_are_rejected(self):
        group = _group(4)
        group[0].ok = False
        group[1].traces = []
        group[2].traces = []
        self.assertFalse(check_group(group).valid)

    def test_group_advantages_are_centered_once_and_survive_batch_observation(self):
        class FakeAlgorithm:
            calls = 0

            async def finalize_episode(self, _episode):
                return None

            async def finalize_group(self, group):
                type(self).calls += 1
                traces = [
                    trace
                    for episode in group
                    for trace in episode.traces
                    if trace.agent.trainable
                ]
                rewards = [trace.info["reward"] for trace in traces]
                mean = sum(rewards) / len(rewards)
                for trace in traces:
                    trace.advantage = trace.info["reward"] - mean

        class FakeTrainSource:
            observed = []

            def on_result(self, group):
                type(self).observed.append(group)
                return True

        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
            "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
            "prime_rl.orchestrator.algo.base": types.ModuleType("prime_rl.orchestrator.algo.base"),
            "prime_rl.orchestrator.train_source": types.ModuleType("prime_rl.orchestrator.train_source"),
        }
        modules["prime_rl.orchestrator.algo.base"].Algorithm = FakeAlgorithm
        modules["prime_rl.orchestrator.train_source"].TrainSource = FakeTrainSource
        with patch.dict(sys.modules, modules):
            from painter.consistent_group_admission import install_prime_guard

            result = install_prime_guard(group_size=4)
            self.assertTrue(result["installed"])
            group = _group_with_uncertain()
            for index, reward in enumerate((0.0, 1.0, 0.5, None)):
                group[index].traces[0].info["reward"] = reward
            algorithm = FakeAlgorithm()
            asyncio.run(algorithm.finalize_group(group))
            advantages = [
                episode.traces[0].advantage
                for episode in group[:3]
            ]
            self.assertEqual(advantages, [-0.5, 0.5, 0.0])
            self.assertFalse(hasattr(group[3].traces[0], "advantage"))
            source = FakeTrainSource()
            self.assertTrue(source.on_result(group))
            self.assertEqual(FakeAlgorithm.calls, 1)
            self.assertEqual(
                [episode.traces[0].advantage for episode in group[:3]], advantages
            )
            self.assertEqual(len(FakeTrainSource.observed[0]), 4)

    def test_episode_guard_precedes_prime_survivor_snapshot_and_sample_compilation(self):
        class FakeAlgorithm:
            async def finalize_episode(self, _episode):
                return None

            async def finalize_group(self, group):
                traces = [
                    trace
                    for episode in group
                    for trace in episode.traces
                    if trace.agent.trainable
                ]
                mean = sum(trace.info["reward"] for trace in traces) / len(traces)
                for trace in traces:
                    trace.advantage = trace.info["reward"] - mean

        class FakeTrainSource:
            def on_result(self, _group):
                return True

        class FakeTrainSink:
            def __init__(self, algorithm):
                self.algorithm = algorithm
                self.survivors_before_finalize = []
                self.compiled_trace_ids = []

            async def process_group(self, group):
                # This is the ordering in pinned Prime 26b3131: the local
                # survivor snapshot precedes Algorithm.finalize_group().
                survivors = [
                    trace
                    for episode in group
                    for trace in episode.traces
                    if trace.agent.trainable
                ]
                self.survivors_before_finalize = [trace.id for trace in survivors]
                await self.algorithm.finalize_group(group)
                for trace in survivors:
                    if not hasattr(trace, "advantage"):
                        raise AssertionError(
                            f"unscored trace reached sample compilation: {trace.id}"
                        )
                    self.compiled_trace_ids.append(trace.id)

        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
            "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
            "prime_rl.orchestrator.algo.base": types.ModuleType("prime_rl.orchestrator.algo.base"),
            "prime_rl.orchestrator.train_source": types.ModuleType("prime_rl.orchestrator.train_source"),
        }
        modules["prime_rl.orchestrator.algo.base"].Algorithm = FakeAlgorithm
        modules["prime_rl.orchestrator.train_source"].TrainSource = FakeTrainSource
        with patch.dict(sys.modules, modules):
            from painter.consistent_group_admission import install_prime_guard

            result = install_prime_guard(group_size=4)
            self.assertTrue(result["installed"])
            self.assertTrue(result["episode_finalize_guard"])
            group = _group_with_uncertain()
            for index, reward in enumerate((0.0, 1.0, 0.5, None)):
                group[index].traces[0].info["reward"] = reward
            algorithm = FakeAlgorithm()
            for episode in group:
                asyncio.run(algorithm.finalize_episode(episode))
            sink = FakeTrainSink(algorithm)
            asyncio.run(sink.process_group(group))
            self.assertEqual(sink.survivors_before_finalize, ["trace-0", "trace-1", "trace-2"])
            self.assertEqual(sink.compiled_trace_ids, ["trace-0", "trace-1", "trace-2"])
            self.assertNotIn("trace-3", sink.compiled_trace_ids)
            self.assertAlmostEqual(
                sum(group[index].traces[0].advantage for index in range(3)), 0.0
            )
            self.assertFalse(hasattr(group[3].traces[0], "advantage"))

    def test_environment_entrypoint_is_strictly_opt_in(self):
        result = install_from_environment()
        self.assertFalse(result["installed"])
        self.assertFalse(result["enabled"])

    def test_prime_installation_scores_bounded_partial_groups(self):
        class FakeAlgorithm:
            calls = 0

            async def finalize_episode(self, _episode):
                return None

            async def finalize_group(self, _group):
                type(self).calls += 1

        class FakeTrainSource:
            observed = []

            def on_result(self, group):
                type(self).observed.append(group)
                return True

        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
            "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
            "prime_rl.orchestrator.algo.base": types.ModuleType("prime_rl.orchestrator.algo.base"),
            "prime_rl.orchestrator.train_source": types.ModuleType("prime_rl.orchestrator.train_source"),
        }
        modules["prime_rl.orchestrator.algo.base"].Algorithm = FakeAlgorithm
        modules["prime_rl.orchestrator.train_source"].TrainSource = FakeTrainSource
        with patch.dict(sys.modules, modules):
            from painter.consistent_group_admission import install_prime_guard

            result = install_prime_guard(group_size=8)
            self.assertTrue(result["installed"])
            source = FakeTrainSource()
            self.assertTrue(source.on_result(_group(7)))
            self.assertTrue(source.on_result(_group(8)))
            algorithm = FakeAlgorithm()
            asyncio.run(algorithm.finalize_group(_group(7)))
            self.assertEqual(FakeAlgorithm.calls, 1)
            asyncio.run(algorithm.finalize_group(_group(8)))
            self.assertEqual(FakeAlgorithm.calls, 2)
            self.assertEqual(len(FakeTrainSource.observed), 2)

            again = install_prime_guard(group_size=8)
            self.assertFalse(again["installed"])
            self.assertTrue(again["already_installed"])
            self.assertTrue(again["episode_finalize_guard"])

    def test_enabled_environment_entrypoint_installs_with_configured_size(self):
        class FakeAlgorithm:
            async def finalize_episode(self, _episode):
                return None

            async def finalize_group(self, _group):
                return None

        class FakeTrainSource:
            def on_result(self, _group):
                return True

        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
            "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
            "prime_rl.orchestrator.algo.base": types.ModuleType("prime_rl.orchestrator.algo.base"),
            "prime_rl.orchestrator.train_source": types.ModuleType("prime_rl.orchestrator.train_source"),
        }
        modules["prime_rl.orchestrator.algo.base"].Algorithm = FakeAlgorithm
        modules["prime_rl.orchestrator.train_source"].TrainSource = FakeTrainSource
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {
            "PAINTER_JUDGE_PROTOCOL": CONSISTENT_PROTOCOL,
            "PAINTER_WHOLE_GROUP_ADMISSION": "1",
            "PAINTER_WHOLE_GROUP_SIZE": "8",
        }):
            from painter.consistent_group_admission import install_from_environment

            result = install_from_environment()
        self.assertTrue(result["installed"])
        self.assertEqual(result["group_size"], 8)


if __name__ == "__main__":
    unittest.main()
