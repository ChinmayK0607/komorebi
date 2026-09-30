"""No-Prime tests for the whole-group consistent-judge admission guard."""

from __future__ import annotations

import asyncio
import sys
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
            has_error=index == bad_index,
            error=None,
            info=dict(trace_info or {}),
        )
        rows.append(SimpleNamespace(id=f"episode-{index}", ok=True, error=None, traces=[trace]))
    return rows


class ConsistentGroupAdmissionTests(unittest.TestCase):
    def test_variable_group_lengths_are_rejected_and_observed(self):
        observed = []

        def original(group):
            observed.append(group)
            return True

        guard = WholeGroupAdmissionGuard(original, group_size=8)
        self.assertFalse(guard(_group(7)))
        self.assertEqual(guard.last_check.reason, "wrong_group_size")
        self.assertFalse(guard(_group(9)))
        self.assertEqual(len(observed), 2)
        self.assertEqual(guard.stats()["groups_observed"], 2)
        self.assertEqual(guard.stats()["groups_rejected"], 2)
        self.assertEqual(guard.stats()["rejections_by_reason"], {"wrong_group_size": 2})

    def test_one_error_in_eight_rejects_whole_group(self):
        observed = []
        guard = WholeGroupAdmissionGuard(lambda group: observed.append(group) or True, group_size=8)
        result = guard(_group(8, bad_index=3))
        self.assertFalse(result)
        self.assertEqual(guard.last_check.reason, "uncertain_or_error")
        self.assertIn("episode-3:trace_error", guard.last_check.invalid_members)
        self.assertEqual(len(observed), 1)
        self.assertEqual(guard.stats()["groups_rejected"], 1)

    def test_uncertain_or_null_reward_rejects_without_zeroing(self):
        for info in (
            {"reward_status": "judge_uncertain", "final_canvas_score": None},
            {"score_evidence": {"status": "uncertain"}, "final_canvas_score": None},
        ):
            guard = WholeGroupAdmissionGuard(lambda _group: True, group_size=8)
            self.assertFalse(guard(_group(8, trace_info=info)))
            self.assertEqual(guard.last_check.reason, "uncertain_or_error")

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
        group = _group(8)
        group[0].ok = False
        self.assertFalse(check_group(group).valid)
        group = _group(8)
        group[0].traces = []
        self.assertFalse(check_group(group).valid)

    def test_environment_entrypoint_is_strictly_opt_in(self):
        result = install_from_environment()
        self.assertFalse(result["installed"])
        self.assertFalse(result["enabled"])

    def test_prime_installation_skips_invalid_group_before_group_scoring(self):
        class FakeAlgorithm:
            calls = 0

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
            self.assertFalse(source.on_result(_group(7)))
            self.assertTrue(source.on_result(_group(8)))
            algorithm = FakeAlgorithm()
            asyncio.run(algorithm.finalize_group(_group(7)))
            self.assertEqual(FakeAlgorithm.calls, 0)
            asyncio.run(algorithm.finalize_group(_group(8)))
            self.assertEqual(FakeAlgorithm.calls, 1)
            self.assertEqual(len(FakeTrainSource.observed), 2)

            again = install_prime_guard(group_size=8)
            self.assertFalse(again["installed"])
            self.assertTrue(again["already_installed"])

    def test_enabled_environment_entrypoint_installs_with_configured_size(self):
        class FakeAlgorithm:
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
