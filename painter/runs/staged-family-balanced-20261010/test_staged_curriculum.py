"""Offline tests for the staged family-balanced preparation and controller."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("staged_curriculum", ROOT / "staged_curriculum.py")
assert SPEC is not None and SPEC.loader is not None
curriculum = importlib.util.module_from_spec(SPEC)
sys.modules["staged_curriculum"] = curriculum
SPEC.loader.exec_module(curriculum)
PURGE_SPEC = importlib.util.spec_from_file_location("prime_tail_purge", ROOT / "prime_tail_purge.py")
assert PURGE_SPEC is not None and PURGE_SPEC.loader is not None
prime_tail_purge = importlib.util.module_from_spec(PURGE_SPEC)
sys.modules["prime_tail_purge"] = prime_tail_purge
PURGE_SPEC.loader.exec_module(prime_tail_purge)


def valid_group(slot: dict[str, object], reward: float = 0.25) -> dict[str, object]:
    rollouts = []
    for rollout_index in range(curriculum.ROLLOUTS_PER_PROMPT):
        rollouts.append({
            "group_id": slot["group_id"],
            "rollout_index": rollout_index,
            "trace_id": f"{slot['group_id']}-t{rollout_index:02d}",
            "row_id": slot["prompt_row_id"],
            "family": slot["family"],
            "stage": slot["stage"],
            "ownership": "model",
            "status": "scored",
            "evidence": {"source": "offline-test", "receipt_id": f"{slot['group_id']}-{rollout_index}"},
            "code_valid": True,
            "canvas_valid": True,
            "current_canvas_valid": True,
            "terminal_action_valid": True,
            "reward_status": "scored",
            "reward": reward,
        })
    return {
        "group_id": slot["group_id"],
        "row_id": slot["prompt_row_id"],
        "family": slot["family"],
        "stage": slot["stage"],
        "rollouts": rollouts,
    }


def cycle_slots(plan: dict[str, object], stage: int, cycle: int) -> list[dict[str, object]]:
    info = plan["stages"][stage - 1]
    return [slot for slot in info["schedule"] if slot["cycle"] == cycle]


class FakeSample:
    def __init__(self, token_count: int) -> None:
        self.token_ids = list(range(token_count))


class FakeTrace:
    def __init__(self, trace_id: str, token_count: int) -> None:
        self.id = trace_id
        self.num_total_tokens = token_count


class FakeEpisode:
    def __init__(self, trace: FakeTrace) -> None:
        self.traces = [trace]


class FakePendingEpisodes:
    def __init__(self, trace_ids: list[str]) -> None:
        self.sampled_trace_ids = set(trace_ids)


class FakeTrainSink:
    def __init__(self, trace_ids: list[str], *, token_mode: bool = False) -> None:
        self.batch_size = None if token_mode else 24
        self.token_batch_size = 1000 if token_mode else None
        self.pending_batch = {}
        self.episode_by_trace = {}
        for index, trace_id in enumerate(trace_ids):
            trace = FakeTrace(trace_id, index + 3)
            self.pending_batch[trace_id] = [FakeSample(index + 3)]
            self.episode_by_trace[trace_id] = FakeEpisode(trace)
        # Prime resets pending_episodes after a shipped batch; buffered tail
        # IDs therefore need not appear in this collection.
        self.pending_episodes = FakePendingEpisodes([])
        self.pending_tokens = sum(index + 3 for index in range(len(trace_ids))) if token_mode else 0


def prepared_tail() -> tuple[curriculum.StageController, FakeTrainSink, dict[str, list[str]]]:
    plan = curriculum.build_curriculum()
    controller = curriculum.StageController(plan)
    cycle_receipts = []
    for cycle in range(1, 6):
        cycle_receipts.append(controller.observe_cycle([valid_group(slot) for slot in cycle_slots(plan, 1, cycle)]))
    for cycle_receipt in cycle_receipts[:4]:
        controller.record_optimizer_update(trace_ids_by_cycle={cycle_receipt["cycle_id"]: list(controller.trace_ids_by_cycle[cycle_receipt["cycle_id"]])})
    trace_ids = list(controller.trace_ids_by_cycle["s1-c05"])
    cycle_id = "s1-c05"
    ids_by_cycle = {cycle_id: trace_ids}
    return controller, FakeTrainSink(trace_ids), ids_by_cycle


class PreparationTests(unittest.TestCase):
    def test_deterministic_source_and_cycle_shape(self) -> None:
        first = curriculum.build_curriculum()
        second = curriculum.build_curriculum()
        self.assertEqual(first, second)
        self.assertEqual(first["source"]["generator_sha256"], curriculum.PRIOR_GENERATOR_SHA256)
        self.assertEqual(first["sampling"]["stage_cycles"], {"1": 4, "2": 4, "3": 8, "4": 4})
        self.assertFalse(first["sampling"]["cyclic_mixed_sampling_claim"])
        self.assertEqual(first["sampling"]["sampling_cycles"], 20)
        self.assertEqual(first["sampling"]["sampling_cycle_cap"], 40)
        self.assertEqual(first["sampling"]["refill_cycle_multiplier"], 2)
        self.assertEqual(first["sampling"]["optimizer_updates_target"], 20)

    def test_each_sampling_cycle_has_equal_groups_and_trace_capacity(self) -> None:
        plan = curriculum.build_curriculum()
        for stage in plan["stages"]:
            self.assertEqual(len(stage["train_rows"]), 24)
            self.assertEqual(len(stage["heldout_rows"]), 8)
            self.assertEqual(len(stage["schedule"]), stage["max_sampling_cycles"] * 6)
            self.assertEqual(stage["task_groups_per_cycle"], 6)
            self.assertEqual(stage["rollouts_per_prompt"], 4)
            self.assertEqual(stage["trace_capacity_per_cycle"], 24)
            self.assertEqual(stage["max_sampling_cycles"], stage["cycles"] * 2)
            for cycle in range(1, stage["max_sampling_cycles"] + 1):
                slots = [slot for slot in stage["schedule"] if slot["cycle"] == cycle]
                self.assertEqual(len(slots), 6)
                counts = {family: sum(slot["family"] == family for slot in slots) for family in stage["train_families"]}
                self.assertEqual(counts, {family: 1 for family in stage["train_families"]})
                self.assertEqual(sum(slot["rollouts_per_prompt"] for slot in slots), 24)
                self.assertEqual(sum(slot["trace_capacity"] for slot in slots), 24)
            totals = {family: sum(slot["family"] == family for slot in stage["schedule"]) for family in stage["train_families"]}
            self.assertEqual(len(set(totals.values())), 1)
            self.assertEqual({slot["variant"] for slot in stage["schedule"] if slot["family"] == stage["train_families"][0]}, set(range(4)))

    def test_heldout_panel_is_fixed_and_disjoint(self) -> None:
        plan = curriculum.build_curriculum()
        panel = plan["matched_heldout_panel"]
        self.assertEqual(panel["cases"], 32)
        self.assertEqual(len(panel["rows"]), 32)
        train_ids = set(plan["train_row_ids"])
        heldout_ids = {row["id"] for row in panel["rows"]}
        self.assertTrue(train_ids.isdisjoint(heldout_ids))
        self.assertTrue(panel["same_panel_for_start_and_final"])
        by_stage = {stage: [row for row in panel["rows"] if row["stage"] == stage] for stage in range(1, 5)}
        self.assertEqual({stage: len(rows) for stage, rows in by_stage.items()}, {1: 8, 2: 8, 3: 8, 4: 8})
        self.assertEqual({stage: len({row["family"] for row in rows}) for stage, rows in by_stage.items()}, {1: 4, 2: 4, 3: 4, 4: 4})

    def test_config_keeps_flash12_and_disables_warmup(self) -> None:
        config = curriculum.build_config()
        self.assertEqual(config["starting_policy"], curriculum.STARTING_POLICY)
        self.assertEqual(config["base_model"], curriculum.BASE_MODEL)
        self.assertEqual(config["renderer"], curriculum.RENDERER)
        self.assertEqual(config["judge"]["effort"], "medium")
        self.assertEqual(config["judge"]["rubric"], "existing-generic-rubric")
        self.assertFalse(config["supervision"]["enabled"])
        self.assertEqual(config["supervision"]["updates"], 0)
        self.assertTrue(config["rl"]["require_valid_final_action"])
        self.assertEqual(config["curriculum"]["batch_size"], 24)
        self.assertEqual(config["curriculum"]["sampling_cycle_cap"], 40)
        self.assertEqual(config["curriculum"]["task_groups_per_cycle"], 6)
        self.assertEqual(config["curriculum"]["rollouts_per_prompt"], 4)
        self.assertEqual(config["curriculum"]["trace_capacity_per_cycle"], 24)
        self.assertEqual(config["rl"]["validity_admission"]["minimum_scored_rollouts_per_group"], 3)

    def test_prime_contract_is_explicit_and_node_free_when_unavailable(self) -> None:
        contract = curriculum.build_prime_compatibility_config()
        self.assertEqual(contract["orchestrator"]["batch_size"], 24)
        self.assertEqual(contract["orchestrator"]["group_size"], 4)
        self.assertEqual(contract["orchestrator"]["concurrency"]["max_inflight"], 24)
        self.assertEqual(contract["inference"]["vllm"]["max_num_seqs"], 24)
        result = curriculum.validate_prime_compatibility()
        self.assertEqual(result["prime_revision"], curriculum.PRIME_REVISION)
        self.assertEqual(result["trace_capacity_per_cycle"], 24)
        self.assertIn(result["status"], {"unavailable", "passed", "failed"})
        self.assertTrue(result["no_gpu_or_external_calls"])
        pinned_checkout = ROOT.parents[2] / "work" / "prime-rl-upstream"
        if pinned_checkout.is_dir():
            probed = curriculum.validate_prime_compatibility(pinned_checkout)
            self.assertEqual(probed["observed_revision"], curriculum.PRIME_REVISION)
            self.assertIn(probed["status"], {"passed", "failed", "unavailable"})
            self.assertEqual(probed["batch_size"], 24)

    def test_write_preparation_has_small_hash_bound_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = curriculum.write_preparation(Path(directory))
            self.assertTrue(receipt["no_gpu_or_external_calls"])
            names = sorted(path.name for path in Path(directory).iterdir())
            self.assertEqual(names, ["config.json", "curriculum-manifest.json", "preparation-receipt.json"])
            manifest = json.loads((Path(directory) / "curriculum-manifest.json").read_text())
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(config["curriculum"]["manifest_sha256"], curriculum.sha256_bytes(curriculum._canonical(manifest)))
            surface = config["run_surface"]
            self.assertEqual(surface["files"]["bootstrap"]["path"], "bootstrap.py")
            self.assertEqual(surface["files"]["bootstrap_shell"]["path"], "bootstrap.sh")
            self.assertEqual(surface["files"]["launch_prepared"]["path"], "launch_prepared.py")
            self.assertEqual(surface["bootstrap_scope"]["status"], "prepared_not_run")
            self.assertEqual(surface["bootstrap_scope"]["receipt_status"], "ready_no_launch")
            self.assertEqual(surface["launch_scope"]["receipt_derived_launcher"], "launch_prepared.py")
            self.assertEqual(surface["launch_scope"]["requires_owner_only_gateway_mode"], "0600")



class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = curriculum.build_curriculum()

    def _observe_cycle(self, controller: curriculum.StageController, stage: int, cycle: int, *, groups: list[dict[str, object]] | None = None) -> dict[str, object]:
        records = groups if groups is not None else [valid_group(slot) for slot in cycle_slots(self.plan, stage, cycle)]
        return controller.observe_cycle(records)

    def _complete_stage(self, controller: curriculum.StageController, stage: int) -> list[dict[str, object]]:
        receipts: list[dict[str, object]] = []
        target = self.plan["stages"][stage - 1]["optimizer_updates_target"]
        for _ in range(target):
            cycle = controller.next_cycle
            cycle_receipt = self._observe_cycle(controller, stage, cycle)
            self.assertEqual(cycle_receipt["status"], "cycle_complete")
            self.assertEqual(cycle_receipt["decision"], "pass")
            receipts.append(controller.record_optimizer_update(trace_ids_by_cycle={cycle_receipt["cycle_id"]: list(controller.trace_ids_by_cycle[cycle_receipt["cycle_id"]])}))
        return receipts

    def test_stage_advances_only_on_optimizer_target(self) -> None:
        controller = curriculum.StageController(self.plan)
        cycle_receipt = self._observe_cycle(controller, 1, 1)
        self.assertEqual(cycle_receipt["status"], "cycle_complete")
        self.assertEqual(cycle_receipt["decision"], "pass")
        self.assertEqual(controller.active_stage, 1)
        self.assertEqual(controller.stage_optimizer_updates_observed, 0)
        update_receipt = controller.record_optimizer_update(trace_ids_by_cycle={cycle_receipt["cycle_id"]: list(controller.trace_ids_by_cycle[cycle_receipt["cycle_id"]])})
        self.assertEqual(update_receipt["decision"], "continue_stage")
        self.assertEqual(controller.active_stage, 1)
        self.assertEqual(controller.stage_optimizer_updates_observed, 1)

    def test_schedule_receipts_advance_and_final_stop_without_quality_gate(self) -> None:
        controller = curriculum.StageController(self.plan)
        for stage in range(1, 5):
            receipts = self._complete_stage(controller, stage)
            self.assertEqual(receipts[-1]["decision"], "stop" if stage == 4 else "advance")
            self.assertEqual(receipts[-1]["scored_traces"], 24)
            self.assertEqual(receipts[-1]["stage_optimizer_update"], self.plan["stages"][stage - 1]["optimizer_updates_target"])
            if stage < 4:
                self.assertEqual(receipts[-1]["next_stage"], stage + 1)
        self.assertTrue(controller.halted)

    def test_model_owned_invalid_is_scored_zero_and_does_not_halt(self) -> None:
        controller = curriculum.StageController(self.plan)
        slots = cycle_slots(self.plan, 1, 1)
        records = [valid_group(slot) for slot in slots]
        rollout = records[0]["rollouts"][0]
        rollout["status"] = "model_invalid"
        rollout["canvas_valid"] = False
        rollout["reward"] = 0.0
        rollout["evidence"]["reason"] = "model emitted invalid canvas operation"
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "cycle_complete")
        self.assertEqual(receipt["decision"], "pass")
        self.assertFalse(controller.halted)
        self.assertEqual(receipt["scored_rollouts"], 24)
        self.assertEqual(receipt["family_model_invalid_counts"][slots[0]["family"]], 1)
        self.assertEqual(receipt["family_scored_trace_counts"][slots[0]["family"]], 4)
        self.assertEqual(receipt["model_invalid_evidence"][0]["evidence"]["source"], "offline-test")

    def test_censored_retries_consume_finite_sampling_attempt_cap(self) -> None:
        controller = curriculum.StageController(self.plan)
        slots = cycle_slots(self.plan, 1, 1)
        records = [valid_group(slot) for slot in slots]
        for rollout in records[0]["rollouts"][2:]:
            rollout.update({
                "ownership": "judge",
                "status": "judge_uncertain",
                "reward_status": "unscored",
                "reward": None,
                "evidence": {"source": "offline-test", "reason": "uncertain"},
            })
        cap = self.plan["stages"][0]["max_sampling_cycles"]
        for _ in range(cap):
            receipt = controller.observe_cycle(records)
            self.assertEqual(receipt["status"], "censored")
        self.assertTrue(controller.halted)
        self.assertEqual(receipt["decision"], "stop")
        self.assertEqual(receipt["sampling_attempt"], cap)

    def test_judge_uncertainty_excludes_known_valid_rollout(self) -> None:
        controller = curriculum.StageController(self.plan)
        slots = cycle_slots(self.plan, 1, 1)
        records = [valid_group(slot) for slot in slots]
        rollout = records[0]["rollouts"][0]
        rollout.update({
            "ownership": "judge",
            "status": "judge_uncertain",
            "reward_status": "unscored",
            "reward": None,
        })
        rollout["evidence"]["reason"] = "judge response was uncertain after valid render"
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "cycle_complete")
        self.assertEqual(receipt["scored_rollouts"], 23)
        self.assertEqual(receipt["excluded_rollouts"], 1)
        self.assertEqual(receipt["family_scored_trace_counts"][slots[0]["family"]], 3)
        self.assertEqual(receipt["family_excluded_rollout_counts"][slots[0]["family"]], 1)
        self.assertEqual(receipt["excluded_evidence"][0]["status"], "judge_uncertain")
        self.assertFalse(controller.halted)

    def test_two_exclusions_censor_cycle_without_fabricated_zeros(self) -> None:
        controller = curriculum.StageController(self.plan)
        slots = cycle_slots(self.plan, 1, 1)
        records = [valid_group(slot) for slot in slots]
        for rollout in records[0]["rollouts"][:2]:
            rollout.update({
                "ownership": "service",
                "status": "service_failure",
                "code_valid": None,
                "canvas_valid": None,
                "current_canvas_valid": None,
                "terminal_action_valid": None,
                "reward_status": "unscored",
                "reward": None,
            })
            rollout["evidence"]["reason"] = "service failure"
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "censored")
        self.assertEqual(receipt["decision"], "retry_cycle")
        self.assertFalse(controller.halted)
        self.assertEqual(controller.next_cycle, 1)
        self.assertEqual(receipt["family_scheduled_trace_counts"][slots[0]["family"]], 4)
        self.assertEqual(receipt["family_scored_trace_counts"][slots[0]["family"]], 2)
        self.assertEqual(receipt["family_excluded_rollout_counts"][slots[0]["family"]], 2)

    def test_malformed_ownership_status_or_evidence_stops_fail_closed(self) -> None:
        for field in ("ownership", "status", "evidence"):
            controller = curriculum.StageController(self.plan)
            records = [valid_group(slot) for slot in cycle_slots(self.plan, 1, 1)]
            records[0]["rollouts"][0].pop(field)
            receipt = controller.observe_cycle(records)
            self.assertEqual(receipt["status"], "stop", field)
            self.assertEqual(receipt["admission"], "fail_closed", field)
            self.assertTrue(controller.halted)

    def test_out_of_order_stage_is_rejected(self) -> None:
        controller = curriculum.StageController(self.plan)
        records = [valid_group(slot) for slot in cycle_slots(self.plan, 1, 1)]
        records[0]["rollouts"][0]["stage"] = 2
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "stop")
        self.assertEqual(receipt["reason"], "action_admission_failed")

    def test_group_requires_exactly_four_rollouts(self) -> None:
        controller = curriculum.StageController(self.plan)
        records = [valid_group(slot) for slot in cycle_slots(self.plan, 1, 1)]
        records[0]["rollouts"].pop()
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["reason"], "action_admission_failed")

    def test_duplicate_native_trace_ids_fail_closed(self) -> None:
        controller = curriculum.StageController(self.plan)
        records = [valid_group(slot) for slot in cycle_slots(self.plan, 1, 1)]
        records[0]["rollouts"][1]["trace_id"] = records[0]["rollouts"][0]["trace_id"]
        receipt = controller.observe_cycle(records)
        self.assertEqual(receipt["status"], "stop")
        self.assertEqual(receipt["reason"], "action_admission_failed")

    def test_optimizer_update_binds_active_stage_cycles_and_exact_batch(self) -> None:
        controller = curriculum.StageController(self.plan)
        first = self._observe_cycle(controller, 1, 1)
        second = self._observe_cycle(controller, 1, 2)
        first_ids = list(controller.trace_ids_by_cycle[first["cycle_id"]])
        second_ids = list(controller.trace_ids_by_cycle[second["cycle_id"]])
        receipt = controller.record_optimizer_update(trace_ids_by_cycle={first["cycle_id"]: first_ids[:12], second["cycle_id"]: second_ids[:12]})
        self.assertEqual(receipt["optimizer_update"], 1)
        self.assertEqual(receipt["sampling_cycles"], ["s1-c01", "s1-c02"])
        self.assertEqual(receipt["trace_ids_by_cycle"], {"s1-c01": first_ids[:12], "s1-c02": second_ids[:12]})
        self.assertEqual(controller.optimizer_updates_observed, 1)
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_optimizer_update(trace_ids_by_cycle={"s1-c01": first_ids[:11] + second_ids[:12]})

    def test_optimizer_update_cannot_bind_prior_stage_after_advance(self) -> None:
        controller = curriculum.StageController(self.plan)
        for _ in range(4):
            cycle_receipt = self._observe_cycle(controller, 1, controller.next_cycle)
            update = controller.record_optimizer_update(trace_ids_by_cycle={cycle_receipt["cycle_id"]: list(controller.trace_ids_by_cycle[cycle_receipt["cycle_id"]])})
        self.assertEqual(update["decision"], "advance")
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_optimizer_update(trace_ids_by_cycle={"s1-c01": ["old-trace"] * 24})

    def test_optimizer_update_rejects_false_cycle_attribution(self) -> None:
        controller = curriculum.StageController(self.plan)
        first = self._observe_cycle(controller, 1, 1)
        first_ids = list(controller.trace_ids_by_cycle[first["cycle_id"]])
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_optimizer_update(trace_ids_by_cycle={first["cycle_id"]: first_ids[:12], "s1-c02": first_ids[12:]})
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_optimizer_update(trace_ids_by_cycle={first["cycle_id"]: first_ids + ["forged-trace"]})

    def test_stage_target_requires_explicit_tail_drain_before_advance(self) -> None:
        controller, sink, ids_by_cycle = prepared_tail()
        # The helper leaves the controller at the target with an unconsumed c05 tail.
        self.assertEqual(controller.stage_optimizer_updates_observed, 4)
        self.assertEqual(len(controller.trace_ids_by_cycle["s1-c05"]), 24)
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_optimizer_update(trace_ids_by_cycle={"s1-c05": list(controller.trace_ids_by_cycle["s1-c05"])})
        with self.assertRaises(curriculum.CurriculumError):
            controller.record_stage_tail_drop(
                trace_ids_by_cycle={"s1-c05": list(controller.trace_ids_by_cycle["s1-c05"])},
                reason="missing physical purge receipt",
                purge_receipt={},
            )
        purge = prime_tail_purge.purge_pending_stage_tail(
            sink,
            controller,
            ids_by_cycle,
            prime_revision=curriculum.PRIME_REVISION,
            receipt_id="tail-test-1",
        )
        tail = prime_tail_purge.commit_purged_stage_tail(
            controller,
            purge,
            reason="runtime Prime tail purge required before stage transition",
        )
        self.assertEqual(tail["decision"], "advance")
        self.assertEqual(tail["trace_inventory_after"], {})
        self.assertEqual(controller.active_stage, 2)
        self.assertEqual(controller.trace_inventory_by_cycle, {})

    def test_exact_batch_mode_purge_and_replay_are_auditable(self) -> None:
        controller, sink, ids_by_cycle = prepared_tail()
        before_ids = set(sink.pending_batch)
        purge = prime_tail_purge.purge_pending_stage_tail(
            sink,
            controller,
            ids_by_cycle,
            prime_revision=curriculum.PRIME_REVISION,
            receipt_id="tail-test-batch",
        )
        self.assertEqual(purge["removed_trace_count"], 24)
        self.assertEqual(purge["pending_trace_count_after"], 0)
        self.assertEqual(set(sink.pending_batch), before_ids - set(sum(ids_by_cycle.values(), [])))
        self.assertEqual(sink.pending_tokens, 0)
        self.assertEqual(sink.pending_episodes.sampled_trace_ids, set())
        with self.assertRaises(prime_tail_purge.PrimeTailPurgeError):
            prime_tail_purge.purge_pending_stage_tail(
                sink,
                controller,
                ids_by_cycle,
                prime_revision=curriculum.PRIME_REVISION,
                receipt_id="tail-test-replay",
            )
        committed = prime_tail_purge.commit_purged_stage_tail(
            controller,
            purge,
            reason="test replay commit",
        )
        self.assertEqual(committed["decision"], "advance")
        with self.assertRaises(curriculum.CurriculumError):
            prime_tail_purge.commit_purged_stage_tail(
                controller,
                purge,
                reason="replayed controller commit",
            )

    def test_token_mode_decrements_payload_tokens(self) -> None:
        controller, _sink, ids_by_cycle = prepared_tail()
        trace_ids = ids_by_cycle["s1-c05"]
        sink = FakeTrainSink(trace_ids, token_mode=True)
        before_tokens = sink.pending_tokens
        purge = prime_tail_purge.purge_pending_stage_tail(
            sink,
            controller,
            ids_by_cycle,
            prime_revision=curriculum.PRIME_REVISION,
            receipt_id="tail-test-tokens",
        )
        self.assertTrue(purge["token_mode"])
        self.assertEqual(purge["removed_token_count"], before_tokens)
        self.assertEqual(sink.pending_tokens, 0)

    def test_wrong_ids_counts_stage_and_revision_leave_sink_unchanged(self) -> None:
        cases = ("unknown", "count", "cross_cycle", "revision")
        for case in cases:
            controller, sink, ids_by_cycle = prepared_tail()
            before = (set(sink.pending_batch), set(sink.episode_by_trace), set(sink.pending_episodes.sampled_trace_ids), sink.pending_tokens)
            ids = {cycle_id: list(trace_ids) for cycle_id, trace_ids in ids_by_cycle.items()}
            revision = curriculum.PRIME_REVISION
            if case == "unknown":
                ids["s1-c05"][0] = "unknown-trace"
            elif case == "count":
                ids["s1-c05"] = ids["s1-c05"][:-1]
            elif case == "cross_cycle":
                ids["s1-c06"] = ids.pop("s1-c05")
            else:
                revision = "unsupported-prime"
            with self.assertRaises(prime_tail_purge.PrimeTailPurgeError):
                prime_tail_purge.purge_pending_stage_tail(
                    sink,
                    controller,
                    ids,
                    prime_revision=revision,
                    receipt_id=f"tail-test-{case}",
                )
            self.assertEqual((set(sink.pending_batch), set(sink.episode_by_trace), set(sink.pending_episodes.sampled_trace_ids), sink.pending_tokens), before, case)

    def test_refill_cycle_cap_stops_without_stage_progression(self) -> None:
        controller = curriculum.StageController(self.plan)
        for cycle in range(1, 9):
            receipt = self._observe_cycle(controller, 1, cycle)
        self.assertEqual(receipt["decision"], "pass")
        self.assertTrue(receipt["cycle_cap_reached"])
        self.assertFalse(controller.halted)
        stopped = controller.observe_cycle([])
        self.assertEqual(stopped["status"], "stop")
        self.assertEqual(stopped["reason"], "sampling_cycle_cap_exhausted")
        self.assertTrue(controller.halted)


if __name__ == "__main__":
    unittest.main()
