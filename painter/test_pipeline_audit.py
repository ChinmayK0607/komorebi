import json
import tempfile
import unittest
from pathlib import Path

from painter.pipeline_audit import audit


class PipelineAuditTests(unittest.TestCase):
    def write_json(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_fixture(self, root: Path) -> Path:
        arm = root / "pro"
        self.write_json(
            arm / "source-identity.json",
            {
                "base_model_revision": "base-sha",
                "initializer_revision": "init-rev",
                "initializer_prefix": "step_16",
                "initializer_sha256": "init-sha",
                "frozen_sft_model_revision": "sft-rev",
                "frozen_sft_initializer": {"revision": "sft-rev", "prefix": "step_512", "adapter_sha256": "sft-sha"},
            },
        )
        self.write_json(
            arm / "supervisor-state.json",
            {
                "completed_optimizer_steps": 14,
                "last_optimizer_step": 14,
                "last_admitted_batch": 15,
                "budget": {"gpu_rental_cap_usd": 20.0, "price_usd_per_hour": 2.0, "reserve_seconds": 60},
            },
        )
        self.write_json(arm / "analysis-summary.json", {"judge_status": {"scored": 1}})
        self.write_json(
            arm / "consistent-group-admission.jsonl",
            {},
        )
        (arm / "consistent-group-admission.jsonl").write_text(
            json.dumps({"decision": "admit", "groups_observed": 15, "groups_admitted": 14, "groups_rejected": 1}) + "\n",
            encoding="utf-8",
        )
        trainer = arm / "async-output/run/logs/attempt_1/trainer.log"
        trainer.parent.mkdir(parents=True, exist_ok=True)
        trainer.write_text(
            "\x1b[32mSUCCESS\x1b[0m Step 14 | Loss 0.1 | Grad. Norm 0.0045 | Peak Mem. 512 MiB\n",
            encoding="utf-8",
        )
        orchestrator = arm / "async-output/run/monitors/file/traces/annotations/orch.index.jsonl"
        orchestrator.parent.mkdir(parents=True, exist_ok=True)
        orchestrator.write_text(json.dumps({"info": {"ship": {"step": 15}}}) + "\n", encoding="utf-8")
        orchestrator_log = arm / "async-output/run/logs/attempt_1/orchestrator.log"
        orchestrator_log.write_text(
            "\x1b[33m3 stale queued traces dropped\x1b[0m\n2 active episodes cancelled\n"
            "\x1b[31mDropped 4 queued traces past max_off_policy_steps=1.\x1b[0m\n"
            "Cancelled 84 train episodes past max_off_policy_steps=1.\n",
            encoding="utf-8",
        )
        self.write_json(
            arm / "rl-rollouts/a/painting.json",
            {"valid": True, "elapsed_seconds": "1.5 s"},
        )
        self.write_json(
            arm / "rl-rollouts/b/painting.json",
            {"valid": False, "timed_out": True, "elapsed_seconds": "2.5 s"},
        )
        self.write_json(arm / "rl-rollouts/c/painting.json", {"valid": True})
        success = {
            "elapsed_seconds": "2.5 s",
            "finish_reason": "stop",
            "observed_reasoning_tokens": 2,
            "raw_response": "opaque",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "cost": None},
        }
        failed = {"elapsed_seconds": 4, "finish_reason": "error", "usage": {"cost": None}}
        self.write_json(
            arm / "rl-rollouts/a/text-judge.json",
            {"views": [{"calls": [success]}], "recheck_views": [{"calls": [dict(success)]}],},
        )
        self.write_json(arm / "rl-rollouts/b/text-judge.json", {"views": [{"calls": [failed]}]})
        self.write_json(
            arm / "compact-traces-metrics.json",
            {
                "episodes": [
                    {
                        "reward_status": "scored",
                        "final_canvas_valid": True,
                        "episode_ok": True,
                        "trace_ok": True,
                        "timing": {"agent": {"model": {"duration": "1 s"}, "harness": {"duration": "2 s"}}, "scoring": {"start": 10, "end": 13}},
                    },
                    {
                        "reward_status": "model_invalid_final_action",
                        "final_canvas_valid": True,
                        "episode_ok": True,
                        "trace_ok": True,
                        "timing": {"agent": {"model": {"duration": "2 s"}, "harness": {"duration": "3 s"}}, "scoring": {"duration": 4}},
                    },
                    {
                        "reward_status": "model_invalid_final_action",
                        "final_canvas_valid": False,
                        "episode_ok": False,
                        "trace_ok": False,
                        "timing": {"agent": {"model": {"duration": "3 s"}, "harness": {"duration": "4 s"}}, "scoring": {"duration": 5}},
                    },
                ]
            },
        )
        (arm / "logs").mkdir(parents=True, exist_ok=True)
        (arm / "logs/gpu.csv").write_text(
            "timestamp,index,utilization.gpu [%]\n"
            "2026/10/02 00:00:00.000,0,50 %\n"
            "2026/10/02 00:00:00.000,7,25 %\n"
            "2026/10/02 00:00:10.000,0,75 %\n"
            "2026/10/02 00:00:10.500,7,50 %\n",
            encoding="utf-8",
        )
        return root

    def test_empty_input_is_compact_and_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(Path(directory))
        self.assertEqual(result["arms"], {})
        self.assertFalse(result["initializer"]["exact_match_across_arms"])
        self.assertEqual(result["schema"], "painter-pipeline-audit-v1")

    def test_incomplete_initializer_is_explicitly_censored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "pro"
            self.write_json(arm / "source-identity.json", {"base_model_revision": "base-sha"})
            result = audit(root)
        initializer = result["initializer"]
        self.assertFalse(initializer["exact_match_across_arms"])
        self.assertIn("initializer_revision", initializer["missing_fields"]["pro"])
        self.assertIn("frozen_sft_initializer.adapter_sha256", initializer["missing_fields"]["pro"])

    def test_actual_completion_is_not_admitted_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self.make_fixture(Path(directory)))
        arm = result["arms"]["pro"]
        self.assertEqual(arm["completion"]["completed_optimizer_steps"], 14)
        self.assertEqual(arm["completion"]["admitted_batch_counter"], 15)
        self.assertEqual(arm["completion"]["orchestrator_batch_steps"], 15)
        self.assertEqual(arm["training"]["grad_norm"]["count"], 1)
        self.assertEqual(arm["training"]["peak_memory"]["unit"], "GiB")
        self.assertAlmostEqual(arm["training"]["peak_memory"]["mean"], 0.5)
        self.assertEqual(arm["admission"]["groups_censored"], 1)
        self.assertEqual(arm["queue_and_cancellation"]["stale_queued_trace_drops"]["count"], 7)
        self.assertEqual(arm["queue_and_cancellation"]["active_episode_cancels"]["count"], 86)

    def test_completed_trace_metrics_split_invalid_final_action_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self.make_fixture(Path(directory)))
        traces = result["arms"]["pro"]["traces"]
        self.assertEqual(traces["completed_trace_count"], 3)
        self.assertEqual(traces["reward_status_counts"]["model_invalid_final_action"], 2)
        self.assertEqual(traces["final_action_invalid_count"], 2)
        self.assertEqual(traces["final_action_invalid_split"]["retained_final_canvas_valid"], 1)
        self.assertEqual(traces["final_action_invalid_split"]["final_canvas_invalid"], 1)
        self.assertEqual(traces["timing_seconds"]["model"]["count"], 3)
        self.assertEqual(traces["timing_seconds"]["harness"]["p90"], 4.0)
        self.assertEqual(traces["timing_seconds"]["scoring"]["count"], 3)
        self.assertIn("completed trace records", traces["censor_caveat"])

    def test_nested_calls_deduplicate_failed_and_null_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self.make_fixture(Path(directory)))
        judge = result["arms"]["pro"]["judge"]
        self.assertEqual(judge["views"]["raw_call_count"], 2)
        self.assertEqual(judge["views"]["deduplicated_call_count"], 2)
        self.assertEqual(judge["recheck_views"]["deduplicated_call_count"], 0)
        self.assertEqual(judge["recheck_views"]["exact_record_reuse_count"], 1)
        self.assertEqual(judge["total_failed_call_count"], 1)
        self.assertEqual(judge["total_usage"]["unknown_cost_count"], 2)
        self.assertNotIn("raw_response", json.dumps(result))

    def test_render_and_gpu_units(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self.make_fixture(Path(directory)))
        arm = result["arms"]["pro"]
        self.assertEqual(arm["render"]["valid_receipts"], 2)
        self.assertEqual(arm["render"]["invalid_receipts"], 1)
        self.assertEqual(arm["render"]["timing_unknown_receipts"], 1)
        self.assertEqual(arm["render"]["timed_out_receipts"], 1)
        self.assertEqual(arm["gpu"]["roles"]["inference"]["physical_indices"], list(range(7)))
        self.assertEqual(arm["gpu"]["roles"]["inference"]["samples"], 2)
        self.assertEqual(arm["gpu"]["roles"]["trainer"]["samples"], 2)
        self.assertEqual(arm["gpu"]["roles"]["inference"]["per_gpu_cadence_seconds"]["0"], 10.0)
        self.assertEqual(arm["gpu"]["roles"]["trainer"]["per_gpu_cadence_seconds"]["7"], 10.5)

    def test_streamed_trace_fallback_keeps_only_episode_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "pro"
            self.write_json(arm / "source-identity.json", {"base_model_revision": "base-sha"})
            stream_dir = arm / "async-output/run/monitors/file/traces/stream"
            stream_dir.mkdir(parents=True, exist_ok=True)
            def trace(trace_id):
                return {
                    "id": trace_id,
                    "ok": True,
                    "info": {"reward_status": "model_invalid_final_action", "final_canvas_valid": True},
                    "timing": {"agent": {"model": {"duration": 1}, "harness": {"duration": 2}}},
                }
            (stream_dir / "00000.jsonl").write_text(
                json.dumps({"id": "batch-0", "ok": True, "program": "opaque", "traces": [trace("one")]}) + "\n",
                encoding="utf-8",
            )
            (stream_dir / "00001.jsonl").write_text(
                json.dumps({"id": "batch-1", "ok": True, "traces": [trace("two"), trace("one")]}) + "\n",
                encoding="utf-8",
            )
            result = audit(root)
        traces = result["arms"]["pro"]["traces"]
        self.assertEqual(traces["source"], "streamed-traces")
        self.assertEqual(traces["completed_trace_count"], 2)
        self.assertEqual(traces["final_action_invalid_count"], 2)
        self.assertEqual(traces["final_action_invalid_split"]["retained_final_canvas_valid"], 2)
        self.assertEqual(traces["timing_seconds"]["model"]["count"], 2)
        self.assertNotIn("program", json.dumps(result))

    def test_stream_index_without_completed_aggregates_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "pro"
            self.write_json(arm / "source-identity.json", {"base_model_revision": "base-sha"})
            index = arm / "async-output/run/monitors/file/traces/stream.index.jsonl"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps({"id": "trace-one", "ok": True, "reward": 0.0}) + "\n", encoding="utf-8")
            result = audit(root)
        traces = result["arms"]["pro"]["traces"]
        self.assertEqual(traces["status"], "unknown")
        self.assertIsNone(traces["completed_trace_count"])
        self.assertIn("completed_trace_aggregate", traces["missing_fields"])
        self.assertIn("not treated as completed traces", traces["censor_caveat"])


if __name__ == "__main__":
    unittest.main()
