from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from painter.profile_rl_evidence import _zstandard, profile_run, render_markdown


class ProfileRLEvidenceTests(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def test_profile_separates_updates_render_failures_and_no_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text("\n".join([
                json.dumps({"producer": "trainer", "step": 1, "time/step": 4.0, "time/wait_for_batch": 6.0}),
                json.dumps({"producer": "trainer", "step": 2, "time/step": 5.0, "time/wait_for_batch": 1.0}),
                json.dumps({"producer": "orch", "step": 1, "train/agg/effective/agent/is_admitted/mean": 1.0, "inference/server0.0/generation_tokens_total": 100.0}),
                json.dumps({"producer": "orch", "step": 2, "train/agg/effective/agent/reward/mean": 0.5, "inference/server0.0/generation_tokens_total": 140.0}),
                "malformed",
            ]) + "\n")
            (root / "logs").mkdir()
            (root / "logs" / "orchestrator.log").write_text(
                "Discarded 4/8 episodes: stale=2, errored=1, no_signal=1\n"
            )
            self.write_json(root / "rollouts" / "a" / "turn-1" / "painting.json", {
                "valid": True, "elapsed_seconds": 2.0, "errors": [],
            })
            self.write_json(root / "rollouts" / "a" / "turn-2" / "painting.json", {
                "valid": False, "elapsed_seconds": 180.5, "errors": ["SyntaxError: unexpected token"],
            })
            self.write_json(root / "rollouts" / "a" / "text-judge.json", {
                "status": "scored", "consensus": "initial_agreement",
                "views": [{"elapsed_seconds": 0.4}],
            })
            self.write_json(root / "rollouts" / "b" / "text-judge.json", {
                "status": "no_signal", "consensus": "no_signal",
                "views": [{"elapsed_seconds": 0.8, "provider_error": "timeout"}],
            })
            (root / "rollout-traces.jsonl").write_text(json.dumps({
                "traces": [{
                    "info": {"turns": [{"error": "Unexpected token"}, {"error": "Unexpected token"}]},
                    "calls": [{"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}},
                              {"usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}],
                }],
            }) + "\n")

            report = profile_run(root)
            self.assertEqual(report["training"]["completed_optimizer_updates"]["value"], 2)
            self.assertEqual(report["training"]["admitted_reward_batches"]["value"], 1)
            self.assertIn("reward/mean", report["training"]["admitted_reward_batches"]["basis"])
            self.assertEqual(report["training"]["trainer_wait_vs_active"]["wait_seconds"], 7.0)
            self.assertEqual(report["training"]["trainer_wait_vs_active"]["active_seconds"], 4.0)
            self.assertEqual(report["render"]["valid"]["window"], 1)
            self.assertEqual(report["render"]["invalid"]["window"], 1)
            self.assertEqual(report["render"]["timeouts"]["window"], 1)
            self.assertEqual(report["render"]["failed_trajectories"]["window"], 0)
            self.assertEqual(report["judge"]["statuses"]["scored"]["window"], 1)
            self.assertEqual(report["judge"]["statuses"]["no_signal"]["window"], 1)
            self.assertNotIn("uncertain", report["judge"]["statuses"])
            self.assertEqual(report["discarded"]["stale"]["window"], 2)
            self.assertEqual(report["discarded"]["error"]["window"], 1)
            self.assertEqual(report["discarded"]["no_signal"]["window"], 1)
            self.assertEqual(report["output"]["turns"]["window"], 2)
            self.assertEqual(report["output"]["completion_tokens"]["window"], 7)
            self.assertGreater(report["failures"]["error_categories"]["unexpected_token"], 1)
            self.assertEqual(report["evidence"]["malformed_json_lines"], 1)

    def test_trainer_active_time_subtracts_wait_from_step_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({
                "producer": "trainer", "step": 1,
                "time/step": 780.0, "time/wait_for_batch": 600.0,
            }) + "\n")
            report = profile_run(root)
            timing = report["training"]["trainer_wait_vs_active"]
            self.assertEqual(timing["wait_seconds"], 600.0)
            self.assertEqual(timing["active_seconds"], 180.0)
            self.assertEqual(timing["denominator_seconds"], 780.0)
            self.assertEqual(timing["wait_fraction"], round(600 / 780, 6))
            self.assertEqual(timing["active_fraction"], round(180 / 780, 6))
            self.assertEqual(timing["measurement_window"], "trainer metric rows with matched wait and active values")

    def test_zero_wait_is_a_matched_trainer_timing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({
                "producer": "trainer", "step": 1,
                "time/step": 180.0, "time/wait_for_batch": 0.0,
            }) + "\n")
            report = profile_run(root)
            timing = report["training"]["trainer_wait_vs_active"]
            self.assertEqual(timing["wait_seconds"], 0.0)
            self.assertEqual(timing["active_seconds"], 180.0)
            self.assertEqual(timing["denominator_seconds"], 180.0)
            self.assertEqual(timing["measurement_window"], "trainer metric rows with matched wait and active values")

    def test_judge_views_are_separate_from_nested_provider_call_leaves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "text-judge.json", {
                "status": "scored",
                "views": [
                    {"calls": [{"elapsed_seconds": 1.2, "model": "judge", "usage": {"total_tokens": 10}}]},
                    {"calls": [{"attempt": 2, "elapsed_seconds": 2.3, "model": "judge", "provider_error": "timeout"}]},
                ],
            })
            report = profile_run(root)
            self.assertEqual(report["judge"]["views"]["window"], 2)
            self.assertEqual(report["judge"]["calls"]["window"], 2)
            self.assertEqual(report["judge"]["timeouts"]["window"], 1)
            self.assertEqual(report["judge"]["call_latency_seconds"]["count"], 2)

    def test_optimizer_steps_require_trainer_log_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trainer" / "rank0" / "stdout.log").parent.mkdir(parents=True)
            (root / "trainer" / "rank0" / "stdout.log").write_text("2026 INFO SUCCESS Step 19 | loss=1\n")
            (root / "orchestrator.log").write_text("2026 INFO SUCCESS Step 20 | dispatched=1\n")
            (root / "optimizer-update-audit.jsonl").write_text(json.dumps({"completed_updates": 20}) + "\n")
            report = profile_run(root)
            self.assertEqual(report["training"]["completed_optimizer_updates"]["value"], 19)
            self.assertIn("disagreeing audit", report["training"]["completed_optimizer_updates"]["basis"])

    def test_step_zero_metric_is_not_an_optimizer_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({
                "producer": "trainer", "step": 0,
                "time/step": 10.0, "time/wait_for_batch": 2.0,
            }) + "\n")
            report = profile_run(root)
            self.assertEqual(report["training"]["completed_optimizer_updates"]["status"], "unavailable")

    def test_excludes_teacher_validation_and_anchors_from_training_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("teacher-candidates", "validation", "anchors-frozen"):
                self.write_json(root / folder / "painting.json", {
                    "valid": True, "elapsed_seconds": 181.0,
                })
            self.write_json(root / "training" / "painting.json", {
                "valid": False, "timed_out": True, "error_code": "render_timeout",
            })
            self.write_json(root / "training" / "render-error" / "painting.json", {
                "valid": False, "elapsed_seconds": 0.5, "error": "render_timeout",
            })
            report = profile_run(root)
            self.assertEqual(report["render"]["observations"]["window"], 2)
            self.assertEqual(report["render"]["timeouts"]["window"], 2)
            self.assertEqual(report["render"]["failed_trajectories"]["status"], "unavailable")

    def test_streams_rollout_records_from_tar_without_extracting_raw_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "evidence.tar.gz"
            payload = (json.dumps({
                "traces": [{"info": {"turns": [{}]}, "calls": [{"usage": {"completion_tokens": 9}}]}]
            }) + "\n").encode()
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("rollouts/stream/00000.jsonl")
                member.size = len(payload)
                output.addfile(member, io.BytesIO(payload))
            report = profile_run(root)
            self.assertEqual(report["evidence"]["sources"]["archives_scanned"], 1)
            self.assertEqual(report["output"]["completion_tokens"]["window"], 9)
            self.assertFalse(report["evidence"]["raw_prompts_or_programs_in_report"])

    def test_unavailable_zstd_is_reported_without_inferred_trajectory_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rollout-stream.jsonl.zst").write_bytes(b"not a zstd stream")
            report = profile_run(root)
            if _zstandard is None:
                self.assertEqual(report["evidence"]["sources"]["unsupported_zstd_jsonl"], 1)
                self.assertTrue(any("jsonl.zst" in item for item in report["limitations"]))
            self.assertEqual(report["render"]["failed_trajectories"]["status"], "unavailable")

    def test_effective_reward_metric_is_used_when_admitted_flag_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text("\n".join([
                json.dumps({"producer": "orch", "step": 7, "train/agg/effective/agent/reward/mean": 0.25}),
                json.dumps({"producer": "orch", "step": 8, "train/agg/effective/agent/reward/mean": 0.5}),
            ]) + "\n")
            report = profile_run(root)
            self.assertEqual(report["training"]["admitted_reward_batches"]["value"], 2)
            self.assertIn("reward/mean", report["training"]["admitted_reward_batches"]["basis"])

    def test_zero_effective_reward_metric_still_counts_as_an_admitted_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({
                "producer": "orch", "step": 3,
                "train/agg/effective/agent/reward/mean": 0.0,
            }) + "\n")
            report = profile_run(root)
            self.assertEqual(report["training"]["admitted_reward_batches"]["value"], 1)

    def test_markdown_is_compact_and_does_not_include_private_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "painting.json", {
                "valid": False, "source": "/private/node/prompt/program.js", "errors": [],
            })
            markdown = render_markdown(profile_run(root))
            self.assertIn("intermediate invalid renders excluded", markdown)
            self.assertNotIn("/private/node", markdown)
            self.assertNotIn("program.js", markdown)

    def test_discard_counts_are_scoped_to_logged_warning_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "orchestrator.log").write_text(
                "Discarded 6/8 episodes: stale=3, errored=1, no_signal=2\n"
                "Discarded 4/8 episodes: stale=2, errored=1, no_signal=1\n"
            )
            report = profile_run(root)
            discarded = report["discarded"]
            self.assertEqual(discarded["discard_events"]["window"], 2)
            self.assertEqual(discarded["warning_windows"]["window"], 1)
            self.assertIn("not a run-total", discarded["scope"])
            markdown = render_markdown(report)
            self.assertIn("logged summary counters only, not run totals", markdown)
            self.assertIn(">50% warning windows: 1", markdown)

    def test_browser_launch_failure_is_operational_not_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "rollouts" / "a" / "painting.json", {
                "valid": False,
                "errors": ["BrowserType.launch: executable doesn't exist", "SyntaxError: unexpected token"],
            })
            report = profile_run(root)
            failures = report["failures"]
            self.assertNotIn("browser_launch_operational", failures["error_categories"])
            self.assertEqual(failures["operational_error_categories"]["browser_launch_operational"], 1)
            self.assertEqual(failures["error_categories"]["unexpected_token"], 1)
            self.assertIn("overlapping", failures["error_category_scope"])


if __name__ == "__main__":
    unittest.main()
