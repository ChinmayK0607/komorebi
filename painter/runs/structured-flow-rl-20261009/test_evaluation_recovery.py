"""Offline regression checks for the bounded evaluation-only recovery."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationRecoveryTest(unittest.TestCase):
    def test_bootstrap_gpu_probe_uses_actual_setup_script_without_installing(self) -> None:
        """Exercise the real shared bootstrap boundary before package/model work."""
        script = HERE.parent / "text-pairwise-rl-20260928" / "setup_node.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "uname").write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
            (fake_bin / "nvidia-smi").write_text(
                "#!/bin/sh\n"
                "i=0\n"
                "while [ \"$i\" -lt \"${MOCK_GPU_COUNT:-0}\" ]; do printf '%s\\n' \"$i\"; i=$((i + 1)); done\n"
            )
            (fake_bin / "python3").write_text("#!/bin/sh\nprintf '%s\\n' invoked > \"$MOCK_INSTALL_MARKER\"\nexit 99\n")
            for helper in fake_bin.iterdir():
                helper.chmod(0o755)
            marker = root / "install-marker"

            def probe(gpu_count: int, evaluation_only: bool) -> subprocess.CompletedProcess[str]:
                environment = {
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                    "MOCK_GPU_COUNT": str(gpu_count),
                    "MOCK_INSTALL_MARKER": str(marker),
                }
                if evaluation_only:
                    environment["PAINTER_EVALUATION_ONLY"] = "1"
                else:
                    environment.pop("PAINTER_EVALUATION_ONLY", None)
                return subprocess.run(
                    ["bash", str(script), "--gpu-probe-only"],
                    cwd=HERE,
                    env=environment,
                    text=True,
                    capture_output=True,
                )

            evaluation = probe(1, True)
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
            self.assertIn("mode=evaluation_only visible_gpus=1", evaluation.stdout)
            training_one = probe(1, False)
            self.assertNotEqual(training_one.returncode, 0)
            self.assertIn("two visible GPUs required", training_one.stderr)
            training_two = probe(2, False)
            self.assertEqual(training_two.returncode, 0, training_two.stderr)
            self.assertIn("mode=training visible_gpus=2", training_two.stdout)
            self.assertFalse(marker.exists(), "probe must stop before Python/package/model setup")

    def test_controller_evaluation_mode_routes_detached_job_to_recovery(self) -> None:
        path = HERE / "operate_single.py"
        spec = importlib.util.spec_from_file_location("evaluation_operation_contract", path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"PAINTER_EVALUATION_ONLY": "1"}, clear=False):
            spec.loader.exec_module(module)
        command = module._detached_job_command(source_commit="a" * 40, rental_start_epoch=100.0, price_usd_per_hour=5.75, cap_usd=3.0)
        self.assertIn("recover_evaluation.sh", command)
        self.assertNotIn("execute_single.sh", command)
        self.assertIn("PAINTER_GPUS_PER_NODE=1", command)
        self.assertIn("PAINTER_NUM_TRAIN_GPUS=0", command)
        self.assertIn("PAINTER_NUM_INFER_GPUS=1", command)
        self.assertIn("PAINTER_EVALUATION_ONLY=1", command)
        self.assertIn("PAINTER_RECOVERY_TOKEN_FILE=/root/painter/private/hf-token", command)
        result = subprocess.run(
            ["bash", "-c", 'source "$1/runtime_env.sh"; validate_structured_bindings; printf "%s %s %s" "$PAINTER_GPUS_PER_NODE" "$PAINTER_NUM_TRAIN_GPUS" "$PAINTER_NUM_INFER_GPUS"', "runtime", str(HERE)],
            env={**os.environ, "PAINTER_EVALUATION_ONLY": "1", "PAINTER_GPUS_PER_NODE": "1", "PAINTER_NUM_TRAIN_GPUS": "0", "PAINTER_NUM_INFER_GPUS": "1"},
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "1 0 1")

    def test_receipt_is_evaluation_only_and_single_gpu(self) -> None:
        recovery = load("evaluation_recovery")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{"id": f"case-{i}"} for i in range(16)]}))
            receipt = recovery.recovery_receipt(run_root=root, source_commit="a7dd0fa", manifest=manifest)
        self.assertEqual(receipt["mode"], "evaluation_only")
        self.assertEqual(receipt["training"]["new_optimizer_updates"], 0)
        self.assertEqual(receipt["topology"], {"gpus_per_node": 1, "trainer_gpus": 0, "inference_gpus": 1, "evaluation_gpu": 0})
        self.assertEqual(receipt["evaluation"]["expected_cases"], 16)

    def test_viewer_rejects_invalid_terminal_even_with_prior_canvas(self) -> None:
        viewer = load("build_recovery_review")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            cases = []
            for i in range(16):
                brief = root / f"brief-{i}.txt"
                brief.write_text(f"[USER]\npaint case {i}\n")
                cases.append({"id": f"case-{i}", "reference": str(brief)})
            manifest.write_text(json.dumps({"cases": cases}))
            baseline, candidate = root / "baseline", root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            image = candidate / "candidate.png"
            image.write_bytes(b"prior")
            trace = {
                "task": {"data": {"id": "case-0"}},
                "info": {
                    "finished": True,
                    "final_canvas": "/root/painter/candidate.png",
                    "final_canvas_valid": False,
                    "turns": [{"turn": 1, "valid": True, "canvas": "/root/painter/candidate.png"}, {"turn": 3, "valid": False, "canvas": "/root/painter/candidate.png"}],
                },
            }
            (candidate / "traces.jsonl").write_text(json.dumps(trace) + "\n")
            output = root / "review"
            review = viewer.build(manifest=manifest, baseline_root=baseline, candidate_root=candidate, output=output)
            item = review["rows"][0]["checkpoint4"]
            self.assertEqual(item["status"], "censored_invalid_terminal_fallback_prior")
            self.assertFalse(item["terminal_valid"])
            self.assertIsNone(item["path"])
            self.assertTrue(item["fallback_path"])
            self.assertEqual(review["rows"][0]["brief"], "paint case 0")
            self.assertNotIn("file://", (output / "index.html").read_text())

    def test_bounded_user_close_with_valid_final_canvas_is_terminal_valid(self) -> None:
        viewer = load("build_recovery_review")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            cases = []
            for i in range(16):
                brief = root / f"brief-{i}.txt"
                brief.write_text(f"[USER]\npaint case {i}\n")
                cases.append({"id": f"case-{i}", "reference": str(brief)})
            manifest.write_text(json.dumps({"cases": cases}))
            baseline = root / "baseline"
            baseline.mkdir()
            image = baseline / "final.png"
            image.write_bytes(b"final")
            trace = {
                "task": {"data": {"id": "case-0"}},
                "stop_condition": "user_closed",
                "info": {
                    "final_canvas": "/root/painter/final.png",
                    "final_canvas_valid": True,
                    "operational_cutoff": True,
                    "turns": [{"turn": 3, "valid": True, "action": "paint", "canvas": "/root/painter/final.png"}],
                },
            }
            (baseline / "traces.jsonl").write_text(json.dumps(trace) + "\n")
            review = viewer.collect(manifest=manifest, baseline_root=baseline, candidate_root=root / "missing")
            item = review["rows"][0]["flash12"]
            self.assertEqual(item["status"], "valid_terminal")
            self.assertTrue(item["terminal_valid"])
            self.assertTrue(item["completed_at_cap"])

    def test_preflight_is_portable_and_declares_no_trainer(self) -> None:
        script = HERE / "recover_evaluation.sh"
        result = subprocess.run(["bash", str(script), "1", "5.70", "3", "--preflight"], cwd=HERE, text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertFalse(value["training"])
        self.assertEqual(value["trainer_gpus"], 0)
        self.assertEqual(value["inference_gpus"], 1)

    def test_score_validation_rejects_nonfinite_bool_and_out_of_range(self) -> None:
        scorer = load("score_recovery")
        self.assertTrue(scorer._score_value(0.5))
        for value in (True, float("nan"), float("inf"), -0.01, 1.01):
            self.assertFalse(scorer._score_value(value))


if __name__ == "__main__":
    unittest.main()
