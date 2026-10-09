"""Offline tests for bounded train-only anchor censoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnchorAdmissionTests(unittest.TestCase):
    def test_one_model_invalid_train_case_is_censored_after_retry(self):
        bank = load("structured_flow_anchor_admission", RUN / "anchor_bank.py")
        train = [{"id": f"train-{index:02d}"} for index in range(32)]
        heldout = [{"id": f"heldout-{index:02d}"} for index in range(16)]
        excluded, records = bank._classify_train_censoring(
            train,
            heldout,
            {
                "train-07": {
                    "initial": {"status": "invalid_model_output", "retry_index": 0, "case_seed": 1},
                    "retry": {"status": "invalid_model_output", "retry_index": 1, "case_seed": 2},
                }
            },
        )
        self.assertEqual(excluded, {"train-07"})
        self.assertEqual(records[0]["reason"], "invalid_model_output_after_bounded_retry")
        self.assertEqual([attempt["retry_index"] for attempt in records[0]["attempts"]], [0, 1])

    def test_censoring_below_minimum_fails_closed(self):
        bank = load("structured_flow_anchor_admission_min", RUN / "anchor_bank.py")
        train = [{"id": f"train-{index:02d}"} for index in range(32)]
        heldout = [{"id": f"heldout-{index:02d}"} for index in range(16)]
        failures = {
            row["id"]: {"status": "invalid_model_output", "retry_index": 1}
            for row in train[:9]
        }
        with self.assertRaisesRegex(RuntimeError, "minimum"):
            bank._classify_train_censoring(train, heldout, failures)

    def test_heldout_and_infrastructure_failures_cannot_be_censored(self):
        bank = load("structured_flow_anchor_admission_boundaries", RUN / "anchor_bank.py")
        train = [{"id": f"train-{index:02d}"} for index in range(32)]
        heldout = [{"id": f"heldout-{index:02d}"} for index in range(16)]
        with self.assertRaisesRegex(RuntimeError, "heldout"):
            bank._classify_train_censoring(
                train, heldout, {"heldout-03": {"status": "invalid_model_output"}}
            )
        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            bank._classify_train_censoring(
                train, heldout, {"train-03": {"status": "unavailable"}}
            )
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            bank._classify_train_censoring(
                train, heldout, {"train-04": {"status": "invalid_model_output", "retry_index": 0}}
            )

    def test_repeated_materialization_returns_filtered_count_without_generation(self):
        bank = load("structured_flow_anchor_admission_repeat", RUN / "anchor_bank.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def rows(prefix: str, count: int) -> list[dict[str, str]]:
                result = []
                for index in range(count):
                    opponent = root / f"{prefix}-{index:02d}.png"
                    opponent.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
                    result.append({"id": f"{prefix}-{index:02d}", "opponent": str(opponent), "opponent_sha256": "a" * 64})
                return result
            train_path = root / "train.json"
            eval_path = root / "eval.json"
            train_path.write_text(json.dumps({"rows": rows("train", 31)}))
            eval_path.write_text(json.dumps({"cases": rows("heldout", 16)}))
            result = bank.materialize(
                train_manifest=train_path,
                heldout_manifest=eval_path,
                anchor_root=root / "anchors",
                output_dir=root / "staging",
                run_root=root,
            )
            self.assertEqual(result, {"status": "already_materialized", "train_rows": 31, "heldout_cases": 16})

    def test_final_config_receipt_rebinds_expected_count_to_admitted_manifest(self):
        config = load("structured_flow_config_for_admission", RUN / "prepare_structured_config.py")
        runner = load("structured_flow_runner_for_admission", RUN / "run_from_sft.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def make_rows(count: int) -> list[dict[str, str]]:
                result = []
                for index in range(count):
                    result.append({
                        "id": f"train-{index:02d}",
                        "reference_kind": "text",
                        "reference": f"brief-{index:02d}.txt",
                        "opponent": f"anchor-{index:02d}.png",
                        "opponent_sha256": "b" * 64,
                        "anchor_status": "verified_file",
                    })
                return result
            manifest = root / "curriculum/train.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({"rows": make_rows(32)}))
            native = root / "native.toml"
            config.prepare(output=native, model=Path("/tmp/model"), run_root=root, manifest=manifest, expected_count=32)
            admitted = root / "curriculum/admitted.json"
            admitted.write_text(json.dumps({"rows": make_rows(31)}))
            runner._refresh_final_config(native, admitted, 31)
            parsed = tomllib.loads(native.read_text())
            taskset = parsed["orchestrator"]["train"]["source"][0]["env"]["taskset"]
            receipt = json.loads((root / "rl-config-receipt.json").read_text())
            self.assertEqual(taskset["expected_count"], 31)
            self.assertEqual(receipt["curriculum"]["pilot_rows"], 31)
            self.assertEqual(receipt["manifest_sha256"], hashlib.sha256(admitted.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
