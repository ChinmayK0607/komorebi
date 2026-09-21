"""Focused tests for manifest-derived native evaluation counts."""

from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from painter.native_painting_eval import config_text, main, validate_manifest


class NativePaintingEvalTests(unittest.TestCase):
    def test_count_is_inferred_for_scaled_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index in range(3):
                image = root / f"reference-{index}.png"
                image.write_bytes(f"reference-{index}".encode())
                rows.append({
                    "id": f"case-{index}",
                    "source_group": f"group-{index}",
                    "reference": image.name,
                    "reference_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                })
            manifest = root / "dev.json"
            manifest.write_text(json.dumps({"cases": rows}))
            audit = validate_manifest(root, manifest)
            self.assertEqual(audit["count"], 3)
            text = config_text(
                taskset_id="native_painting",
                root=root,
                manifest=manifest,
                policy_label="dev",
                model="model",
                output_dir=root / "out",
                count=audit["count"],
            )
            self.assertIn("num_tasks = 3", text)
            self.assertIn("expected_count = 3", text)
            self.assertNotIn("num_tasks = 12", text)

    def test_controlled_comparison_keeps_default_count_argument_compatibility(self):
        text = config_text(
            taskset_id="native_painting",
            root=Path("/tmp/root"),
            manifest=Path("/tmp/eval.json"),
            policy_label="baseline",
            model="model",
            output_dir=Path("/tmp/output"),
        )
        self.assertIn("num_tasks = 12", text)
        self.assertIn("expected_count = 12", text)
        self.assertIn("max_tokens = 8192", text)
        self.assertIn("max_turns = 6", text)
        self.assertIn("rollout = 1200", text)
        self.assertIn("context_length = 32768", text)

    def test_runtime_parameters_round_trip_into_config_and_metadata_values(self):
        text = config_text(
            taskset_id="native_painting",
            root=Path("/tmp/root"),
            manifest=Path("/tmp/eval.json"),
            policy_label="base",
            model="base",
            output_dir=Path("/tmp/output"),
            max_tokens=8192,
            max_turns=6,
            rollout_timeout=3600,
            context_length=98304,
        )
        self.assertIn("max_tokens = 8192", text)
        self.assertIn("max_turns = 6", text)
        self.assertIn("rollout = 3600", text)
        self.assertIn("context_length = 98304", text)

    def test_runtime_parameters_reject_non_positive_values(self):
        kwargs = dict(
            taskset_id="native_painting",
            root=Path("/tmp/root"),
            manifest=Path("/tmp/eval.json"),
            policy_label="base",
            model="base",
            output_dir=Path("/tmp/output"),
        )
        for name in ("max_tokens", "max_turns", "rollout_timeout", "context_length"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    config_text(**kwargs, **{name: 0})

    def test_cli_report_round_trips_runtime_parameters(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "reference.png"
            image.write_bytes(b"reference")
            manifest = root / "eval.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "case",
                "source_group": "group",
                "reference": image.name,
            }]}))
            config = root / "config.toml"
            output = io.StringIO()
            with redirect_stdout(output):
                result = main([
                    "--root", str(root),
                    "--manifest", str(manifest),
                    "--policy-label", "base",
                    "--model", "base",
                    "--output-dir", str(root / "out"),
                    "--config", str(config),
                    "--max-tokens", "8192",
                    "--max-turns", "6",
                    "--rollout-timeout", "3600",
                    "--context-length", "98304",
                ])
            self.assertEqual(result, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["protocol"]["context_length"], 98304)
            self.assertEqual(report["protocol"]["rollout_timeout"], 3600)
            self.assertIn("rollout = 3600", config.read_text())

    def test_expected_count_can_still_guard_a_controlled_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(2):
                (root / f"{index}.png").write_bytes(str(index).encode())
            manifest = root / "eval.jsonl"
            manifest.write_text("\n".join(json.dumps({
                "id": str(index),
                "source_group": str(index),
                "reference": f"{index}.png",
            }) for index in range(2)) + "\n")
            with self.assertRaisesRegex(ValueError, "expected 12"):
                validate_manifest(root, manifest, expected_count=12)


if __name__ == "__main__":
    unittest.main()
