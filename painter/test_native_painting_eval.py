"""Focused tests for manifest-derived native evaluation counts."""

from __future__ import annotations

import hashlib
import io
import json
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from painter.native_painting_eval import audit_evaluation_artifacts, config_text, main, validate_manifest


class NativePaintingEvalTests(unittest.TestCase):
    def test_config_places_render_artifacts_under_absolute_output_root(self):
        with TemporaryDirectory() as temp:
            run_root = Path(temp) / "archived-run"
            output = run_root / "eval"
            text = config_text(
                taskset_id="native_painting",
                root=run_root,
                manifest=run_root / "cases.json",
                policy_label="step-16",
                model="model",
                output_dir=output,
            )
            config = tomllib.loads(text)
            configured_output = Path(config["output_dir"])
            artifact_root = Path(config["env"]["taskset"]["output_root"])
            self.assertTrue(configured_output.is_absolute())
            self.assertTrue(artifact_root.is_absolute())
            self.assertEqual(artifact_root, output.resolve() / "evaluation-rollouts")
            self.assertTrue(artifact_root.is_relative_to(configured_output))

    def test_artifact_audit_requires_hash_verified_png_for_valid_receipts(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "evaluation-rollouts"
            valid = root / "policy" / "case-valid" / "episode" / "turn-1"
            invalid = root / "policy" / "case-invalid" / "episode" / "turn-1"
            valid.mkdir(parents=True)
            invalid.mkdir(parents=True)
            canvas = b"verified-canvas"
            (valid / "painting.png").write_bytes(canvas)
            (valid / "painting.json").write_text(json.dumps({
                "valid": True,
                "png_sha256": hashlib.sha256(canvas).hexdigest(),
            }))
            (invalid / "painting.json").write_text(json.dumps({
                "valid": False,
                "png_sha256": "unused",
            }))
            audit = audit_evaluation_artifacts(root)
            self.assertEqual(audit["receipt_count"], 2)
            self.assertEqual(audit["valid_receipts"], 1)
            self.assertEqual(audit["valid_receipts_with_png"], 1)
            self.assertEqual(audit["invalid_receipts"], 1)

    def test_artifact_audit_rejects_valid_receipt_without_canvas(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "evaluation-rollouts" / "policy" / "case" / "episode" / "turn-1"
            root.mkdir(parents=True)
            (root / "painting.json").write_text(json.dumps({
                "valid": True,
                "png_sha256": "0" * 64,
            }))
            with self.assertRaisesRegex(ValueError, "missing_png=1"):
                audit_evaluation_artifacts(root.parents[3])

    def test_artifact_audit_rejects_symlinked_receipt_and_png(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "evaluation-rollouts"
            turn = root / "policy" / "case" / "episode" / "turn-1"
            turn.mkdir(parents=True)
            outside = base / "outside"
            outside.mkdir()
            outside_png = outside / "painting.png"
            outside_png.write_bytes(b"outside-canvas")
            receipt = {
                "valid": True,
                "png_sha256": hashlib.sha256(outside_png.read_bytes()).hexdigest(),
            }
            (outside / "painting.json").write_text(json.dumps(receipt))
            try:
                (turn / "painting.json").symlink_to(outside / "painting.json")
            except OSError as exc:  # pragma: no cover - platform permission guard
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "receipt must not be a symlink"):
                audit_evaluation_artifacts(root)

            (turn / "painting.json").unlink()
            (turn / "painting.json").write_text(json.dumps(receipt))
            (turn / "painting.png").symlink_to(outside_png)
            with self.assertRaisesRegex(ValueError, "PNG must not be a symlink"):
                audit_evaluation_artifacts(root)

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
        self.assertIn("max_concurrent = 4", text)
        self.assertIn("max_tokens = 8192", text)
        self.assertIn("max_turns = 6", text)
        self.assertIn("rollout = 1200", text)
        self.assertIn("context_length = 32768", text)
        self.assertNotIn("reasoning_effort", text)
        self.assertIn("enable_thinking = false", text)

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
            max_concurrent=16,
        )
        self.assertIn("max_tokens = 8192", text)
        self.assertIn("max_turns = 6", text)
        self.assertIn("rollout = 3600", text)
        self.assertIn("context_length = 98304", text)
        self.assertIn("max_concurrent = 16", text)

    def test_runtime_parameters_reject_non_positive_values(self):
        kwargs = dict(
            taskset_id="native_painting",
            root=Path("/tmp/root"),
            manifest=Path("/tmp/eval.json"),
            policy_label="base",
            model="base",
            output_dir=Path("/tmp/output"),
        )
        for name in ("max_tokens", "max_turns", "rollout_timeout", "context_length", "max_concurrent"):
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
                    "--max-concurrent", "16",
                ])
            self.assertEqual(result, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["protocol"]["context_length"], 98304)
            self.assertEqual(report["protocol"]["rollout_timeout"], 3600)
            self.assertEqual(report["protocol"]["max_concurrent"], 16)
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
