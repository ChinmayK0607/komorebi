"""Offline checks for the evidence archive's native-training coverage."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mixed_publish_evidence", HERE / "publish_evidence.py")
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class EvidenceSelectionTests(unittest.TestCase):
    def test_step_four_requires_and_packages_native_traces_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old, evaluation, pack = (base / name for name in ("old", "evaluation", "pack"))
            telemetry = base / "gpu.csv"
            run = old / "async-output" / publisher.RUN_NAME
            matched = evaluation / "matched-eval-step-4"
            files = [
                old / "native-rl-mixed-anchors.toml",
                *(pack / name for name in ("train-manifest.json", "heldout-cases.json", "provenance.json")),
                *(matched / name for name in (
                    "identity.json", "cases.json", "completion.json", "review/index.html",
                    "review/report.json", "review/judge.json")),
                telemetry,
                run / "artifacts/adapters/step_4/hf-upload.json",
                run / "monitors/file/traces/stream/00000.jsonl",
                run / "monitors/file/metrics.jsonl",
                run / "logs/attempt_1/orchestrator.log",
                run / "logs/attempt_1/trainer.log",
            ]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}\n")
            selected = publisher.select(old, evaluation, pack, telemetry, 4)
            names = {name for _, name in selected}
            self.assertIn("prime/traces/stream/00000.jsonl", names)
            self.assertIn("prime/metrics.jsonl", names)
            self.assertIn("prime/orchestrator.log", names)
            self.assertIn("prime/trainer.log", names)
            self.assertEqual(len(names), len(selected))
            (run / "monitors/file/traces/stream/00000.jsonl").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "prime/traces"):
                publisher.select(old, evaluation, pack, telemetry, 4)


if __name__ == "__main__":
    unittest.main()
