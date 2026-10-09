"""Offline contract checks for the strict same-run evidence resume."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent
PINNED_URL = (
    "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
    "c35b77c6d3a858ab224f0812f1ed73e0150d1525/runs/diverse-flow-rl-20261009/evidence.tar.gz"
)
PINNED_SHA = "ddd90296e2c6562cb24bc8f8033f439c4925e77002ad8eabca2723c38323eaac"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SameRunResumeTests(unittest.TestCase):
    def test_strict_archive_pin_rejects_revision_or_hash_mismatch(self):
        restore = load("diverse_flow_resume_restore_contract", RUN / "restore_anchor_evidence.py")
        self.assertEqual(restore.SAME_RUN_ARCHIVE_URL, PINNED_URL)
        self.assertEqual(restore.SAME_RUN_ARCHIVE_SHA256, PINNED_SHA)
        restore.validate_same_run_archive_pin(url=PINNED_URL, expected_sha256=PINNED_SHA)
        with self.assertRaises(ValueError):
            restore.validate_same_run_archive_pin(url=PINNED_URL.replace("c35b77c6", "main0000"), expected_sha256=PINNED_SHA)
        with self.assertRaises(ValueError):
            restore.validate_same_run_archive_pin(url=PINNED_URL, expected_sha256="0" * 64)

    def test_schema_shaped_manifest_uses_prepare_curriculum_contract(self):
        restore = load("diverse_flow_resume_manifest_contract", RUN / "restore_anchor_evidence.py")
        prepare = load("diverse_flow_resume_curriculum_contract", RUN / "prepare_curriculum.py")
        train = {
            "schema": prepare.NATIVE_SCHEMA,
            "run_id": "diverse-flow-rl-20261009",
            "split": "train",
            "native_rl_compatible": True,
            "rows": [{"id": f"case-{index}"} for index in range(prepare.PILOT_TRAIN_COUNT)],
        }
        self.assertEqual(prepare.PILOT_TRAIN_COUNT, 96)
        self.assertEqual(prepare.PILOT_HELDOUT_COUNT, 32)
        self.assertEqual(restore._manifest_rows(train, name="train", expected_count=96), train["rows"])
        train["schema"] = "native-painting-rl-Diverse-flow-v1"
        with self.assertRaisesRegex(ValueError, "pinned native"):
            restore._manifest_rows(train, name="train", expected_count=1)

    def test_archived_curriculum_paths_rebase_to_current_run(self):
        restore = load("diverse_flow_resume_path_contract", RUN / "restore_anchor_evidence.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = root / "curriculum" / "train-proposal.json"
            prompt = root / "curriculum" / "prompts" / "train" / "case-1.txt"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("a brief\n", encoding="utf-8")
            proposal.write_text("{}", encoding="utf-8")
            proposal_row = {"reference": str(prompt), "reference_sha256": restore.sha256_file(prompt)}
            anchor_root = root / "curriculum" / "anchors"
            row = {
                "id": "case-1",
                "reference": "/root/painter/repo/curriculum/prompts/train/case-1.txt",
                "source_prompt": "/root/painter/repo/curriculum/prompts/train/case-1.txt",
                "casebrief": "/root/painter/repo/curriculum/.materialized-pilot/casebriefs/train/case-1.txt",
                "opponent": "/root/painter/repo/curriculum/anchors/case-1.png",
                "anchor_provenance_file": "/root/painter/repo/curriculum/anchors/anchor-provenance.json",
            }
            rebased = restore._rebase_cached_row(
                row, proposal_row, proposal_path=proposal,
                anchor_root=anchor_root, run_root=root,
            )
            self.assertEqual(rebased["reference"], str(prompt))
            self.assertEqual(rebased["source_prompt"], str(prompt))
            self.assertEqual(
                rebased["casebrief"],
                str(root / "curriculum" / ".materialized-pilot" / "casebriefs" / "train" / "case-1.txt"),
            )
            self.assertEqual(rebased["opponent"], str(anchor_root / "case-1.png"))
            self.assertEqual(rebased["anchor_provenance_file"], str(anchor_root / "anchor-provenance.json"))

    def test_actual_disabled_casebrief_marker_is_required(self):
        restore = load("diverse_flow_resume_casebrief_contract", RUN / "restore_anchor_evidence.py")
        actual = "[SYSTEM]\nStructured-flow hint is disabled for this run.\n\n[USER]\nbrief text\n"
        restore._validate_cached_casebrief_text(actual, "brief text")
        with self.assertRaisesRegex(ValueError, "canonical disabled"):
            restore._validate_cached_casebrief_text(
                actual.replace("Structured-flow", "Diverse-flow"), "brief text",
            )

    def test_prepare_orders_anchor_then_manifest_restore_and_gates_lazy_generation(self):
        prepare = (RUN / "prepare_node.sh").read_text(encoding="utf-8")
        self.assertLess(prepare.index('"$PY_PRIME" "$RESTORE_SCRIPT"'), prepare.index('restore_cached_pilot_manifests'))
        self.assertLess(prepare.index('restore_cached_pilot_manifests'), prepare.index('anchor_bank.py" prepare-lazy'))
        self.assertIn('if [[ "$SAME_RUN_RESUME" == 0 ]]; then', prepare)
        self.assertIn('generator_calls") != 0', prepare)

    def test_validated_baseline_resume_receipt_has_no_generation_calls(self):
        shell = (RUN / "matched_baseline.sh").read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", shell, flags=re.DOTALL)
        block = next(item for item in blocks if 'reused_cached_baseline_no_generation' in item)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            run = root / "run"
            run.mkdir()
            sha = PINNED_SHA
            (run / "baseline-restore.json").write_text(json.dumps({
                "source_archive_sha256": sha,
                "validation": {
                    "eval": {"seed": 918210, "max_turns": 3, "max_tokens": 8192},
                    "prompt": {"case_count": 32},
                    "rollouts": {"receipt_count": 32},
                    "invalid_or_censored_retained": True,
                },
            }), encoding="utf-8")
            result = subprocess.run(
                ["python3", "-", str(out), str(run), sha],
                input=block, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((out / "baseline-reuse-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["generator_calls"], 0)
            self.assertEqual(receipt["model_calls"], 0)
            self.assertEqual(receipt["renderer_calls"], 0)
            self.assertEqual(receipt["judge_calls"], 0)

    def test_controller_forwards_only_the_pinned_credential_free_coordinates(self):
        operator = load("diverse_flow_resume_controller_contract", RUN / "operate_single.py")
        with patch.dict(os.environ, {"PAINTER_RESUME_SAME_RUN_ARCHIVE": "1"}, clear=False):
            command = operator._detached_job_command(
                source_commit="a" * 40, rental_start_epoch=100.0,
                price_usd_per_hour=1.52, cap_usd=15.0,
            )
        self.assertIn("PAINTER_RESUME_SAME_RUN_ARCHIVE=1", command)
        self.assertIn(f"PAINTER_SAME_RUN_ARCHIVE_SHA256={PINNED_SHA}", command)
        self.assertIn(f"PAINTER_SAME_RUN_ARCHIVE_URL={PINNED_URL}", command)
        self.assertNotIn("PAINTER_ANCHOR_EVIDENCE_ARCHIVE", command)
        with patch.dict(os.environ, {
            "PAINTER_RESUME_SAME_RUN_ARCHIVE": "1",
            "PAINTER_SAME_RUN_ARCHIVE_SHA256": "0" * 64,
        }, clear=False):
            with self.assertRaisesRegex(ValueError, "pinned public archive"):
                operator._detached_job_command(
                    source_commit="a" * 40, rental_start_epoch=100.0,
                    price_usd_per_hour=1.52, cap_usd=15.0,
                )

    def test_same_run_url_callers_enforce_the_python_pin(self):
        restore = load("diverse_flow_resume_url_contract", RUN / "restore_anchor_evidence.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "pinned public revision"):
                restore.restore(run_root=root, expected_sha256=PINNED_SHA, url=PINNED_URL.replace("c35b77c6", "main0000"))

    def test_baseline_binds_archived_and_installed_manifest_hashes(self):
        restore = load("diverse_flow_resume_baseline_hash_contract", RUN / "restore_anchor_evidence.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cached-anchor-reuse.json").write_text(json.dumps({
                "schema": "Diverse-flow-cached-anchor-reuse-v1",
                "manifest_hashes": {
                    "archived": {"matched32-manifest.json": PINNED_SHA},
                    "installed": {"matched32-manifest.json": "a" * 64},
                },
            }), encoding="utf-8")
            binding = restore._validate_cached_manifest_rebase(
                root, archived_manifest_sha256=PINNED_SHA, installed_manifest_sha256="a" * 64,
            )
            self.assertEqual(binding, {"archived": PINNED_SHA, "installed": "a" * 64})
            with self.assertRaisesRegex(ValueError, "archived matched32"):
                restore._validate_cached_manifest_rebase(
                    root, archived_manifest_sha256="b" * 64, installed_manifest_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
