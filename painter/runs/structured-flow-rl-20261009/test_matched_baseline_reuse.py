"""Metadata-only regression tests for the paid baseline cache path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import re


RUN = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MatchedBaselineReuseTests(unittest.TestCase):
    def test_restores_exact_archived_admitted_manifests_and_disabled_casebriefs(self):
        restore = load_module(RUN / "restore_anchor_evidence.py", "structured_flow_restore_admitted_test")
        curriculum = load_module(RUN / "prepare_curriculum.py", "structured_flow_curriculum_admitted_test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            shutil.copyfile(RUN / "flow_demonstrations.py", run / "flow_demonstrations.py")
            curriculum_root = run / "curriculum"
            curriculum.build_proposal(curriculum_root)
            train_proposal = curriculum_root / "train-proposal.json"
            heldout_proposal = curriculum_root / "heldout-proposal.json"
            train_doc = json.loads(train_proposal.read_text(encoding="utf-8"))
            heldout_doc = json.loads(heldout_proposal.read_text(encoding="utf-8"))
            plan = curriculum.build_pilot_plan(train_doc["rows"], heldout_doc["rows"])
            by_id = {str(row["id"]): row for row in [*train_doc["rows"], *heldout_doc["rows"]]}
            anchors = curriculum_root / "anchors"
            anchors.mkdir()
            for case_id in [*plan["train_ids"], *plan["heldout_ids"]]:
                (anchors / f"{case_id}.png").write_bytes(f"anchor:{case_id}".encode())
            stage = curriculum_root / ".materialized-pilot_m_tgr81x"
            casebrief_root = stage / "casebriefs"
            casebrief_root.mkdir(parents=True)
            def admitted(case_id: str) -> dict[str, object]:
                proposal = by_id[case_id]
                split = str(proposal["split"])
                prompt = Path(str(proposal["reference"]))
                if not prompt.is_absolute():
                    prompt = train_proposal.parent / prompt
                anchor = anchors / f"{case_id}.png"
                casebrief = casebrief_root / split / f"{case_id}.txt"
                casebrief.parent.mkdir(parents=True, exist_ok=True)
                casebrief.write_text(
                    "[SYSTEM]\nStructured-flow hint is disabled for this run.\n\n[USER]\n" + prompt.read_text(encoding="utf-8").strip() + "\n",
                    encoding="utf-8",
                )
                return {
                    **proposal,
                    "reference": str(prompt.absolute()),
                    "opponent": str(anchor.absolute()),
                    "opponent_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
                    "opponent_source_group": proposal["source_group"],
                    "frozen_sft512_checkpoint_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
                    "admission_status": "ready_for_native_painting_rl",
                    "anchor_status": "verified_file",
                    "casebrief": str(casebrief.absolute()),
                    "casebrief_sha256": hashlib.sha256(casebrief.read_bytes()).hexdigest(),
                    "system_hint_sha256": hashlib.sha256(("Structured-flow hint is disabled for this run.\n").encode()).hexdigest(),
                    "max_turns": 3,
                }
            train_rows = [admitted(case_id) for case_id in plan["train_ids"][:-1]]
            heldout_rows = [admitted(case_id) for case_id in plan["heldout_ids"]]
            common = {"schema": "painter.native-painting-rl-structured-flow-v1", "run_id": curriculum.RUN_ID, "status": "admitted_with_verified_frozen_sft512_anchors"}
            train_value = {**common, "split": "train", "native_rl_compatible": True, "rows": train_rows}
            heldout_value = {**common, "split": "heldout", "native_rl_compatible": False, "rows": heldout_rows, "cases": heldout_rows}
            archive_root = root / "archive" / "runs" / curriculum.RUN_ID
            archive_stage = archive_root / "curriculum" / stage.name
            archive_stage.mkdir(parents=True)
            for name, value in (("pilot-train-manifest.json", train_value), ("pilot-heldout-manifest.json", heldout_value), ("matched16-manifest.json", heldout_value)):
                (archive_stage / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            shutil.copytree(casebrief_root, archive_stage / "casebriefs")
            shutil.copyfile(run / "flow_demonstrations.py", archive_root / "flow_demonstrations.py")
            archive = root / "cached.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(root / "archive" / "runs", arcname="runs")
            train_manifest = curriculum_root / "pilot-train-manifest.json"
            heldout_manifest = curriculum_root / "matched16-manifest.json"
            train_manifest.write_text("{}\n", encoding="utf-8")
            heldout_manifest.write_text("{}\n", encoding="utf-8")
            result = restore.restore_cached_pilot_manifests(
                run_root=run,
                expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                train_proposal=train_proposal,
                heldout_proposal=heldout_proposal,
                anchor_root=anchors,
                train_manifest=train_manifest,
                heldout_manifest=heldout_manifest,
                flow_source=run / "flow_demonstrations.py",
                archive=archive,
            )
            self.assertEqual(result["train_rows"], 31)
            self.assertEqual(result["heldout_cases"], 16)
            self.assertEqual(result["casebrief_mode"], "disabled_casebrief_with_flow_hook")
            self.assertEqual(result["train_manifest_sha256"], hashlib.sha256((archive_stage / "pilot-train-manifest.json").read_bytes()).hexdigest())
            self.assertEqual(result["heldout_manifest_sha256"], hashlib.sha256((archive_stage / "matched16-manifest.json").read_bytes()).hexdigest())

    def test_shell_reuse_receipt_inline_block_executes_with_string_archive_sha(self):
        shell = (RUN / "matched_baseline.sh").read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", shell, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            run = root / "run"
            stage = root / "stage"
            run.mkdir()
            stage.mkdir()
            archive_sha = "f" * 64
            (run / "baseline-restore.json").write_text(json.dumps({
                "source_archive_sha256": archive_sha,
                "validation": {"eval": {"seed": 918210}, "invalid_or_censored_retained": True},
            }), encoding="utf-8")
            (run / "cached-anchor-reuse.json").write_text(json.dumps({
                "generator_calls": 0, "train_rows": 31, "heldout_cases": 16,
            }), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-", str(out), str(run), archive_sha, str(stage)],
                input=blocks[1], text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((out / "baseline-reuse-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["source_archive_sha256"], archive_sha)
            self.assertEqual(receipt["train_rows"], 31)
            self.assertEqual(receipt["heldout_cases"], 16)
            self.assertEqual(receipt["generator_calls"], 0)

    def test_baseline_allowlist_keeps_invalid_receipts_as_raw_evidence(self):
        restore = load_module(RUN / "restore_anchor_evidence.py", "structured_flow_restore_allowlist_test")
        with self.assertRaises(ValueError):
            restore._normalized_baseline_member("matched-baseline/pre-step12/model/weights.bin")
        self.assertEqual(
            restore._normalized_baseline_member("public/matched-baseline/pre-step12/evaluation-rollouts/case/painting.json").as_posix(),
            "matched-baseline/pre-step12/evaluation-rollouts/case/painting.json",
        )

    def test_restores_hash_bound_baseline_and_reports_censored_receipts(self):
        restore = load_module(RUN / "restore_anchor_evidence.py", "structured_flow_restore_validation_test")
        flow = load_module(RUN / "flow_demonstrations.py", "structured_flow_baseline_prompt_test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer = root / "renderer.py"
            renderer.write_text("pinned renderer\n", encoding="utf-8")
            manifest = root / "matched16.json"
            briefs = root / "briefs"
            briefs.mkdir()
            references = root / "references"
            opponents = root / "opponents"
            references.mkdir()
            opponents.mkdir()
            rows = []
            for index in range(16):
                brief = briefs / f"{index}.txt"
                prompt = references / f"{index}.txt"
                opponent = opponents / f"{index}.png"
                prompt.write_text(f"brief {index}\n", encoding="utf-8")
                opponent.write_bytes(f"opponent {index}\n".encode())
                brief.write_text(f"[SYSTEM]\n{flow.FLOW_HINT}\n\n[USER]\nbrief {index}\n", encoding="utf-8")
                rows.append({
                    "id": str(index),
                    "reference": str(prompt),
                    "reference_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                    "opponent": str(opponent),
                    "opponent_sha256": hashlib.sha256(opponent.read_bytes()).hexdigest(),
                    "casebrief": str(brief),
                    "casebrief_sha256": hashlib.sha256(brief.read_bytes()).hexdigest(),
                    "system_hint_sha256": hashlib.sha256((flow.FLOW_HINT + "\n").encode()).hexdigest(),
                })
            manifest.write_text(json.dumps({"cases": rows}), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            renderer_sha = hashlib.sha256(renderer.read_bytes()).hexdigest()
            evidence = root / "evidence" / "matched-baseline" / "pre-step12"
            rollout = evidence / "evaluation-rollouts" / "case-00"
            rollout.mkdir(parents=True)
            canvas = rollout / "painting.png"
            canvas.write_bytes(b"png\n")
            (rollout / "painting.json").write_text(json.dumps({
                "valid": True, "renderer_sha256": renderer_sha,
                "png_sha256": hashlib.sha256(canvas.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            censored = evidence / "evaluation-rollouts" / "case-01"
            censored.mkdir()
            (censored / "painting.json").write_text(json.dumps({"valid": False, "reason": "censored"}), encoding="utf-8")
            (evidence / "baseline-receipt.json").write_text(json.dumps({
                "schema": "structured-flow-matched-flash12-baseline-v1", "status": "complete",
                "policy": {"label": "flash12-pre-warmup", "revision": restore.BASELINE_POLICY_REVISION, "prefix": restore.BASELINE_POLICY_PREFIX, "adapter_sha256": restore.BASELINE_POLICY_SHA256},
                "manifest_sha256": manifest_sha, "manifest_cases": 16, "max_turns": 3, "seed": 918210,
                "judge_deferred": True, "quality_scoring": "offline_only",
            }), encoding="utf-8")
            (evidence / "eval.toml").write_text(
                'model = "flash12-baseline"\nnum_tasks = 16\nnum_rollouts = 1\nmax_tokens = 8192\nseed = 918210\nmax_turns = 3\npolicy_label = "flash12-pre-warmup"\nid = "native_painting"\npush = false\nrenderer = "renderer.py"\nrenderer_python = "python"\n',
                encoding="utf-8",
            )
            (evidence / "structured-flow-hook-receipt.json").write_text(json.dumps({
                "schema": "structured-flow-prompt-hook-v1", "installed": True, "enabled": True,
                "evaluation_taskset_patched": True, "eval_manifest_sha256": manifest_sha, "eval_rows": 16,
            }), encoding="utf-8")
            archive = root / "baseline.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(root / "evidence", arcname="runs/structured-flow-rl-20261009")
            run = root / "run"
            run.mkdir()
            receipt = restore.restore_baseline(
                run_root=run,
                expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                renderer_sha256=renderer_sha,
                renderer_path=renderer,
                eval_manifest=manifest,
                flow_source=RUN / "flow_demonstrations.py",
                archive=archive,
            )
            self.assertTrue(receipt["validation"]["invalid_or_censored_retained"])
            self.assertEqual(receipt["validation"]["rollouts"]["invalid_receipts"], 1)


if __name__ == "__main__":
    unittest.main()
