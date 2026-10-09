"""Offline tests for the opt-in public anchor evidence restore."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


RUN = Path(__file__).resolve().parent
SCRIPT = RUN / "restore_anchor_evidence.py"


def load_module():
    spec = importlib.util.spec_from_file_location("structured_flow_anchor_restore", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("restore helper is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnchorRestoreTests(unittest.TestCase):
    def _archive(self, root: Path, *, bad_seed: bool = False, bad_painting_hash: bool = False) -> tuple[Path, str]:
        module = load_module()
        evidence = root / "evidence-source"
        batch = evidence / "runs/structured-flow-rl-20261009/curriculum/anchors/native-generation/batch-old"
        turn = batch / "case-01/turn-1"
        turn.mkdir(parents=True)
        source = turn / "program.js"
        source.write_text("function draw(){background('white');noLoop();}\n", encoding="utf-8")
        canvas = turn / "painting.png"
        canvas.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        png_sha = hashlib.sha256(canvas.read_bytes()).hexdigest()
        painting = {
            "valid": True,
            "source": "/old/node/case-01/turn-1/program.js",
            "source_sha256": source_sha if not bad_painting_hash else "0" * 64,
            "png_sha256": png_sha,
        }
        (turn / "painting.json").write_text(json.dumps(painting), encoding="utf-8")
        reference_sha = "a" * 64
        seed = module._case_seed("case-01", reference_sha, 0)
        if bad_seed:
            seed += 1
        manifest = {
            "cases": [{
                "id": "case-01",
                "source_group": "case-01",
                "reference_sha256": reference_sha,
                "reference_kind": "text",
                "max_turns": 3,
                "case_seed": seed,
            }],
        }
        (batch / "native-eval-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (batch / "native-eval.toml").write_text('seed = 20261009\n', encoding="utf-8")
        archive = root / "evidence.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(evidence / "runs", arcname="runs")
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_restores_hash_verified_evidence_and_records_public_identity(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root)
            (root / "run").mkdir()
            receipt = module.restore(run_root=root / "run", expected_sha256=digest, archive=archive)
            self.assertEqual(receipt["status"], "restored_verified_public_evidence")
            self.assertEqual(receipt["source_archive_sha256"], digest)
            restored = root / "run/curriculum/anchors/native-generation/batch-old/case-01/turn-1/painting.png"
            self.assertTrue(restored.is_file())
            receipt_path = root / "run/anchor-restore-receipt.json"
            self.assertTrue(receipt_path.is_file())
            original_receipt = receipt_path.read_bytes()
            replay = module.restore(run_root=root / "run", expected_sha256=digest, archive=archive)
            self.assertEqual(replay, receipt)
            self.assertEqual(receipt_path.read_bytes(), original_receipt)


    def test_restores_and_replays_hash_bound_supervision_subset(self):
        module = load_module()
        renderer_sha = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence" / "public-wrapper" / "code-supervision"
            programs = evidence / "programs"
            renders = evidence / "renders"
            briefs = evidence / "supervision-briefs"
            rows, sources, results, candidates = [], [], [], []
            for index in range(16):
                candidate_id = f"candidate-{index:02d}"
                split = "train" if index < 12 else "dev"
                program_paths = []
                for version in ("block_in", "improved_final"):
                    rel = Path("programs") / candidate_id / f"{version}.js"
                    path = evidence / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"// {candidate_id} {version}\n", encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    sources.append({"path": rel.as_posix(), "sha256": digest, "bytes": path.stat().st_size, "candidate_id": candidate_id, "version": version, "standalone": True})
                    program_paths.append(rel.as_posix())
                brief_rel = Path("supervision-briefs") / f"{candidate_id}.txt"
                brief = evidence / brief_rel
                brief.parent.mkdir(parents=True, exist_ok=True)
                brief.write_text(f"Brief {candidate_id}\n", encoding="utf-8")
                rows.append({"id": candidate_id, "split": split, "programs": program_paths, "source_group": candidate_id})
                stage_hashes = {}
                for version in ("block_in", "improved_final"):
                    rel = Path("renders") / candidate_id / f"{version}.png"
                    path = evidence / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"PNG {candidate_id} {version}".encode())
                    png_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                    source_sha = next(x["sha256"] for x in sources if x["path"] == (Path("programs") / candidate_id / f"{version}.js").as_posix())
                    results.append({"candidate_id": candidate_id, "version": version, "valid": True, "source_sha256": source_sha, "png_sha256": png_sha, "receipt": {"valid": True, "renderer_sha256": renderer_sha}})
                    stage_hashes[version] = png_sha
                candidates.append({"id": candidate_id, "brief": brief_rel.as_posix(), "brief_sha256": hashlib.sha256(brief.read_bytes()).hexdigest(), "initial_png_sha256": stage_hashes["block_in"], "final_png_sha256": stage_hashes["improved_final"], "decision": "approved" if index < 6 else "censored"})
            source_manifest = {"schema": "painter.structured-flow-program.v1", "run_id": module.RUN_ID, "sources": sources}
            source_path = evidence / "source-manifest.json"
            source_path.write_text(json.dumps(source_manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            manifest = {"schema": "painter.structured-flow-candidates.v1", "run_id": module.RUN_ID, "program_count": 32, "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "rows": rows}
            (evidence / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (evidence / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            render_manifest = {"schema": "painter.structured-flow-render.v1", "run_id": module.RUN_ID, "program_count": 32, "valid_program_count": 32, "complete_hash_verified": True, "renderer": {"renderer_sha256": renderer_sha}, "source_manifest_sha256": manifest["source_manifest_sha256"], "results": results}
            (evidence / "render-manifest.json").write_text(json.dumps(render_manifest), encoding="utf-8")
            approval = {"schema": "painter.structured-flow-approval-candidates.v1", "run_id": module.RUN_ID, "status": "reviewed", "review_protocol": "painter.text_pairwise_consistent_side_judge_v2_generic", "judge_model": "xiaomi/mimo-v2.6-flash", "candidate_count": 16, "approved_count": 6, "candidates": candidates}
            (evidence / "approval-candidates.json").write_text(json.dumps(approval), encoding="utf-8")
            archive = root / "evidence.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(evidence.parent.parent, arcname="public-wrapper")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            run = root / "run"
            run.mkdir()
            receipt = module.restore_supervision(run_root=run, expected_sha256=digest, renderer_sha256=renderer_sha, archive=archive)
            self.assertEqual(receipt["validation"]["approved_count"], 6)
            self.assertEqual(receipt["member_count"], 4 + 32 + 32 + 16)
            replay = module.restore_supervision(run_root=run, expected_sha256=digest, renderer_sha256=renderer_sha, archive=archive)
            self.assertEqual(replay, receipt)
            other = root / "other-run"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "renderer hash"):
                module.restore_supervision(run_root=other, expected_sha256=digest, renderer_sha256="e" * 64, archive=archive)

    def test_rejects_archive_hash_mismatch_before_extraction(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _digest = self._archive(root)
            (root / "run").mkdir()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.restore(run_root=root / "run", expected_sha256="0" * 64, archive=archive)
            self.assertFalse((root / "run/curriculum").exists())

    def test_rejects_canonical_seed_mismatch_and_painting_hash_mismatch(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root, bad_seed=True)
            (root / "run").mkdir()
            with self.assertRaisesRegex(ValueError, "case seed mismatch"):
                module.restore(run_root=root / "run", expected_sha256=digest, archive=archive)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root, bad_painting_hash=True)
            (root / "run").mkdir()
            with self.assertRaisesRegex(ValueError, "source_sha256 mismatch"):
                module.restore(run_root=root / "run", expected_sha256=digest, archive=archive)

    def test_rejects_traversal_and_symlink_members(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar"
            (root / "run").mkdir()
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("curriculum/anchors/../escape.json")
                payload = b"bad"
                info.size = len(payload)
                import io
                output.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                module.restore(run_root=root / "run", expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), archive=archive)

    def test_resume_reuses_real_native_trace_and_copies_only_its_final_canvas(self):
        anchor_bank = load_module_from(RUN / "anchor_bank.py", "structured_flow_anchor_bank_restore_test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor_root = root / "anchors"
            attempt = anchor_root / "native-generation/batch-paid"
            image = attempt / "eval-output/evaluation-rollouts/case-01/turn-1/painting.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"\x89PNG\r\n\x1a\npaid")
            renderer = root / "renderer.py"
            renderer.write_text("# reviewed renderer\n")
            renderer_python = root / "renderer-python"
            renderer_python.write_text("#!/bin/sh\n")
            reference_sha = "b" * 64
            case_seed = anchor_bank.deterministic_anchor_seed("case-01", reference_sha, 0)
            (attempt / "native-eval.toml").write_text('model = "sft-step512"\nseed = 20261009\n')
            (attempt / "native-eval-manifest.json").write_text(json.dumps({"cases": [{
                "id": "case-01", "source_group": "case-01", "reference_sha256": reference_sha,
                "reference_kind": "text", "max_turns": 3,
                "case_seed": case_seed, "renderer": str(renderer), "renderer_python": str(renderer_python),
            }]}))
            (attempt / "traces.jsonl").write_text(json.dumps({
                "task": {"data": {"task_id": "case-01", "source_group": "case-01"}},
                "info": {
                    "task_id": "case-01", "source_group": "case-01",
                "final_canvas": "/old/painter/repo/painter/runs/structured-flow-rl-20261009/curriculum/anchors/native-generation/batch-paid/eval-output/evaluation-rollouts/case-01/turn-1/painting.png", "final_canvas_valid": True,
                "turns": [{"turn": 1, "action": "paint", "valid": True, "canvas": "/old/painter/repo/painter/runs/structured-flow-rl-20261009/curriculum/anchors/native-generation/batch-paid/eval-output/evaluation-rollouts/case-01/turn-1/painting.png"}],
                },
            }) + "\n")
            rows = [{"id": "case-01", "source_group": "case-01", "reference_sha256": reference_sha, "reference": "brief.txt"}]
            environment = {
                "PAINTER_FROZEN_OPPONENT_RENDERER": str(renderer),
                "PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON": str(renderer_python),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                provenance, failures = anchor_bank._resume_existing_batches(
                    rows, anchor_root=anchor_root, receipt_root=anchor_root / "native-generation",
                    run_root=root / "run",
                )
            self.assertEqual(failures, {})
            self.assertEqual(provenance["case-01"]["generation_mode"], "resumed_paid_batch")
            self.assertEqual((anchor_root / "case-01.png").read_bytes(), image.read_bytes())


def load_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
