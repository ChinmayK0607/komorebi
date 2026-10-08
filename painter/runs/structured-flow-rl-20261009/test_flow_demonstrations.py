"""Pure contract tests for the structured-flow candidate supervision run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("structured_flow_demonstrations", HERE / "flow_demonstrations.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StructuredFlowCandidateTests(unittest.TestCase):
    def test_authored_pool_has_four_categories_and_two_complete_programs_each(self):
        categories = [candidate.category for candidate in MODULE.CANDIDATES]
        self.assertEqual(len(MODULE.CANDIDATES), 16)
        self.assertEqual({category: categories.count(category) for category in set(categories)}, {
            "single_forms": 4,
            "connected_objects": 4,
            "spatial_pairs": 4,
            "simple_scenes": 4,
        })
        with tempfile.TemporaryDirectory() as directory:
            manifest = MODULE.build(directory)
            self.assertEqual(manifest["count"], 16)
            self.assertEqual(manifest["program_count"], 32)
            self.assertEqual(manifest["row_count_after_pairing"], 32)
            source_manifest = json.loads((Path(directory) / "source-manifest.json").read_text())
            self.assertEqual(len(source_manifest["sources"]), 32)
            for source in source_manifest["sources"]:
                path = Path(directory) / source["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])
                MODULE.validate_program(path.read_text())

    def test_warmup_candidates_do_not_use_curriculum_heldout_families(self):
        heldout_family_names = {"single-stone", "connected-key", "pair-vase-branch", "scene-tram-street"}
        ids = {candidate.id for candidate in MODULE.CANDIDATES}
        self.assertFalse(ids & heldout_family_names)
        self.assertIn("single-orb", ids)
        self.assertIn("connected-teapot", ids)

    def test_program_validation_rejects_raw_p5_shapes_and_affine_frames(self):
        good = MODULE._program(MODULE.CANDIDATES[0], False)
        MODULE.validate_program(good)
        with self.assertRaisesRegex(ValueError, "raw p5 shape"):
            MODULE.validate_program(good.replace("brush.polygon", "ellipse", 1))
        with self.assertRaisesRegex(ValueError, "exactly one translate"):
            MODULE.validate_program(good.replace("translate(-300,-300)", "translate(-300,-300);translate(5,5)", 1))

    def test_export_requires_two_hash_verified_renders_and_keeps_visual_review_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.build(root)
            results = []
            for candidate in MODULE.CANDIDATES:
                for version in ("block_in", "improved_final"):
                    source = root / "programs" / candidate.id / f"{version}.js"
                    png = root / "renders" / candidate.id / f"{version}.png"
                    png.parent.mkdir(parents=True, exist_ok=True)
                    png.write_bytes(b"fake-png-" + candidate.id.encode() + version.encode())
                    receipt = {
                        "valid": True,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                    }
                    results.append({
                        "candidate_id": candidate.id,
                        "version": version,
                        "valid": True,
                        "source_sha256": receipt["source_sha256"],
                        "png_sha256": receipt["png_sha256"],
                    })
            (root / "render-manifest.json").write_text(json.dumps({
                "schema": "painter.structured-flow-render.v1",
                "run_id": MODULE.RUN_ID,
                "source_manifest_sha256": json.loads((root / "manifest.json").read_text())["source_manifest_sha256"],
                "renderer": {"renderer_sha256": "a" * 64, "assets_sha256": "b" * 64},
                "program_count": 32,
                "valid_program_count": 32,
                "complete_hash_verified": True,
                "results": results,
            }))
            receipt = MODULE.export_sft(root)
            self.assertEqual(receipt, {"rows": 32, "teacher_rows": 32, "finish_rows": 0, "reviewed_code_rows": 0, "train": 24, "dev": 8, "quality_status": "visual_review_pending", "quality_claim": False})
            rows = [json.loads(line) for line in (root / "sft.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            teacher_rows = [json.loads(line) for line in (root / "teacher.jsonl").read_text().splitlines()]
            self.assertEqual(len(teacher_rows), 32)
            self.assertTrue(all(set(row) == {"case_id", "source_group", "system", "user", "next_assistant_token"} for row in teacher_rows))
            self.assertTrue(all("p5.brush drawing API reference v2" in row["system"] for row in teacher_rows))
            self.assertTrue(all(MODULE.FLOW_HINT in row["system"] for row in teacher_rows))
            self.assertTrue(all(row["quality_status"] == "visual_review_pending" for row in rows))
            self.assertTrue(all(row["quality_claim"] is False for row in rows))
            self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in rows))
            self.assertTrue(all(row["messages"][-1]["content"][0]["text"].startswith("```javascript\n") for row in rows))
            self.assertTrue(all(row["messages"][-1]["content"][0]["text"].count("```") == 2 for row in rows))
            finals = [row for row in rows if row["flow_stage"] == "improved_final"]
            self.assertEqual(len(finals), 16)
            self.assertTrue(all(row["previous_program_in_user_text"] for row in finals))
            self.assertTrue(all(len(row["messages"][1]["content"]) == 2 for row in finals))
            self.assertTrue(all(MODULE.TEXT_INSPECT_CHOICE in row["messages"][1]["content"][0]["text"] for row in finals))
            teacher_finals = [row for row in teacher_rows if row["case_id"].endswith("-improved_final")]
            self.assertEqual(len(teacher_finals), 16)
            self.assertTrue(all("CURRENT CANVAS SHA256:" in row["user"] for row in teacher_finals))
            self.assertTrue(all("PREVIOUS PROGRAM" in row["user"] for row in teacher_finals))

            with self.assertRaisesRegex(ValueError, "actual visual review"):
                MODULE.export_sft(root, include_finish=True)
            approval = {
                "status": "reviewed",
                "candidates": [
                    {
                        "id": candidate.id,
                        "render_valid": True,
                        "decision": "approved",
                        "improved_reviewed": True,
                        "final_png_sha256": hashlib.sha256((root / "renders" / candidate.id / "improved_final.png").read_bytes()).hexdigest(),
                    }
                    for candidate in MODULE.CANDIDATES[:8]
                ],
            }
            reviewed_receipt = MODULE.export_sft(root, approval_manifest=approval)
            self.assertEqual(reviewed_receipt["reviewed_code_rows"], 16)
            reviewed_rows = [json.loads(line) for line in (root / "sft.jsonl").read_text().splitlines()]
            approved_rows = [row for row in reviewed_rows if row["source_group"] in {candidate.id for candidate in MODULE.CANDIDATES[:8]}]
            self.assertEqual(len(approved_rows), 16)
            self.assertTrue(all(row["admission"] == "reviewed_improvement" for row in approved_rows))
            self.assertTrue(all(row["target_receipt"]["verified"] and row["target_receipt"]["approved"] for row in approved_rows))
            self.assertTrue(all(row["target_receipt"]["png_sha256"] == row["render"]["png_sha256"] for row in approved_rows))
            finish_receipt = MODULE.export_sft(root, include_finish=True, approval_manifest=approval)
            self.assertEqual(finish_receipt["rows"], 40)
            self.assertEqual(finish_receipt["teacher_rows"], 40)
            self.assertEqual(finish_receipt["finish_rows"], 8)
            all_rows = [json.loads(line) for line in (root / "sft.jsonl").read_text().splitlines()]
            finishes = [row for row in all_rows if row["flow_stage"] == "finish"]
            self.assertEqual(len(finishes), 8)
            self.assertTrue(all(row["messages"][-1]["content"][0]["text"] == "FINISHED" for row in finishes))
            self.assertTrue(all(row["prior_canvas"] and row["previous_program_in_user_text"] for row in finishes))
            self.assertTrue(all(MODULE.TEXT_INSPECT_CHOICE in row["messages"][1]["content"][0]["text"] for row in finishes))

    def test_pending_approval_is_not_correction_admission(self):
        with self.assertRaisesRegex(ValueError, "reviewed approval manifest"):
            MODULE.admit_reviewed_corrections({
                "status": "candidate_approval_pending",
                "candidates": [{"id": "single-apple", "decision": "pending", "improved_reviewed": False}],
            })
        with self.assertRaisesRegex(ValueError, "explicit improvement decision"):
            MODULE.admit_reviewed_corrections({
                "status": "flash_decision",
                "candidates": [{"id": "single-apple", "decision": "flash_decision", "improved_reviewed": False}],
            })
        admitted = MODULE.admit_reviewed_corrections({
            "status": "reviewed",
            "candidates": [{"id": "single-apple", "decision": "approved", "improved_reviewed": True}],
        })
        self.assertEqual(admitted[0]["admission"], "reviewed_improvement")

    def test_render_is_explicitly_linux_only(self):
        with patch.object(MODULE.sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Linux-node-only"):
                MODULE.render("/tmp/no-run", "/tmp/renderer.py", "/tmp/python")

    def test_build_cli_alias_keeps_authoring_step_explicit_in_output(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(MODULE.main(["--output", directory]), 0)
            self.assertTrue((Path(directory) / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
