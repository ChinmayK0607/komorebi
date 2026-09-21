"""Pure curriculum-order and contract-placement tests."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"curriculum_repair_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curriculum = _load("prepare_curriculum")
teacher_checks = _load("teacher_program_checks")


class CurriculumRepairTest(unittest.TestCase):
    def _manifest(self):
        rows = []
        for family_index in range(12):
            family = f"family-{family_index:02d}"
            for group_index in range(2):
                group = f"v2-{family}-{group_index:03d}"
                rows.append({
                    "id": f"{group}-task",
                    "source_group": group,
                    "family": family,
                    "reference": f"data/{group}/reference.png",
                    "mode": "both",
                    "initial_canvas": f"canvas/{group}.png",
                    "initial_program": f"program/{group}.js",
                    "anchors": [
                        {"id": f"{group}-early", "canvas": f"anchor/{group}-early.png"},
                        {"id": f"{group}-final", "canvas": f"anchor/{group}-final.png"},
                    ],
                })
        return {
            "schema": "painter.curriculum-candidate.v1",
            "reviewed": True,
            "tasks": rows,
        }

    def test_fresh_first_prefix_covers_families_and_collapses_anchors(self):
        manifest = self._manifest()
        prepared = curriculum.prepare_mode(manifest, "fresh", limit=24, first_n=12)
        self.assertEqual(prepared["counts"]["tasks"], 24)
        self.assertEqual(prepared["counts"]["source_groups"], 24)
        self.assertEqual(len(prepared["coverage"]["first_n_groups"]), 12)
        self.assertEqual(len(prepared["coverage"]["first_n_families"]), 12)
        self.assertEqual({row["mode"] for row in prepared["tasks"]}, {"fresh"})
        self.assertEqual(len({row["source_group"] for row in prepared["tasks"]}), 24)

    def test_correction_order_keeps_anchor_rows_after_group_balanced_prefix(self):
        manifest = self._manifest()
        prepared = curriculum.prepare_mode(manifest, "correction", limit=8, first_n=8)
        tasks = prepared["tasks"]
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({row["source_group"] for row in tasks}), 8)
        self.assertEqual({row["mode"] for row in tasks}, {"correction"})
        self.assertTrue(all(row["initial_canvas"] and row["initial_program"] for row in tasks))
        self.assertEqual([row["order"] for row in tasks], list(range(8)))

    def test_order_is_deterministic_and_missing_correction_start_fails(self):
        manifest = self._manifest()
        a = curriculum.prepare_mode(manifest, "fresh", limit=24, first_n=12)
        b = curriculum.prepare_mode(manifest, "fresh", limit=24, first_n=12)
        self.assertEqual(a, b)
        broken = self._manifest()
        broken["tasks"][0].pop("initial_program")
        with self.assertRaises(ValueError):
            curriculum.prepare_mode(broken, "correction", limit=1, first_n=1)

    def test_teacher_transform_must_be_inside_draw_and_canonical_copy_matches(self):
        draw_program = "function setup(){createCanvas(600,600,WEBGL);} function draw(){translate(-300,-300);noLoop();}"
        setup_program = "function setup(){createCanvas(600,600,WEBGL);translate(-300,-300);} function draw(){noLoop();}"
        teacher_checks.validate_teacher_program(draw_program)
        with self.assertRaises(ValueError):
            teacher_checks.validate_teacher_program(setup_program)
        canonical = ROOT.parents[1] / "contract.py"
        copied = ROOT / "contract.py"
        self.assertEqual(hashlib.sha256(canonical.read_bytes()).hexdigest(), teacher_checks.canonical_contract_sha256())
        self.assertEqual(canonical.read_bytes(), copied.read_bytes())

    def test_repaired_taskset_references_shared_contract_and_modes(self):
        source = (ROOT / "taskset_impl.py").read_text()
        self.assertIn("from contract import FINISH_WITHOUT_CANVAS, NEXT_VERSION, NO_CHANGE, SYSTEM", source)
        self.assertIn("mode == \"fresh\"", source)
        self.assertIn("NO_CHANGE if result.get(\"no_change\")", source)
        self.assertIn("curriculum-plan.json", source)
        self.assertIn("no_valid_canvas_at_cap", source)
        self.assertIn("taskset imported {name} from a different contract", source)

    def test_turn_cap_is_explicit_in_prepared_plan(self):
        prepared = curriculum.prepare_mode(self._manifest(), "fresh", limit=24, max_turns=6)
        self.assertEqual(prepared["max_turns"], 6)

    def test_reviewed_candidate_manifest_builds_fresh_and_repair_modes(self):
        path = ROOT / "candidate-manifest.json"
        manifest = curriculum._load_manifest(path)
        fresh = curriculum.prepare_mode(manifest, "fresh", limit=24, max_turns=6)
        repair = curriculum.prepare_mode(manifest, "correction", limit=8, max_turns=6)
        self.assertEqual(fresh["counts"], {"tasks": 24, "families": 12, "source_groups": 24})
        self.assertEqual(repair["counts"], {"tasks": 8, "families": 8, "source_groups": 8})
        self.assertTrue(all(row["mode"] == "correction" for row in repair["tasks"]))
        self.assertTrue(all(row["anchor"] != row["reference"] for row in repair["tasks"]))
        self.assertTrue(all(row["initial_canvas"] == row["anchor"] for row in repair["tasks"]))


if __name__ == "__main__":
    unittest.main()
