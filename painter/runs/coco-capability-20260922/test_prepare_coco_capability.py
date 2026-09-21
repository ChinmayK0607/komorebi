"""Unit checks for the bounded COCO capability preparation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[2]
SOURCE = REPO_ROOT / "painter/runs/scaled-curriculum-20260921/real-pool/reference-manifest.json"
FROZEN_SOURCE_IDS = [
    "000000000127",
    "000000000009",
    "000000000419",
    "000000000599",
    "000000000133",
    "000000000089",
    "000000000208",
    "000000000357",
    "000000000382",
    "000000000542",
    "000000000641",
    "000000000078",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_module = _load("prepare_coco_capability", RUN_DIR / "prepare_coco_capability.py")
gallery_module = _load("build_gallery", RUN_DIR / "build_gallery.py")
contact_module = _load("make_contact_sheets", RUN_DIR / "make_contact_sheets.py")


class CocoCapabilityPreparationTests(unittest.TestCase):
    def test_selection_is_dev_only_deterministic_and_covers_buckets(self):
        source = json.loads(SOURCE.read_text())
        first = prepare_module.select_cases(source["references"])
        second = prepare_module.select_cases(source["references"])
        self.assertEqual([row["source_id"] for row in first], [row["source_id"] for row in second])
        self.assertEqual([row["source_id"] for row in first], FROZEN_SOURCE_IDS)
        self.assertEqual(len(first), 12)
        self.assertTrue(all(row["split"] == "dev" for row in first))
        self.assertEqual(len({row["source_id"] for row in first}), 12)
        self.assertEqual(len({row["source_group"] for row in first}), 12)
        self.assertEqual(
            {prepare_module._object_bucket(len(row["annotations"])) for row in first},
            {"one", "two_to_four", "five_to_nine", "ten_plus"},
        )

    def test_prepare_copies_hash_verified_refs_and_native_manifest_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            result = prepare_module.prepare(
                source_manifest=SOURCE,
                output=output,
                eval_root=Path("/root/painter"),
            )
            manifest_path = output / "eval-manifest.json"
            self.assertEqual(result["status"], "prepared_no_inference")
            self.assertEqual(len(result["cases"]), 12)
            audit = prepare_module.validate_manifest(output, manifest_path, expected_count=12)
            self.assertEqual(audit["count"], 12)
            for case in result["cases"]:
                reference = output / case["reference"]
                self.assertEqual(prepare_module.sha256(reference), case["reference_sha256"])
                self.assertTrue(case["selection_reason"])
            base_config = tomllib.loads((output / "base.toml").read_text())
            sft_config = tomllib.loads((output / "sft.toml").read_text())
            self.assertEqual(base_config["model"], "base")
            self.assertEqual(sft_config["model"], "sft")
            self.assertEqual(base_config["env"]["taskset"]["policy_label"], "base")
            self.assertEqual(sft_config["env"]["taskset"]["policy_label"], "sft")
            self.assertEqual(base_config["sampling"]["max_tokens"], 8192)
            self.assertEqual(sft_config["env"]["taskset"]["max_turns"], 6)

    def test_gallery_keeps_missing_policy_cells_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            prepare_module.prepare(source_manifest=SOURCE, output=output, eval_root=Path("/root/painter"))
            gallery = gallery_module.build_gallery(
                evidence_root=root / "empty-evidence",
                reference_manifest=output / "eval-manifest.json",
                outdir=output / "gallery",
            )
            self.assertTrue(gallery["partial"])
            self.assertEqual(gallery["discovered"]["matched_episodes"], 0)
            self.assertEqual(len(gallery["cases"]), 12)
            self.assertTrue(all(not case["policies"]["base"]["trace_found"] for case in gallery["cases"]))
            self.assertTrue(all(not case["policies"]["sft"]["trace_found"] for case in gallery["cases"]))
            self.assertNotIn("placeholder", (output / "gallery/index.html").read_text().lower())

    def test_gallery_maps_actual_base_and_sft_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            prepare_module.prepare(source_manifest=SOURCE, output=output, eval_root=Path("/root/painter"))
            evidence = root / "evidence"
            evidence.mkdir()
            case = json.loads((output / "eval-manifest.json").read_text())["cases"][0]
            records = []
            for label in ("base", "sft"):
                turn_one = evidence / label / "turn-1" / "painting.png"
                turn_two = evidence / label / "turn-2" / "painting.png"
                turn_one.parent.mkdir(parents=True)
                turn_two.parent.mkdir(parents=True)
                turn_one.write_bytes(b"first")
                turn_two.write_bytes(b"final")
                records.append(
                    {
                        "id": f"trace-{label}",
                        "task": {"data": {"task_id": case["id"], "source_group": case["source_group"], "policy_label": label}},
                        "calls": [{"usage": {"prompt_tokens": 10, "completion_tokens": 4}}],
                        "info": {
                            "task_id": case["id"],
                            "source_group": case["source_group"],
                            "policy_label": label,
                            "turns": [
                                {"turn": 1, "valid": True, "canvas": str(turn_one)},
                                {"turn": 2, "valid": True, "canvas": str(turn_two)},
                            ],
                            "finished": True,
                            "final_canvas_valid": True,
                            "final_canvas": str(turn_two),
                        },
                    }
                )
            (evidence / "traces.jsonl").write_text(json.dumps({"traces": records}) + "\n")
            gallery = gallery_module.build_gallery(
                evidence_root=evidence,
                reference_manifest=output / "eval-manifest.json",
                outdir=output / "gallery",
            )
            first_case = gallery["cases"][0]
            self.assertEqual(first_case["policies"]["base"]["trace_id"], "trace-base")
            self.assertEqual(first_case["policies"]["sft"]["trace_id"], "trace-sft")
            self.assertEqual(gallery["policy_labels"], {"base": "base", "sft": "sft"})
            self.assertEqual(gallery["expected"]["policies"], ["base", "sft"])
            self.assertNotIn("rl", json.dumps(gallery).lower())
            page = (output / "gallery/index.html").read_text()
            self.assertIn("Base unadapted", page)
            self.assertIn("SFT step24", page)
            self.assertNotIn("RL", page)

    def test_contact_sheet_slots_use_recorded_base_sft_paths_only(self):
        case = {
            "id": "case-1",
            "reference": "/node/reference.jpg",
            "policies": {
                "base": {"status": "complete", "first_canvas": "/node/base/turn-1/painting.png", "final_canvas": "/node/base/turn-6/painting.png"},
                "sft": {"status": "censored", "first_canvas": "/node/sft/turn-1/painting.png", "final_canvas": None},
            },
        }
        slots = contact_module._slot_records(case)
        self.assertEqual([slot["label"] for slot in slots], [
            "Reference",
            "Base unadapted · first",
            "Base unadapted · final",
            "SFT step24 · first",
            "SFT step24 · final",
        ])
        self.assertEqual(slots[1]["recorded_path"], "/node/base/turn-1/painting.png")
        self.assertIsNone(slots[4]["recorded_path"])


if __name__ == "__main__":
    unittest.main()
