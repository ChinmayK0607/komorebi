"""Focused offline checks for the diverse-flow curriculum handoff."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).with_name("prepare_curriculum.py")
SPEC = importlib.util.spec_from_file_location("diverse_flow_prepare", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ANCHOR_SCRIPT = Path(__file__).with_name("anchor_bank.py")
ANCHOR_SPEC = importlib.util.spec_from_file_location("diverse_flow_anchor_bank", ANCHOR_SCRIPT)
assert ANCHOR_SPEC and ANCHOR_SPEC.loader
ANCHOR_MODULE = importlib.util.module_from_spec(ANCHOR_SPEC)
ANCHOR_SPEC.loader.exec_module(ANCHOR_MODULE)

CHECKPOINT = "a" * 64


class DiverseFlowCurriculumTests(unittest.TestCase):
    def test_fixed_balanced_selection_is_idempotent_and_family_disjoint(self):
        rows = MODULE.generate_rows()
        self.assertEqual(rows, MODULE.generate_rows())
        self.assertEqual(len(rows), 128)
        self.assertEqual(sum(row["split"] == "train" for row in rows), 96)
        self.assertEqual(sum(row["split"] == "heldout" for row in rows), 32)
        self.assertEqual(len({row["base_family_id"] for row in rows}), 40)

        train_families = {row["base_family_id"] for row in rows if row["split"] == "train"}
        heldout_families = {row["base_family_id"] for row in rows if row["split"] == "heldout"}
        self.assertEqual(len(train_families), 24)
        self.assertEqual(len(heldout_families), 16)
        self.assertFalse(train_families & heldout_families)
        self.assertEqual(len({row["subject"] for row in rows if row["split"] == "heldout"}), 16)
        teacher_families = {
            "single-apple", "single-vase", "single-orb", "single-lantern",
            "connected-cup-saucer", "connected-bird-branch", "connected-boat-sail", "connected-teapot",
            "pair-apple-pear", "pair-book-mug", "pair-rock-flower", "pair-sun-cloud",
            "scene-window-plant",
        }
        self.assertTrue(teacher_families <= train_families)
        self.assertFalse(teacher_families & heldout_families)
        self.assertEqual({"scene-hill-cabin", "scene-pond-reeds"} <= heldout_families, True)

        for stage in range(1, 5):
            stage_rows = [row for row in rows if row["stage"] == stage]
            self.assertEqual(len(stage_rows), 32)
            self.assertEqual(sum(row["split"] == "train" for row in stage_rows), 24)
            self.assertEqual(sum(row["split"] == "heldout" for row in stage_rows), 8)
            self.assertEqual(len({row["base_family_id"] for row in stage_rows}), 10)

        for family in train_families:
            self.assertEqual(len([row for row in rows if row["base_family_id"] == family]), 4)
        for family in heldout_families:
            self.assertEqual(len([row for row in rows if row["base_family_id"] == family]), 2)

        train = [row for row in rows if row["split"] == "train"]
        heldout = [row for row in rows if row["split"] == "heldout"]
        first_plan = MODULE.build_pilot_plan(train, heldout)
        self.assertEqual(first_plan, MODULE.build_pilot_plan(train, heldout))
        self.assertEqual(first_plan["counts"], {"train": 96, "heldout": 32, "total": 128})
        self.assertEqual(first_plan["stage_counts"], {
            "train": {"1": 24, "2": 24, "3": 24, "4": 24},
            "heldout": {"1": 8, "2": 8, "3": 8, "4": 8},
        })

    def test_build_proposal_writes_hashed_96_32_pending_rows(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = MODULE.build_proposal(root / "proposal")
            self.assertEqual(receipt["status"], "prepared_unrendered")
            train = json.loads((root / "proposal" / "train-proposal.json").read_text())
            heldout = json.loads((root / "proposal" / "heldout-proposal.json").read_text())
            self.assertEqual(train["counts"]["total"], 96)
            self.assertEqual(heldout["counts"]["total"], 32)
            self.assertTrue(all(row["anchor_required"] for row in train["rows"] + heldout["rows"]))
            self.assertTrue(all("opponent" not in row and "opponent_sha256" not in row for row in train["rows"] + heldout["rows"]))
            self.assertEqual(json.loads((root / "proposal" / "pilot-plan.json").read_text())["counts"], {"train": 96, "heldout": 32, "total": 128})
            for row in train["rows"][:3] + heldout["rows"][:3]:
                prompt = root / "proposal" / row["reference"]
                self.assertEqual(MODULE.sha256_file(prompt), row["reference_sha256"])
                self.assertIn("comfortable canvas margins", prompt.read_text())

    def test_briefs_keep_semantics_and_materials_natural(self):
        rows = MODULE.generate_rows()
        by_family = {row["base_family_id"]: row for row in rows if row["variant"] == 0}

        cup = by_family["connected-cup-saucer"]
        self.assertIn("cup resting on a saucer", cup["prompt_text"])
        self.assertIn("without fusing", cup["prompt_text"])
        self.assertNotIn("connected to its saucer", cup["prompt_text"])
        self.assertTrue(all("cup" in material and "saucer" in material for material in next(f["materials"] for f in MODULE.BASE_FAMILIES if f["id"] == "connected-cup-saucer")))

        fruit = by_family["pair-apple-pear"]
        self.assertIn("real red apple", fruit["prompt_text"])
        self.assertNotIn("ceramic and linen", fruit["prompt_text"])

        sky = by_family["pair-sun-cloud"]
        self.assertIn("golden light and blue-gray cloud", sky["prompt_text"])
        self.assertNotIn("wool", sky["prompt_text"])
        self.assertNotIn("gauze", sky["prompt_text"])

        palette_text = {text for _, text in MODULE.PALETTES}
        self.assertTrue(any("slate blue" in text for text in palette_text))
        self.assertTrue(any("blue-gray" in text for text in palette_text))

    def _build_with_anchors(self, root: Path) -> tuple[Path, Path, Path]:
        proposal_root = root / "proposal"
        MODULE.build_proposal(proposal_root)
        anchor_root = root / "anchors"
        anchor_root.mkdir()
        provenance = {}
        for name in ("train-proposal.json", "heldout-proposal.json"):
            rows = json.loads((proposal_root / name).read_text())["rows"]
            for row in rows:
                anchor = anchor_root / f"{row['id']}.png"
                anchor.write_bytes(("verified-anchor:" + row["id"]).encode())
                provenance[row["id"]] = {
                    "checkpoint_sha256": CHECKPOINT,
                    "reference_sha256": row["reference_sha256"],
                    "source_group": row["source_group"],
                    "png_sha256": MODULE.sha256_file(anchor),
                    "valid": True,
                    "generation_receipt": f"native-generation/{row['id']}.json",
                }
        (anchor_root / "anchor-provenance.json").write_text(json.dumps({
            "schema": "structured-flow-anchor-provenance-v1",
            "anchors": provenance,
        }, sort_keys=True))
        return proposal_root / "train-proposal.json", anchor_root, proposal_root

    def test_materialize_pilot_admits_exact_96_32_and_casebrief_is_hash_bound(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            train_proposal, anchors, proposal_root = self._build_with_anchors(root)
            output = root / "materialized"
            receipt = MODULE.materialize_pilot_manifests(
                train_proposal,
                proposal_root / "heldout-proposal.json",
                output,
                anchors,
                frozen_sft512_checkpoint_sha256=CHECKPOINT,
            )
            self.assertEqual(receipt["train"]["rows"], 96)
            self.assertEqual(receipt["heldout"]["rows"], 32)
            train = json.loads((output / "pilot-train-manifest.json").read_text())
            heldout = json.loads((output / "pilot-heldout-manifest.json").read_text())
            self.assertTrue(train["native_rl_compatible"])
            self.assertFalse(heldout["native_rl_compatible"])
            self.assertEqual(len(train["rows"]), 96)
            self.assertEqual(len(heldout["cases"]), 32)
            for row in train["rows"] + heldout["rows"]:
                self.assertEqual(row["anchor_status"], "verified_file")
                self.assertEqual(row["opponent_source_group"], row["source_group"])
                self.assertEqual(row["frozen_sft512_checkpoint_sha256"], CHECKPOINT)
                self.assertEqual(MODULE.sha256_file(Path(row["opponent"])), row["opponent_sha256"])

            sample = train["rows"][0]
            MODULE.write_casebrief(sample, proposal_root, output, enabled=True)
            casebrief = Path(sample["casebrief"])
            self.assertEqual(MODULE.sha256_file(casebrief), sample["casebrief_sha256"])
            text = casebrief.read_text()
            self.assertIn("[SYSTEM]", text)
            self.assertIn("[USER]", text)
            self.assertIn("Plan before writing code", text)

    def test_anchor_bank_provenance_feeds_real_materializer_and_keeps_native_schema(self):
        """Exercise the generator's emitted provenance through admission.

        The native batch is replaced only at its model/render boundary with a
        deterministic PNG-producing stub. The anchor-bank materializer,
        provenance writer, curriculum loader, casebrief propagation, and final
        manifest writes all remain real code paths.
        """
        with TemporaryDirectory() as temp:
            root = Path(temp)
            curriculum_root = root / "curriculum"
            MODULE.build_proposal(curriculum_root)
            train_manifest = curriculum_root / "pilot-train-manifest.json"
            heldout_manifest = curriculum_root / "pilot-heldout-manifest.json"
            anchor_root = curriculum_root / "anchors"
            ANCHOR_MODULE.prepare_lazy(
                train_proposal=curriculum_root / "train-proposal.json",
                heldout_proposal=curriculum_root / "heldout-proposal.json",
                train_output=train_manifest,
                heldout_output=heldout_manifest,
            )

            def fake_batch(rows, *, anchor_root, receipt_root, proposal_root, run_root, workers, retry_index=0, failed_cases=None):
                del receipt_root, proposal_root, run_root, workers, retry_index, failed_cases
                provenance = {}
                for row in rows:
                    anchor = anchor_root / f"{row['id']}.png"
                    anchor.parent.mkdir(parents=True, exist_ok=True)
                    anchor.write_bytes(b"\x89PNG\r\n\x1a\n" + str(row["id"]).encode())
                    provenance[str(row["id"])] = {
                        "checkpoint_sha256": ANCHOR_MODULE.CHECKPOINT_SHA256,
                        "reference_sha256": row["reference_sha256"],
                        "source_group": row["source_group"],
                        "png_sha256": MODULE.sha256_file(anchor),
                        "valid": True,
                        "generation_receipt": f"native-generation/{row['id']}.json",
                    }
                return provenance, {}

            original_batch = ANCHOR_MODULE._batch_with_bounded_resample
            ANCHOR_MODULE._batch_with_bounded_resample = fake_batch
            try:
                result = ANCHOR_MODULE.materialize(
                    train_manifest=train_manifest,
                    heldout_manifest=heldout_manifest,
                    anchor_root=anchor_root,
                    output_dir=curriculum_root / ".materialized-pilot",
                    run_root=root,
                )
            finally:
                ANCHOR_MODULE._batch_with_bounded_resample = original_batch

            self.assertEqual(result["status"], "materialized")
            provenance = json.loads((anchor_root / MODULE.ANCHOR_PROVENANCE_FILE).read_text())
            self.assertEqual(provenance["schema"], MODULE.ANCHOR_PROVENANCE_SCHEMA)
            loaded, _ = MODULE._load_anchor_provenance(anchor_root)
            self.assertEqual(len(loaded), 128)

            train = json.loads(train_manifest.read_text())
            heldout = json.loads(heldout_manifest.read_text())
            self.assertEqual(train["schema"], MODULE.NATIVE_SCHEMA)
            self.assertEqual(heldout["schema"], MODULE.NATIVE_SCHEMA)
            self.assertEqual(len(train["rows"]), 96)
            self.assertEqual(len(heldout["cases"]), 32)
            self.assertTrue(all(row["anchor_status"] == "verified_file" for row in train["rows"] + heldout["rows"]))

    def test_stage_manifests_have_expected_current_plus_replay_counts(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal, anchors, _ = self._build_with_anchors(root)
            output = root / "stages"
            receipt = MODULE.materialize_stage_manifests(proposal, output, anchors, frozen_sft512_checkpoint_sha256=CHECKPOINT)
            self.assertEqual([item["total_count"] for item in receipt["manifests"]], [24, 30, 36, 42])
            for stage, expected in enumerate((24, 30, 36, 42), 1):
                manifest = json.loads((output / f"stage-{stage}-train-manifest.json").read_text())
                self.assertEqual(len(manifest["rows"]), expected)
                self.assertEqual(len({row["task_id"] for row in manifest["rows"]}), expected)
                self.assertEqual(sum(row["sampling_role"] == "replay" for row in manifest["rows"]), expected - 24)


if __name__ == "__main__":
    unittest.main()
