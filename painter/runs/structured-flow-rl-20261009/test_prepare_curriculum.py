"""Offline invariants for the structured-flow curriculum proposal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("prepare_curriculum.py")
SPEC = importlib.util.spec_from_file_location("structured_flow_prepare", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CHECKPOINT = "a" * 64


class StructuredFlowCurriculumTests(unittest.TestCase):
    def test_counts_and_family_first_split_are_disjoint(self):
        rows = MODULE.generate_rows()
        self.assertEqual(len(rows), 300)
        self.assertEqual(sum(row["split"] == "train" for row in rows), 240)
        self.assertEqual(sum(row["split"] == "heldout" for row in rows), 60)
        train_families = {row["base_family_id"] for row in rows if row["split"] == "train"}
        heldout_families = {row["base_family_id"] for row in rows if row["split"] == "heldout"}
        self.assertEqual(len(train_families), 16)
        self.assertEqual(len(heldout_families), 4)
        self.assertFalse(train_families & heldout_families)
        for stage in range(1, 5):
            stage_rows = [row for row in rows if row["stage"] == stage]
            self.assertEqual(len(stage_rows), 75)
            self.assertEqual(sum(row["split"] == "train" for row in stage_rows), 60)
            self.assertEqual(sum(row["split"] == "heldout" for row in stage_rows), 15)
        for family in {row["base_family_id"] for row in rows}:
            variants = [row for row in rows if row["base_family_id"] == family]
            self.assertEqual(len(variants), MODULE.VARIANTS_PER_FAMILY)
            self.assertEqual(len({row["prompt_signature"] for row in variants}), MODULE.VARIANTS_PER_FAMILY)
            self.assertGreaterEqual(len({row["palette"] for row in variants}), 5)
            self.assertGreaterEqual(len({row["orientation"] for row in variants}), 3)
            self.assertGreaterEqual(len({row["material"] for row in variants}), 3)
            self.assertTrue(all("exactly one a " not in row["prompt_text"] for row in variants))

        by_family = {row["base_family_id"]: row for row in rows if row["variant"] == 0}
        self.assertIn("small clear gap", by_family["pair-cup-book"]["prompt_text"])
        self.assertIn("contact with the folded cloth", by_family["pair-apple-cloth"]["prompt_text"])
        self.assertIn("on top of the table", by_family["pair-lamp-table"]["prompt_text"])
        self.assertIn("entering the vase opening", by_family["pair-vase-branch"]["prompt_text"])
        self.assertIn("otherwise finish", by_family["single-apple"]["flow_guidance"])

    def test_build_writes_hashed_prompts_and_pending_non_runnable_proposals(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "proposal"
            receipt = MODULE.build_proposal(output)
            self.assertEqual(receipt["status"], "prepared_unrendered")
            train = json.loads((output / "train-proposal.json").read_text())
            heldout = json.loads((output / "heldout-proposal.json").read_text())
            self.assertEqual(train["counts"]["total"], 240)
            self.assertEqual(heldout["counts"]["total"], 60)
            self.assertEqual(train["status"], "proposal_pending_anchors")
            self.assertTrue(all(row["anchor_required"] for row in train["rows"]))
            self.assertTrue(all(row["admission_status"] == "awaiting_frozen_sft512_anchor" for row in train["rows"]))
            self.assertTrue(all("opponent" not in row and "opponent_sha256" not in row for row in train["rows"]))
            pilot = json.loads((output / "pilot-plan.json").read_text())
            self.assertEqual(pilot["counts"], {"train": 32, "heldout": 16, "total": 48})
            self.assertEqual(pilot["stage_counts"], {"train": {"1": 16, "2": 16}, "heldout": {"1": 8, "2": 8}})
            self.assertTrue(all("flow_guidance" in row for row in train["rows"]))
            for row in train["rows"][:4] + heldout["rows"][:4]:
                prompt = output / row["reference"]
                self.assertEqual(MODULE.sha256_file(prompt), row["reference_sha256"])
            files = {item["path"] for item in receipt["outputs"]}
            self.assertIn("train-proposal.json", files)
            self.assertIn("heldout-proposal.json", files)

    def _build_with_anchors(self, root: Path) -> tuple[Path, Path, Path]:
        proposal_root = root / "proposal"
        MODULE.build_proposal(proposal_root)
        anchor_root = root / "anchors"
        anchor_root.mkdir()
        rows = json.loads((proposal_root / "train-proposal.json").read_text())["rows"]
        provenance = {}
        for row in rows:
            anchor = anchor_root / f"{row['id']}.png"
            anchor.write_bytes(("anchor:" + row["id"]).encode())
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

    def test_materialization_requires_real_anchors_and_emits_native_fields(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_root = root / "proposal"
            MODULE.build_proposal(proposal_root)
            output = root / "train-manifest.json"
            with self.assertRaises(FileNotFoundError):
                MODULE.materialize_manifest(
                    proposal_root / "train-proposal.json", output, root / "missing-anchors",
                    frozen_sft512_checkpoint_sha256=CHECKPOINT,
                )
            self.assertFalse(output.exists())

            proposal, anchors, _ = self._build_with_anchors(root / "complete")
            result = MODULE.materialize_manifest(proposal, output, anchors, frozen_sft512_checkpoint_sha256=CHECKPOINT)
            self.assertTrue(result["native_rl_compatible"])
            manifest = json.loads(output.read_text())
            self.assertEqual(manifest["schema"], MODULE.NATIVE_SCHEMA)
            self.assertEqual(len(manifest["rows"]), 240)
            for row in manifest["rows"]:
                self.assertEqual(row["split"], "train")
                self.assertEqual(row["reference_kind"], "text")
                self.assertEqual(row["opponent_source_group"], row["source_group"])
                self.assertEqual(row["frozen_sft512_checkpoint_sha256"], CHECKPOINT)
                self.assertEqual(MODULE.sha256_file(Path(row["opponent"])), row["opponent_sha256"])

    def test_selected_materialization_preserves_registered_order_and_rejects_missing_ids(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal, anchors, _ = self._build_with_anchors(root / "complete")
            proposal_rows = json.loads(proposal.read_text())["rows"]
            selected = [str(proposal_rows[1]["id"]), str(proposal_rows[0]["id"])]
            output = root / "selected.json"
            MODULE.materialize_manifest(
                proposal, output, anchors,
                frozen_sft512_checkpoint_sha256=CHECKPOINT,
                selected_ids=selected,
            )
            self.assertEqual(
                [row["id"] for row in json.loads(output.read_text())["rows"]],
                selected,
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                MODULE.materialize_manifest(
                    proposal, root / "missing.json", anchors,
                    frozen_sft512_checkpoint_sha256=CHECKPOINT,
                    selected_ids=[selected[0], "not-registered"],
                )

    def test_stage_manifests_keep_unique_ids_and_exact_quarter_replay(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal, anchors, _ = self._build_with_anchors(root / "complete")
            output_dir = root / "stages"
            receipt = MODULE.materialize_stage_manifests(
                proposal, output_dir, anchors,
                frozen_sft512_checkpoint_sha256=CHECKPOINT,
            )
            self.assertEqual([row["total_count"] for row in receipt["manifests"]], [60, 75, 90, 105])
            for stage, expected_replay in enumerate((0, 15, 30, 45), 1):
                manifest = json.loads((output_dir / f"stage-{stage}-train-manifest.json").read_text())
                rows = manifest["rows"]
                self.assertEqual(len(rows), 60 + expected_replay)
                self.assertEqual(len({row["task_id"] for row in rows}), len(rows))
                self.assertEqual(sum(row["sampling_role"] == "replay" for row in rows), expected_replay)
                self.assertTrue(all(row["admission_status"] == "ready_for_native_painting_rl" for row in rows))

    def test_pilot_materialization_keeps_full_pool_pending_and_marks_heldout_non_rl(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_root = root / "proposal"
            MODULE.build_proposal(proposal_root)
            anchors = root / "anchors"
            anchors.mkdir()
            plan = json.loads((proposal_root / "pilot-plan.json").read_text())
            provenance = {}
            for ident in plan["train_ids"] + plan["heldout_ids"]:
                anchor = anchors / f"{ident}.png"
                anchor.write_bytes(("pilot:" + ident).encode())
                source = next(
                    row for proposal_name in ("train-proposal.json", "heldout-proposal.json")
                    for row in json.loads((proposal_root / proposal_name).read_text())["rows"]
                    if row["id"] == ident
                )
                provenance[ident] = {
                    "checkpoint_sha256": CHECKPOINT,
                    "reference_sha256": source["reference_sha256"],
                    "source_group": source["source_group"],
                    "png_sha256": MODULE.sha256_file(anchor),
                    "valid": True,
                    "trace_id": f"trace-{ident}",
                }
            (anchors / "anchor-provenance.json").write_text(json.dumps({
                "schema": "structured-flow-anchor-provenance-v1",
                "anchors": provenance,
            }, sort_keys=True))
            output = root / "pilot"
            receipt = MODULE.materialize_pilot_manifests(
                proposal_root / "train-proposal.json",
                proposal_root / "heldout-proposal.json",
                output,
                anchors,
                frozen_sft512_checkpoint_sha256=CHECKPOINT,
            )
            self.assertEqual(receipt["train"]["rows"], 32)
            self.assertEqual(receipt["heldout"]["rows"], 16)
            train = json.loads((output / "pilot-train-manifest.json").read_text())
            heldout = json.loads((output / "pilot-heldout-manifest.json").read_text())
            self.assertTrue(train["native_rl_compatible"])
            self.assertFalse(heldout["native_rl_compatible"])
            self.assertEqual({row["stage"] for row in train["rows"]}, {1, 2})
            self.assertEqual({row["stage"] for row in heldout["rows"]}, {1, 2})

    def test_checkpoint_identity_is_required(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_root = root / "proposal"
            MODULE.build_proposal(proposal_root)
            with self.assertRaises(ValueError):
                MODULE.materialize_manifest(
                    proposal_root / "train-proposal.json", root / "out.json", root / "anchors",
                    frozen_sft512_checkpoint_sha256="not-a-hash",
                )

    def test_anchor_provenance_rejects_prompt_checkpoint_and_png_mismatches(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal, anchors, _ = self._build_with_anchors(root / "complete")
            train = json.loads(proposal.read_text())
            first = train["rows"][0]

            # The brief file is the source of truth; changing only its
            # manifest hash must prevent admission before any output exists.
            first["reference_sha256"] = "b" * 64
            changed = root / "complete" / "proposal" / "changed-proposal.json"
            changed.write_text(json.dumps(train))
            with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
                MODULE.materialize_manifest(changed, root / "prompt-mismatch.json", anchors, frozen_sft512_checkpoint_sha256=CHECKPOINT)

            # Restore the proposal and alter the provenance checkpoint.
            proposal, anchors, _ = self._build_with_anchors(root / "checkpoint")
            provenance_path = anchors / "anchor-provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["anchors"][first["id"]]["checkpoint_sha256"] = "c" * 64
            provenance_path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "checkpoint provenance mismatch"):
                MODULE.materialize_manifest(proposal, root / "checkpoint-mismatch.json", anchors, frozen_sft512_checkpoint_sha256=CHECKPOINT)

            # A changed PNG must disagree with the node-written PNG hash.
            proposal, anchors, _ = self._build_with_anchors(root / "png")
            target = anchors / f"{first['id']}.png"
            target.write_bytes(b"changed-anchor")
            with self.assertRaisesRegex(ValueError, "PNG provenance mismatch"):
                MODULE.materialize_manifest(proposal, root / "png-mismatch.json", anchors, frozen_sft512_checkpoint_sha256=CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
