from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("staged_materialize", RUN / "materialize.py")
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class MaterializeFixtures:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.assets = root / "assets"
        self.assets.mkdir()
        self.prior_run_id = "prior-run-fixture"
        self.staged_run_id = "staged-run-fixture"
        self.checkpoint = M.EXPECTED_CHECKPOINT_SHA256
        self.train_rows: list[dict[str, object]] = []
        self.heldout_rows: list[dict[str, object]] = []
        self._build_rows()
        self.provenance = self._write_provenance()
        self.curriculum = self._write_curriculum()
        self.train_manifest = self._write_prior_manifest("train", self.train_rows)
        self.heldout_manifest = self._write_prior_manifest("heldout", self.heldout_rows)

    def _write_rows(self, rows: list[dict[str, object]], split: str) -> None:
        for row in rows:
            ident = str(row["id"])
            prompt = f"{self.assets}/prompts/{split}/stage-{row['stage']}/{ident}.txt"
            prompt_path = Path(prompt)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_bytes = f"prompt for {ident}\n".encode()
            prompt_path.write_bytes(prompt_bytes)
            anchor_path = self.assets / "anchors" / f"{ident}.png"
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_path.write_bytes(f"anchor for {ident}\n".encode())
            row["reference_sha256"] = _sha(prompt_bytes)
            row["source_prompt_sha256"] = row["reference_sha256"]
            row["opponent_sha256"] = _sha(anchor_path.read_bytes())
            row["opponent_source_group"] = row["source_group"]
            row["frozen_sft512_checkpoint_sha256"] = self.checkpoint
            row["reference"] = f"/old/checkout/painter/runs/{self.prior_run_id}/curriculum/prompts/{split}/stage-{row['stage']}/{ident}.txt"
            row["source_prompt"] = row["reference"]
            row["opponent"] = f"/old/checkout/painter/runs/{self.prior_run_id}/curriculum/anchors/{ident}.png"
            row["anchor_provenance_file"] = f"/old/checkout/painter/runs/{self.prior_run_id}/curriculum/anchors/anchor-provenance.json"
            # Exercise optional casebrief copying on train rows and omit it
            # from one heldout row to prove the field is genuinely optional.
            if split == "train" or ident != self.heldout_rows[0]["id"]:
                brief_path = self.assets / ".materialized-old" / "casebriefs" / split / f"{ident}.txt"
                brief_path.parent.mkdir(parents=True, exist_ok=True)
                brief_bytes = f"brief for {ident}\n".encode()
                brief_path.write_bytes(brief_bytes)
                row["casebrief"] = f"/old/checkout/painter/runs/{self.prior_run_id}/curriculum/.materialized-old/casebriefs/{split}/{ident}.txt"
                row["casebrief_sha256"] = _sha(brief_bytes)

    def _build_rows(self) -> None:
        for stage in range(1, 5):
            for index in range(24):
                ident = f"df-s{stage}-tr-family-{stage}-{index:02d}"
                self.train_rows.append({
                    "id": ident,
                    "source_group": ident,
                    "family": f"family-{stage}",
                    "family_kind": "single" if stage == 1 else "connected",
                    "prompt_signature": f"{stage:02d}{index:062d}",
                    "split": "train",
                    "stage": stage,
                    "variant": index % 4,
                    "reference_kind": "text",
                    "task_id": ident,
                    "max_turns": 3,
                })
            for index in range(8):
                ident = f"df-s{stage}-he-family-{stage}-{index:02d}"
                self.heldout_rows.append({
                    "id": ident,
                    "source_group": ident,
                    "family": f"heldout-family-{stage}",
                    "family_kind": "single" if stage == 1 else "connected",
                    "prompt_signature": f"{stage:02d}{index:062d}",
                    "split": "heldout",
                    "stage": stage,
                    "variant": index % 2,
                    "reference_kind": "text",
                    "task_id": ident,
                    "max_turns": 3,
                })
        self._write_rows(self.train_rows, "train")
        self._write_rows(self.heldout_rows, "heldout")

    def _write_provenance(self) -> Path:
        entries = {}
        for row in [*self.train_rows, *self.heldout_rows]:
            ident = str(row["id"])
            entries[ident] = {
                "checkpoint_sha256": self.checkpoint,
                "generation_receipt": f"native-generation/{ident}.json",
                "png_sha256": row["opponent_sha256"],
                "reference_sha256": row["reference_sha256"],
                "source_group": row["source_group"],
                "valid": True,
            }
        path = self.assets / "anchors" / "anchor-provenance.json"
        path.write_text(json.dumps({
            "anchors": entries,
            "checkpoint_sha256": self.checkpoint,
            "generation_protocol": "fixture",
            "run_id": self.prior_run_id,
            "schema": M.PROVENANCE_SCHEMA,
        }, indent=2, sort_keys=True) + "\n")
        return path

    def _write_curriculum(self) -> Path:
        stages = []
        for stage in range(1, 5):
            stages.append({
                "stage": stage,
                "name": f"stage-{stage}",
                "train_rows": [row for row in self.train_rows if row["stage"] == stage],
                "heldout_rows": [row for row in self.heldout_rows if row["stage"] == stage],
            })
        value = {
            "matched_heldout_panel": {"rows": self.heldout_rows},
            "prior_run_id": self.prior_run_id,
            "run_id": self.staged_run_id,
            "schema": M.CURRICULUM_SCHEMA,
            "stages": stages,
            "train_row_ids": [str(row["id"]) for row in self.train_rows],
        }
        path = self.root / "curriculum-manifest.json"
        path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        return path

    def _write_prior_manifest(self, split: str, rows: list[dict[str, object]]) -> Path:
        provenance_hash = _sha(self.provenance.read_bytes())
        document = {
            "frozen_opponent": {
                "checkpoint_sha256": self.checkpoint,
                "model_alias": "sft-step512",
                "policy": "frozen-sft512",
                "provenance_file": f"/old/checkout/painter/runs/{self.prior_run_id}/curriculum/anchors/anchor-provenance.json",
                "provenance_sha256": provenance_hash,
                "same_source_group_required": True,
            },
            "native_rl_compatible": split == "train",
            "run_id": self.prior_run_id,
            "schema": M.NATIVE_SCHEMA,
            "split": split,
            "status": "admitted_with_verified_frozen_sft512_anchors",
            "rows": rows,
        }
        path = self.root / f"prior-{split}.json"
        path.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode())
        return path


class MaterializeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture = MaterializeFixtures(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _run(self, output: str) -> tuple[dict[str, object], Path]:
        destination = self.root / output
        receipt = M.materialize(
            train_manifest=self.fixture.train_manifest,
            heldout_manifest=self.fixture.heldout_manifest,
            asset_root=self.fixture.assets,
            output_root=destination,
            curriculum_manifest=self.fixture.curriculum,
        )
        return receipt, destination

    def test_rebases_stale_paths_and_output_resolves_like_native_loader(self) -> None:
        receipt, output = self._run("materialized")
        self.assertEqual(receipt["counts"]["train_rows"], 96)
        self.assertEqual(receipt["counts"]["heldout_rows"], 32)
        train = json.loads((output / "train-manifest.json").read_text())
        heldout = json.loads((output / "heldout-manifest.json").read_text())
        self.assertEqual(train["run_id"], self.fixture.staged_run_id)
        self.assertEqual(train["source_run_id"], self.fixture.prior_run_id)
        self.assertEqual(len(train["rows"]), 96)
        self.assertEqual(len(heldout["rows"]), 32)
        for row in [*train["rows"], *heldout["rows"]]:
            self.assertFalse(str(row["reference"]).startswith("/"))
            self.assertFalse(str(row["opponent"]).startswith("/"))
            self.assertTrue((output / row["reference"]).is_file())
            self.assertTrue((output / row["opponent"]).is_file())
            self.assertEqual(M.sha256_file(output / row["reference"]), row["reference_sha256"])
            self.assertEqual(M.sha256_file(output / row["opponent"]), row["opponent_sha256"])
        self.assertNotIn("casebrief", heldout["rows"][0])
        self.assertEqual(train["frozen_opponent"]["provenance_file"], "assets/anchors/anchor-provenance.json")
        self.assertTrue((output / "assets/anchors/anchor-provenance.json").is_file())

    def test_deterministic_output_has_stable_bytes(self) -> None:
        _, first = self._run("first")
        _, second = self._run("second")
        first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
        second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
        self.assertEqual(first_files, second_files)

    def test_anchor_tampering_is_rejected(self) -> None:
        target = self.fixture.assets / "anchors" / f"{self.fixture.train_rows[0]['id']}.png"
        target.write_bytes(b"tampered")
        with self.assertRaises(M.MaterializationError):
            self._run("tampered")
        self.assertFalse((self.root / "tampered").exists())

    def test_membership_mismatch_is_rejected(self) -> None:
        document = json.loads(self.fixture.train_manifest.read_text())
        document["rows"][0]["id"] = "not-in-curriculum"
        self.fixture.train_manifest.write_text(json.dumps(document))
        with self.assertRaises(M.MaterializationError):
            self._run("membership")

    def test_symlink_asset_is_rejected(self) -> None:
        row = self.fixture.train_rows[0]
        target = self.fixture.assets / "prompts" / "train" / f"stage-{row['stage']}" / f"{row['id']}.txt"
        real = target.with_name(target.name + ".real")
        target.rename(real)
        target.symlink_to(real)
        with self.assertRaises(M.MaterializationError):
            self._run("symlink")

    def test_path_traversal_is_rejected(self) -> None:
        document = json.loads(self.fixture.train_manifest.read_text())
        document["rows"][0]["reference"] = "../../outside.txt"
        self.fixture.train_manifest.write_text(json.dumps(document))
        with self.assertRaises(M.MaterializationError):
            self._run("escape")


if __name__ == "__main__":
    unittest.main()
