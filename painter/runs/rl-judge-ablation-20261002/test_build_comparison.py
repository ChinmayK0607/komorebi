from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_comparison", RUN / "build_comparison.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonGalleryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.eval_roots: dict[str, Path] = {}
        self.png_good = b"fake-png-good"
        self.png_bad = b"fake-png-bad"
        for arm in ("flash", "pro", "luna"):
            self.eval_roots[arm] = self.root / arm
            self.eval_roots[arm].mkdir()
        for arm in ("flash", "pro", "luna"):
            for i in range(1, 17):
                if arm == "pro" and i == 2:
                    # c02 is intentionally absent from Pro: the builder must
                    # show missing/censored evidence, never borrow another arm.
                    continue
                self._write_case(arm, f"c{i:02d}")
        self._write_case("flash", "c01", invalid_final=True)
        self._write_case("luna", "c01", hash_mismatch=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _write_case(self, arm: str, case_id: str, *, invalid_final: bool = False, hash_mismatch: bool = False) -> None:
        case = self.eval_roots[arm] / case_id
        (case / "turn-1").mkdir(parents=True, exist_ok=True)
        (case / "turn-2").mkdir(parents=True, exist_ok=True)
        first = self.png_good
        (case / "turn-1" / "painting.png").write_bytes(first)
        (case / "turn-1" / "painting.json").write_text(json.dumps({
            "valid": True,
            "png_sha256": self._sha(first),
            "seed": 918210,
        }) + "\n", encoding="utf-8")
        final = self.png_bad
        (case / "turn-2" / "painting.png").write_bytes(final)
        expected = self._sha(self.png_good) if hash_mismatch else self._sha(final)
        (case / "turn-2" / "painting.json").write_text(json.dumps({
            "valid": not invalid_final,
            "png_sha256": expected,
            "seed": 918210,
        }) + "\n", encoding="utf-8")
        (case / "episode.json").write_text(json.dumps({
            "turns": [{"turn": 1, "valid": True}, {"turn": 2, "valid": not invalid_final}]
        }) + "\n", encoding="utf-8")

    def _manifest(self, *, panel_hash: str | None = None) -> Path:
        cases = [{
            "id": f"c{i:02d}",
            "source_group": f"fixture-{i:02d}",
            "prompt": f"fixture prompt {i}",
            "seed": 918210,
            "provenance": "offline-fixture",
            "max_turns": 2,
        } for i in range(1, 17)]
        arms = [{
            "id": arm,
            "label": {"flash": "Flash judge", "pro": "Pro judge", "luna": "Luna judge"}[arm],
            "checkpoint": "step-12",
            "checkpoints": [{
                "id": "step-12",
                "label": f"{arm} step 12",
                "eval_root": str(self.eval_roots[arm]),
                "checkpoint_sha256": f"{arm}-checkpoint-hash",
            }],
            "seed": 42,
            "source_id": f"fixture-source-{arm}",
        } for arm in ("flash", "pro", "luna")]
        payload = {
            "schema": MODULE.SCHEMA,
            "cases": cases,
            "arms": arms,
            "references": [],
        }
        if panel_hash is not None:
            payload["cases_sha256"] = panel_hash
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_final_invalid_is_visible_and_never_replaced_by_earlier_valid(self) -> None:
        manifest = self._manifest()
        output = self.root / "gallery"
        result = MODULE.build_gallery(manifest, output)
        gallery = json.loads((output / "gallery.json").read_text(encoding="utf-8"))
        row = next(row for row in gallery["rows"] if row["arm_id"] == "flash" and row["case_id"] == "c01")
        self.assertEqual(row["final_turn"], 2)
        self.assertFalse(row["final_valid"])
        self.assertEqual(row["final_status"], "final invalid action")
        self.assertTrue(row["final_png"].endswith("painting-invalid.png"))
        self.assertEqual(row["turns"][0]["status"], "valid")
        self.assertIn("final invalid action", (output / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["final_invalid"], 1)

    def test_missing_and_unverified_final_cases_are_counted(self) -> None:
        manifest = self._manifest()
        output = self.root / "gallery"
        MODULE.build_gallery(manifest, output)
        gallery = json.loads((output / "gallery.json").read_text(encoding="utf-8"))
        missing = next(row for row in gallery["rows"] if row["arm_id"] == "pro" and row["case_id"] == "c02")
        mismatch = next(row for row in gallery["rows"] if row["arm_id"] == "luna" and row["case_id"] == "c01")
        self.assertIsNone(missing["final_turn"])
        self.assertEqual(missing["final_status"], "final missing")
        self.assertIsNone(missing["final_png"])
        self.assertIn("unverified", mismatch["final_status"])
        self.assertIsNone(mismatch["final_valid"])
        self.assertTrue(mismatch["terminal_action_valid"])
        self.assertFalse(mismatch["turns"][1]["png_hash_ok"])
        self.assertEqual(gallery["summary"]["final_missing_or_censored"], 2)

    def test_manifest_panel_hash_and_case_identity_are_bound(self) -> None:
        cases = [{"id": f"c{i:02d}"} for i in range(1, 17)]
        expected = MODULE.canonical_sha(cases)
        manifest = self._manifest(panel_hash=expected)
        # The fixture manifest has richer case fields, so a stale hash is
        # rejected before any evidence is copied.
        with self.assertRaises(ValueError):
            MODULE.build_gallery(manifest, self.root / "bad-hash")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["cases"][1]["id"] = payload["cases"][0]["id"]
        bad = self.root / "duplicate.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            MODULE.load_manifest(bad)

    def test_blind_fixed_pairs_and_reveal_control_are_present(self) -> None:
        manifest = self._manifest()
        output = self.root / "gallery"
        MODULE.build_gallery(manifest, output)
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="reveal"', page)
        self.assertIn('id="pair"', page)
        self.assertIn("fixed", page)
        self.assertIn("Flash judge", page)  # available to the explicit reveal control
        self.assertIn('"blind_label":"A vs B"', page)
        self.assertNotIn('new Option(`${p[0]} vs ${p[1]}`', page)

    def test_checkpoint_selection_requires_a_shared_id(self) -> None:
        manifest = self._manifest()
        with self.assertRaises(ValueError):
            MODULE.build_gallery(manifest, self.root / "gallery", checkpoint_id="step-4")

    def test_optional_baseline_reference_keeps_its_natural_checkpoint_identity(self) -> None:
        manifest = self._manifest()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["references"] = [{
            "id": "sft512",
            "label": "Frozen SFT512 baseline",
            "kind": "frozen_baseline",
            "checkpoints": [{
                "id": "sft512",
                "label": "SFT512",
                "eval_root": str(self.eval_roots["flash"]),
                "checkpoint_sha256": "sft-reference-hash",
            }],
        }]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        output = self.root / "gallery-with-reference"
        MODULE.build_gallery(manifest, output)
        gallery = json.loads((output / "gallery.json").read_text(encoding="utf-8"))
        self.assertIn(["sft512", "flash"], gallery["pairs"])
        reference_row = next(row for row in gallery["rows"] if row["arm_id"] == "sft512")
        self.assertEqual(reference_row["checkpoint_id"], "step-12")
        self.assertEqual(reference_row["actual_checkpoint_id"], "sft512")

    def test_text_reference_path_is_hash_bound_and_copied_as_prompt(self) -> None:
        manifest = self._manifest()
        prompt = self.root / "brief-c01.txt"
        raw = b"A declared text brief for c01.\n"
        prompt.write_bytes(raw)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        case = payload["cases"][0]
        case.pop("prompt")
        case.update({
            "reference_kind": "text",
            "reference": prompt.name,
            "reference_sha256": self._sha(raw),
        })
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        output = self.root / "gallery-with-prompt"
        MODULE.build_gallery(manifest, output)
        gallery = json.loads((output / "gallery.json").read_text(encoding="utf-8"))
        selected = next(case for case in gallery["cases"] if case["id"] == "c01")
        self.assertEqual(selected["prompt_text"], "A declared text brief for c01.")
        prompt_asset = output / selected["prompt_asset"]
        self.assertEqual(prompt_asset.read_bytes(), raw)
        case["reference_sha256"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            MODULE.build_gallery(manifest, self.root / "bad-prompt")


if __name__ == "__main__":
    unittest.main()
