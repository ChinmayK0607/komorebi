from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_cpu_recovery_gallery import _judgment_text, build_gallery


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CpuRecoveryGalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "cpu-recovery-results"
        self.cases = self.results / "cases"
        self.judgments = self.results / "judgments"
        self.anchors = self.root / "anchors-all" / "anchors"
        self.prompts = self.root / "anchors-all" / "prompts"
        for directory in (self.cases, self.judgments, self.anchors, self.prompts):
            directory.mkdir(parents=True)

        self.manifest_cases = []
        self._add_case("exact", b"same", b"same", "recovered-identical", "verified")
        self._add_case("changed", b"target", b"replayed", "new-replay", "verified")
        self._add_case("failed", b"target", None, "unavailable", "failed", ["render_timeout"])
        (self.anchors / "unused.png").write_bytes(b"unused")
        (self.prompts / "unused.txt").write_text("unused", encoding="utf-8")
        (self.judgments / "changed.json").write_text(
            json.dumps({"verdict": "pending-human-review"}), encoding="utf-8"
        )
        (self.results / "manifest.json").write_text(json.dumps({
            "schema": "text-sft512-cpu-replay-results-v1",
            "summary": {"verified": 2, "failed": 1},
            "cases": self.manifest_cases,
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _add_case(
        self,
        case_id: str,
        target: bytes,
        replay: bytes | None,
        comparison: str,
        status: str,
        failure_reasons: list[str] | None = None,
    ) -> None:
        case_dir = self.cases / case_id
        case_dir.mkdir()
        (self.anchors / f"{case_id}.png").write_bytes(b"anchor-" + case_id.encode())
        (self.prompts / f"{case_id}.txt").write_text(f"brief-{case_id}", encoding="utf-8")
        target_hash = digest(target)
        replay_hash = digest(replay) if replay is not None else None
        if replay is not None:
            (case_dir / "painting.png").write_bytes(replay)
            (case_dir / "replay-receipt.json").write_text(
                json.dumps({"id": case_id, "status": status}), encoding="utf-8"
            )
        self.manifest_cases.append({
            "id": case_id,
            "family": "complex",
            "source_turn": 3,
            "source_sha256": "source-" + case_id,
            "target_png_sha256": target_hash,
            "replay_png_sha256": replay_hash,
            "status": status,
            "comparison": comparison,
            "failure_reasons": failure_reasons or [],
            "png_file": f"cases/{case_id}/painting.png",
            "replay_receipt_file": f"cases/{case_id}/replay-receipt.json",
        })

    def test_gallery_copies_only_used_assets_and_labels_provenance(self) -> None:
        output = self.results / "review"
        report = build_gallery(self.results / "manifest.json", self.anchors, self.prompts, output)
        self.assertEqual(report["counts"]["recoverable"], 3)
        self.assertEqual(report["counts"]["recovered_identical"], 1)
        self.assertEqual(report["counts"]["new_cpu_replay"], 1)
        self.assertEqual(report["counts"]["recovery_failures"], 1)
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("Frozen SFT512", page)
        self.assertIn("Replayed RL step16", page)
        self.assertIn("Scores are relative preference against frozen SFT; 0.5 means tie with frozen SFT", page)
        self.assertIn("<b>Prompt:</b> brief-exact", page)
        self.assertIn("<details><summary>Provenance and receipts</summary>", page)
        self.assertIn("new CPU replay", page)
        self.assertIn("recovery failure: render_timeout (not model failure)", page)
        self.assertIn("pending-human-review", page)
        self.assertIn("../cases/changed/painting.png", page)
        self.assertIn("../cases/exact/replay-receipt.json", page)
        self.assertTrue((output / "assets/anchors/exact.png").is_file())
        self.assertTrue((output / "assets/prompts/changed.txt").is_file())
        self.assertFalse((output / "assets/anchors/unused.png").exists())
        self.assertFalse((output / "assets/prompts/unused.txt").exists())

    def test_judgment_labels_are_directional_and_rounded(self) -> None:
        self.assertEqual(_judgment_text({"status": "scored", "score": 2 / 3}), "RL preferred (0.667)")
        self.assertEqual(_judgment_text({"status": "scored", "score": 1 / 3}), "SFT preferred (0.333)")
        self.assertEqual(_judgment_text({"status": "scored", "score": 0.5}), "Tie (0.500)")
        self.assertEqual(_judgment_text({"status": "uncertain"}), "Judge disagreed (uncertain)")

    def test_manifest_file_path_cannot_escape_result_root(self) -> None:
        raw = json.loads((self.results / "manifest.json").read_text(encoding="utf-8"))
        raw["cases"][0]["png_file"] = "../outside.png"
        (self.results / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes result root"):
            build_gallery(self.results / "manifest.json", self.anchors, self.prompts)


if __name__ == "__main__":
    unittest.main()
