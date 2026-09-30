from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_spatial_teacher_review import build_review


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_candidate(data: Path) -> None:
    candidate = data / "astra" / "01"
    candidate.mkdir(parents=True)
    (candidate / "prompt.txt").write_text("Draw a readable birdhouse on cream paper.", encoding="utf-8")
    (candidate / "candidate.json").write_text(
        json.dumps(
            {
                "candidate_id": "fixture-astra-01",
                "model": "gpt-6-astra",
                "name": "Fixture birdhouse",
                "style": "hybrid opaque illustration with pencil accents",
                "focus": ["roof", "entry"],
                "status": "authored-unrendered",
                "seed": 93001,
            }
        ),
        encoding="utf-8",
    )


def write_turn(collected: Path, turn: int, *, valid: bool = True, png_hash: str | None = None) -> tuple[bytes, bytes]:
    turn_dir = collected / "linux-turn1" / "astra" / "01" / f"turn-{turn}"
    turn_dir.mkdir(parents=True)
    source = f"function setup() {{ background({turn}); }}\n".encode()
    png = b"fixture-png-" + str(turn).encode()
    (turn_dir / "program.js").write_bytes(source)
    (turn_dir / "painting.png").write_bytes(png)
    receipt = {
        "valid": valid,
        "finished": valid,
        "timed_out": not valid,
        "error_code": None if valid else "render_timeout",
        "seed": 93001,
        "source_sha256": digest(source),
        "png_sha256": png_hash if png_hash is not None else digest(png),
        "elapsed_seconds": 1.25,
    }
    (turn_dir / "painting.json").write_text(json.dumps(receipt), encoding="utf-8")
    return source, png


class SpatialTeacherReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.collected = self.root / "collected"
        self.output = self.root / "review"
        write_candidate(self.data)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_verified_initial_and_selected_final_are_side_by_side(self) -> None:
        write_turn(self.collected, 1)
        write_turn(self.collected, 3)

        manifest = build_review(self.data, self.collected, self.output)

        task = manifest["tasks"][0]
        self.assertEqual(manifest["task_count"], 1)
        self.assertEqual(task["selected_final_render_id"], "astra-01-turn-3")
        self.assertEqual(task["selection_status"], "pending_final_metadata")
        self.assertFalse(task["selected_final_accepted"])
        self.assertIsNone(task["human_label"])
        self.assertFalse(task["training_export"])
        self.assertEqual(task["style_class"], "hybrid")
        self.assertTrue((self.output / "assets/astra-01-turn-1.png").is_file())
        self.assertTrue((self.output / "assets/astra-01-turn-3.png").is_file())
        page = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn("Initial render", page)
        self.assertIn("Selected final render", page)
        self.assertIn("Draw a readable birdhouse", page)

    def test_png_hash_mismatch_is_preserved_but_not_copied(self) -> None:
        write_turn(self.collected, 1)
        write_turn(self.collected, 3, png_hash="0" * 64)

        manifest = build_review(self.data, self.collected, self.output)

        task = manifest["tasks"][0]
        self.assertIsNone(task["selected_final_render_id"])
        self.assertEqual(task["selection_status"], "failed_selected_render")
        self.assertIn("astra-01-turn-3", task["preserved_failed_turns"])
        self.assertFalse((self.output / "assets/astra-01-turn-3.png").exists())
        selected = next(row for row in task["turns"] if row["render_id"] == "astra-01-turn-3")
        self.assertIn("png_hash_mismatch", selected["verification"])

    def test_invalid_selected_final_is_not_autoaccepted(self) -> None:
        write_turn(self.collected, 1)
        write_turn(self.collected, 3, valid=False)

        manifest = build_review(self.data, self.collected, self.output)

        task = manifest["tasks"][0]
        self.assertIsNone(task["selected_final_render_id"])
        self.assertFalse(task["selected_final_accepted"])
        self.assertEqual(task["review_status"], "failed_selected_render")
        self.assertFalse(task["training_benefit_claim"])
        self.assertEqual(task["preserved_failed_turns"], ["astra-01-turn-3"])


if __name__ == "__main__":
    unittest.main()
