from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_comparison import ARM_SPECS, CASE_IDS, build


class MatchedComparisonBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "completed"
        self.output = Path(self.temp.name) / "comparison" / "index.html"
        self.cases = [
            {"id": case_id, "category": "fixture", "prompt": "Paint a small paper fan." if i == 0 else "fixture"}
            for i, case_id in enumerate(CASE_IDS)
        ]
        for arm_id, _, step in ARM_SPECS:
            validation = self.root / arm_id / "validation" / f"step-{step}-balanced16"
            validation.mkdir(parents=True)
            (validation / "cases.json").write_text(json.dumps({"cases": self.cases}), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _episode(root: Path, arm_id: str, step: int, case_id: str, *, uuid: str = "episode") -> Path:
        episode = root / arm_id / "validation" / f"step-{step}-balanced16" / "eval" / "evaluation-rollouts" / f"consistent-step-{step}" / case_id / uuid
        episode.mkdir(parents=True)
        return episode

    @staticmethod
    def _painting(turn: Path, *, valid: bool, png: bytes | None = None, error: str | None = None) -> None:
        payload: dict[str, object] = {"valid": valid, "finished": valid}
        if error:
            payload["errors"] = [error]
        if png is not None:
            path = turn / "painting.png"
            path.write_bytes(png)
            payload["png_sha256"] = hashlib.sha256(png).hexdigest()
        (turn / "painting.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_ordered_arms_pairs_prompt_and_relative_image_links(self) -> None:
        for arm_id, _, step in ARM_SPECS:
            episode = self._episode(self.root, arm_id, step, CASE_IDS[0])
            turn = episode / "turn-1"
            turn.mkdir()
            self._painting(turn, valid=True, png=f"{arm_id}-image".encode())
            (episode / "native_trace.json").write_text(
                json.dumps(
                    {
                        "info": {
                            "task_id": CASE_IDS[0],
                            "turns": [{"turn": 1, "valid": True, "canvas": "/node/turn-1/painting.png"}],
                        }
                    }
                ),
                encoding="utf-8",
            )

        result = build(input_root=self.root, output=self.output)
        self.assertEqual([case["id"] for case in result["cases"]], list(CASE_IDS))
        self.assertEqual([arm["label"] for arm in result["arms"]], [spec[1] for spec in ARM_SPECS])
        self.assertEqual(len(result["results"]), 64)
        first_rows = [row for row in result["results"] if row["case_id"] == CASE_IDS[0]]
        self.assertEqual([row["arm_id"] for row in first_rows], [spec[0] for spec in ARM_SPECS])
        self.assertTrue(all(row["status"] == "VALID TERMINAL" for row in first_rows))
        self.assertTrue(all(row["image"].startswith("../completed/") for row in first_rows))
        page = self.output.read_text(encoding="utf-8")
        self.assertIn("Paint a small paper fan.", page)
        self.assertIn("Pairwise comparison", page)
        self.assertIn("RL • Luna judge • 12 updates", page)

    def test_invalid_native_terminal_never_promotes_prior_image(self) -> None:
        episode = self._episode(self.root, "baseline", 1, CASE_IDS[0])
        turn1 = episode / "turn-1"
        turn1.mkdir()
        self._painting(turn1, valid=True, png=b"earlier-image")
        turn2 = episode / "turn-2"
        turn2.mkdir()
        self._painting(turn2, valid=True, png=b"receipt-says-valid")
        (episode / "native_trace.json").write_text(
            json.dumps(
                {
                    "info": {
                        "task_id": CASE_IDS[0],
                        "final_canvas_valid": True,
                        "turns": [
                            {"turn": 1, "valid": True, "canvas": "/node/turn-1/painting.png"},
                            {"turn": 2, "valid": False, "canvas": "/node/turn-1/painting.png", "error": "terminal action invalid"},
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

        result = build(input_root=self.root, output=self.output)
        row = next(item for item in result["results"] if item["arm_id"] == "baseline" and item["case_id"] == CASE_IDS[0])
        self.assertEqual(row["status"], "INVALID TERMINAL")
        self.assertEqual(row["terminal_turn"], 2)
        self.assertIsNone(row["image"])
        self.assertIn("terminal action invalid", row["error"])
        self.assertTrue(row["turns"][0]["png"])

    def test_missing_and_censored_terminal_outcomes_remain_explicit(self) -> None:
        episode = self._episode(self.root, "flash", 12, CASE_IDS[0])
        turn = episode / "turn-1"
        turn.mkdir()
        self._painting(turn, valid=True)  # valid action, but no image artifact

        retained = self._episode(self.root, "pro", 12, CASE_IDS[0])
        retained_turn = retained / "turn-1"
        retained_turn.mkdir()
        self._painting(retained_turn, valid=True, png=b"retained-without-trace")

        result = build(input_root=self.root, output=self.output)
        flash = next(item for item in result["results"] if item["arm_id"] == "flash" and item["case_id"] == CASE_IDS[0])
        pro = next(item for item in result["results"] if item["arm_id"] == "pro" and item["case_id"] == CASE_IDS[0])
        luna = next(item for item in result["results"] if item["arm_id"] == "luna" and item["case_id"] == CASE_IDS[0])
        self.assertEqual(flash["status"], "CENSORED TERMINAL")
        self.assertIsNone(flash["image"])
        self.assertEqual(pro["status"], "CENSORED TERMINAL")
        self.assertIsNone(pro["image"])
        self.assertEqual(luna["status"], "MISSING TERMINAL")
        page = self.output.read_text(encoding="utf-8")
        self.assertIn("CENSORED TERMINAL", page)
        self.assertIn("MISSING TERMINAL", page)


if __name__ == "__main__":
    unittest.main()
