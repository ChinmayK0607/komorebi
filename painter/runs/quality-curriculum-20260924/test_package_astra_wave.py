"""Check that partial MiMo prefixes do not block Astra coverage."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_astra_wave


class PriorLookupTest(unittest.TestCase):
    def test_missing_episode_has_no_prior(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(package_astra_wave, "WAVE5", Path(directory)):
            self.assertEqual(package_astra_wave.latest_valid("easy-b", "coco128-000000000036"), (None, None, None))

    def test_original_prefix_can_supply_valid_prior(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(package_astra_wave, "WAVE5", Path(directory)):
            episode = Path(directory) / "easy-b/episodes/job--coco128-000000000036--s01"
            episode.mkdir(parents=True)
            canvas = episode / "turn-01.png"
            program = episode / "turn-01.program.js"
            canvas.write_bytes(b"canvas")
            program.write_text("program")
            episode.joinpath("episode.json").write_text(json.dumps({"turns": [{
                "turn": 1, "render": {"valid": True},
                "canvas": f"episodes/{episode.name}/{canvas.name}",
                "program": f"episodes/{episode.name}/{program.name}",
                "current_canvas_sha256": package_astra_wave.sha(canvas),
            }]}))
            self.assertEqual(package_astra_wave.latest_valid("easy-b", "coco128-000000000036"),
                             (canvas, program, 1))


if __name__ == "__main__":
    unittest.main()
