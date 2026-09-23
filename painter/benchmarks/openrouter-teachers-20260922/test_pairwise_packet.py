import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairwise_packet


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path, format="PNG")


def _episode(root: Path, model: str, reference_sha: str, *, status: str, invalid: bool, color: tuple[int, int, int]) -> None:
    job = f"job-{model.replace('/', '-')}-quality"
    episode_dir = root / "episodes" / job
    canvas = episode_dir / "turn-01.png"
    _image(canvas, color)
    canvas_rel = canvas.relative_to(root).as_posix()
    canvas_sha = _sha(canvas)
    (episode_dir / "episode.json").write_text(json.dumps({
        "schema": "painter.ai-gateway-teachers.v1",
        "job_id": job,
        "model": model,
        "reference_id": "ref-1",
        "reference_image": "references/ref-1.png",
        "reference_sha256": reference_sha,
        "status": status,
        "last_turn_invalid": invalid,
        "final_valid_canvas": canvas_rel,
        "settings": {"track": "quality", "provider": "fixture-provider"},
        "turns": [{
            "turn": 1,
            "canvas": canvas_rel,
            "current_canvas_sha256": canvas_sha,
            "render": {"valid": True, "canvas_sha256": canvas_sha},
        }],
    }, indent=2))


class PairwisePacketTests(unittest.TestCase):
    def _fixture(self) -> tuple[Path, Path]:
        temp = Path(tempfile.mkdtemp())
        root = temp / "results"
        root.mkdir()
        reference = temp / "references" / "ref-1.png"
        _image(reference, (240, 240, 240))
        reference_sha = _sha(reference)
        (temp / "refs.json").write_text(json.dumps({"references": [{
            "id": "ref-1", "image": "references/ref-1.png", "sha256": reference_sha,
        }]}))
        _episode(root, "model/one", reference_sha, status="complete", invalid=False, color=(255, 0, 0))
        _episode(root, "model/two", reference_sha, status="complete", invalid=False, color=(0, 255, 0))
        # This has a real PNG and a render-valid turn, but the episode ended
        # invalid.  It must remain censored rather than enter the queue.
        _episode(root, "model/three", reference_sha, status="invalid", invalid=True, color=(0, 0, 255))
        _episode(root, "model/four", reference_sha, status="deadline_censored", invalid=False, color=(255, 255, 0))
        return root, temp

    def test_builds_blinded_eligible_queue_and_censored_accounting(self):
        root, temp = self._fixture()
        with self.subTest("build"):
            result = pairwise_packet.build_pairwise_packet(root, temp / "packet", track="quality", seed="test-seed")
        public = json.loads(Path(result["packet"]).read_text())
        private = json.loads(Path(result["private_mapping"]).read_text())
        self.assertEqual(public["coverage"], {
            "attempted_pairs": 6,
            "eligible_comparisons": 1,
            "one_invalid_censored": 4,
            "both_invalid_censored": 1,
        })
        self.assertEqual(len(public["comparisons"]), 1)
        self.assertEqual(len(public["censored"]), 5)
        self.assertEqual(len(private["pairs"]), 6)
        comparison = public["comparisons"][0]
        self.assertIsNone(comparison["judgment"])
        self.assertTrue(comparison["candidate_a"]["valid"])
        self.assertTrue(comparison["candidate_b"]["valid"])
        for key in ("model", "provider", "track", "turn", "job_id", "reference_id"):
            self.assertNotIn(key, json.dumps(public))
        self.assertEqual(comparison["reference"]["sha256"], _sha(temp / "references" / "ref-1.png"))
        for candidate in (comparison["candidate_a"], comparison["candidate_b"]):
            self.assertTrue((Path(result["packet"]).parent / candidate["path"]).is_file())

    def test_orientation_and_outputs_are_deterministic(self):
        root, temp = self._fixture()
        first = pairwise_packet.build_pairwise_packet(root, temp / "one", track="quality", seed="stable")
        second = pairwise_packet.build_pairwise_packet(root, temp / "two", track="quality", seed="stable")
        first_public = json.loads(Path(first["packet"]).read_text())
        second_public = json.loads(Path(second["packet"]).read_text())
        self.assertEqual(first_public, second_public)
        first_private = json.loads(Path(first["private_mapping"]).read_text())
        second_private = json.loads(Path(second["private_mapping"]).read_text())
        # Private paths are artifact-relative, so the mapping is reproducible
        # even when the destination directory changes.
        self.assertEqual(first_private, second_private)

    def test_reference_hash_mismatch_fails_closed(self):
        root, temp = self._fixture()
        episode = next((root / "episodes").glob("*/episode.json"))
        value = json.loads(episode.read_text())
        value["reference_sha256"] = "0" * 64
        episode.write_text(json.dumps(value))
        with self.assertRaisesRegex(pairwise_packet.PairwisePacketError, "reference hash mismatch"):
            pairwise_packet.build_pairwise_packet(root, temp / "packet", track="quality")

    def test_screen_track_builds_its_own_packet(self):
        root, temp = self._fixture()
        for path in (root / "episodes").glob("*/episode.json"):
            row = json.loads(path.read_text())
            row["settings"]["track"] = "screen"
            path.write_text(json.dumps(row))
        output = temp / "screen-packet"
        self.assertEqual(pairwise_packet.main(["--root", str(root), "--track", "screen", "--output", str(output)]), 0)
        packet = json.loads((output / "judge-packet.json").read_text())
        self.assertEqual(packet["coverage"]["attempted_pairs"], 6)
        self.assertEqual(packet["coverage"]["eligible_comparisons"], 1)


if __name__ == "__main__":
    unittest.main()
