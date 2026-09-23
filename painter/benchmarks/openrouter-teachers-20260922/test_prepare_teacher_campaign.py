from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_teacher_campaign", HERE / "prepare_teacher_campaign.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_image(path: Path, marker: bytes) -> str:
    path.write_bytes(marker)
    return hashlib.sha256(marker).hexdigest()


class TestPrepareTeacherCampaign(unittest.TestCase):
    def test_generated_node_campaign_parent_is_ignored(self):
        ignore = (HERE / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("teacher-campaigns/", ignore)

    def _fixture(self, count: int = 6) -> tuple[Path, Path, dict[str, str]]:
        root = Path(tempfile.mkdtemp(prefix="teacher-campaign-test-"))
        images = root / "images"
        images.mkdir()
        rows = []
        hashes: dict[str, str] = {}
        for index in range(count):
            ident = f"train-{index:03d}"
            marker = f"fixture-image-{index}".encode()
            image = images / f"{ident}.jpg"
            digest = _write_image(image, marker)
            hashes[ident] = digest
            rows.append({"id": ident, "image": f"images/{image.name}", "sha256": digest, "source_split": "train", "family": ["flower", "bus", "animal"][index % 3], "difficulty": index})
        training = root / "training.json"
        training.write_text(json.dumps({"references": rows}) + "\n", encoding="utf-8")
        holdout_image = root / "holdout.jpg"
        holdout_digest = _write_image(holdout_image, b"evaluation-holdout")
        holdout = root / "holdout.json"
        holdout.write_text(json.dumps({"references": [{"id": "eval", "image": holdout_image.name, "sha256": holdout_digest}]}) + "\n", encoding="utf-8")
        return root, training, hashes

    def test_requires_dry_run_marker_and_never_needs_a_key(self):
        result = __import__("subprocess").run(
            ["python3", str(HERE / "prepare_teacher_campaign.py"), "--training-manifest", "/missing", "--output", "/missing-output", "--teacher-model", "zai/glm-5.3-flash"],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dry-run", result.stderr)

    def test_writes_disjoint_stage_roots_native_quality_and_launch_commands(self):
        root, training, _ = self._fixture()
        output = root / "campaign"
        summary = module.prepare_campaign(output=output, training_manifest=training, reference_root=root, model="zai/glm-5.3-flash", benchmark_refs=root / "holdout.json", counts={"short": 2, "medium": 2, "long": 2})
        self.assertTrue(summary["no_provider_calls"])
        self.assertTrue((output / "launch_commands.sh").is_file())
        launch = (output / "launch_commands.sh").read_text(encoding="utf-8")
        self.assertEqual(launch.count("--track quality"), 3)
        self.assertIn("--models 'zai/glm-5.3-flash'", launch)
        printed = subprocess.run(["bash", str(output / "launch_commands.sh")], capture_output=True, text=True, check=True).stdout
        self.assertIn(str(output / "short-02"), printed)
        self.assertIn(str(output / "medium-05"), printed)
        self.assertIn(str(output / "long-12"), printed)
        stage_ids: list[set[str]] = []
        for stage, max_turns in (("short-02", 2), ("medium-05", 5), ("long-12", 12)):
            stage_root = output / stage
            config = json.loads((stage_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["tracks"]["quality"]["max_turns"], max_turns)
            self.assertEqual(config["tracks"]["quality"]["max_tokens"], "native")
            self.assertTrue((stage_root / "gateway_transport.ts").is_file())
            refs = json.loads((stage_root / "refs.json").read_text(encoding="utf-8"))["references"]
            stage_ids.append({row["id"] for row in refs})
        self.assertEqual(len(set.union(*stage_ids)), 6)
        self.assertFalse(set.intersection(*stage_ids))
        self.assertTrue((output / "campaign-manifest.sha256").is_file())
        self.assertTrue((output / "file-hashes.sha256").is_file())
        self.assertFalse(os_access_writable(output / "campaign-manifest.json"))

    def test_excludes_and_records_benchmark_holdout_overlap_by_hash(self):
        root, training, hashes = self._fixture()
        holdout = root / "holdout.json"
        holdout.write_text(json.dumps({"references": [{"id": "eval", "image": "images/train-000.jpg", "sha256": hashes["train-000"]}]}), encoding="utf-8")
        summary = module.prepare_campaign(output=root / "campaign", training_manifest=training, reference_root=root, model="zai/glm-5.3-flash", benchmark_refs=holdout, counts={"short": 1, "medium": 1, "long": 1})
        self.assertEqual(summary["excluded_benchmark_holdout_overlap_count"], 1)
        self.assertEqual(summary["excluded_benchmark_holdout_overlap"][0]["id"], "train-000")
        for stage in ("short-02", "medium-05", "long-12"):
            refs = json.loads((root / "campaign" / stage / "refs.json").read_text(encoding="utf-8"))["references"]
            self.assertNotIn("train-000", {entry["id"] for entry in refs})

    def test_rejects_benchmark_refs_as_training_manifest(self):
        root, _, _ = self._fixture()
        with self.assertRaisesRegex(module.CampaignError, "evaluation holdout"):
            module.prepare_campaign(output=root / "campaign", training_manifest=HERE / "refs.json", reference_root=root, model="zai/glm-5.3-flash", benchmark_refs=HERE / "refs.json", counts={"short": 2, "medium": 2, "long": 2})


def os_access_writable(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


if __name__ == "__main__":
    unittest.main()
