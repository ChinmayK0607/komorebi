import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_curriculum


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="PNG")


def _episode(root: Path, model: str, reference_sha: str, *, job: str, turns: int, status: str = "complete", invalid: bool = False) -> None:
    episode_dir = root / "episodes" / job
    episode_dir.mkdir(parents=True, exist_ok=True)
    canvas = episode_dir / f"turn-{turns:02d}.png"
    _image(canvas, (turns, 30, 90))
    canvas_rel = canvas.relative_to(root).as_posix()
    canvas_sha = _sha(canvas)
    turn_rows = []
    for number in range(1, turns + 1):
        turn_canvas = canvas_rel if number == turns else None
        turn_rows.append({
            "turn": number,
            "canvas": turn_canvas,
            "render": {"valid": number == turns},
            "current_canvas_after": turn_canvas,
        })
    (episode_dir / "turn-01.js").write_text("// preserved program\n", encoding="utf-8")
    (episode_dir / "episode.json").write_text(json.dumps({
        "schema": "painter.ai-gateway-teachers.v1",
        "job_id": job,
        "model": model,
        "reference_id": "train-ref",
        "reference_image": "references/train-ref.png",
        "reference_sha256": reference_sha,
        "status": status,
        "last_turn_invalid": invalid,
        "first_valid_canvas": canvas_rel,
        "final_valid_canvas": canvas_rel,
        "settings": {"track": "quality"},
        "turns": turn_rows,
        "total_tokens": 123,
    }, indent=2), encoding="utf-8")


class CurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp())
        self.source = self.temp / "results"
        self.reference_root = self.temp / "training"
        self.source.mkdir()
        reference = self.reference_root / "references" / "train-ref.png"
        _image(reference, (240, 240, 240))
        self.reference_sha = _sha(reference)
        self.training_manifest = self.temp / "training-refs.json"
        self.training_manifest.write_text(json.dumps({"references": [{
            "id": "train-ref", "image": "references/train-ref.png", "sha256": self.reference_sha, "source_split": "train",
        }]}), encoding="utf-8")
        holdout_image = self.temp / "holdout" / "references" / "eval.png"
        _image(holdout_image, (1, 2, 3))
        self.holdout_manifest = self.temp / "holdout-refs.json"
        self.holdout_manifest.write_text(json.dumps({"references": [{
            "id": "eval", "image": "references/eval.png", "sha256": _sha(holdout_image),
        }]}), encoding="utf-8")
        _episode(self.source, "teacher/best", self.reference_sha, job="short-job", turns=2)
        _episode(self.source, "teacher/best", self.reference_sha, job="medium-job", turns=4)
        _episode(self.source, "teacher/best", self.reference_sha, job="bad-job", turns=3, status="invalid", invalid=True)
        _episode(self.source, "other/model", self.reference_sha, job="other-job", turns=6)
        self.labels = self.temp / "labels.jsonl"
        self.labels.write_text("\n".join(json.dumps(row) for row in [
            {"job_id": "short-job", "admitted": True, "quality_score": 4, "reason": "clean"},
            {"job_id": "medium-job", "admitted": True, "quality_score": 5, "reason": "strong finish"},
            {"job_id": "bad-job", "admitted": True, "quality_score": 5},
            {"job_id": "other-job", "admitted": True, "quality_score": 5},
        ]) + "\n", encoding="utf-8")

    def _build(self, output: Path) -> dict:
        return build_curriculum.build_curriculum(
            self.source, output, model="teacher/best", labels_path=self.labels,
            reference_manifest=self.training_manifest, holdout_manifest=self.holdout_manifest,
            reference_root=self.reference_root, split="train", track="quality",
        )

    def test_admits_only_reviewed_valid_selected_model_and_stages(self) -> None:
        summary = self._build(self.temp / "curriculum")
        self.assertEqual(summary["accepted_total"], 2)
        self.assertEqual(summary["stages"]["short"]["accepted"], 1)
        self.assertEqual(summary["stages"]["medium"]["accepted"], 1)
        self.assertEqual(summary["stages"]["long"]["accepted"], 0)
        record = json.loads((self.temp / "curriculum/stages/short.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["teacher_model"], "teacher/best")
        self.assertTrue((self.temp / "curriculum" / record["reference"]["path"]).is_file())
        self.assertTrue((self.temp / "curriculum" / record["episode_receipt"]["path"]).is_file())
        reasons = {row["job_id"]: row["reason"] for row in map(json.loads, (self.temp / "curriculum/excluded.jsonl").read_text().splitlines())}
        self.assertEqual(reasons["bad-job"], "episode_status:invalid")

    def test_accepts_split_field_from_campaign_manifest(self) -> None:
        manifest = json.loads(self.training_manifest.read_text())
        entry = manifest["references"][0]
        entry["split"] = entry.pop("source_split")
        self.training_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        summary = self._build(self.temp / "split-alias")
        self.assertEqual(summary["accepted_total"], 2)

    def test_conflicting_split_fields_fail_closed(self) -> None:
        manifest = json.loads(self.training_manifest.read_text())
        manifest["references"][0]["split"] = "validation"
        self.training_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(build_curriculum.CurriculumError, "conflicting reference split"):
            self._build(self.temp / "conflicting-split")

    def test_evaluation_overlap_fails_closed(self) -> None:
        overlapping = self.temp / "holdout-overlap.json"
        overlapping.write_text(json.dumps({"references": [{"id": "train-ref", "sha256": self.reference_sha}]}), encoding="utf-8")
        with self.assertRaisesRegex(build_curriculum.CurriculumError, "overlaps the evaluation holdout"):
            build_curriculum.build_curriculum(
                self.source, self.temp / "overlap", model="teacher/best", labels_path=self.labels,
                reference_manifest=self.training_manifest, holdout_manifest=overlapping,
                reference_root=self.reference_root,
            )

    def test_missing_labels_are_never_admitted(self) -> None:
        labels = self.temp / "empty.jsonl"
        labels.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(build_curriculum.CurriculumError, "no reviewed"):
            build_curriculum.build_curriculum(
                self.source, self.temp / "empty", model="teacher/best", labels_path=labels,
                reference_manifest=self.training_manifest, holdout_manifest=self.holdout_manifest,
                reference_root=self.reference_root,
            )

    def test_turn_limit_needs_explicit_final_quality_review(self) -> None:
        episode = self.source / "episodes" / "medium-job" / "episode.json"
        value = json.loads(episode.read_text())
        value["status"] = "turn_limit"
        episode.write_text(json.dumps(value), encoding="utf-8")
        summary = self._build(self.temp / "turn-limit-excluded")
        self.assertEqual(summary["accepted_total"], 1)
        reasons = {row["job_id"]: row["reason"] for row in map(json.loads, (self.temp / "turn-limit-excluded/excluded.jsonl").read_text().splitlines())}
        self.assertEqual(reasons["medium-job"], "turn_limit_requires_reviewed_final_quality")

        labels = self.temp / "turn-limit-labels.jsonl"
        rows = [json.loads(line) for line in self.labels.read_text().splitlines()]
        for row in rows:
            if row["job_id"] == "medium-job":
                row["final_quality"] = True
        labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        summary = build_curriculum.build_curriculum(
            self.source, self.temp / "turn-limit-admitted", model="teacher/best", labels_path=labels,
            reference_manifest=self.training_manifest, holdout_manifest=self.holdout_manifest,
            reference_root=self.reference_root, split="train", track="quality",
        )
        self.assertEqual(summary["accepted_total"], 2)

    def test_secret_in_turn_artifact_is_rejected_before_copy(self) -> None:
        (self.source / "episodes" / "short-job" / "turn-01.json").write_text(
            json.dumps({"response": "sk-or-v1-0123456789abcdef"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(build_curriculum.CurriculumError, "possible credential"):
            self._build(self.temp / "secret")
        self.assertFalse((self.temp / "secret/episodes/short-job/episode.json").exists())


if __name__ == "__main__":
    unittest.main()
