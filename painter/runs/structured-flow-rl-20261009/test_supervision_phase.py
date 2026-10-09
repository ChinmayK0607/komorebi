"""Offline contract tests for the Linux-only supervision approval phase."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("structured_flow_supervision_phase", HERE / "supervision_phase.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WARMUP_SPEC = importlib.util.spec_from_file_location("structured_flow_warmup_for_supervision", HERE / "warmup.py")
assert WARMUP_SPEC and WARMUP_SPEC.loader
WARMUP = importlib.util.module_from_spec(WARMUP_SPEC)
WARMUP_SPEC.loader.exec_module(WARMUP)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SupervisionPhaseTests(unittest.TestCase):
    def _fake_render(self, root: Path, invalid_candidate: str | None = None, **_: object) -> dict:
        results = []
        manifest = json.loads((root / "manifest.json").read_text())
        for row in manifest["rows"]:
            for version, source_name in (("block_in", "block_in.js"), ("improved_final", "improved_final.js")):
                source = root / "programs" / row["id"] / source_name
                png = root / "renders" / row["id"] / f"{version}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                png.write_bytes(f"{row['id']}:{version}".encode())
                results.append({
                    "candidate_id": row["id"],
                    "version": version,
                    "valid": row["id"] != invalid_candidate,
                    "source_sha256": _sha(source),
                    "png_sha256": _sha(png),
                    "receipt": {"valid": row["id"] != invalid_candidate, "renderer_sha256": "a" * 64},
                })
        value = {
            "schema": "painter.structured-flow-render.v1",
            "run_id": MODULE.RUN_ID,
            "renderer": {"renderer_sha256": "a" * 64, "assets_sha256": "b" * 64},
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "complete_hash_verified": invalid_candidate is None,
            "program_count": 32,
            "valid_program_count": 32,
            "results": results,
        }
        (root / "render-manifest.json").write_text(json.dumps(value))
        return value

    def _make_key(self, root: Path) -> Path:
        key = root / "gateway.env"
        key.write_text("AI_GATEWAY_API_KEY=test-only\n")
        key.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return key

    def test_review_filters_censored_families_and_writes_finish_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.flow.build(root)
            key = self._make_key(root)
            accepted = {candidate.id for candidate in (*MODULE.flow.CANDIDATES[:5], MODULE.flow.CANDIDATES[12])}
            calls: list[dict] = []

            async def fake_judge(*, brief: Path, anchor: Path, candidate: Path, key_file: Path, model: str) -> dict:
                candidate_id = candidate.parent.name
                calls.append({"candidate": candidate_id, "key_file": key_file, "model": model})
                anchor_sha = _sha(anchor)
                candidate_sha = _sha(candidate)
                if candidate_id in accepted:
                    return {
                        "status": "scored",
                        "consensus": "initial_agreement",
                        "score": 2 / 3,
                        "preference": "moderate",
                        "winner_id": "candidate-image",
                        "image_ids": {"anchor": "anchor-image", "candidate": "candidate-image"},
                        "image_sha256": {"anchor": anchor_sha, "candidate": candidate_sha},
                        "brief_sha256": _sha(brief),
                    }
                return {
                    "status": "scored",
                    "consensus": "initial_agreement",
                    "score": 0.5,
                    "preference": "tied",
                    "winner_id": "tie",
                    "image_ids": {"anchor": "anchor-image", "candidate": "candidate-image"},
                    "image_sha256": {"anchor": anchor_sha, "candidate": candidate_sha},
                    "brief_sha256": _sha(brief),
                }

            with patch.object(MODULE.sys, "platform", "linux"):
                receipt = MODULE.run_phase(
                    root,
                    Path("/renderer.py"),
                    Path("/renderer-python"),
                    "a" * 64,
                    key,
                    workers=4,
                    render_fn=self._fake_render,
                    judge_fn=fake_judge,
                )
            self.assertEqual(receipt["status"], "reviewed_ready_for_warmup")
            self.assertEqual(receipt["actual"], {"train_families": 5, "dev_families": 1, "finish_rows": 5, "train_rows": 15, "dev_rows": 2})
            self.assertEqual(len(calls), 16)
            self.assertTrue(all(call["model"] == MODULE.FLASH_MODEL for call in calls))
            approvals = json.loads((root / "approval-candidates.json").read_text())
            self.assertEqual(approvals["approved_count"], 6)
            self.assertEqual(len(approvals["candidates"]), 16)
            self.assertTrue(all("raw_decision" in item for item in approvals["candidates"]))
            train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(line) for line in (root / "dev.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["flow_stage"] == "finish" for row in train), 5)
            self.assertTrue(all(row["source_group"] in accepted for row in [*train, *dev]))
            self.assertTrue(all(row["target_receipt"]["approved"] for row in [*train, *dev]))
            self.assertTrue(all(row["messages"][-1]["content"][0]["text"] == "FINISHED" for row in train if row["flow_stage"] == "finish"))
            warmup_receipt = WARMUP.validate_inputs(root / "train.jsonl", root / "dev.jsonl", root / "approval-candidates.json")
            self.assertEqual(warmup_receipt["counts"], {"train": 15, "dev": 2})
            scheduled = WARMUP.build_schedule(train)
            self.assertEqual(len(scheduled), 12)
            self.assertEqual(
                {row["source_group"] for row in scheduled},
                {candidate.id for candidate in MODULE.flow.CANDIDATES[:5]},
            )

    def test_cached_paid_review_refreshes_filtered_15_2_without_judge_or_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.flow.build(root)
            key = self._make_key(root)
            accepted = {candidate.id for candidate in (*MODULE.flow.CANDIDATES[:5], MODULE.flow.CANDIDATES[12])}

            async def strong_judge(*, brief: Path, anchor: Path, candidate: Path, key_file: Path, model: str) -> dict:
                candidate_id = candidate.parent.name
                return {
                    "status": "scored", "consensus": "initial_agreement", "score": 2 / 3,
                    "preference": "moderate" if candidate_id in accepted else "tied",
                    "winner_id": "candidate-image" if candidate_id in accepted else "tie",
                    "image_ids": {"anchor": "anchor-image", "candidate": "candidate-image"},
                    "image_sha256": {"anchor": _sha(anchor), "candidate": _sha(candidate)},
                    "brief_sha256": _sha(brief),
                }

            with patch.object(MODULE.sys, "platform", "linux"):
                MODULE.run_phase(root, Path("/renderer.py"), Path("/renderer-python"), "a" * 64, key, render_fn=self._fake_render, judge_fn=strong_judge)
            (root / "train.jsonl").unlink()
            (root / "dev.jsonl").unlink()
            (root / "supervision-phase.json").unlink()
            with patch.object(MODULE.sys, "platform", "linux"), patch.object(MODULE, "render_and_verify", side_effect=AssertionError("cached refresh rendered")):
                refreshed = MODULE.refresh_filtered_exports(root, "a" * 64)
            self.assertEqual(refreshed["status"], "reviewed_ready_for_warmup")
            self.assertEqual(refreshed["actual"], {"train_families": 5, "dev_families": 1, "finish_rows": 5, "train_rows": 15, "dev_rows": 2})
            self.assertEqual(len((root / "train.jsonl").read_text().splitlines()), 15)
            self.assertEqual(len((root / "dev.jsonl").read_text().splitlines()), 2)

    def test_one_invalid_pair_is_censored_while_other_families_are_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.flow.build(root)
            key = self._make_key(root)
            invalid_id = MODULE.flow.CANDIDATES[7].id
            calls: list[str] = []

            async def strong_judge(*, brief: Path, anchor: Path, candidate: Path, key_file: Path, model: str) -> dict:
                candidate_id = candidate.parent.name
                calls.append(candidate_id)
                return {
                    "status": "scored", "consensus": "three_of_four", "score": 2 / 3,
                    "preference": "moderate", "winner_id": "candidate-image",
                    "image_ids": {"anchor": "anchor-image", "candidate": "candidate-image"},
                    "image_sha256": {"anchor": _sha(anchor), "candidate": _sha(candidate)},
                    "brief_sha256": _sha(brief),
                }

            with patch.object(MODULE.sys, "platform", "linux"):
                receipt = MODULE.run_phase(
                    root,
                    Path("/renderer.py"),
                    Path("/renderer-python"),
                    "a" * 64,
                    key,
                    workers=4,
                    render_fn=lambda render_root, **kwargs: self._fake_render(render_root, invalid_candidate=invalid_id, **kwargs),
                    judge_fn=strong_judge,
                )
            self.assertEqual(receipt["status"], "reviewed_ready_for_warmup")
            self.assertEqual(len(calls), 15)
            approval = json.loads((root / "approval-candidates.json").read_text())
            invalid = next(item for item in approval["candidates"] if item["id"] == invalid_id)
            self.assertFalse(invalid["render_valid"])
            self.assertEqual(invalid["decision"], "censored")
            self.assertIsNone(invalid["raw_decision"])
            self.assertEqual(invalid["judge_error"]["type"], "render_invalid")
            train_groups = {json.loads(line)["source_group"] for line in (root / "train.jsonl").read_text().splitlines()}
            self.assertNotIn(invalid_id, train_groups)

    def test_insufficient_approval_keeps_raw_censored_evidence_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.flow.build(root)
            key = self._make_key(root)

            async def weak_judge(*, brief: Path, anchor: Path, candidate: Path, key_file: Path, model: str) -> dict:
                return {
                    "status": "uncertain",
                    "consensus": "no_three_of_four",
                    "score": None,
                    "winner_id": None,
                    "image_sha256": {"anchor": _sha(anchor), "candidate": _sha(candidate)},
                    "brief_sha256": _sha(brief),
                }

            with patch.object(MODULE.sys, "platform", "linux"):
                receipt = MODULE.run_phase(
                    root,
                    Path("/renderer.py"),
                    Path("/renderer-python"),
                    "a" * 64,
                    key,
                    render_fn=self._fake_render,
                    judge_fn=weak_judge,
                )
            self.assertEqual(receipt["status"], "reviewed_insufficient_for_warmup")
            self.assertFalse((root / "train.jsonl").exists())
            approval = json.loads((root / "approval-candidates.json").read_text())
            self.assertTrue(all(item["decision"] == "censored" for item in approval["candidates"]))
            self.assertTrue(all(item["raw_decision"]["status"] == "uncertain" for item in approval["candidates"]))

    def test_normal_cli_reports_missing_runtime_inputs(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            MODULE.main(["--data", "/tmp/structured-flow-missing", "--renderer-sha256", "a" * 64])
        self.assertEqual(raised.exception.code, 2)

    def test_judge_concurrency_is_bounded_and_linux_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.flow.build(root)
            key = self._make_key(root)
            render_manifest = self._fake_render(root)
            manifest = json.loads((root / "manifest.json").read_text())
            briefs = MODULE._materialize_briefs(root, manifest)
            active = 0
            maximum = 0

            async def fake_judge(*, brief: Path, anchor: Path, candidate: Path, key_file: Path, model: str) -> dict:
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0)
                active -= 1
                return {
                    "status": "uncertain", "consensus": "no_three_of_four", "score": None,
                    "image_sha256": {"anchor": _sha(anchor), "candidate": _sha(candidate)},
                    "brief_sha256": _sha(brief),
                }

            with patch.object(MODULE.sys, "platform", "linux"):
                decisions = asyncio.run(MODULE.judge_candidates(root, manifest, render_manifest, briefs, key, workers=4, judge_fn=fake_judge))
            self.assertEqual(len(decisions), 16)
            self.assertLessEqual(maximum, 4)
            with patch.object(MODULE.sys, "platform", "darwin"):
                with self.assertRaisesRegex(RuntimeError, "Linux-node-only"):
                    MODULE.render_and_verify(root, renderer=Path("/renderer"), renderer_python=Path("/python"), renderer_sha256="a" * 64, workers=4)


if __name__ == "__main__":
    unittest.main()
