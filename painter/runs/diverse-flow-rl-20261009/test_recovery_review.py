"""Offline display and comparison contracts for the diverse recovery viewer."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VIEWER = load("build_recovery_review")
SCORER = load("score_recovery")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> Path:
    rows = []
    for index in range(32):
        brief = root / f"brief-{index}.txt"
        brief.write_text(f"[USER]\npaint case {index}\n", encoding="utf-8")
        anchor = root / f"anchor-{index}.png"
        anchor.write_bytes(f"anchor-{index}".encode())
        rows.append({
            "id": f"case-{index}",
            "family": f"family-{index}",
            "source_group": f"source-{index}",
            "reference": str(brief),
            "reference_sha256": _sha(brief),
            "opponent": str(anchor),
            "opponent_sha256": _sha(anchor),
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cases": rows}), encoding="utf-8")
    return manifest


def _cells(root: Path, cases: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Provide one valid and one censored cell without touching model/render code."""
    image = root / "case-0.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"verified-painting")
    return {
        "case-0": {
            "path": str(image),
            "status": "valid_terminal",
            "terminal_valid": True,
            "terminal_recorded": True,
            "fallback_path": None,
        },
        "case-1": {
            "path": None,
            "status": "censored_invalid_terminal_fallback_prior",
            "terminal_valid": False,
            "terminal_recorded": True,
            "fallback_path": str(image),
        },
    }


class RecoveryReviewDisplayTest(unittest.TestCase):
    def test_optional_postwarmup_builds_four_generic_panels_and_keeps_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest(root)
            starting, postwarmup, final_rl = (root / name for name in ("starting", "postwarmup", "final-rl"))
            output = root / "review"

            def fake_cells(policy_root, cases):
                return _cells(Path(policy_root), cases)

            with mock.patch.object(VIEWER, "_trace_cells", side_effect=fake_cells):
                review = VIEWER.build(
                    manifest=manifest,
                    baseline_root=starting,
                    candidate_root=final_rl,
                    postwarmup_root=postwarmup,
                    output=output,
                )

            self.assertEqual(review["policies"], [
                "starting_painter", "after_code_sft", "after_rl", "frozen_sft_reference",
            ])
            self.assertEqual(review["rows"][0]["after_code_sft"]["status"], "valid_terminal")
            self.assertEqual(review["rows"][1]["after_rl"]["status"], "censored_invalid_terminal_fallback_prior")
            html = (output / "index.html").read_text(encoding="utf-8")
            for label in ("Starting painter", "After code SFT", "After RL", "Frozen SFT reference"):
                self.assertIn(label, html)
            self.assertIn("review.policies.map", html)
            self.assertIn("repeat(auto-fit", html)
            self.assertIn("Previous", html)
            self.assertIn("Next", html)
            self.assertNotIn("After 4 more RL updates", html)
            self.assertNotIn("Flash RL step 12", html)
            self.assertNotIn("Checkpoint 4", html)

    def test_existing_two_root_collect_api_keeps_three_panel_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest(root)
            with mock.patch.object(VIEWER, "_trace_cells", side_effect=lambda path, cases: _cells(Path(path), cases)):
                review = VIEWER.collect(
                    manifest=manifest,
                    baseline_root=root / "starting",
                    candidate_root=root / "final-rl",
                )
            self.assertEqual(review["policies"], ["starting_painter", "after_rl", "frozen_sft_reference"])
            self.assertNotIn("after_code_sft", review["rows"][0])
            self.assertEqual(review["failed_or_missing"], 31)

    def test_real_trace_loader_keeps_invalid_terminal_censored_with_prior_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest(root)
            cases = VIEWER._manifest_rows(manifest)
            trace_root = root / "native_painting--consistent--trace"
            trace_root.mkdir(parents=True)
            image = root / "evaluation-rollouts" / "case-0" / "trace-0" / "turn-3" / "painting.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"prior-terminal-image")
            node_path = (
                "/root/painter/repo/painter/runs/diverse-flow-rl-20261009/"
                "validation/eval/evaluation-rollouts/case-0/trace-0/turn-3/painting.png"
            )
            (trace_root / "traces.jsonl").write_text(json.dumps({
                "task": {"data": {"id": "case-0"}},
                "info": {
                    "finished": True,
                    "final_canvas": node_path,
                    "final_canvas_valid": False,
                    "turns": [
                        {"turn": 1, "valid": True, "canvas": node_path, "canvas_sha256": _sha(image)},
                        {"turn": 3, "valid": False, "canvas": node_path},
                    ],
                },
            }) + "\n", encoding="utf-8")
            cells = VIEWER._trace_cells(root, cases)
            self.assertEqual(cells["case-0"]["status"], "censored_invalid_terminal_fallback_prior")
            self.assertFalse(cells["case-0"]["terminal_valid"])
            self.assertTrue(cells["case-0"]["fallback_path"])

    def test_score_reports_explicit_primary_and_postwarmup_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest(root)
            image = root / "verified.png"
            image.write_bytes(b"verified")
            brief = root / "brief.txt"
            brief.write_text("paint", encoding="utf-8")
            rows = []
            review_rows = []
            for index in range(32):
                ident = f"case-{index}"
                row = {"id": ident, "family": f"family-{index}"}
                rows.append(row)
                valid = {
                    "path": str(image), "status": "valid_terminal", "terminal_valid": True,
                }
                review_rows.append({
                    **row,
                    "brief": "paint",
                    "starting_painter": dict(valid),
                    "after_code_sft": dict(valid),
                    "after_rl": dict(valid),
                })
            manifest.write_text(json.dumps({"cases": rows}), encoding="utf-8")
            key_file = root / "judge.key"
            key_file.write_text("test-key", encoding="utf-8")
            key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
            output = root / "scores"
            calls: list[tuple[Path, Path]] = []

            async def fake_judge_pair(*, brief, anchor, candidate, **kwargs):
                del brief, kwargs
                calls.append((anchor, candidate))
                return {
                    "protocol": SCORER.PROTOCOL,
                    "format_version": SCORER.FORMAT_VERSION,
                    "status": "scored",
                    "score": 0.75,
                }

            fake_judge_module = types.ModuleType("painter.text_pairwise_consistent_side_judge_v2_generic")
            fake_judge_module.judge_pair = fake_judge_pair
            fake_review = {
                "rows": review_rows,
            }
            with mock.patch.object(SCORER, "collect", return_value=fake_review), mock.patch.dict(
                sys.modules, {"painter.text_pairwise_consistent_side_judge_v2_generic": fake_judge_module}
            ):
                summary = asyncio.run(SCORER.score(
                    manifest=manifest,
                    baseline_root=root / "starting",
                    candidate_root=root / "final-rl",
                    postwarmup_root=root / "postwarmup",
                    output=output,
                    key_file=key_file,
                    concurrency=8,
                ))

            self.assertEqual(calls.__len__(), 64)
            self.assertEqual(summary["comparison_labels"], ["final_rl_vs_start", "final_rl_vs_postwarmup"])
            self.assertEqual(summary["comparisons"]["final_rl_vs_start"]["comparable_visual"], 32)
            self.assertEqual(summary["comparisons"]["final_rl_vs_postwarmup"]["comparable_visual"], 32)
            first = summary["rows"][0]
            self.assertEqual(first["comparisons"]["final_rl_vs_start"]["anchor_label"], "Starting painter")
            self.assertEqual(first["comparisons"]["final_rl_vs_start"]["candidate_label"], "After RL")
            self.assertEqual(first["comparisons"]["final_rl_vs_postwarmup"]["anchor_label"], "After code SFT")
            self.assertEqual(first["status"], "scored")
            self.assertEqual(first["flash12_status"], "valid_terminal")
            self.assertEqual(first["checkpoint4_status"], "valid_terminal")


if __name__ == "__main__":
    unittest.main()
