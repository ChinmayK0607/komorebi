"""Pure tests for the shared native painting RL integration.

These tests use local files and a queue double.  They do not render, contact a
judge, start Prime, or validate the Linux-node runtime configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from painter.test_native_painting import _load_module as _load_native_module


def _load_rl_module():
    native = _load_native_module()
    vf = native.vf
    vf.reward = lambda **_kwargs: (lambda function: function)
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = vf
    with patch.dict(sys.modules, {"verifiers": verifiers, "verifiers.v1": vf}, clear=False):
        sys.modules.pop("painter.native_painting_rl", None)
        return importlib.import_module("painter.native_painting_rl")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Trace:
    id = "trace-test"

    def __init__(self, **info):
        self.info = dict(info)


class NativePaintingRLTests(unittest.TestCase):
    def test_consistent_judge_format_selector_is_explicit_and_fail_closed(self):
        module = _load_rl_module()
        self.assertEqual(
            module._consistent_judge_module_name("identity-joint-schema-v1"),
            "painter.text_pairwise_consistent_judge",
        )
        self.assertEqual(
            module._consistent_judge_module_name("identity-joint-side-schema-v1"),
            "painter.text_pairwise_consistent_side_judge",
        )
        self.assertEqual(
            module._consistent_judge_module_name("identity-joint-side-schema-v2-pro-v1"),
            "painter.text_pairwise_consistent_side_judge_v2_pro",
        )
        self.assertEqual(
            module._consistent_judge_module_name("identity-joint-side-schema-v2-generic-v1"),
            "painter.text_pairwise_consistent_side_judge_v2_generic",
        )
        with self.assertRaises(ValueError):
            module._consistent_judge_module_name("unregistered-format")

    def test_consistent_judge_model_argument_is_opt_in_and_receipt_keeps_identity(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-pro", source_group="text-pro", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts",
                mode="fresh",
            )
            result = {
                "protocol": "text-paint-sft-pairwise-consistent-v1",
                "format_version": "identity-joint-side-schema-v2-pro-v1",
                "status": "scored", "score": 0.75,
                "requested_model": "xiaomi/mimo-v2.6-pro",
                "returned_models": ["xiaomi/mimo-v2.6-pro"],
            }
            seen = {}

            async def fake_judge(**kwargs):
                seen.update(kwargs)
                return result

            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                "PAINTER_JUDGE_FORMAT": "identity-joint-side-schema-v2-pro-v1",
                "PAINTER_JUDGE_MODEL": "xiaomi/mimo-v2.6-pro",
            }), patch("painter.text_pairwise_consistent_side_judge_v2_pro.judge_pair", fake_judge):
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.75)
            self.assertEqual(seen["model"], "xiaomi/mimo-v2.6-pro")
            receipt = json.loads((root / "rl-rollouts/trace-test/text-judge.json").read_text())
            self.assertEqual(receipt["requested_model"], "xiaomi/mimo-v2.6-pro")
            self.assertEqual(receipt["returned_models"], ["xiaomi/mimo-v2.6-pro"])

    def test_text_reward_uses_fixed_anchor_pairwise_score(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-1", source_group="text-1", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor),
                output_root="rl-rollouts", mode="fresh",
            )
            async def fake_judge(**kwargs):
                self.assertEqual((kwargs["brief"], kwargs["anchor"], kwargs["candidate"]),
                                 (brief, anchor, canvas))
                return {"protocol": "text-paint-sft-pairwise-v1", "score": 0.25,
                        "status": "scored", "views": []}
            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {"PAINTER_GATEWAY_ENV_FILE": str(root / "key")}), \
                 patch("painter.text_pairwise_judge.judge_pair", fake_judge):
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.25)
            self.assertEqual(trace.info["score_evidence"]["protocol_version"],
                             "text-paint-sft-pairwise-v1")
            self.assertTrue((root / "rl-rollouts/trace-test/text-judge.json").is_file())

    def test_text_consistent_protocol_preserves_decision_metadata(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-consistent", source_group="text-consistent", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts",
                mode="fresh",
            )
            result = {
                "protocol": "text-paint-sft-pairwise-consistent-v1",
                "format_version": "identity-joint-schema-v1",
                "status": "scored", "score": 5 / 6,
                "winner_id": "img_candidate", "preference": "clear",
                "consensus": "initial_agreement", "acceptance": "two_of_two",
                "initial_conflict": False, "initial_votes": ["img_candidate", "img_candidate"],
                "recheck_votes": [],
            }
            async def fake_judge(**kwargs):
                self.assertEqual((kwargs["brief"], kwargs["anchor"], kwargs["candidate"]),
                                 (brief, anchor, canvas))
                return result
            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
            }), patch("painter.text_pairwise_consistent_judge.judge_pair", fake_judge):
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 5 / 6)
            self.assertEqual(trace.info["reward_status"], "scored")
            self.assertEqual(trace.info["score_evidence"]["protocol_version"],
                             "text-paint-sft-pairwise-consistent-v1")
            self.assertEqual(trace.info["score_evidence"]["decision"]["winner_id"], "img_candidate")
            self.assertEqual(json.loads((root / "rl-rollouts/trace-test/text-judge.json").read_text()), result)

    def test_text_uncertain_judge_is_not_zero_or_service_failure(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-uncertain", source_group="text-uncertain", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts",
                mode="fresh",
            )
            result = {
                "protocol": "text-paint-sft-pairwise-consistent-v1",
                "status": "uncertain", "score": None,
                "consensus": "no_three_of_four", "acceptance": "censored",
                "initial_conflict": True, "initial_votes": ["img_a", "img_b"],
                "recheck_votes": ["img_a", "tie"],
            }
            async def fake_judge(**_kwargs):
                return result
            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
            }), patch("painter.text_pairwise_consistent_judge.judge_pair", fake_judge):
                with self.assertRaisesRegex(ValueError, "missing or uncertain"):
                    asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertIsNone(trace.info["final_canvas_score"])
            self.assertEqual(trace.info["reward_status"], "judge_uncertain")
            self.assertNotIn("service_failure", trace.info)
            evidence = trace.info["score_evidence"]
            self.assertEqual(evidence["status"], "uncertain")
            self.assertEqual(evidence["decision"]["consensus"], "no_three_of_four")
            self.assertEqual(json.loads((root / "rl-rollouts/trace-test/text-judge.json").read_text()), result)

    def test_text_malformed_score_is_uncertain_instead_of_zero(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-malformed", source_group="text-malformed", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts",
                mode="fresh",
            )
            async def fake_judge(**_kwargs):
                return {"protocol": "text-paint-sft-pairwise-consistent-v1", "status": "scored", "score": "bad"}
            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
            }), patch("painter.text_pairwise_consistent_judge.judge_pair", fake_judge):
                with self.assertRaisesRegex(ValueError, "malformed"):
                    asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(trace.info["reward_status"], "judge_uncertain")
            self.assertNotIn("service_failure", trace.info)

    def test_text_protocol_mismatch_and_boolean_score_are_uncertain(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-strict", source_group="text-strict", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts",
                mode="fresh",
            )
            cases = [
                ({"protocol": "text-paint-sft-pairwise-v1", "status": "scored", "score": 0.5},
                 "missing or uncertain"),
                ({"protocol": "text-paint-sft-pairwise-consistent-v1", "status": "scored", "score": True},
                 "malformed"),
            ]
            for index, (result, message) in enumerate(cases):
                async def fake_judge(**_kwargs):
                    return result
                trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
                trace.id = f"trace-strict-{index}"
                with patch.dict("os.environ", {
                    "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
                    "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                }), patch("painter.text_pairwise_consistent_judge.judge_pair", fake_judge):
                    with self.assertRaisesRegex(ValueError, message):
                        asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
                self.assertEqual(trace.info["reward_status"], "judge_uncertain")

    def test_text_loader_requires_matching_fixed_sft_canvas(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "brief.txt").write_text("Paint one pear beside a cup.\n")
            (root / "sft.png").write_bytes(b"sft canvas")
            row = {"id": "text-1", "source_group": "text-1", "family": "simple",
                   "split": "train", "reference_kind": "text", "reference": "brief.txt",
                   "reference_sha256": _digest(root / "brief.txt"), "opponent": "sft.png",
                   "opponent_sha256": _digest(root / "sft.png")}
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"rows": [row]}))
            task = list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(
                root=str(root), manifest=manifest.name,
            )).load())[0]
            self.assertEqual(task.data.reference_kind, "text")
            self.assertIn("PAINTING BRIEF", task.data.system_prompt)
            self.assertNotIn("reference", task.data.system_prompt.lower())
            self.assertTrue(all(not hasattr(part, "image_url")
                                for part in task.data.prompt[0].content))
            manifest.write_text(json.dumps({"rows": [{k: v for k, v in row.items()
                                                     if not k.startswith("opponent") }]}))
            with self.assertRaisesRegex(ValueError, "pre-rendered fixed opponent"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(
                    root=str(root), manifest=manifest.name, lazy_opponent=True,
                )).load())

    def test_rl_task_declares_rl_wire_data_type(self):
        module = _load_rl_module()
        self.assertIs(module.NativePaintingRLTask.data_type(), module.NativePaintingRLData)

    def test_rl_env_reuses_shared_native_loop(self):
        module = _load_rl_module()
        self.assertIs(module.NativePaintingRLEnv.run, module.NativePaintingEnv.run)

    def test_interleave_covers_each_group_before_second_row(self):
        module = _load_rl_module()
        rows = [
            {"id": "a0-1", "family": "a", "source_group": "a0"},
            {"id": "a0-2", "family": "a", "source_group": "a0"},
            {"id": "a1-1", "family": "a", "source_group": "a1"},
            {"id": "b0-1", "family": "b", "source_group": "b0"},
            {"id": "b0-2", "family": "b", "source_group": "b0"},
        ]
        ordered = module._interleave(rows)
        first_groups = [row["source_group"] for row in ordered[:3]]
        self.assertEqual(set(first_groups), {"a0", "a1", "b0"})
        self.assertEqual(len(first_groups), len(set(first_groups)))

    def test_interleave_balances_large_unequal_family_prefix(self):
        module = _load_rl_module()
        rows = []
        for family_index in range(12):
            family = f"family-{family_index:02d}"
            for group_index in range(200):
                group = f"{family}-group-{group_index:03d}"
                for row_index in range(2):
                    rows.append({
                        "id": f"{group}-row-{row_index}",
                        "family": family,
                        "source_group": group,
                    })

        ordered = module._interleave(rows)
        first_round = ordered[: 12 * 200]
        self.assertEqual(len({row["source_group"] for row in first_round}), 12 * 200)
        self.assertTrue(all(row["id"].endswith("row-0") for row in first_round))
        prefix = ordered[:512]
        self.assertEqual({row["family"] for row in prefix}, {
            f"family-{index:02d}" for index in range(12)
        })
        family_counts = Counter(row["family"] for row in prefix)
        self.assertLessEqual(max(family_counts.values()) - min(family_counts.values()), 1)
        self.assertEqual(ordered[12 * 200]["id"].split("-")[-1], "1")

    def test_loader_interleaves_all_rows_and_preserves_hash_bound_pair(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            for name, contents in (
                ("ref-a.png", b"ref-a"),
                ("opp-a.png", b"opp-a"),
                ("ref-b.png", b"ref-b"),
                ("opp-b.png", b"opp-b"),
            ):
                paths[name] = root / name
                paths[name].write_bytes(contents)
            rows = [
                {"id": "a-1", "source_group": "group-a", "family": "a", "split": "train", "reference": "ref-a.png", "opponent": "opp-a.png"},
                {"id": "a-2", "source_group": "group-a", "family": "a", "split": "train", "reference": "ref-a.png", "opponent": "opp-a.png"},
                {"id": "b-1", "source_group": "group-b", "family": "b", "split": "train", "reference": "ref-b.png", "opponent": "opp-b.png"},
                {"id": "b-2", "source_group": "group-b", "family": "b", "split": "train", "reference": "ref-b.png", "opponent": "opp-b.png"},
            ]
            rows[0]["reference_sha256"] = _digest(paths["ref-a.png"])
            rows[0]["opponent_sha256"] = _digest(paths["opp-a.png"])
            manifest = root / "scaled.json"
            manifest.write_text(json.dumps({"training_rows": rows}))

            config = module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")
            tasks = list(module.NativePaintingRLTaskset(config).load())

            self.assertEqual(len(tasks), 4)
            self.assertEqual([task.data.family for task in tasks], ["a", "b", "a", "b"])
            first = tasks[0].data
            self.assertEqual(first.reference_sha256, _digest(paths["ref-a.png"]))
            self.assertEqual(first.opponent_sha256, _digest(paths["opp-a.png"]))
            self.assertEqual(first.opponent_source_group, "group-a")
            self.assertIn("group-a:fresh:", tasks[0].key)
            self.assertNotEqual(tasks[0].key, tasks[2].key)
            self.assertTrue(tasks[0].data.prompt)

            jsonl = root / "scaled.jsonl"
            jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            jsonl_tasks = list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.jsonl")).load())
            self.assertEqual([task.data.task_id for task in jsonl_tasks], [task.data.task_id for task in tasks])

    def test_loader_requires_distinct_fixed_same_group_opponent(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, contents in (("reference.png", b"reference"), ("opponent.png", b"opponent"), ("other.png", b"other")):
                (root / name).write_bytes(contents)
            manifest = root / "scaled.json"
            base = {"source_group": "same", "family": "shape", "split": "train", "reference": "reference.png"}
            manifest.write_text(json.dumps({"rows": [dict(base, id="one", opponent="opponent.png")] }))
            tasks = list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())
            self.assertEqual(len(tasks), 1)

            unsplit = dict(base, id="unsplit", opponent="opponent.png")
            unsplit.pop("split")
            manifest.write_text(json.dumps({"rows": [unsplit]}))
            with self.assertRaisesRegex(ValueError, "must declare split=train"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())

            manifest.write_text(json.dumps({"rows": [
                dict(base, id="one", opponent="opponent.png"),
                dict(base, id="two", opponent="other.png"),
            ]}))
            self.assertEqual(len(list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())), 2)

            manifest.write_text(json.dumps({"rows": [dict(base, id="one"),]}))
            with self.assertRaisesRegex(ValueError, "no fixed opponent"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())

            manifest.write_text(json.dumps({"rows": [
                dict(base, id="one", rollout_group="pair", opponent="opponent.png"),
                dict(base, id="two", rollout_group="pair", opponent="other.png"),
            ]}))
            with self.assertRaisesRegex(ValueError, "rollout_group pair has multiple fixed opponents"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())

            manifest.write_text(json.dumps({"rows": [dict(base, id="one", opponent={"path": "opponent.png", "source_group": "other"})]}))
            with self.assertRaisesRegex(ValueError, "expected source_group"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())

    def test_correction_rows_require_canvas_and_program(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reference.png").write_bytes(b"reference")
            (root / "opponent.png").write_bytes(b"opponent")
            (root / "canvas.png").write_bytes(b"canvas")
            (root / "program.js").write_text("draw();")
            manifest = root / "scaled.json"
            row = {
                "id": "repair-1", "source_group": "group", "family": "shape",
                "split": "train",
                "reference": "reference.png", "opponent": "opponent.png", "mode": "correction",
                "initial_canvas": "canvas.png", "initial_program": "program.js",
            }
            manifest.write_text(json.dumps([row]))
            task = list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())[0]
            self.assertEqual(task.data.mode, "correction")
            self.assertEqual(task.data.initial_program, "draw();")
            self.assertEqual(len(task.data.prompt[0].content), 6)

            row.pop("initial_program")
            manifest.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "needs initial_canvas and initial_program"):
                list(module.NativePaintingRLTaskset(module.NativePaintingRLConfig(root=str(root), manifest="scaled.json")).load())

    def test_reward_preserves_paths_hashes_and_receipt_without_additive_signals(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            opponent = root / "opponent.png"
            candidate = root / "candidate.png"
            reference.write_bytes(b"reference")
            opponent.write_bytes(b"opponent")
            candidate.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-1", source_group="group", family="shape",
                reference=str(reference), reference_sha256=_digest(reference),
                opponent=str(opponent), opponent_sha256=_digest(opponent),
                mode="fresh", max_turns=2, output_root="rl-rollouts",
            )
            task = module.NativePaintingRLTask(data)
            calls = {}

            class QueueDouble:
                class Judge:
                    PROTOCOL_VERSION = "aesthetic-pairwise-v1"

                judge = Judge()

                def enqueue_comparison(self, *, root, trace_id, reference, anchor, candidate):
                    calls["paths"] = (root, trace_id, reference, anchor, candidate)
                    request = "request-1"
                    folder = root / "judge-queue" / request
                    folder.mkdir(parents=True)
                    (folder / "receipt.json").write_text(json.dumps({"winner": "A"}))
                    return request

                async def await_receipt(self, **_kwargs):
                    return {"winner": "A", "judgment": "external"}

                def score_receipt(self, *, receipt, request_id, reference, anchor, candidate):
                    calls["score"] = (receipt, request_id, reference, anchor, candidate)
                    return 0.75

            trace = _Trace(final_canvas=str(candidate), final_canvas_valid=True)
            with patch.object(module, "_QueueContract", return_value=QueueDouble()):
                score = asyncio.run(task.final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.75)
            self.assertEqual(calls["paths"][2:], (reference, opponent, candidate))
            self.assertEqual(trace.info["comparison_images"]["image_sha256"]["candidate"], _digest(candidate))
            self.assertEqual(trace.info["score_evidence"]["request_id"], "request-1")
            self.assertTrue((root / "rl-rollouts" / "trace-test" / "reward.json").is_file())

    def test_queue_uncertain_receipt_is_masked_with_raw_evidence(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            opponent = root / "opponent.png"
            candidate = root / "candidate.png"
            reference.write_bytes(b"reference")
            opponent.write_bytes(b"opponent")
            candidate.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-uncertain", source_group="group", family="shape",
                reference=str(reference), reference_sha256=_digest(reference),
                opponent=str(opponent), opponent_sha256=_digest(opponent),
                mode="fresh", max_turns=2, output_root="rl-rollouts",
            )

            class QueueDouble:
                class Judge:
                    PROTOCOL_VERSION = "aesthetic-pairwise-v1"

                judge = Judge()

                def enqueue_comparison(self, *, root, trace_id, reference, anchor, candidate):
                    request = "request-uncertain"
                    folder = root / "judge-queue" / request
                    folder.mkdir(parents=True)
                    (folder / "receipt.json").write_text(json.dumps({"status": "uncertain"}))
                    return request

                async def await_receipt(self, **_kwargs):
                    return {"status": "uncertain"}

                def score_receipt(self, **_kwargs):
                    return None

            trace = _Trace(final_canvas=str(candidate), final_canvas_valid=True)
            with patch.object(module, "_QueueContract", return_value=QueueDouble()):
                with self.assertRaisesRegex(ValueError, "missing or uncertain"):
                    asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertIsNone(trace.info["final_canvas_score"])
            self.assertEqual(trace.info["reward_status"], "judge_uncertain")
            self.assertEqual(trace.info["score_evidence"]["status"], "uncertain")
            self.assertEqual(trace.info["score_evidence"]["request_id"], "request-uncertain")
            self.assertEqual(trace.info["judge_receipt"], {"status": "uncertain"})

    def test_reward_scores_model_invalid_canvas_zero_but_masks_service_failure(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            opponent = root / "opponent.png"
            reference.write_bytes(b"reference")
            opponent.write_bytes(b"opponent")
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-1", source_group="group", family="shape",
                reference=str(reference), reference_sha256=_digest(reference),
                opponent=str(opponent), opponent_sha256=_digest(opponent), mode="fresh",
            )
            task = module.NativePaintingRLTask(data)
            trace = _Trace(final_canvas_valid=False)
            self.assertEqual(asyncio.run(task.final_canvas_aesthetic(trace)), 0.0)
            self.assertEqual(trace.info["reward_status"], "model_invalid_canvas")
            self.assertEqual(trace.info["final_canvas_score"], 0.0)
            with self.assertRaisesRegex(RuntimeError, "not a visual score"):
                asyncio.run(task.final_canvas_aesthetic(_Trace(service_failure="renderer unavailable")))

    def test_opt_in_invalid_final_action_gets_zero_but_retains_canvas_provenance(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-final-invalid", source_group="text-final-invalid",
                family="simple", reference=str(brief), reference_sha256=_digest(brief),
                reference_kind="text", opponent=str(anchor), opponent_sha256=_digest(anchor),
                output_root="rl-rollouts", mode="fresh",
            )
            trace = _Trace(
                final_canvas=str(canvas), final_canvas_valid=True,
                turns=[{"turn": 2, "action": "paint", "valid": False, "failure_kind": "program"}],
            )
            with patch.dict("os.environ", {
                "PAINTER_REQUIRE_VALID_FINAL_ACTION": "1",
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
            }), patch("painter.text_pairwise_judge.judge_pair") as judge:
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.0)
            self.assertEqual(trace.info["reward_status"], "model_invalid_final_action")
            self.assertEqual(trace.info["final_canvas"], str(canvas))
            self.assertTrue(trace.info["final_canvas_valid"])
            self.assertEqual(trace.info["final_action_validation"]["status"], "invalid")
            judge.assert_not_called()

    def test_opt_in_valid_final_action_remains_judge_eligible(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief, anchor, canvas = (root / name for name in ("brief.txt", "anchor.png", "canvas.png"))
            brief.write_text("Paint a pear beside a cup.\n")
            anchor.write_bytes(b"anchor")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="text-final-valid", source_group="text-final-valid", family="simple",
                reference=str(brief), reference_sha256=_digest(brief), reference_kind="text",
                opponent=str(anchor), opponent_sha256=_digest(anchor), output_root="rl-rollouts", mode="fresh",
            )
            async def fake_judge(**_kwargs):
                return {"protocol": "text-paint-sft-pairwise-v1", "score": 0.25,
                        "status": "scored", "views": []}
            trace = _Trace(
                final_canvas=str(canvas), final_canvas_valid=True,
                turns=[{"turn": 3, "action": "finish", "valid": True}],
            )
            with patch.dict("os.environ", {
                "PAINTER_REQUIRE_VALID_FINAL_ACTION": "1",
                "PAINTER_GATEWAY_ENV_FILE": str(root / "key"),
            }), patch("painter.text_pairwise_judge.judge_pair", fake_judge):
                score = asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(score, 0.25)
            self.assertEqual(trace.info["final_action_validation"]["status"], "valid")

    def test_opt_in_missing_final_action_evidence_is_operational(self):
        module = _load_rl_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference, opponent, canvas = (root / name for name in ("reference.png", "opponent.png", "canvas.png"))
            reference.write_bytes(b"reference")
            opponent.write_bytes(b"opponent")
            canvas.write_bytes(b"candidate")
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-final-missing", source_group="group", family="shape",
                reference=str(reference), reference_sha256=_digest(reference), opponent=str(opponent),
                opponent_sha256=_digest(opponent), mode="fresh",
            )
            trace = _Trace(final_canvas=str(canvas), final_canvas_valid=True)
            with patch.dict("os.environ", {"PAINTER_REQUIRE_VALID_FINAL_ACTION": "1"}):
                with self.assertRaisesRegex(RuntimeError, "last turn"):
                    asyncio.run(module.NativePaintingRLTask(data).final_canvas_aesthetic(trace))
            self.assertEqual(trace.info["reward_status"], "final_action_evidence_error")
            self.assertNotIn("final_canvas_score", trace.info)

    def test_opt_in_service_failure_stays_operational(self):
        module = _load_rl_module()
        task = module.NativePaintingRLTask(module.NativePaintingRLData(
            root=".", task_id="task-service", source_group="group", family="shape",
            reference="reference.png", reference_sha256="a" * 64,
            opponent="opponent.png", opponent_sha256="b" * 64, mode="fresh",
        ))
        trace = _Trace(
            service_failure="renderer unavailable",
            turns=[{"turn": 2, "action": "paint", "valid": False, "failure_kind": "service"}],
        )
        with patch.dict("os.environ", {"PAINTER_REQUIRE_VALID_FINAL_ACTION": "1"}):
            with self.assertRaisesRegex(RuntimeError, "not a visual score"):
                asyncio.run(task.final_canvas_aesthetic(trace))
        self.assertNotIn("final_action_validation", trace.info)


if __name__ == "__main__":
    unittest.main()
