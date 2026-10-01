from __future__ import annotations

import asyncio
import base64
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_pair_repeat_runner", HERE / "calibrate_frozen_pair_repeats.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from painter import text_pairwise_consistent_side_judge_v2 as draft


def _result(module, pair: dict, *, winner: str, preference: str = "slight", status: str = "scored") -> dict:
    anchor_id = pair["image_ids"]["anchor"]
    candidate_id = pair["image_ids"]["candidate"]
    return {
        "protocol": module.PROTOCOL,
        "format_version": module.FORMAT_VERSION,
        "status": status,
        "winner_id": winner if status == "scored" else None,
        "preference": preference if status == "scored" else None,
        "score": 2 / 3 if status == "scored" else None,
        "consensus": "initial_agreement" if status == "scored" else "no_three_of_four",
        "brief_sha256": pair["brief_sha256"],
        "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
        "image_sha256": pair["image_sha256"],
        "views": [{"assessment": {"winner_id": winner, "preference": preference}, "calls": []},
                  {"assessment": {"winner_id": winner, "preference": preference}, "calls": []}]
                  if status == "scored" else [],
        "recheck_views": [],
        "initial_votes": [winner, winner] if status == "scored" else [],
        "recheck_votes": [],
    }


class DraftPromptTests(unittest.TestCase):
    def test_draft_is_versioned_general_and_strict(self):
        self.assertNotEqual(draft.PROTOCOL, runner.old_judge.PROTOCOL)
        self.assertNotEqual(draft.FORMAT_VERSION, runner.old_judge.FORMAT_VERSION)
        prompt = draft.PROMPT.lower()
        for phrase in ("whole connected form", "negative space", "spatial relationships", "precision", "muddy"):
            self.assertIn(phrase, prompt)
        self.assertNotIn("horseshoe", prompt)
        self.assertNotIn("feather", prompt)
        schema = draft._response_schema()
        self.assertTrue(schema["additionalProperties"] is False)
        self.assertEqual(schema["properties"]["winner"]["enum"], ["LEFT", "RIGHT", "TIE"])

    def test_parser_preserves_swapped_side_to_id_mapping_and_rejects_extra_field(self):
        assessment = {
            "brief_fidelity": "clear",
            "visible_evidence": "The connected subject and its opening are visible.",
            "weaknesses": ["Some edges are soft."],
        }
        raw = json.dumps({
            "left_assessment": assessment, "right_assessment": assessment,
            "reason": "The LEFT subject is more coherent.",
            "winner": "LEFT", "preference": "slight",
        })
        first = draft._parse(raw, ("img_a", "img_b"))
        second = draft._parse(raw, ("img_b", "img_a"))
        self.assertEqual(first["winner_id"], "img_a")
        self.assertEqual(second["winner_id"], "img_b")
        value = json.loads(raw)
        value["extra"] = "reject"
        with self.assertRaises(ValueError):
            draft._parse(json.dumps(value), ("img_a", "img_b"))

    def test_view_uses_real_sheet_payload_and_preserves_swapped_ids(self):
        assessment = {
            "brief_fidelity": "clear",
            "visible_evidence": "The connected subject and its opening are visible.",
            "weaknesses": ["Some edges are soft."],
        }
        raw = json.dumps({
            "left_assessment": assessment, "right_assessment": assessment,
            "reason": "The LEFT panel has the stronger connected form.",
            "winner": "LEFT", "preference": "slight",
        })
        body = json.dumps({
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 19},
        }).encode()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return body

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "#d35f4f").save(left)
            Image.new("RGB", (8, 12), "#4f75d3").save(right)
            ids = (draft.image_id(left), draft.image_id(right))
            requests = []

            def open_url(request, timeout):
                requests.append((json.loads(request.data), timeout))
                return Response()

            with mock.patch("urllib.request.urlopen", side_effect=open_url):
                first = draft._view(key="external", brief="Paint a connected subject.",
                                    a=left, b=right, a_id=ids[0], b_id=ids[1])
                second = draft._view(key="external", brief="Paint a connected subject.",
                                     a=right, b=left, a_id=ids[1], b_id=ids[0])
            self.assertEqual(len(requests), 2)
            self.assertEqual(first["assessment"]["winner_id"], ids[0])
            self.assertEqual(second["assessment"]["winner_id"], ids[1])
            self.assertEqual(first["usage"]["total_tokens"], 19)
            self.assertEqual(first["image_ids"], {"A": ids[0], "B": ids[1]})
            self.assertEqual(second["image_ids"], {"A": ids[1], "B": ids[0]})
            self.assertEqual(first["image_sha256"], {"A": draft.digest(left), "B": draft.digest(right)})
            self.assertEqual(second["image_sha256"], {"A": draft.digest(right), "B": draft.digest(left)})
            self.assertNotEqual(first["sheet_sha256"], second["sheet_sha256"])
            for payload, timeout in requests:
                self.assertEqual(timeout, 240)
                self.assertEqual(payload["messages"][0]["content"], draft.PROMPT)
                self.assertEqual(payload["response_format"], draft._response_format())
                image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
                sheet = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
                self.assertEqual(sheet.size, (1248, 696))

    def test_draft_censors_incomplete_conflict_recheck(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            anchor = root / "anchor.png"
            candidate = root / "candidate.png"
            key = root / "gateway.env"
            brief.write_text("Paint a connected subject.", encoding="utf-8")
            anchor.write_bytes(b"anchor")
            candidate.write_bytes(b"candidate")
            key.write_text("not used", encoding="utf-8")
            ids = (draft.image_id(anchor), draft.image_id(candidate))
            calls = 0

            async def fake(**_kwargs):
                nonlocal calls
                calls += 1
                if calls <= 2:
                    winner = ids[0] if calls == 1 else ids[1]
                    return {"assessment": {"winner_id": winner, "preference": "slight"}, "calls": []}
                return {"assessment": None, "calls": [], "error": "incomplete_evidence"}

            with mock.patch.object(draft, "load_key", return_value="external"), mock.patch.object(
                draft, "_call_with_retry", side_effect=fake
            ):
                result = asyncio.run(draft.judge_pair(
                    brief=brief, anchor=anchor, candidate=candidate, key_file=key,
                ))
            self.assertEqual(result["status"], "uncertain")
            self.assertIsNone(result["score"])
            self.assertEqual(result["acceptance"], "censored")


class FrozenRepeatRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "anchor.png").write_bytes(b"anchor image bytes")
        (self.root / "candidate.png").write_bytes(b"candidate image bytes")
        pairs = {
            "selection_note": "Six calibration and nine untouched confirmation pairs were selected before calls.",
            "pairs": [
                {"pair_id": "cal-1", "split": "calibration", "brief": "Paint a connected subject.",
                 "anchor_path": "anchor.png", "candidate_path": "candidate.png"},
                {"pair_id": "confirm-1", "split": "confirmation", "brief": "Paint an untouched subject.",
                 "anchor_path": "anchor.png", "candidate_path": "candidate.png"},
            ],
        }
        self.pairs_file = self.root / "pairs.json"
        self.pairs_file.write_text(json.dumps(pairs), encoding="utf-8")
        self.output = self.root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_freezes_hashes_and_split_without_calls(self):
        manifest = runner.prepare_manifest(pairs_file=self.pairs_file, output=self.output, repeat_count=2)
        loaded = runner.read_manifest(self.output / "input-manifest.json")
        self.assertEqual(loaded, manifest)
        self.assertEqual(manifest["repeat_count"], 2)
        self.assertEqual({row["split"] for row in manifest["pairs"]}, {"calibration", "confirmation"})
        self.assertEqual(manifest["pairs"][0]["image_sha256"]["anchor"], runner.sha256_file(self.root / "anchor.png"))
        self.assertEqual(manifest["pairs"][0]["image_ids"]["anchor"], runner.old_judge.image_id(self.root / "anchor.png"))

    def test_run_measures_total_repeat_consistency_and_censors_unresolved(self):
        manifest = runner.prepare_manifest(pairs_file=self.pairs_file, output=self.output, repeat_count=2)
        calls = {"old": 0, "draft": 0}

        async def fake_old(**kwargs):
            calls["old"] += 1
            pair = next(row for row in manifest["pairs"] if row["pair_id"] == ("cal-1" if "connected" in kwargs["brief"].read_text() else "confirm-1"))
            return _result(runner.old_judge, pair, winner=pair["image_ids"]["anchor"], preference="slight")

        async def fake_draft(**kwargs):
            calls["draft"] += 1
            brief = kwargs["brief"].read_text()
            pair = next(row for row in manifest["pairs"] if row["pair_id"] == ("cal-1" if "connected" in brief else "confirm-1"))
            if pair["pair_id"] == "confirm-1":
                return _result(runner.draft_judge, pair, winner=pair["image_ids"]["anchor"], status="uncertain")
            return _result(runner.draft_judge, pair, winner=pair["image_ids"]["anchor"], preference="moderate")

        with mock.patch.object(runner.old_judge, "judge_pair", side_effect=fake_old), mock.patch.object(
            runner.draft_judge, "judge_pair", side_effect=fake_draft
        ):
            summary = asyncio.run(runner.run(
                manifest_path=self.output / "input-manifest.json", output=self.output,
                key_file=self.root / "externally-supplied-key", concurrency=2,
            ))
            cached_summary = asyncio.run(runner.run(
                manifest_path=self.output / "input-manifest.json", output=self.output,
                key_file=self.root / "externally-supplied-key", concurrency=2,
            ))
        self.assertEqual(calls, {"old": 4, "draft": 4})
        self.assertEqual(cached_summary, summary)
        old_total = summary["total_repeat_consistency"]["old"]
        self.assertEqual(old_total["all_repeats_scored_pairs"], 2)
        self.assertEqual(old_total["winner_consistent_pairs"], 2)
        self.assertEqual(old_total["exact_decision_consistent_pairs"], 2)
        draft_total = summary["total_repeat_consistency"]["draft"]
        self.assertEqual(draft_total["all_repeats_scored_pairs"], 1)
        self.assertEqual(draft_total["unresolved_or_censored_pairs"], 1)
        self.assertEqual(summary["metrics"]["draft"]["confirmation"]["censored_count"], 2)
        self.assertEqual(summary["paired_old_vs_draft"]["calibration"]["exact_decision_agreement"], 0)
        self.assertEqual(summary["paired_old_vs_draft"]["calibration"]["winner_agreement"], 2)

    def test_changed_frozen_image_is_rejected_before_calls(self):
        runner.prepare_manifest(pairs_file=self.pairs_file, output=self.output, repeat_count=1)
        (self.root / "candidate.png").write_bytes(b"mutated candidate")
        with self.assertRaisesRegex(ValueError, "image identity changed"):
            asyncio.run(runner.run(
                manifest_path=self.output / "input-manifest.json", output=self.output,
                key_file=self.root / "externally-supplied-key", concurrency=1,
            ))

    def test_local_attribute_error_is_separate_from_provider_censorship(self):
        self.assertEqual(runner._exception_status(AttributeError("sheet helper")), "implementation_error")
        self.assertEqual(runner._exception_status(RuntimeError("gateway")), "censored")
        metrics = runner._metrics([{
            "status": "implementation_error", "judge": {},
            "identity": {"method": "old", "pair_id": "x", "repeat": 1},
        }])
        self.assertEqual(metrics["implementation_error_count"], 1)
        self.assertEqual(metrics["censored_count"], 0)


if __name__ == "__main__":
    unittest.main()
