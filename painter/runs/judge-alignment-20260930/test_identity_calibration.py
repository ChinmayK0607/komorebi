import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from painter import text_pairwise_identity_judge as judge


RUNNER_PATH = Path(__file__).with_name("run_identity_calibration.py")
spec = importlib.util.spec_from_file_location("identity_runner_tests", RUNNER_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _png(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 6), color).save(path, format="PNG")
    return path


def _assessment(ids: tuple[str, str], winner: str = "tie", preference: str = "tied") -> dict:
    return {
        "assessments": {
            ident: {
                "brief_fidelity": "clear",
                "visible_evidence": "Visible connected shapes occupy the painted panel.",
                "weaknesses": ["Some edges are simplified."],
            }
            for ident in ids
        },
        "reason": "The visible shapes and color separation provide the comparison.",
        "winner_id": winner,
        "preference": preference,
    }


class IdentityJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.brief = root / "brief.txt"
        self.brief.write_text("Paint a readable subject.", encoding="utf-8")
        self.left = _png(root / "left.png", (230, 80, 60))
        self.right = _png(root / "right.png", (60, 90, 220))
        self.key = root / "key"
        self.key.write_text("test-key", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_is_stable_and_reverse_view_is_balanced(self):
        left_id, right_id = judge.image_id(self.left), judge.image_id(self.right)
        self.assertEqual(left_id, judge.image_id(self.left))
        seen = []

        async def fake(**kwargs):
            seen.append((kwargs["a_id"], kwargs["b_id"]))
            ids = (kwargs["a_id"], kwargs["b_id"])
            return {"assessment": _assessment(ids), "calls": []}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_parse_retry", side_effect=fake):
            result = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
        self.assertEqual(set(seen), {(left_id, right_id), (right_id, left_id)})
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["consensus"], "initial_agreement")

    def test_fixed_image_winner_reverses_oriented_score_but_not_request_order(self):
        left_id, right_id = judge.image_id(self.left), judge.image_id(self.right)
        seen = []

        async def fake(**kwargs):
            seen.append((kwargs["a_id"], kwargs["b_id"]))
            return {"assessment": _assessment((kwargs["a_id"], kwargs["b_id"]), left_id, "clear"), "calls": []}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_parse_retry", side_effect=fake):
            first = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
            second = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.right, candidate=self.left, key_file=self.key))
        self.assertEqual(seen[:2], seen[2:])
        self.assertAlmostEqual(first["score"] + second["score"], 1.0)

    def test_no_majority_is_uncertain_and_has_no_numeric_score(self):
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            calls += 1
            ids = (kwargs["a_id"], kwargs["b_id"])
            if calls == 1:
                winner = ids[0]
                return {"assessment": _assessment(ids, winner, "clear"), "calls": []}
            if calls == 2:
                winner = ids[0]
                return {"assessment": _assessment(ids, winner, "clear"), "calls": []}
            return {"assessment": _assessment(ids, "tie", "tied"), "calls": []}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_parse_retry", side_effect=fake):
            result = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["score"])
        self.assertEqual(result["consensus"], "no_majority")

    def test_parser_rejects_unknown_ids_and_winner_preference_mismatch(self):
        ids = ("img_a", "img_b")
        value = _assessment(ids, ids[0], "slight")
        value["assessments"]["img_extra"] = value["assessments"].pop("img_a")
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), ids)
        value = _assessment(ids, "tie", "tied")
        value["preference"] = "clear"
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), ids)

    def test_identical_bytes_are_real_tie_without_calls(self):
        twin = Path(self.tmp.name) / "twin.png"
        twin.write_bytes(self.left.read_bytes())
        with mock.patch.object(judge, "load_key", side_effect=AssertionError("no call expected")):
            result = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=twin, key_file=self.key))
        self.assertEqual(result["status"], "identical_canvases")
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["views"], [])

    def test_conflict_tie_win_tie_is_two_of_three_tie(self):
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            calls += 1
            ids = (kwargs["a_id"], kwargs["b_id"])
            if calls == 1:
                return {"assessment": _assessment(ids, "tie", "tied"), "calls": []}
            if calls == 2:
                return {"assessment": _assessment(ids, ids[0], "clear"), "calls": []}
            return {"assessment": _assessment(ids, "tie", "tied"), "calls": []}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_parse_retry", side_effect=fake):
            result = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
        self.assertEqual(result["status"], "conflict_resolved")
        self.assertEqual(result["winner_id"], "tie")
        self.assertEqual(result["score"], 0.5)
        self.assertTrue(result["conflict"])

    def test_failed_fresh_view_is_censored_without_vote_type_error(self):
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            calls += 1
            ids = (kwargs["a_id"], kwargs["b_id"])
            if calls == 1:
                return {"assessment": _assessment(ids, ids[0], "clear"), "calls": []}
            if calls == 2:
                return {"assessment": _assessment(ids, ids[0], "clear"), "calls": []}
            return {"assessment": None, "calls": [{"parse_error": "truncated"}], "error": "incomplete_evidence"}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_parse_retry", side_effect=fake):
            result = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["score"])
        self.assertEqual(result["fresh_view"]["error"], "incomplete_evidence")

    def test_malformed_provider_payload_keeps_usage_and_is_not_scored(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"usage": {"total_tokens": 17}}).encode()

        with mock.patch.object(judge.urllib.request, "urlopen", return_value=Response()):
            call = judge._view(key="key", brief="Paint.", a=self.left, b=self.right,
                               a_id=judge.image_id(self.left), b_id=judge.image_id(self.right))
        self.assertEqual(call["parse_error"], "malformed_provider_response")
        self.assertEqual(call["usage"]["total_tokens"], 17)
        self.assertNotIn("assessment", call)

    def test_runner_revised_maps_tie_and_censored_without_fake_score(self):
        anchor_id, candidate_id = judge.image_id(self.left), judge.image_id(self.right)
        pair = {"anchor_role": "anchor", "candidate_role": "candidate"}
        receipt = {"status": "scored", "judge": {
            "status": "scored", "winner_id": "tie", "preference": "tied",
            "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
            "views": [{"assessment": {"winner_id": "tie"}}], "fresh_view": None,
        }}
        row = runner._revised(receipt, pair)
        self.assertEqual(row["choices"], ["tie"])
        censored = runner._revised({"status": "censored", "reason": "judge_failure"}, pair)
        self.assertEqual(censored["status"], "censored")
        self.assertEqual(censored["headline"], "Could not judge")

    def test_censored_fresh_failure_keeps_initial_conflict_denominator(self):
        anchor_id, candidate_id = judge.image_id(self.left), judge.image_id(self.right)
        pair = {"anchor_role": "anchor", "candidate_role": "candidate"}
        uncertain = {
            "status": "uncertain", "consensus": "incomplete_evidence", "conflict": True,
            "image_ids": {"anchor": anchor_id, "candidate": candidate_id},
            "views": [
                {"assessment": {"winner_id": anchor_id}, "calls": []},
                {"assessment": {"winner_id": candidate_id}, "calls": []},
            ],
            "fresh_view": {"assessment": None, "calls": []},
        }
        row = runner._revised({"status": "censored", "reason": "incomplete_evidence", "judge": uncertain}, pair)
        self.assertEqual(row["choices"], ["LEFT", "RIGHT"])
        metrics = runner._metrics([row])
        self.assertEqual(metrics["initial_valid"], 1)
        self.assertEqual(metrics["initial_conflict"], 1)
        self.assertEqual(metrics["remaining_conflict"], 1)


if __name__ == "__main__":
    unittest.main()
