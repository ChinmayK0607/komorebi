import asyncio
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image

from painter import text_pairwise_consistent_judge as judge


def _assessment(ids, winner, preference="slight"):
    return {
        "assessments": {
            ident: {
                "brief_fidelity": "clear",
                "visible_evidence": "Connected visible painted forms occupy the panel.",
                "weaknesses": ["Some edges are simplified."],
            }
            for ident in ids
        },
        "reason": "Visible shape separation is the decisive difference.",
        "winner_id": winner,
        "preference": preference,
    }


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class ConsistentJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.brief = root / "brief.txt"
        self.brief.write_text("Paint a readable subject.", encoding="utf-8")
        self.left = root / "left.png"
        self.right = root / "right.png"
        Image.new("RGB", (8, 6), (220, 40, 40)).save(self.left)
        Image.new("RGB", (8, 6), (40, 60, 220)).save(self.right)
        self.key = root / "key"
        self.key.write_text("test-key", encoding="utf-8")
        self.left_id, self.right_id = judge.image_id(self.left), judge.image_id(self.right)

    def tearDown(self):
        self.tmp.cleanup()

    def test_payload_uses_strict_dynamic_schema_and_ids(self):
        ids = (self.left_id, self.right_id)
        raw = json.dumps(_assessment(ids, ids[0]))
        body = json.dumps({"choices": [{"message": {"content": raw}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}}).encode()
        captured = {}

        def open_url(request, timeout):
            captured["payload"] = json.loads(request.data)
            return _Response(body)

        with mock.patch.object(judge.urllib.request, "urlopen", side_effect=open_url):
            call = judge._view(key="key", brief="Paint.", a=self.left, b=self.right, a_id=ids[0], b_id=ids[1])
        response_format = captured["payload"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "painting_comparison")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["winner_id"]["enum"], [*ids, "tie"])
        assessments = schema["properties"]["assessments"]
        self.assertEqual(assessments["required"], list(ids))
        self.assertFalse(assessments["additionalProperties"])
        self.assertFalse(assessments["properties"][ids[0]]["additionalProperties"])
        self.assertTrue(call.get("assessment"))
        self.assertEqual(call["usage"]["total_tokens"], 9)

    def test_role_flip_keeps_canonical_requests_and_reverses_oriented_score(self):
        seen = []

        async def fake(**kwargs):
            seen.append((kwargs["a_id"], kwargs["b_id"]))
            ids = (kwargs["a_id"], kwargs["b_id"])
            return {"assessment": _assessment(ids, self.left_id, "clear"), "calls": []}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_retry", side_effect=fake):
            first = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))
            second = asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.right, candidate=self.left, key_file=self.key))
        self.assertEqual(seen[:2], seen[2:])
        self.assertAlmostEqual(first["score"] + second["score"], 1.0)

    def _run_sequence(self, winners):
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            winner = winners[calls]
            calls += 1
            ids = (kwargs["a_id"], kwargs["b_id"])
            if winner is None:
                return {"assessment": None, "calls": [{"provider_error": "HTTPError", "http_status": 400}]}
            return {"assessment": _assessment(ids, winner, "moderate"), "calls": [{"usage": {"total_tokens": 3}, "elapsed_seconds": 0.2}]}

        with mock.patch.object(judge, "load_key", return_value="key"), mock.patch.object(judge, "_call_with_retry", side_effect=fake):
            return asyncio.run(judge.judge_pair(brief=self.brief, anchor=self.left, candidate=self.right, key_file=self.key))

    def test_balanced_2_to_2_is_uncertain_but_3_to_1_scores(self):
        # canonical left wins calls 1,3; canonical right wins calls 2,4
        tie = self._run_sequence([self.left_id, self.right_id, self.left_id, self.right_id])
        self.assertEqual(tie["consensus"], "no_three_of_four")
        self.assertIsNone(tie["score"])
        decisive = self._run_sequence([self.left_id, self.right_id, self.left_id, self.left_id])
        self.assertEqual(decisive["consensus"], "three_of_four")
        self.assertIsNotNone(decisive["score"])
        self.assertEqual(len(decisive["recheck_views"]), 2)

    def test_failed_recheck_preserves_initial_and_recheck_receipts(self):
        result = self._run_sequence([self.left_id, self.right_id, None, self.left_id])
        self.assertEqual(result["consensus"], "failed_recheck_call")
        self.assertIsNone(result["score"])
        self.assertEqual(len(result["views"]), 2)
        self.assertEqual(len(result["recheck_views"]), 2)
        self.assertEqual(result["recheck_views"][0]["calls"][0]["http_status"], 400)

    def test_http_400_is_censored_without_error_body(self):
        error = urllib.error.HTTPError("https://gateway.invalid", 400, "Bad Request", {}, io.BytesIO(b"secret response body"))
        with mock.patch.object(judge.urllib.request, "urlopen", side_effect=error):
            call = judge._view(key="key", brief="Paint.", a=self.left, b=self.right, a_id=self.left_id, b_id=self.right_id)
        self.assertEqual(call["http_status"], 400)
        self.assertEqual(call["provider_error"], "HTTPError")
        self.assertEqual(call["raw_response"], "")
        self.assertNotIn("secret", json.dumps(call))

    def test_strict_parser_rejects_extra_fields_and_invalid_winner(self):
        ids = (self.left_id, self.right_id)
        value = _assessment(ids, ids[0])
        value["unexpected"] = 1
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), ids)
        value = _assessment(ids, "img_other")
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), ids)


if __name__ == "__main__":
    unittest.main()
