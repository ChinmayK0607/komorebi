import json
import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from painter import text_pairwise_consistent_side_judge as judge
from PIL import Image


def _assessment():
    return {
        "brief_fidelity": "clear",
        "visible_evidence": "The subject has connected visible forms.",
        "weaknesses": ["Some edges are simplified."],
    }


class SideJudgeTests(unittest.TestCase):
    def setUp(self):
        self.ids = ("img_left_content", "img_right_content")

    def _raw(self, winner="LEFT", preference="moderate"):
        return json.dumps({
            "left_assessment": _assessment(),
            "right_assessment": _assessment(),
            "reason": "The left silhouette is more coherent.",
            "winner": winner,
            "preference": preference,
        })

    def test_side_winner_maps_to_opaque_identity_for_both_request_orders(self):
        first = judge._parse(self._raw("LEFT"), self.ids)
        second = judge._parse(self._raw("LEFT"), self.ids[::-1])
        self.assertEqual(first["winner_id"], self.ids[0])
        self.assertEqual(second["winner_id"], self.ids[1])
        self.assertEqual(first["assessments"].keys(), set(self.ids))

    def test_response_schema_has_fixed_side_keys_and_no_dynamic_ids(self):
        schema = judge._response_schema()
        self.assertEqual(
            set(schema["properties"]),
            {"left_assessment", "right_assessment", "reason", "winner", "preference"},
        )
        self.assertEqual(schema["properties"]["winner"]["enum"], ["LEFT", "RIGHT", "TIE"])
        self.assertTrue(schema["additionalProperties"] is False)

    def test_missing_fields_conflicts_and_non_json_are_censored(self):
        value = json.loads(self._raw())
        value.pop("right_assessment")
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), self.ids)
        value = json.loads(self._raw())
        value["winner"] = "TIE"
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), self.ids)
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            judge._parse("not-json", self.ids)

    def test_nonempty_weaknesses_and_unknown_fields_remain_strict(self):
        value = json.loads(self._raw())
        value["left_assessment"]["weaknesses"] = []
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), self.ids)

    def test_view_renders_side_sheet_maps_winner_and_keeps_raw_response(self):
        class Response:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.body

        raw = self._raw("LEFT", "moderate")
        body = json.dumps({
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 11},
        }).encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "red").save(left)
            Image.new("RGB", (8, 12), "blue").save(right)
            captured = {}

            def open_url(request, timeout):
                captured["payload"] = json.loads(request.data)
                captured["timeout"] = timeout
                return Response(body)

            with mock.patch("urllib.request.urlopen", side_effect=open_url):
                call = judge._view(
                    key="test-key", brief="Paint a readable subject.", a=left, b=right,
                    a_id=self.ids[0], b_id=self.ids[1],
                )
            payload = captured["payload"]
            schema = payload["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["winner"]["enum"], ["LEFT", "RIGHT", "TIE"])
            self.assertNotIn(self.ids[0], json.dumps(schema))
            self.assertNotIn(self.ids[1], json.dumps(schema))
            image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
            image = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
            self.assertEqual(image.size, (1248, 696))
            self.assertEqual(call["assessment"]["winner_id"], self.ids[0])
            self.assertEqual(call["raw_response"], raw)
            self.assertEqual(call["usage"]["total_tokens"], 11)
        value = json.loads(self._raw())
        value["extra"] = "ignored?"
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(value), self.ids)


if __name__ == "__main__":
    unittest.main()
