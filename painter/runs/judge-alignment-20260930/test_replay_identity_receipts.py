import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("replay_identity_receipts.py")
spec = importlib.util.spec_from_file_location("identity_replay", MODULE_PATH)
assert spec and spec.loader
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def _response(ids, winner=None, preference=None, **extra):
    winner = winner or ids[0]
    preference = preference or "slight"
    value = {
        "assessments": {
            ident: {
                "brief_fidelity": "clear",
                "visible_evidence": "A visible connected painted form occupies the panel.",
                "weaknesses": ["Some edges are simplified."],
            }
            for ident in ids
        },
        "reason": "Visible shape separation supports this comparison.",
        "winner_id": winner,
        "preference": preference,
    }
    value.update(extra)
    if "_preference" in extra:
        value.pop("preference", None)
    return json.dumps(value)


class ReplayIdentityTests(unittest.TestCase):
    def setUp(self):
        self.ids = ("img_alpha", "img_beta")

    def test_lossless_extra_top_fields_and_preference_alias(self):
        normalized = replay.normalize_response(
            _response(self.ids, _preference="slight", _preference_note="provider prose", priority="low"),
            self.ids,
        )
        self.assertEqual(normalized["winner_id"], self.ids[0])
        self.assertEqual(normalized["preference"], "slight")
        self.assertIn("_preference_note", normalized["normalization"]["ignored_top_keys"])
        self.assertIn("priority", normalized["normalization"]["ignored_top_keys"])
        self.assertIn("alias:_preference->preference", normalized["normalization"]["transformations"])

    def test_whitespace_known_key_is_normalized_but_alias_collision_rejected(self):
        value = json.loads(_response(self.ids))
        value[" preference"] = value.pop("preference")
        normalized = replay.normalize_response(json.dumps(value), self.ids)
        self.assertEqual(normalized["preference"], "slight")
        self.assertIn("trimmed_key:' preference'->preference", normalized["normalization"]["transformations"])
        value = json.loads(_response(self.ids, _preference="clear"))
        value["preference"] = "slight"
        with self.assertRaises(ValueError):
            replay.normalize_response(json.dumps(value), self.ids)

    def test_unknown_image_id_and_scalar_weakness_are_not_repaired(self):
        value = json.loads(_response(self.ids))
        assessment = value["assessments"].pop(self.ids[0])
        value["assessments"][self.ids[0] + " "] = assessment
        with self.assertRaises(ValueError):
            replay.normalize_response(json.dumps(value), self.ids)
        value = json.loads(_response(self.ids))
        value["assessments"][self.ids[0]]["weaknesses"] = "one weakness"
        with self.assertRaises(ValueError):
            replay.normalize_response(json.dumps(value), self.ids)

    def test_first_semantically_valid_call_selected_and_raw_usage_preserved(self):
        invalid = json.loads(_response(self.ids))
        invalid["assessments"][self.ids[0]]["weaknesses"] = "scalar"
        valid_raw = _response(self.ids, winner=self.ids[1], preference="moderate")
        calls = [
            {"image_ids": {"A": self.ids[0], "B": self.ids[1]}, "raw_response": json.dumps(invalid), "usage": {"total_tokens": 11}},
            {"image_ids": {"A": self.ids[0], "B": self.ids[1]}, "raw_response": valid_raw, "usage": {"total_tokens": 22}},
        ]
        view = replay._replay_view({"calls": calls})
        self.assertEqual(view["selected_call_index"], 1)
        self.assertEqual(view["assessment"]["winner_id"], self.ids[1])
        self.assertEqual(view["calls"][0]["raw_call"]["usage"]["total_tokens"], 11)
        self.assertEqual(view["calls"][1]["usage"]["total_tokens"], 22)
        self.assertFalse(view["calls"][0]["valid"])
        self.assertTrue(view["calls"][1]["valid"])

    def test_disagreeing_initial_views_without_saved_fresh_are_pending(self):
        def view(a, b, winner, preference="slight"):
            return {"calls": [{"image_ids": {"A": a, "B": b}, "raw_response": _response((a, b), winner=winner, preference=preference)}]}

        judge = {
            "status": "uncertain",
            "image_ids": {"anchor": self.ids[0], "candidate": self.ids[1]},
            "views": [view(self.ids[0], self.ids[1], self.ids[0]), view(self.ids[1], self.ids[0], self.ids[1])],
            "fresh_view": None,
        }
        result = replay.replay_judge(judge)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["consensus"], "pending_fresh")
        self.assertTrue(result["replay_pending_fresh"])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["fresh_view"])

        judge["fresh_view"] = view(self.ids[0], self.ids[1], "tie", "tied")
        result = replay.replay_judge(judge)
        self.assertEqual(result["consensus"], "no_majority")
        self.assertFalse(result["replay_pending_fresh"])
        self.assertIsNone(result["score"])

    def test_offline_run_writes_new_receipts_and_leaves_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "receipts"
            source.mkdir()
            receipt = {
                "schema": "old",
                "status": "scored",
                "judge": {
                    "status": "scored",
                    "image_ids": {"anchor": self.ids[0], "candidate": self.ids[1]},
                    "views": [{"calls": [{"image_ids": {"A": self.ids[0], "B": self.ids[1]}, "raw_response": _response(self.ids)}]}, {"calls": [{"image_ids": {"A": self.ids[1], "B": self.ids[0]}, "raw_response": _response((self.ids[1], self.ids[0]), winner=self.ids[0])}]}],
                    "fresh_view": None,
                },
            }
            original = json.dumps(receipt, indent=2) + "\n"
            path = source / "pair.json"
            path.write_text(original, encoding="utf-8")
            replay.run(input_path=source, output=root / "normalized-replay")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            output = json.loads((root / "normalized-replay" / "receipts" / "pair.json").read_text(encoding="utf-8"))
            self.assertEqual(output["schema"], replay.SCHEMA)
            self.assertEqual(output["judge"]["status"], "replay_scored")


if __name__ == "__main__":
    unittest.main()
