from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("oss_judge", HERE / "oss_judge.py")
assert spec and spec.loader
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class JudgeProtocolTest(unittest.TestCase):
    def test_closed_response_and_reversal(self) -> None:
        good = {"winner": "A", "reason": "The subject reads clearly",
                "weaknesses_A": ["Flat shadow"], "weaknesses_B": ["Missing subject"]}
        self.assertEqual(judge.parse_label(json.dumps(good)), good)
        with self.assertRaises(ValueError):
            judge.parse_label(json.dumps({**good, "reward": 1}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, a, b = (root / name for name in ("ref.png", "a.png", "b.png"))
            for path, raw in ((reference, b"ref"), (a, b"better"), (b, b"worse")):
                path.write_bytes(raw)
            with patch.object(judge, "judge_view", side_effect=[
                {"label": good},
                {"label": {**good, "winner": "B"}},
            ]) as call:
                result = judge.judge_pair(endpoint="http://127.0.0.1:8200", model="m",
                                          reference=reference, a=a, b=b)
            self.assertEqual(result["score_A"], 1)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(call.call_args_list[1].kwargs["a"], b)

    def test_identical_candidates_skip_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            path.write_bytes(b"same")
            with patch.object(judge, "judge_view") as call:
                result = judge.judge_pair(endpoint="http://127.0.0.1:8200", model="m",
                                          reference=path, a=path, b=path)
            self.assertEqual(result["score_A"], 0.5)
            call.assert_not_called()

    def test_uncertain_is_not_a_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.png", root / "b.png"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            with patch.object(judge, "judge_view", side_effect=[
                {"label": {"winner": "uncertain"}},
                {"label": {"winner": "B"}},
            ]):
                result = judge.judge_pair(endpoint="http://127.0.0.1:8200", model="m",
                                          reference=a, a=a, b=b)
            self.assertIsNone(result["score_A"])


if __name__ == "__main__":
    unittest.main()
