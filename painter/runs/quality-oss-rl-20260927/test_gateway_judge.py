"""Schema drift may be normalized, but preference and defects must remain."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("gateway_judge", HERE / "gateway_judge.py")
assert spec and spec.loader
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class GatewayLabelTest(unittest.TestCase):
    def test_subject_first_prompt_is_forwarded_without_changing_default(self):
        self.assertIn("principal subjects", gateway.SUBJECT_FIRST_PROMPT)
        self.assertNotEqual(gateway.PROMPT, gateway.SUBJECT_FIRST_PROMPT)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference, a, b = (root / name for name in ("reference.png", "a.png", "b.png"))
            for path, payload in ((reference, b"ref"), (a, b"A"), (b, b"B")):
                path.write_bytes(payload)
            with patch.object(gateway, "judge_view", side_effect=[
                {"label": {"winner": "A"}}, {"label": {"winner": "B"}},
            ]) as judge_view:
                result = gateway.judge_pair(
                    key="unused", reference=reference, a=a, b=b,
                    prompt=gateway.SUBJECT_FIRST_PROMPT,
                )
            self.assertEqual(result["score_A"], 1.0)
            self.assertEqual(judge_view.call_args_list[0].kwargs["prompt"],
                             gateway.SUBJECT_FIRST_PROMPT)

    def test_singular_weakness_keys(self):
        raw = json.dumps({"winner": "A", "reason": "A preserves the bus",
                          "weakness_A": ["Slightly flat"],
                          "weakness_B": ["Wrong subject"]})
        self.assertEqual(gateway.parse_gateway_label(raw)["weaknesses_B"], ["Wrong subject"])

    def test_no_winner_is_rejected(self):
        raw = json.dumps({"reason": "Both are nice", "weakness_A": [], "weakness_B": []})
        with self.assertRaises(ValueError):
            gateway.parse_gateway_label(raw)

    def test_conflicting_aliases_are_rejected(self):
        raw = json.dumps({"winner": "A", "reason": "A is clearer",
                          "weaknesses_A": [], "weakness_A": ["Different"],
                          "weaknesses_B": []})
        with self.assertRaises(ValueError):
            gateway.parse_gateway_label(raw)
