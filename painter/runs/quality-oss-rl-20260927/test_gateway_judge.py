"""Schema drift may be normalized, but preference and defects must remain."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("gateway_judge", HERE / "gateway_judge.py")
assert spec and spec.loader
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class GatewayLabelTest(unittest.TestCase):
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
