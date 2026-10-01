"""Pure tests for the optional terminal-credit contract."""

import unittest

from painter.final_action_credit import (
    REQUIRE_VALID_FINAL_ACTION_ENV,
    require_valid_final_action_enabled,
    validate_last_turn,
)


class FinalActionCreditTests(unittest.TestCase):
    def test_contract_is_default_off_and_requires_literal_one(self):
        self.assertFalse(require_valid_final_action_enabled({}))
        self.assertFalse(require_valid_final_action_enabled({REQUIRE_VALID_FINAL_ACTION_ENV: "true"}))
        self.assertTrue(require_valid_final_action_enabled({REQUIRE_VALID_FINAL_ACTION_ENV: "1"}))

    def test_valid_and_invalid_last_turns_are_summarized(self):
        valid = validate_last_turn({"turns": [{"turn": 4, "action": "finish", "valid": True}]})
        self.assertEqual(valid["status"], "valid")
        invalid = validate_last_turn({"turns": [{"turn": 4, "action": "paint", "valid": False,
                                                  "failure_kind": "program"}]})
        self.assertEqual(invalid["status"], "invalid")
        self.assertEqual(invalid["failure_kind"], "program")

    def test_missing_or_malformed_evidence_is_operational(self):
        cases = [{}, {"turns": []}, {"turns": [None]}, {"turns": [{"valid": 0}]}]
        for info in cases:
            with self.subTest(info=info), self.assertRaisesRegex(RuntimeError, "final-action evidence"):
                validate_last_turn(info)


if __name__ == "__main__":
    unittest.main()
