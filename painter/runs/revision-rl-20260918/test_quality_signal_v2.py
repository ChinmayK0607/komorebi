import unittest

from quality_signal_v2 import PROTOCOL_VERSION, reconcile_v2, reward_v2


def label(winner='A', confidence='high', acceptable_A='yes', acceptable_B='no', reason='raw'):
    return dict(winner=winner, confidence=confidence, acceptable_A=acceptable_A,
                acceptable_B=acceptable_B, reason=reason)


def result(action='paint', valid=True, failure_kind=None, no_change=False):
    return dict(action=action, valid=valid, failure_kind=failure_kind,
                no_change=no_change)


class QualitySignalV2Test(unittest.TestCase):
    def test_relative_agreement_survives_absolute_conflict_for_paint(self):
        judgment = reconcile_v2(label(), label(winner='B', acceptable_A='no', acceptable_B='no'))
        self.assertEqual(judgment['delta'], 1.0)
        self.assertIsNone(judgment['candidate_acceptable'])
        self.assertEqual(judgment['anchor_acceptable'], False)
        self.assertEqual(reward_v2(result('paint'), judgment), 1.0)
        self.assertIsNone(reward_v2(result('finish'), judgment))

    def test_absolute_agreement_survives_relative_conflict_for_finish(self):
        judgment = reconcile_v2(label(), label(winner='A', acceptable_A='no', acceptable_B='yes'))
        self.assertIsNone(judgment['delta'])
        self.assertTrue(judgment['candidate_acceptable'])
        self.assertFalse(judgment['anchor_acceptable'])
        self.assertIsNone(reward_v2(result('paint'), judgment))
        self.assertEqual(reward_v2(result('finish'), judgment), 1.0)

    def test_low_confidence_masks_both_signal_families(self):
        judgment = reconcile_v2(label(confidence='low'), label(winner='B', confidence='medium'))
        self.assertEqual(judgment['protocol_version'], PROTOCOL_VERSION)
        self.assertIsNone(judgment['delta'])
        self.assertIsNone(judgment['candidate_acceptable'])
        self.assertIsNone(reward_v2(result('paint'), judgment))
        self.assertIsNone(reward_v2(result('finish'), judgment))

    def test_tie_zero_and_no_change_are_valid_relative_scores(self):
        judgment = reconcile_v2(label(winner='tie', acceptable_A='no', acceptable_B='no'),
                                label(winner='tie', acceptable_A='no', acceptable_B='no'))
        self.assertEqual(judgment['delta'], 0.0)
        self.assertEqual(reward_v2(result('paint', no_change=True), judgment), 0.0)
        self.assertEqual(reward_v2(result('repair'), judgment), 0.0)
        self.assertEqual(reward_v2(result('finish'), judgment), -1.5)

    def test_failures_and_invalid_program_match_v1_boundaries(self):
        judgment = reconcile_v2(label(), label(winner='B'))
        self.assertIsNone(reward_v2(result('paint', failure_kind='infrastructure'), judgment))
        self.assertIsNone(reward_v2(result('paint', failure_kind='cutoff'), judgment))
        self.assertEqual(reward_v2(result('paint', valid=False, failure_kind='program'), judgment), -2.0)

    def test_malformed_result_labels_are_masked_without_type_errors(self):
        judgment = reconcile_v2(label(), label(winner='B'))
        malformed_failure = result('paint', failure_kind=['infrastructure'])
        malformed_action = result(action={'paint': True})
        self.assertIsNone(reward_v2(malformed_failure, judgment))
        self.assertIsNone(reward_v2(malformed_action, judgment))

    def test_malformed_or_missing_judgments_do_not_score(self):
        self.assertIsNone(reward_v2(result('paint'), None))
        self.assertIsNone(reward_v2(result('finish'), {'delta': 1.0,
                                                        'candidate_acceptable': True,
                                                        'anchor_acceptable': False}))
        malformed = reconcile_v2({'winner': 'A'}, label(winner='B'))
        self.assertIsNone(reward_v2(result('paint'), malformed))
        self.assertIsNone(reward_v2(result('finish'), malformed))

    def test_normalized_schema_rejects_missing_and_boolean_delta_fields(self):
        valid = reconcile_v2(label(), label(winner='B'))
        for missing in ('delta', 'candidate_acceptable', 'anchor_acceptable'):
            incomplete = dict(valid)
            del incomplete[missing]
            self.assertIsNone(reward_v2(result('paint'), incomplete))
            self.assertIsNone(reward_v2(result('finish'), incomplete))
        boolean_delta = dict(valid, delta=True)
        self.assertIsNone(reward_v2(result('paint'), boolean_delta))

    def test_malformed_label_values_leave_independent_absolute_channel_safe(self):
        first = label(acceptable_A='yes', acceptable_B='no')
        reverse = label(winner='B', acceptable_A='no', acceptable_B='yes')
        for field, value in (
                ('confidence', ['high']), ('winner', {'A': 1}),
                ('acceptable_A', ['yes']), ('acceptable_B', {'no': 1})):
            malformed = dict(first)
            malformed[field] = value
            judgment = reconcile_v2(malformed, reverse)
            if field == 'confidence':
                self.assertIsNone(judgment['delta'])
                self.assertIsNone(judgment['candidate_acceptable'])
                self.assertIsNone(judgment['anchor_acceptable'])
            elif field == 'winner':
                self.assertIsNone(judgment['delta'])
                self.assertTrue(judgment['candidate_acceptable'])
                self.assertFalse(judgment['anchor_acceptable'])
            elif field == 'acceptable_A':
                self.assertEqual(judgment['delta'], 1.0)
                self.assertIsNone(judgment['candidate_acceptable'])
                self.assertFalse(judgment['anchor_acceptable'])
            else:
                self.assertEqual(judgment['delta'], 1.0)
                self.assertTrue(judgment['candidate_acceptable'])
                self.assertIsNone(judgment['anchor_acceptable'])

        malformed_relative = dict(first, winner=['A'])
        judgment = reconcile_v2(malformed_relative, reverse)
        self.assertIsNone(judgment['delta'])
        self.assertTrue(judgment['candidate_acceptable'])
        self.assertFalse(judgment['anchor_acceptable'])

    def test_reasons_are_not_copied_into_normalized_record(self):
        judgment = reconcile_v2(label(reason='first secret'),
                                label(winner='B', reason='reverse secret'))
        self.assertNotIn('reason', judgment)


if __name__ == '__main__':
    unittest.main()
