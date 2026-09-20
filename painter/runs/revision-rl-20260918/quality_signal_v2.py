"""Version 2 quality rewards with separate relative and absolute signals.

This module is deliberately standalone.  It addresses the coupling between
pairwise preference and finish acceptability; it does not correct visual
hallucinations or validate the judge itself.  Existing callers continue to use
``quality_signal.py`` until a separately reviewed migration.
"""

PROTOCOL_VERSION = 'quality_signal_v2'
CONFIDENT = frozenset({'high', 'medium'})
WINNERS = {'A', 'B', 'tie'}
INVERSE = {'A': 'B', 'B': 'A', 'tie': 'tie'}
RELATIVE_DELTA = {'A': 1.0, 'B': -1.0, 'tie': 0.0}
VALID_ACTIONS = frozenset({'paint', 'repair', 'finish'})
NORMALIZED_FIELDS = frozenset({
    'protocol_version', 'delta', 'candidate_acceptable', 'anchor_acceptable',
})


def _label(value):
    return value if isinstance(value, dict) else None


def _consensus(first, reverse, first_side, reverse_side):
    first_value = first.get('acceptable_' + first_side)
    reverse_value = reverse.get('acceptable_' + reverse_side)
    if not isinstance(first_value, str) or not isinstance(reverse_value, str):
        return None
    if first_value not in {'yes', 'no'} or first_value != reverse_value:
        return None
    return first_value == 'yes'


def reconcile_v2(first, reverse):
    """Normalize two blinded views without coupling relative and absolute data.

    ``delta`` is the order-consistent relative preference: ``1`` means A won,
    ``-1`` means B won, and ``0`` is a confirmed tie.  Any malformed, unknown,
    or low-confidence winner yields ``None``.  Absolute acceptability is
    independently populated only when both oriented observations agree.
    Reasons are intentionally omitted from this normalized record.
    """
    first = _label(first)
    reverse = _label(reverse)
    result = {
        'protocol_version': PROTOCOL_VERSION,
        'delta': None,
        'candidate_acceptable': None,
        'anchor_acceptable': None,
    }
    if first is None or reverse is None:
        return result
    first_confidence = first.get('confidence')
    reverse_confidence = reverse.get('confidence')
    if (isinstance(first_confidence, str) and isinstance(reverse_confidence, str)
            and first_confidence in CONFIDENT and reverse_confidence in CONFIDENT):
        winner = first.get('winner')
        if (isinstance(winner, str) and winner in WINNERS
                and reverse.get('winner') == INVERSE[winner]):
            result['delta'] = RELATIVE_DELTA[winner]
        result['candidate_acceptable'] = _consensus(first, reverse, 'A', 'B')
        result['anchor_acceptable'] = _consensus(first, reverse, 'B', 'A')
    return result


def _valid_judgment(judgment):
    if not isinstance(judgment, dict):
        return False
    if set(judgment) != NORMALIZED_FIELDS:
        return False
    if judgment.get('protocol_version') != PROTOCOL_VERSION:
        return False
    delta = judgment.get('delta')
    if (delta is not None
            and (isinstance(delta, bool) or not isinstance(delta, (int, float))
                 or delta not in {-1.0, 0.0, 1.0})):
        return False
    return all(judgment.get(name) is None or isinstance(judgment.get(name), bool)
               for name in ('candidate_acceptable', 'anchor_acceptable'))


def reward_v2(result, judgment):
    """Score one transition using the v2 signal protocol.

    Paint and repair transitions use only the relative ``delta`` (including a
    valid zero); finish transitions use only consensus candidate acceptability.
    Infrastructure/cutoff failures mask, and invalid programs retain the v1
    ``-2`` penalty.  Unknown or malformed judgments are always masked.
    """
    if not isinstance(result, dict):
        return None
    failure_kind = result.get('failure_kind')
    if failure_kind is not None and not isinstance(failure_kind, str):
        return None
    if failure_kind in {'infrastructure', 'cutoff'}:
        return None
    if failure_kind == 'program' and result.get('valid') is not True:
        return -2.0
    action = result.get('action')
    if (result.get('valid') is not True
            or not isinstance(action, str)
            or action not in VALID_ACTIONS):
        return None
    if not _valid_judgment(judgment):
        return None
    if action in {'paint', 'repair'}:
        return judgment['delta'] if judgment['delta'] is not None else None
    accepted = judgment['candidate_acceptable']
    if accepted is True:
        return 1.0
    if accepted is False:
        return -1.5
    return None
