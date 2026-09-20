"""Quality-only transition rewards; uncertain judgments are masked, not zeros."""
def transition_reward(*,valid,action='paint',judgment=None,no_change=False,terminal=False):
    if not valid:return dict(reward=-1.,mask=False,reason='invalid_action')
    terminal=terminal or action=='finish'
    if action=='paint' and no_change and not terminal:return dict(reward=0.,mask=False,reason='unchanged_canvas')
    if not judgment or judgment.get('mask') or judgment.get('reward') is None:
        return dict(reward=None,mask=True,reason='missing_or_uncertain_visual_judgment')
    delta=0. if no_change or action=='finish' else 2*judgment['reward']-1
    if terminal:
        if not judgment.get('acceptability_consistent'):
            return dict(reward=None,mask=True,reason='inconsistent_finish_acceptability')
        return dict(reward=delta+(1. if judgment.get('candidate_acceptable') else -1.),mask=False,
                    reason='acceptable_finish' if judgment.get('candidate_acceptable') else 'premature_finish')
    return dict(reward=delta,mask=False,reason='pairwise_visual_change')
