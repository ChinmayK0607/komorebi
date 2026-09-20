"""Quality curriculum decisions, independent of trainer and GPU vendor."""
import math
import random
from collections import defaultdict


def pass_at_k(successes, attempts, k):
    if not 0 <= successes <= attempts or not 1 <= k <= attempts:
        raise ValueError('Need 0 <= successes <= attempts and 1 <= k <= attempts')
    return 1.0 if attempts-successes < k else 1-math.comb(attempts-successes,k)/math.comb(attempts,k)


def sample(rows, level, count, seed=917, weights=None):
    """Group-balanced sampling; fresh/unattempted tasks remain eligible."""
    rng=random.Random(seed)
    by_band=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['split']!='train':continue
        rung=row['level'];band='current' if rung==level else 'review' if rung<level else 'next' if rung==level+1 else None
        if band:by_band[band][row['family']].append(row)
    bands=[b for b in ('current','review','next') if by_band[b]]
    if not bands:raise ValueError('No eligible training references')
    weights=weights or {'current':.6,'review':.25,'next':.15};output=[]
    for _ in range(count):
        band=rng.choices(bands,[weights[b] for b in bands])[0]
        family=rng.choice(sorted(by_band[band]));pool=by_band[band][family]
        def difficulty(row):
            n=row.get('attempts',0);p=row.get('acceptable',0)/n if n else .5
            return .25+4*p*(1-p)
        output.append(rng.choices(pool,[difficulty(x) for x in pool])[0])
    return output


def mastery(records, settings, required_families=()):
    """Judge preferences alone cannot establish absolute quality or mastery."""
    groups=defaultdict(list)
    for r in records:
        if r['split']=='dev':groups[r['source_group']].append(r)
    if len(groups)<settings['minimum_unique_development_references']:
        return {'mastered':False,'reason':'need more distinct development references','references':len(groups)}
    families=defaultdict(list);completed=[];acceptable=[]
    for group in groups.values():
        c=sum(bool(r['completed']) for r in group)/len(group)
        a=sum(bool(r['completed']) and bool(r.get('absolute_acceptable',False)) and not r.get('operational_cutoff',False) for r in group)/len(group)
        completed.append(c);acceptable.append(a);families[group[0]['family']].append(a)
    completion=sum(completed)/len(completed);quality=sum(acceptable)/len(acceptable)
    family_ok=all(f in families and sum(families[f])/len(families[f])>=settings['absolute_acceptable_fraction'] for f in required_families)
    ok=completion>=settings['completed_fraction'] and quality>=settings['absolute_acceptable_fraction'] and family_ok
    return dict(mastered=ok,references=len(groups),completed_fraction=completion,acceptable_fraction=quality,
                families={f:sum(v)/len(v) for f,v in families.items()},reason='advance candidate' if ok else 'continue current difficulty')


def speed_budget(phase, mastered_evaluations, settings, quality_cost, rung=0, mastered_level=None):
    if phase!='speed':return None
    if mastered_level is None or mastered_level<settings['speed']['minimum_mastered_level']:
        raise ValueError('Finish the quality curriculum before speed training')
    if mastered_evaluations<settings['mastery']['consecutive_evaluations']:
        raise ValueError('Speed phase requires sustained quality mastery')
    return max(1,math.ceil(quality_cost*settings['speed']['budget_fractions'][rung]))
