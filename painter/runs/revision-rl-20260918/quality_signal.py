"""Reference-conditioned decision rewards and within-state credit assignment."""
import math

def reconcile(first,reverse):
 inverse={'A':'B','B':'A','tie':'tie','uncertain':'uncertain'}
 for r in [first,reverse]:
  if not r or r.get('winner') not in inverse or r.get('confidence') not in ['high','medium']:
   return None
  if any(r.get('acceptable_'+side) not in ['yes','no'] for side in ['A','B']):return None
 if first['winner']=='uncertain' or inverse[first['winner']]!=reverse['winner']:return None
 if first['acceptable_A']!=reverse['acceptable_B'] or first['acceptable_B']!=reverse['acceptable_A']:return None
 return dict(delta={'A':1.,'B':-1.,'tie':0.}[first['winner']],candidate_acceptable=first['acceptable_A']=='yes',anchor_acceptable=first['acceptable_B']=='yes')

def reward(result,judgment):
 if result.get('failure_kind') in ['infrastructure','cutoff']:return None
 if not result.get('valid'):return -2. if result.get('failure_kind')=='program' else None
 if judgment is None:return None
 accepted=judgment['candidate_acceptable']
 if result['action']=='finish':return 1.1 if accepted else -1.5
 if result['no_change']:return .75 if accepted else -.5
 return (1. if accepted else 0.)+.5*judgment['delta']

def advantages(values,minimum=4):
 scored=[v for v in values if v is not None]
 if len(scored)<minimum:return [None]*len(values)
 mean=sum(scored)/len(scored);var=sum((v-mean)**2 for v in scored)/len(scored)
 if var<1e-12:return [None]*len(values)
 return [None if v is None else (v-mean)/(math.sqrt(var)+1e-8) for v in values]
