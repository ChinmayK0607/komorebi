import unittest
from quality_signal import reconcile,reward,advantages
class SignalTest(unittest.TestCase):
 def test_order_disagreement_and_missing_are_masked(self):
  a=dict(winner='A',confidence='high',acceptable_A='yes',acceptable_B='no');b=dict(winner='B',confidence='high',acceptable_A='no',acceptable_B='yes')
  self.assertEqual(reconcile(a,b)['delta'],1)
  self.assertIsNone(reconcile(a,a));self.assertIsNone(reconcile(a,None))
 def test_finish_is_conditional_and_noops_do_not_win(self):
  good=dict(candidate_acceptable=True,delta=0);bad=dict(candidate_acceptable=False,delta=0)
  finish=dict(valid=True,action='finish',no_change=True);noop=dict(valid=True,action='paint',no_change=True)
  self.assertGreater(reward(finish,good),reward(noop,good));self.assertLess(reward(finish,bad),reward(noop,bad))
  self.assertIsNone(reward(dict(valid=False,failure_kind='infrastructure'),None))
 def test_groups_and_masks(self):
  self.assertEqual(advantages([1]*8),[None]*8)
  a=advantages([1,1,-1,-1,None]);self.assertIsNone(a[-1]);self.assertAlmostEqual(sum(a[:-1]),0)
  self.assertEqual(advantages([1,-1,None]),[None]*3)
if __name__=='__main__':unittest.main()
