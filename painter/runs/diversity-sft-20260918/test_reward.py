import unittest
from reward import transition_reward

class QualityRewardTests(unittest.TestCase):
 def test_uncertain_judge_does_not_become_tie(self):
  self.assertIsNone(transition_reward(valid=True,judgment={'mask':True,'reward':None})['reward'])
 def test_repetition_has_no_quality_gain(self):
  self.assertEqual(transition_reward(valid=True,no_change=True)['reward'],0)
 def test_finishing_bad_canvas_is_not_rewarded(self):
  self.assertEqual(transition_reward(valid=True,action='finish',judgment={'mask':False,'reward':.5,'candidate_acceptable':False,'acceptability_consistent':True})['reward'],-1)
 def test_preference_has_signed_direction(self):
  for preference,reward in [(1,1),(.5,0),(0,-1)]:
   self.assertEqual(transition_reward(valid=True,judgment={'mask':False,'reward':preference})['reward'],reward)
 def test_improved_but_incomplete_final_program_does_not_get_positive_reward(self):
  self.assertEqual(transition_reward(valid=True,terminal=True,judgment={'mask':False,'reward':1,'candidate_acceptable':False,'acceptability_consistent':True})['reward'],0)

if __name__=='__main__':unittest.main()
