import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from select_uesolds_weighted_auc import objective_score

class UESolDSSelectionTests(unittest.TestCase):
 def test_binary_objectives_reward_correct_ranking(self):
  target=np.asarray([0,0,1,1]);good=np.asarray([-2,-1,1,2]);bad=-good
  for objective in ('auroc','auprc','brier'):self.assertGreater(objective_score(target,good,objective),objective_score(target,bad,objective))

if __name__=='__main__':unittest.main()
