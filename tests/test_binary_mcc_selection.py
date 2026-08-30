import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from select_binary_mcc_candidates import exact_max_mcc

class BinaryMCCSelectionTests(unittest.TestCase):
 def test_exact_search_matches_brute_force_with_ties(self):
  target=np.asarray([0,1,0,1,1,0]);score=np.asarray([.1,.7,.1,.4,.7,-.2]);observed,threshold=exact_max_mcc(target,score);expected=max(matthews_corrcoef(target,score>=value) for value in np.unique(score));self.assertAlmostEqual(observed,expected);self.assertAlmostEqual(observed,matthews_corrcoef(target,score>=threshold))

if __name__=='__main__':unittest.main()
