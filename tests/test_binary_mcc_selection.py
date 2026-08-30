import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from select_binary_mcc_candidates import base_components,exact_max_mcc

class BinaryMCCSelectionTests(unittest.TestCase):
 def test_exact_search_matches_brute_force_with_ties(self):
  target=np.asarray([0,1,0,1,1,0]);score=np.asarray([.1,.7,.1,.4,.7,-.2]);observed,threshold=exact_max_mcc(target,score);expected=max(matthews_corrcoef(target,score>=value) for value in np.unique(score));self.assertAlmostEqual(observed,expected);self.assertAlmostEqual(observed,matthews_corrcoef(target,score>=threshold))
 def test_recursive_incremental_report_preserves_component_mass(self):
  components=base_components({'base_run_weights':{'a':.6,'b':.4},'base_weight':.9,'candidate_weights':{'c':.1,'z':0}});self.assertEqual(set(components),{'a','b','c'});self.assertAlmostEqual(components['a'],.54);self.assertAlmostEqual(components['b'],.36);self.assertAlmostEqual(components['c'],.1);self.assertAlmostEqual(sum(components.values()),1)

if __name__=='__main__':unittest.main()
