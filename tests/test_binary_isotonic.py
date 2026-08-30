import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from calibrate_binary_isotonic import probability_logits

class BinaryIsotonicTests(unittest.TestCase):
 def test_probability_logit_roundtrip_is_finite_and_monotone(self):
  probability=np.asarray([0.,.1,.5,.9,1.]);logits=probability_logits(probability);self.assertTrue(np.isfinite(logits).all());self.assertTrue(np.all(np.diff(logits)>0));roundtrip=1/(1+np.exp(-logits));np.testing.assert_allclose(roundtrip[1:-1],probability[1:-1])

if __name__=='__main__':unittest.main()
