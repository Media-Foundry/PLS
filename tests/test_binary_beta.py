import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from calibrate_binary_beta import beta_logits

class BinaryBetaTests(unittest.TestCase):
 def test_constrained_beta_mapping_is_strictly_monotone(self):
  logits=np.linspace(-10,10,1000)
  for parameters in (np.asarray([0.,0.,0.]),np.asarray([-6.,6.,2.]),np.asarray([6.,-6.,-2.])):self.assertTrue(np.all(np.diff(beta_logits(logits,parameters))>0))

if __name__=='__main__':unittest.main()
