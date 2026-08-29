import unittest
import torch
from pls.models.structure_fusion import build_fusion

class FusionTests(unittest.TestCase):
 def test_all_architectures(self):
  x=torch.randn(7,15)
  for name in ('early','late','gated_residual','film'):
   model=build_fusion(name,10,5,16,8,.1)
   y=model(x); self.assertEqual(tuple(y.shape),(7,)); self.assertTrue(torch.isfinite(y).all())
if __name__=='__main__': unittest.main()
