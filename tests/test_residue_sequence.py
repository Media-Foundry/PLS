import unittest,torch
from pls.models.residue_sequence import ResidueSequenceRegressor
class ResidueSequenceTests(unittest.TestCase):
 def test_pooling_modes(self):
  mask=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
  for mode in ('mean','attention','conv_attention'):
   model=ResidueSequenceRegressor(12,8,16,10,.1,mode);y=model(torch.randn(2,12),torch.randn(2,4,8),mask);self.assertEqual(tuple(y.shape),(2,));self.assertTrue(torch.isfinite(y).all())
if __name__=='__main__':unittest.main()
