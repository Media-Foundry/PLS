import unittest,torch
from pls.models.residue_sequence import ResidueSequenceRegressor
class ResidueSequenceTests(unittest.TestCase):
 def test_pooling_modes(self):
  mask=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
  for mode in ('mean','attention','multihead_attention','statistics_attention','gated_statistics_attention','conv_attention','local_attention'):
   for fusion in ('concat','interaction'):
    model=ResidueSequenceRegressor(12,8,16,10,.1,mode,fusion);y=model(torch.randn(2,12),torch.randn(2,4,8),mask);self.assertEqual(tuple(y.shape),(2,));self.assertTrue(torch.isfinite(y).all())
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3);self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3,'concat');self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3,'logit_mixture');self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  with self.assertRaises(ValueError):ResidueSequenceRegressor(12,8,16,10,.1,'attention','interaction',3,'concat')
if __name__=='__main__':unittest.main()
