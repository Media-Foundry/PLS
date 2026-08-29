import unittest,torch
from pls.models.residue_structure import ResidueLateFusion
class ResidueTests(unittest.TestCase):
 def test_pooling_modes_and_padding(self):
  mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool); seq=torch.randn(2,12); x=torch.randn(2,5,10); p=torch.rand(2,5); patch=torch.randn(2,5,5)
  for mode in ('mean','attention','plddt_gate','dual_patch'):
   model=ResidueLateFusion(12,10,16,8,.0,mode); y,aux=model(seq,x,p,patch,mask); self.assertEqual(tuple(y.shape),(2,)); self.assertTrue(torch.isfinite(y).all()); self.assertEqual(tuple(aux['gate'].shape),(2,1))
if __name__=='__main__': unittest.main()
