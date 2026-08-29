import unittest
import torch
from pls.models.geometry_structure import KNNMessageLayer
from pls.models.gvp_structure import GVPMessageLayer

class NeighborValidityTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(19);self.mask=torch.ones(1,4,dtype=torch.bool);self.neighbors=torch.tensor([[[1,2],[0,2],[1,3],[2,1]]]);self.distances=torch.tensor([[[3.,5.],[3.,4.],[4.,3.],[3.,6.]]]);self.padded_neighbors=torch.cat((self.neighbors,self.neighbors[...,:1].expand(-1,-1,3)),-1);self.padded_distances=torch.cat((self.distances,torch.zeros(1,4,3)),-1)
 def test_scalar_layer_ignores_invalid_duplicate_slots(self):
  layer=KNNMessageLayer(12,8,0).eval();z=torch.randn(1,4,12)
  with torch.inference_mode():a=layer(z,self.neighbors,self.distances,self.mask);b=layer(z,self.padded_neighbors,self.padded_distances,self.mask)
  self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
 def test_gvp_layer_ignores_invalid_duplicate_slots(self):
  layer=GVPMessageLayer(12,4,0).eval();s=torch.randn(1,4,12);v=torch.randn(1,4,4,3);coords=torch.randn(1,4,3)
  with torch.inference_mode():sa,va=layer(s,v,coords,self.neighbors,self.distances,self.mask);sb,vb=layer(s,v,coords,self.padded_neighbors,self.padded_distances,self.mask)
  self.assertTrue(torch.allclose(sa,sb,atol=2e-6,rtol=2e-6));self.assertTrue(torch.allclose(va,vb,atol=2e-6,rtol=2e-6))

if __name__=='__main__':unittest.main()
