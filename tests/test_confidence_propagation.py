import unittest
import torch
from pls.models.geometry_structure import GeometryLateFusion

class ConfidencePropagationTests(unittest.TestCase):
 def test_zero_confidence_is_structure_independent_sequence_fallback(self):
  torch.manual_seed(23);model=GeometryLateFusion(32,152,24,16,0,1,'dual_patch',7,True,'global_query_pooling','propagated_moe').eval();b,n,k=2,11,4;sequence=torch.randn(b,32);mask=torch.ones(b,n,dtype=torch.bool);plddt=torch.zeros(b,n);neighbors=torch.randint(0,n,(b,n,k));distances=torch.rand(b,n,k)*8+2;global_features=torch.randn(b,7)
  def run(scale):return model(sequence,torch.randn(b,n,152)*scale,plddt,torch.randn(b,n,5)*scale,mask,neighbors,distances,global_features*scale)
  with torch.inference_mode():a=run(1);bvalue=run(100)
  self.assertTrue(torch.allclose(a,bvalue,atol=2e-6,rtol=2e-6),(a,bvalue))

 def test_zero_confidence_and_padding_have_finite_gradients(self):
  torch.manual_seed(29);model=GeometryLateFusion(32,152,24,16,0,1,'attention',0,True,'concat','propagated_moe').train();b,n,k=2,11,4
  sequence=torch.randn(b,32);residue=torch.randn(b,n,152);residue[...,21]=torch.rand(b,n);plddt=torch.rand(b,n);plddt[:,::3]=0;patch=torch.randn(b,n,5);mask=torch.ones(b,n,dtype=torch.bool);mask[:,-3:]=False
  neighbors=torch.randint(0,n,(b,n,k));distances=torch.rand(b,n,k)*8+2
  loss=model(sequence,residue,plddt,patch,mask,neighbors,distances).square().mean();loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

if __name__=='__main__':unittest.main()
