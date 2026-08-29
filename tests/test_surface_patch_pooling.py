import unittest
import torch
from pls.models.geometry_structure import GeometryLateFusion


class SurfacePatchPoolingTests(unittest.TestCase):
 def inputs(self):
  torch.manual_seed(53);b,n,k=2,10,4;sequence=torch.randn(b,16);residue=torch.randn(b,n,152);residue[...,63]=torch.rand(b,n);plddt=torch.rand(b,n);patch=torch.randn(b,n,5);mask=torch.ones(b,n,dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));distances=torch.rand(b,n,k)*6+2;components=torch.randint(-1,4,(b,n,5));return sequence,residue,plddt,patch,mask,neighbors,distances,None,components

 def test_component_relabeling_is_invariant(self):
  model=GeometryLateFusion(16,152,24,16,0,1,'surface_patches',confidence_mode='propagated_moe').eval();values=self.inputs()
  with torch.inference_mode():expected=model(*values);relabelled=values[-1].clone();relabelled[relabelled>=0]+=17;actual=model(*values[:-1],relabelled)
  torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

 def test_padding_is_invariant(self):
  model=GeometryLateFusion(16,152,24,16,0,1,'surface_patches',confidence_mode='propagated_moe').eval();values=list(self.inputs())
  with torch.inference_mode():expected=model(*values)
  pad=6;values[1]=torch.nn.functional.pad(values[1],(0,0,0,pad));values[2]=torch.nn.functional.pad(values[2],(0,pad));values[3]=torch.nn.functional.pad(values[3],(0,0,0,pad));values[4]=torch.nn.functional.pad(values[4],(0,pad));values[5]=torch.nn.functional.pad(values[5],(0,0,0,pad));values[6]=torch.nn.functional.pad(values[6],(0,0,0,pad));values[8]=torch.nn.functional.pad(values[8],(0,0,0,pad),value=-1)
  with torch.inference_mode():actual=model(*values)
  torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

 def test_residue_tokens_cross_attend_to_patch_tokens(self):
  torch.manual_seed(59);b,n,k=2,9,4;model=GeometryLateFusion(16,160,24,16,0,1,'surface_patches',0,True,'residue_aligned_sparse','propagated_moe',8).train();sequence=torch.randn(b,16);residue=torch.randn(b,n,160);residue[...,63]=torch.rand(b,n);plddt=torch.rand(b,n);patch=torch.randn(b,n,5);mask=torch.ones(b,n,dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));distances=torch.rand(b,n,k)*6+2;components=torch.randint(-1,4,(b,n,5));output=model(sequence,residue,plddt,patch,mask,neighbors,distances,None,components);output.square().mean().backward()
  self.assertIsNotNone(model.surface_residue_attention.in_proj_weight.grad);self.assertTrue(torch.isfinite(model.surface_residue_attention.in_proj_weight.grad).all())


if __name__=='__main__':unittest.main()
