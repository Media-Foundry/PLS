import os
import unittest

import torch

from pls.models.gvp_structure import GVPStructureFusion


class GVPSurfaceFusionTests(unittest.TestCase):
 def inputs(self):
  torch.manual_seed(71);b,n,k=2,9,4;sequence=torch.randn(b,16);residue=torch.randn(b,n,160);residue[...,63]=torch.rand(b,n);residue[...,151]=torch.rand(b,n);vectors=torch.randn(b,n,8,3);coords=torch.randn(b,n,3);mask=torch.ones(b,n,dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));distances=torch.rand(b,n,k)*6+2;patch=torch.randn(b,n,5);components=torch.randint(-1,4,(b,n,5));return sequence,residue,vectors,coords,mask,neighbors,distances,patch,components

 def model(self):return GVPStructureFusion(16,160,24,8,16,0,1,'aligned_moe',8,0,1,True,1).eval()

 def test_component_relabeling_and_padding_are_invariant(self):
  model=self.model();values=list(self.inputs())
  with torch.inference_mode():expected=model(*values);relabelled=values[-1].clone();relabelled[relabelled>=0]+=19;actual=model(*values[:-1],relabelled)
  torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
  pad=5
  for index in (1,3,7):values[index]=torch.nn.functional.pad(values[index],(0,0,0,pad))
  values[2]=torch.nn.functional.pad(values[2],(0,0,0,0,0,pad))
  values[4]=torch.nn.functional.pad(values[4],(0,pad));values[5]=torch.nn.functional.pad(values[5],(0,0,0,pad));values[6]=torch.nn.functional.pad(values[6],(0,0,0,pad));values[8]=torch.nn.functional.pad(values[8],(0,0,0,pad),value=-1)
  with torch.inference_mode():padded=model(*values)
  torch.testing.assert_close(padded,expected,atol=2e-6,rtol=2e-6)

 def test_patch_cross_attention_receives_gradients(self):
  model=self.model().train();output=model(*self.inputs());loss=output.square().mean()+model.last_surface_patch_logit.square().mean();loss.backward();gradient=model.surface_residue_attention.in_proj_weight.grad;self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all());self.assertIsNotNone(model.surface_patch_head[-1].weight.grad)
  self.assertIsNotNone(model.patch_spatial_layers[0].message[1].weight.grad)

 def test_rigid_motion_is_invariant(self):
  model=self.model();values=list(self.inputs())
  with torch.inference_mode():expected=model(*values)
  rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
  values[2]=values[2]@rotation;values[3]=values[3]@rotation+torch.tensor([7.,-3.,11.])
  with torch.inference_mode():actual=model(*values)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)

 @unittest.skipUnless(torch.cuda.is_available() and os.environ.get('HIP_VISIBLE_DEVICES') in {'6','7'},'authorized ROCm device 6/7 is unavailable')
 def test_bfloat16_autocast_forward_backward(self):
  model=self.model().cuda().train();values=[value.cuda() for value in self.inputs()]
  with torch.autocast('cuda',dtype=torch.bfloat16):output=model(*values);loss=output.square().mean()+model.last_surface_patch_logit.square().mean()
  loss.backward();self.assertTrue(torch.isfinite(output).all());self.assertIsNotNone(model.patch_spatial_layers[0].message[1].weight.grad)


if __name__=='__main__':unittest.main()
