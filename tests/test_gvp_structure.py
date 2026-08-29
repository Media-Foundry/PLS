import unittest
import torch
from pls.models.gvp_structure import GVPStructureFusion
from pls.training.train_gvp_structure import attach_vectors

class GVPStructureTests(unittest.TestCase):
 def test_data_adapter_drops_surface_component_field(self):
  base=tuple(range(9));adapted=attach_vectors(base,'vectors','coords');self.assertEqual(adapted,(0,1,2,3,4,5,6,'vectors','coords',8))
 def test_prediction_is_rotation_and_translation_invariant(self):
  torch.manual_seed(7);batch,residues,neighbors=2,13,5;model=GVPStructureFusion(32,20,24,6,16,0,2).eval();sequence=torch.randn(batch,32);scalar=torch.randn(batch,residues,20);vectors=torch.randn(batch,residues,8,3);coords=torch.randn(batch,residues,3);mask=torch.ones(batch,residues,dtype=torch.bool);indices=torch.randint(0,residues,(batch,residues,neighbors));batch_index=torch.arange(batch)[:,None,None];distances=(coords[batch_index,indices]-coords[:,:,None]).norm(dim=-1)
  q=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]);translation=torch.randn(3);rotated_vectors=torch.einsum('bnvc,cd->bnvd',vectors,q);rotated_coords=coords@q+translation
  with torch.inference_mode():original=model(sequence,scalar,vectors,coords,mask,indices,distances);rotated=model(sequence,scalar,rotated_vectors,rotated_coords,mask,indices,distances)
  self.assertTrue(torch.allclose(original,rotated,atol=2e-5,rtol=2e-5),(original,rotated))

 def test_aligned_moe_has_finite_gradients_and_sequence_fallback(self):
  torch.manual_seed(13);b,n,k=2,11,4;model=GVPStructureFusion(32,160,24,6,16,0,2,'aligned_moe',8,.3,.2);sequence=torch.randn(b,32);residue=torch.randn(b,n,160);residue[...,151]=0;vectors=torch.randn(b,n,8,3);coords=torch.randn(b,n,3);mask=torch.ones(b,n,dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));batch=torch.arange(b)[:,None,None];distances=(coords[batch,neighbors]-coords[:,:,None]).norm(dim=-1)
  first=model(sequence,residue,vectors,coords,mask,neighbors,distances);second=model(sequence,residue*10,vectors*10,coords,mask,neighbors,distances);torch.testing.assert_close(first,second,atol=2e-6,rtol=2e-6);first.square().mean().backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

 def test_aligned_moe_remains_rigid_transform_invariant(self):
  torch.manual_seed(17);b,n,k=2,9,4;model=GVPStructureFusion(32,160,24,6,16,0,2,'aligned_moe',8,.3,.2).eval();sequence=torch.randn(b,32);residue=torch.randn(b,n,160);residue[...,151]=.8;vectors=torch.randn(b,n,8,3);coords=torch.randn(b,n,3);mask=torch.ones(b,n,dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));batch=torch.arange(b)[:,None,None];distances=(coords[batch,neighbors]-coords[:,:,None]).norm(dim=-1);rotation=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
  with torch.inference_mode():expected=model(sequence,residue,vectors,coords,mask,neighbors,distances);actual=model(sequence,residue,torch.einsum('bnvc,cd->bnvd',vectors,rotation),coords@rotation+3,mask,neighbors,distances)
  torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)

if __name__=='__main__':unittest.main()
