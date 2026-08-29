import unittest
import torch
from pls.models.gvp_structure import GVPStructureFusion

class GVPStructureTests(unittest.TestCase):
 def test_prediction_is_rotation_and_translation_invariant(self):
  torch.manual_seed(7);batch,residues,neighbors=2,13,5;model=GVPStructureFusion(32,20,24,6,16,0,2).eval();sequence=torch.randn(batch,32);scalar=torch.randn(batch,residues,20);vectors=torch.randn(batch,residues,8,3);coords=torch.randn(batch,residues,3);mask=torch.ones(batch,residues,dtype=torch.bool);indices=torch.randint(0,residues,(batch,residues,neighbors));batch_index=torch.arange(batch)[:,None,None];distances=(coords[batch_index,indices]-coords[:,:,None]).norm(dim=-1)
  q,_=torch.linalg.qr(torch.randn(3,3));translation=torch.randn(3);rotated_vectors=torch.einsum('bnvc,cd->bnvd',vectors,q);rotated_coords=coords@q+translation
  with torch.inference_mode():original=model(sequence,scalar,vectors,coords,mask,indices,distances);rotated=model(sequence,scalar,rotated_vectors,rotated_coords,mask,indices,distances)
  self.assertTrue(torch.allclose(original,rotated,atol=2e-5,rtol=2e-5),(original,rotated))

if __name__=='__main__':unittest.main()
