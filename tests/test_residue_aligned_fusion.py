import unittest
import torch
from pls.models.geometry_structure import GeometryLateFusion


class ResidueAlignedFusionTests(unittest.TestCase):
 def test_forward_and_padding_invariance(self):
  torch.manual_seed(41)
  model=GeometryLateFusion(32,160,24,16,0,1,'attention',0,True,'residue_aligned_sparse','propagated_moe',8).eval()
  b,n,k=2,9,4
  sequence=torch.randn(b,32);residue=torch.randn(b,n,160);residue[...,21]=torch.rand(b,n);plddt=torch.rand(b,n);patch=torch.randn(b,n,5);mask=torch.ones(b,n,dtype=torch.bool)
  neighbors=torch.stack([torch.stack([torch.tensor([(i+j+1)%n for j in range(k)]) for i in range(n)]) for _ in range(b)])
  distances=torch.rand(b,n,k)*8+2
  expected=model(sequence,residue,plddt,patch,mask,neighbors,distances)
  pad=7
  padded_residue=torch.nn.functional.pad(residue,(0,0,0,pad));padded_plddt=torch.nn.functional.pad(plddt,(0,pad));padded_patch=torch.nn.functional.pad(patch,(0,0,0,pad));padded_mask=torch.nn.functional.pad(mask,(0,pad));padded_neighbors=torch.nn.functional.pad(neighbors,(0,0,0,pad));padded_distances=torch.nn.functional.pad(distances,(0,0,0,pad))
  actual=model(sequence,padded_residue,padded_plddt,padded_patch,padded_mask,padded_neighbors,padded_distances)
  torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

 def test_requires_residue_sequence_tokens(self):
  with self.assertRaisesRegex(ValueError,'residue_sequence_dimension'):
   GeometryLateFusion(fusion='residue_aligned_sparse')


if __name__=='__main__':unittest.main()
