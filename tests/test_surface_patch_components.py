import unittest
import numpy as np
from preparation.build_surface_patch_components import components


class SurfacePatchComponentTests(unittest.TestCase):
 def test_spatial_components_do_not_use_sequence_adjacency(self):
  sequence='AVDKFW';rsa=np.ones(len(sequence),np.float32);neighbors=np.array([[4,2],[2,3],[3,1],[2,5],[0,2],[4,3]]);distances=np.full_like(neighbors,5,dtype=np.float32)
  labels=components(sequence,rsa,neighbors,distances,.25,8)
  hydrophobic=labels[:,1]
  self.assertEqual(hydrophobic[0],hydrophobic[4])
  self.assertNotEqual(hydrophobic[1],hydrophobic[0])
  self.assertEqual(hydrophobic[2],-1)

 def test_buried_residues_are_excluded(self):
  labels=components('AAAA',np.array([1,.1,1,1]),np.tile(np.arange(4),(4,1)),np.ones((4,4)),.25,8)
  self.assertTrue((labels[1] == -1).all())


if __name__=='__main__':unittest.main()
