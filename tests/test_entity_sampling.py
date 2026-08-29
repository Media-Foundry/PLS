import unittest
import numpy as np
from pls.training.train_residue_structure import BalancedLengthBatchSampler


class EntityAwareSamplingTests(unittest.TestCase):
 def test_duplicate_observations_share_entity_mass(self):
  labels=np.array([0,0,0,0,1,1]);entities=np.array([10,10,10,11,20,21]);sampler=BalancedLengthBatchSampler(labels,[1]*6,2,7,entities)
  weights=sampler.weights.numpy()
  self.assertAlmostEqual(weights[:3].sum(),weights[3],places=12)
  self.assertAlmostEqual(weights[:4].sum(),weights[4:].sum(),places=12)


if __name__=='__main__':unittest.main()
