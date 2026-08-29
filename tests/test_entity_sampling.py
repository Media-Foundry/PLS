import unittest
import numpy as np
from pls.training.train_residue_structure import BalancedLengthBatchSampler
from pls.training.train_plm_heads import entity_task_weights


class EntityAwareSamplingTests(unittest.TestCase):
 def test_duplicate_observations_share_entity_mass(self):
  labels=np.array([0,0,0,0,1,1]);entities=np.array([10,10,10,11,20,21]);sampler=BalancedLengthBatchSampler(labels,[1]*6,2,7,entities)
  weights=sampler.weights.numpy()
  self.assertAlmostEqual(weights[:3].sum(),weights[3],places=12)
  self.assertAlmostEqual(weights[:4].sum(),weights[4:].sum(),places=12)

 def test_multitask_observation_weights_balance_tasks_and_entities(self):
  records=[(1,'esol',.1),(1,'esol',.2),(2,'esol',.3),(1,'pdbsol',1),(3,'pdbsol',0)]
  weights=np.asarray(entity_task_weights(records))
  self.assertAlmostEqual(weights[:2].sum(),weights[2],places=12)
  self.assertAlmostEqual(weights[:3].sum(),weights[3:].sum(),places=12)


if __name__=='__main__':unittest.main()
