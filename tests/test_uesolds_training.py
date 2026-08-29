import unittest,torch
from pls.training.train_uesolds_residue import binary_rank_loss
class UESolDSTrainingTests(unittest.TestCase):
 def test_binary_rank_loss_rewards_separation(self):
  target=torch.tensor([1.,1.,0.,0.]);good=torch.tensor([2.,1.,-1.,-2.]);bad=-good
  self.assertLess(binary_rank_loss(good,target),binary_rank_loss(bad,target))
if __name__=='__main__':unittest.main()
