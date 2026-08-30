import unittest,torch
from pls.training.train_uesolds_residue import binary_rank_loss
class UESolDSTrainingTests(unittest.TestCase):
 def test_binary_rank_loss_rewards_separation(self):
  target=torch.tensor([1.,1.,0.,0.]);good=torch.tensor([2.,1.,-1.,-2.]);bad=-good
  self.assertLess(binary_rank_loss(good,target),binary_rank_loss(bad,target))
 def test_hard_pair_loss_focuses_on_worst_pairs(self):
  target=torch.tensor([1.,1.,0.,0.]);logits=torch.tensor([2.,.1,-.1,-2.]);full=binary_rank_loss(logits,target);hard=binary_rank_loss(logits,target,temperature=.5,hard_fraction=.25,margin=.2);self.assertGreater(hard,full)
 def test_rank_loss_rejects_invalid_controls(self):
  logits=torch.tensor([1.,-1.]);target=torch.tensor([1.,0.])
  with self.assertRaisesRegex(ValueError,'temperature'):binary_rank_loss(logits,target,temperature=0)
  with self.assertRaisesRegex(ValueError,'fraction'):binary_rank_loss(logits,target,hard_fraction=0)
if __name__=='__main__':unittest.main()
