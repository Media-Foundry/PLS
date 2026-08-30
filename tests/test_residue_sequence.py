import tempfile,unittest,torch
from pathlib import Path
from pls.models.residue_sequence import ResidueSequenceRegressor
from pls.training.train_esol_residue import load_pretrained_weights
class ResidueSequenceTests(unittest.TestCase):
 def test_pretrained_loader_transfers_backbone_and_resets_selected_head(self):
  torch.manual_seed(2);source=ResidueSequenceRegressor(16,8,12,12,0,'mean');target=ResidueSequenceRegressor(16,8,12,12,0,'mean');original=target.head[-1].weight.detach().clone()
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'best.pt';torch.save({'model':source.state_dict(),'epoch':4},path);report=load_pretrained_weights(target,path,('head.4',))
  torch.testing.assert_close(target.residue_encoder[1].weight,source.residue_encoder[1].weight);torch.testing.assert_close(target.head[-1].weight,original);self.assertEqual(report['source_epoch'],4)
 def test_pooling_modes(self):
  mask=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
  for mode in ('mean','attention','multihead_attention','statistics_attention','gated_statistics_attention','conv_attention','local_attention','multiscale_attention'):
   for fusion in ('concat','interaction'):
    model=ResidueSequenceRegressor(12,8,16,10,.1,mode,fusion);y=model(torch.randn(2,12),torch.randn(2,4,8),mask);self.assertEqual(tuple(y.shape),(2,));self.assertTrue(torch.isfinite(y).all())
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3);self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3,'concat');self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  model=ResidueSequenceRegressor(12,8,16,10,.1,'attention','concat',3,'logit_mixture');self.assertEqual(tuple(model(torch.randn(2,12),torch.randn(2,4,8),mask).shape),(2,))
  with self.assertRaises(ValueError):ResidueSequenceRegressor(12,8,16,10,.1,'attention','interaction',3,'concat')
if __name__=='__main__':unittest.main()
