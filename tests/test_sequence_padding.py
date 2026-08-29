import unittest
import torch
from pls.models.residue_sequence import ResidueSequenceRegressor

class SequencePaddingTests(unittest.TestCase):
 def test_pooling_predictions_ignore_extra_padding(self):
  torch.manual_seed(11);global_x=torch.randn(2,32);short=torch.randn(2,9,12);mask=torch.tensor([[1]*7+[0]*2,[1]*9],dtype=torch.bool);long=torch.zeros(2,16,12);long[:,:9]=short;long_mask=torch.zeros(2,16,dtype=torch.bool);long_mask[:,:9]=mask
  for pooling in ('attention','conditioned_attention','tcn_conditioned_attention','shift_tcn_conditioned_attention','conv_attention','local_attention','multi_query_pooling','statistics_attention'):
   model=ResidueSequenceRegressor(32,12,24,16,0,pooling,'concat').eval()
   with torch.inference_mode():a=model(global_x,short,mask);b=model(global_x,long,long_mask)
   self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6),pooling)
 def test_segmented_conditioned_pooling_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'global_segments=1'):ResidueSequenceRegressor(64,12,24,16,0,'conditioned_attention','concat',global_segments=2)

if __name__=='__main__':unittest.main()
