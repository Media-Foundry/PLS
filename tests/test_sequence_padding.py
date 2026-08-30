import unittest
import torch
from pls.models.residue_sequence import ResidueSequenceRegressor

class SequencePaddingTests(unittest.TestCase):
 def test_pooling_predictions_ignore_extra_padding(self):
  torch.manual_seed(11);global_x=torch.randn(2,32);short=torch.randn(2,9,12);mask=torch.tensor([[1]*7+[0]*2,[1]*9],dtype=torch.bool);long=torch.zeros(2,48,12);long[:,:9]=short;long_mask=torch.zeros(2,48,dtype=torch.bool);long_mask[:,:9]=mask
  for pooling in ('attention','conditioned_attention','tcn_conditioned_attention','shift_tcn_conditioned_attention','gated_shift_tcn_conditioned_attention','bilstm_textcnn','conv_attention','local_attention','multi_query_pooling','statistics_attention','segment_transformer'):
   model=ResidueSequenceRegressor(32,12,24,16,0,pooling,'concat').eval()
   with torch.inference_mode():a=model(global_x,short,mask);b=model(global_x,long,long_mask)
   self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6),pooling)
 def test_segmented_conditioned_pooling_is_rejected(self):
  model=ResidueSequenceRegressor(69,12,24,16,0,'conditioned_attention','concat',global_segments=2,descriptor_dimension=5);output=model(torch.randn(2,69),torch.randn(2,7,12),torch.ones(2,7,dtype=torch.bool));self.assertEqual(tuple(output.shape),(2,))
  with self.assertRaisesRegex(ValueError,'weighted_sum'):ResidueSequenceRegressor(64,12,24,16,0,'conditioned_attention','concat',global_segments=2,global_segment_fusion='concat')
 def test_physchem_experts_are_normalized_and_trainable(self):
  torch.manual_seed(13);model=ResidueSequenceRegressor(32,12,24,16,0,'shift_tcn_conditioned_attention','concat',descriptor_dimension=5,physchem_experts=4);global_x=torch.randn(3,32);residues=torch.randn(3,11,12);mask=torch.ones(3,11,dtype=torch.bool);output=model(global_x,residues,mask);output.square().mean().backward();torch.testing.assert_close(model.last_expert_weights.sum(1),torch.ones(3));self.assertIsNotNone(model.physchem_gate[-1].weight.grad);self.assertTrue(torch.isfinite(model.physchem_gate[-1].weight.grad).all())
 def test_lower_physchem_gate_temperature_sharpens_assignments(self):
  torch.manual_seed(17);base=ResidueSequenceRegressor(32,12,24,16,0,'shift_tcn_conditioned_attention','concat',descriptor_dimension=5,physchem_experts=4);sharp=ResidueSequenceRegressor(32,12,24,16,0,'shift_tcn_conditioned_attention','concat',descriptor_dimension=5,physchem_experts=4,physchem_gate_temperature=.5);sharp.load_state_dict(base.state_dict());global_x=torch.randn(3,32);residues=torch.randn(3,11,12);mask=torch.ones(3,11,dtype=torch.bool);base(global_x,residues,mask);sharp(global_x,residues,mask);base_entropy=-(base.last_expert_weights*base.last_expert_weights.log()).sum(1);sharp_entropy=-(sharp.last_expert_weights*sharp.last_expert_weights.log()).sum(1);self.assertTrue(torch.all(sharp_entropy<base_entropy))
 def test_physchem_gate_temperature_must_be_positive(self):
  with self.assertRaisesRegex(ValueError,'temperature'):ResidueSequenceRegressor(32,12,24,16,0,'attention','concat',descriptor_dimension=5,physchem_experts=4,physchem_gate_temperature=0)
 def test_sparse_physchem_gate_uses_exactly_top_k_experts(self):
  torch.manual_seed(19);model=ResidueSequenceRegressor(32,12,24,16,0,'attention','concat',descriptor_dimension=5,physchem_experts=4,physchem_top_k=2);model(torch.randn(3,32),torch.randn(3,7,12),torch.ones(3,7,dtype=torch.bool));self.assertTrue(torch.equal((model.last_expert_weights>0).sum(1),torch.full((3,),2)));torch.testing.assert_close(model.last_expert_weights.sum(1),torch.ones(3))
 def test_sparse_physchem_gate_rejects_excess_top_k(self):
  with self.assertRaisesRegex(ValueError,'top-k'):ResidueSequenceRegressor(32,12,24,16,0,'attention','concat',descriptor_dimension=5,physchem_experts=4,physchem_top_k=5)

if __name__=='__main__':unittest.main()
