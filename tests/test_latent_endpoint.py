import unittest

import torch

from pls.models.latent_endpoint import LatentEndpointModel
from pls.training.train_latent_endpoint import source_entity_weights


class LatentEndpointTests(unittest.TestCase):
 def test_all_endpoint_slopes_are_strictly_positive(self):
  model=LatentEndpointModel(12,16,0,3);self.assertTrue(torch.all(model.slopes>0));model.raw_slopes.data.fill_(-100);self.assertTrue(torch.all(model.slopes>0))
 def test_continuous_endpoint_is_bounded_and_binary_heads_are_logits(self):
  torch.manual_seed(7);model=LatentEndpointModel(12,16,0,3);features=torch.randn(4,12);endpoint=torch.tensor([0,1,2,2]);output=model(features,endpoint);self.assertTrue(torch.all((output[2:]>0)&(output[2:]<1)));self.assertEqual(tuple(output.shape),(4,))
 def test_higher_latent_score_cannot_reduce_any_endpoint(self):
  model=LatentEndpointModel(12,16,0,3);score=torch.tensor([-1.,1.]);logits=score[:,None]*model.slopes[None]+model.intercepts[None];self.assertTrue(torch.all(logits[1]>logits[0]))
 def test_all_heads_and_encoder_receive_gradients(self):
  torch.manual_seed(11);model=LatentEndpointModel(12,16,0,3);features=torch.randn(6,12);endpoint=torch.tensor([0,1,2,0,1,2]);model(features,endpoint).sum().backward();self.assertTrue(torch.isfinite(model.raw_slopes.grad).all());self.assertTrue(torch.isfinite(model.encoder[1].weight.grad).all())
 def test_sampling_weights_are_source_balanced_and_entity_aware(self):
  rows=[(1,0,0.),(1,0,1.),(2,0,0.),(3,1,0.),(4,2,.5)];weights=source_entity_weights(rows);self.assertAlmostEqual(float(weights[:3].sum()),1/3);self.assertAlmostEqual(float(weights[3]),1/3);self.assertAlmostEqual(float(weights[4]),1/3);self.assertAlmostEqual(float(weights[0]),float(weights[1]))


if __name__=='__main__':unittest.main()
