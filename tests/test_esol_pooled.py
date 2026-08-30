import unittest
import numpy as np
from pls.training.train_esol_pooled import train_only_embedding_projection

class EsPooledTests(unittest.TestCase):
 def test_embedding_projection_is_fit_only_on_training_entities(self):
  rng=np.random.default_rng(3);embedding=rng.normal(size=(12,8)).astype(np.float32);train=np.arange(8);splits={'train':train,'validation':np.arange(8,12)};first,stats=train_only_embedding_projection(embedding,train,splits,3,7);changed=embedding.copy();changed[8:]+=1000;second,changed_stats=train_only_embedding_projection(changed,train,splits,3,7);np.testing.assert_array_equal(stats['mean'],changed_stats['mean']);np.testing.assert_array_equal(stats['components'],changed_stats['components']);np.testing.assert_array_equal(first['train'],second['train']);self.assertEqual(first['validation'].shape,(4,3))

if __name__=='__main__':unittest.main()
