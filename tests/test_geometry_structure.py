import unittest,torch
from pls.models.geometry_structure import GeometryLateFusion
class GeometryTests(unittest.TestCase):
 def test_knn_models(self):
  b,n,k=2,7,4;mask=torch.tensor([[1]*5+[0]*2,[1]*7],dtype=torch.bool);neighbors=torch.randint(0,n,(b,n,k));dist=torch.rand(b,n,k)*10
  for layers in (1,2):
   for pooling in ('attention','dual_patch'):
    m=GeometryLateFusion(12,20,16,8,.0,layers,pooling);y=m(torch.randn(b,12),torch.randn(b,n,20),torch.rand(b,n),torch.randn(b,n,5),mask,neighbors,dist);self.assertEqual(tuple(y.shape),(b,));self.assertTrue(torch.isfinite(y).all())
  m=GeometryLateFusion(12,20,16,8,.0,1,'attention',global_dimension=6);y=m(torch.randn(b,12),torch.randn(b,n,20),torch.rand(b,n),torch.randn(b,n,5),mask,neighbors,dist,torch.randn(b,6));self.assertEqual(tuple(y.shape),(b,))
if __name__=='__main__':unittest.main()
