"""Pure-PyTorch CA-kNN message passing for solubility structure encoding."""
from __future__ import annotations
import torch
from torch import nn

class KNNMessageLayer(nn.Module):
 def __init__(self,dimension=256,rbf_dimension=16,dropout=.1):
  super().__init__();self.register_buffer('centers',torch.linspace(0,20,rbf_dimension));self.scale=20/(rbf_dimension-1)
  self.message=nn.Sequential(nn.LayerNorm(dimension+rbf_dimension),nn.Linear(dimension+rbf_dimension,dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(dimension,dimension));self.attention=nn.Linear(dimension+rbf_dimension,1);self.update=nn.Sequential(nn.LayerNorm(dimension*2),nn.Linear(dimension*2,dimension),nn.GELU(),nn.Dropout(dropout))
 def forward(self,z,neighbors,distances,mask):
  batch=torch.arange(len(z),device=z.device)[:,None,None];neighbor=z[batch,neighbors.long()];rbf=torch.exp(-((distances[...,None]-self.centers)/self.scale).square());edge=torch.cat((neighbor-z[:,:,None,:],rbf),-1);message=self.message(edge);logits=self.attention(edge).squeeze(-1);weights=torch.softmax(logits,2);aggregate=(message*weights[...,None]).sum(2);return z+self.update(torch.cat((z,aggregate),-1))*mask[...,None]

class GeometryLateFusion(nn.Module):
 def __init__(self,seq_dimension=1280,input_dimension=152,hidden_dimension=256,representation_dimension=256,dropout=.15,layers=1,pooling='attention'):
  super().__init__();self.pooling=pooling;self.sequence=nn.Sequential(nn.LayerNorm(seq_dimension),nn.Linear(seq_dimension,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,representation_dimension),nn.GELU());self.input=nn.Sequential(nn.LayerNorm(input_dimension),nn.Linear(input_dimension,hidden_dimension),nn.GELU());self.layers=nn.ModuleList([KNNMessageLayer(hidden_dimension,16,dropout) for _ in range(layers)]);self.project=nn.Linear(hidden_dimension,representation_dimension);self.attention=nn.Linear(representation_dimension+1,1);self.patch_attention=nn.Sequential(nn.Linear(representation_dimension+5,64),nn.GELU(),nn.Linear(64,1));self.merge=nn.Sequential(nn.Linear(representation_dimension*2,representation_dimension),nn.GELU());self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid());self.head=nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def forward(self,sequence,residue,plddt,patch,mask,neighbors,distances):
  z=self.input(residue)
  for layer in self.layers:z=layer(z,neighbors,distances,mask)
  z=self.project(z);count=mask.sum(1).clamp_min(1);logits=self.attention(torch.cat((z,plddt[...,None]),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);weight=torch.softmax(logits,1);pooled=(z*weight[...,None]).sum(1)
  if self.pooling=='dual_patch':
   logits=self.patch_attention(torch.cat((z,patch),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);pw=torch.softmax(logits,1);pooled=self.merge(torch.cat((pooled,(z*pw[...,None]).sum(1)),1))
  quality=torch.stack(((plddt*mask).sum(1)/count,((plddt<.7)&mask).sum(1)/count,torch.log1p(count)/10),1);pooled=pooled*self.gate(quality);return self.head(torch.cat((self.sequence(sequence),pooled),1)).squeeze(-1)
