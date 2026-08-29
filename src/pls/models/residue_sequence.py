"""Frozen residue-PLM pooling models for quantitative solubility regression."""
from __future__ import annotations
import torch
from torch import nn
class ResidueSequenceRegressor(nn.Module):
 def __init__(self,global_dimension=1280,residue_dimension=256,hidden_dimension=256,representation_dimension=256,dropout=.2,pooling='attention',fusion='concat',global_segments=1):
  super().__init__();self.pooling,self.fusion,self.global_segments=pooling,fusion,global_segments
  if global_dimension%global_segments:raise ValueError('global dimension must be divisible by global segments')
  if global_segments==1:self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,representation_dimension),nn.GELU());self.global_segment_encoders=None
  else:
   segment_dimension=global_dimension//global_segments;self.global_encoder=None;self.global_segment_encoders=nn.ModuleList([nn.Sequential(nn.LayerNorm(segment_dimension),nn.Linear(segment_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) for _ in range(global_segments)]);self.global_segment_logits=nn.Parameter(torch.zeros(global_segments))
  self.residue_encoder=nn.Sequential(nn.LayerNorm(residue_dimension),nn.Linear(residue_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.local=nn.Sequential(nn.Conv1d(representation_dimension,representation_dimension,5,padding=2,groups=representation_dimension),nn.GELU(),nn.Conv1d(representation_dimension,representation_dimension,1),nn.Dropout(dropout));self.attention=nn.Linear(representation_dimension,1);self.multi_attention=nn.Linear(representation_dimension,4);self.statistics_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.multi_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));head_dimension=representation_dimension*(4 if fusion=='interaction' else 2);self.head=nn.Sequential(nn.LayerNorm(head_dimension),nn.Linear(head_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def forward(self,global_embedding,residues,mask):
  z=self.residue_encoder(residues)
  if self.pooling=='conv_attention':z=z+self.local(z.transpose(1,2)).transpose(1,2)*mask[...,None]
  if self.pooling=='local_attention':
   left=torch.roll(z,1,1);right=torch.roll(z,-1,1);left[:,0]=0;right[:,-1]=0;z=z+(left+right)*.5*mask[...,None]
  count=mask.sum(1).clamp_min(1)[:,None];mean=(z*mask[...,None]).sum(1)/count
  if self.pooling=='mean':pooled=mean
  elif self.pooling=='multihead_attention':
   logits=self.multi_attention(z).masked_fill(~mask[...,None],-torch.inf);weights=torch.softmax(logits,1);pooled=self.multi_projection(torch.einsum('bnh,bnd->bhd',weights,z).flatten(1))
  else:
   logits=self.attention(z).squeeze(-1).masked_fill(~mask,-torch.inf);attention=(z*torch.softmax(logits,1)[...,None]).sum(1)
   if self.pooling=='statistics_attention':
    variance=((z-mean[:,None]).square()*mask[...,None]).sum(1)/count
    maximum=z.masked_fill(~mask[...,None],-torch.inf).amax(1)
    pooled=self.statistics_projection(torch.cat((mean,variance.clamp_min(0).sqrt(),maximum,attention),1))
   else:pooled=attention
  if self.global_segments==1:global_encoded=self.global_encoder(global_embedding)
  else:
   segments=global_embedding.chunk(self.global_segments,1);encoded=torch.stack([encoder(segment) for encoder,segment in zip(self.global_segment_encoders,segments)],1);global_encoded=(encoded*torch.softmax(self.global_segment_logits,0)[None,:,None]).sum(1)
  features=(global_encoded,pooled,global_encoded*pooled,(global_encoded-pooled).abs()) if self.fusion=='interaction' else (global_encoded,pooled)
  return self.head(torch.cat(features,1)).squeeze(-1)
