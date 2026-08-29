"""Frozen residue-PLM pooling models for quantitative solubility regression."""
from __future__ import annotations
import torch
from torch import nn
class ResidueSequenceRegressor(nn.Module):
 def __init__(self,global_dimension=1280,residue_dimension=256,hidden_dimension=256,representation_dimension=256,dropout=.2,pooling='attention'):
  super().__init__();self.pooling=pooling;self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,representation_dimension),nn.GELU());self.residue_encoder=nn.Sequential(nn.LayerNorm(residue_dimension),nn.Linear(residue_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.local=nn.Sequential(nn.Conv1d(representation_dimension,representation_dimension,5,padding=2,groups=representation_dimension),nn.GELU(),nn.Conv1d(representation_dimension,representation_dimension,1),nn.Dropout(dropout));self.attention=nn.Linear(representation_dimension,1);self.head=nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def forward(self,global_embedding,residues,mask):
  z=self.residue_encoder(residues)
  if self.pooling=='conv_attention':z=z+self.local(z.transpose(1,2)).transpose(1,2)*mask[...,None]
  if self.pooling=='mean':pooled=(z*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
  else:
   logits=self.attention(z).squeeze(-1).masked_fill(~mask,-torch.inf);pooled=(z*torch.softmax(logits,1)[...,None]).sum(1)
  return self.head(torch.cat((self.global_encoder(global_embedding),pooled),1)).squeeze(-1)
