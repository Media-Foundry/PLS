"""Residue-level structure encoders with confidence-aware pooling."""
from __future__ import annotations
import torch
from torch import nn

class ResidueStructureEncoder(nn.Module):
 def __init__(self,input_dimension=152,hidden_dimension=256,output_dimension=256,dropout=.15,mode='attention'):
  super().__init__(); self.mode=mode
  self.encoder=nn.Sequential(nn.LayerNorm(input_dimension),nn.Linear(input_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,output_dimension),nn.GELU())
  self.attention=nn.Linear(output_dimension+1,1); self.patch_attention=nn.Sequential(nn.Linear(output_dimension+5,64),nn.GELU(),nn.Linear(64,1))
  self.confidence_gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid())
  self.patch_merge=nn.Sequential(nn.Linear(output_dimension*2,output_dimension),nn.GELU())
 def forward(self,features,plddt,patch_features,mask):
  z=self.encoder(features); count=mask.sum(1).clamp_min(1); mean=(z*mask[...,None]).sum(1)/count[:,None]
  if self.mode=='mean': pooled=mean
  else:
   logits=self.attention(torch.cat((z,plddt[...,None]),-1)).squeeze(-1).masked_fill(~mask,-torch.inf); w=torch.softmax(logits,1); pooled=(z*w[...,None]).sum(1)
  if self.mode=='dual_patch':
   logits=self.patch_attention(torch.cat((z,patch_features),-1)).squeeze(-1).masked_fill(~mask,-torch.inf); pw=torch.softmax(logits,1); patch=(z*pw[...,None]).sum(1); pooled=self.patch_merge(torch.cat((pooled,patch),1))
  quality=torch.stack(((plddt*mask).sum(1)/count,((plddt<.7)&mask).sum(1)/count,torch.log1p(count)/10),1)
  gate=self.confidence_gate(quality) if self.mode in ('plddt_gate','dual_patch') else torch.ones_like(quality[:,:1])
  return gate*pooled,{'gate':gate,'quality':quality}

class ResidueLateFusion(nn.Module):
 def __init__(self,seq_dimension=1280,residue_dimension=152,hidden_dimension=512,representation_dimension=256,dropout=.15,pooling='attention'):
  super().__init__(); self.sequence=nn.Sequential(nn.LayerNorm(seq_dimension),nn.Linear(seq_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,representation_dimension),nn.GELU())
  self.structure=ResidueStructureEncoder(residue_dimension,hidden_dimension,representation_dimension,dropout,pooling)
  self.head=nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def forward(self,sequence,residue,plddt,patch,mask,structure_dropout=0.):
  s=self.sequence(sequence); z,aux=self.structure(residue,plddt,patch,mask)
  if self.training and structure_dropout>0:
   keep=(torch.rand(len(z),1,device=z.device)>=structure_dropout); z=z*keep
  return self.head(torch.cat((s,z),1)).squeeze(-1),aux
