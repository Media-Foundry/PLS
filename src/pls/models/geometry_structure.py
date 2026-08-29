"""Pure-PyTorch CA-kNN message passing for solubility structure encoding."""
from __future__ import annotations
import torch
from torch import nn

class KNNMessageLayer(nn.Module):
 def __init__(self,dimension=256,rbf_dimension=16,dropout=.1,use_sequence_separation=False):
  super().__init__();self.use_sequence_separation=use_sequence_separation;self.register_buffer('centers',torch.linspace(0,20,rbf_dimension));self.scale=20/(rbf_dimension-1);edge_dimension=dimension+rbf_dimension+(3 if use_sequence_separation else 0)
  self.message=nn.Sequential(nn.LayerNorm(edge_dimension),nn.Linear(edge_dimension,dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(dimension,dimension));self.attention=nn.Linear(edge_dimension,1);self.update=nn.Sequential(nn.LayerNorm(dimension*2),nn.Linear(dimension*2,dimension),nn.GELU(),nn.Dropout(dropout))
 def forward(self,z,neighbors,distances,mask):
  batch=torch.arange(len(z),device=z.device)[:,None,None];neighbor_indices=neighbors.long();neighbor=z[batch,neighbor_indices];rbf=torch.exp(-((distances[...,None]-self.centers)/self.scale).square());parts=[neighbor-z[:,:,None,:],rbf]
  if self.use_sequence_separation:
   center=torch.arange(z.shape[1],device=z.device)[None,:,None];delta=neighbor_indices-center;length=mask.sum(1).clamp_min(1)[:,None,None];absolute=delta.abs();parts.append(torch.stack((delta/length,torch.log1p(absolute)/torch.log1p(length),(absolute<=4).to(z.dtype)),-1))
  edge=torch.cat(parts,-1);message=self.message(edge);logits=self.attention(edge).squeeze(-1);weights=torch.softmax(logits,2);aggregate=(message*weights[...,None]).sum(2);return z+self.update(torch.cat((z,aggregate),-1))*mask[...,None]

class GeometryLateFusion(nn.Module):
 def __init__(self,seq_dimension=1280,input_dimension=152,hidden_dimension=256,representation_dimension=256,dropout=.15,layers=1,pooling='attention',global_dimension=0,use_sequence_separation=False,fusion='concat'):
  super().__init__();self.pooling=pooling;self.fusion=fusion;self.sequence=nn.Sequential(nn.LayerNorm(seq_dimension),nn.Linear(seq_dimension,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,representation_dimension),nn.GELU());self.input=nn.Sequential(nn.LayerNorm(input_dimension),nn.Linear(input_dimension,hidden_dimension),nn.GELU());self.layers=nn.ModuleList([KNNMessageLayer(hidden_dimension,16,dropout,use_sequence_separation) for _ in range(layers)]);self.project=nn.Linear(hidden_dimension,representation_dimension);self.attention=nn.Linear(representation_dimension+1,1);self.patch_attention=nn.Sequential(nn.Linear(representation_dimension+5,64),nn.GELU(),nn.Linear(64,1));self.merge=nn.Sequential(nn.Linear(representation_dimension*2,representation_dimension),nn.GELU());self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid());self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) if global_dimension else None
  if fusion=='cross_attention':self.cross_query=nn.Linear(representation_dimension,representation_dimension,bias=False);self.cross_key=nn.Linear(representation_dimension,representation_dimension,bias=False);self.cross_scale=representation_dimension**-.5
  branches=(5 if fusion=='cross_attention' else 2)+(1 if global_dimension else 0);self.head=nn.Sequential(nn.LayerNorm(representation_dimension*branches),nn.Linear(representation_dimension*branches,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def forward(self,sequence,residue,plddt,patch,mask,neighbors,distances,global_features=None):
  z=self.input(residue)
  for layer in self.layers:z=layer(z,neighbors,distances,mask)
  z=self.project(z);count=mask.sum(1).clamp_min(1);logits=self.attention(torch.cat((z,plddt[...,None]),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);weight=torch.softmax(logits,1);pooled=(z*weight[...,None]).sum(1)
  if self.pooling=='dual_patch':
   logits=self.patch_attention(torch.cat((z,patch),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);pw=torch.softmax(logits,1);pooled=self.merge(torch.cat((pooled,(z*pw[...,None]).sum(1)),1))
  quality=torch.stack(((plddt*mask).sum(1)/count,((plddt<.7)&mask).sum(1)/count,torch.log1p(count)/10),1);pooled=pooled*self.gate(quality);seq=self.sequence(sequence)
  if self.fusion=='cross_attention':
   cross_logits=(self.cross_key(z)*self.cross_query(seq)[:,None,:]).sum(-1)*self.cross_scale;cross_logits=cross_logits.masked_fill(~mask,-torch.inf);cross=(z*torch.softmax(cross_logits,1)[...,None]).sum(1);branches=[seq,pooled,cross,seq*cross,(seq-cross).abs()]
  else:branches=[seq,pooled]
  if self.global_encoder is not None:branches.append(self.global_encoder(global_features))
  return self.head(torch.cat(branches,1)).squeeze(-1)
