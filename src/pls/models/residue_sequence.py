"""Frozen residue-PLM pooling models for quantitative solubility regression."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
class ResidueSequenceRegressor(nn.Module):
 def __init__(self,global_dimension=1280,residue_dimension=256,hidden_dimension=256,representation_dimension=256,dropout=.2,pooling='attention',fusion='concat',global_segments=1,global_segment_fusion='weighted_sum'):
  super().__init__();self.pooling,self.fusion,self.global_segments,self.global_segment_fusion=pooling,fusion,global_segments,global_segment_fusion
  if global_segment_fusion not in ('weighted_sum','concat','logit_mixture'):raise ValueError('global segment fusion must be weighted_sum, concat or logit_mixture')
  if global_segments==1 and global_segment_fusion!='weighted_sum':raise ValueError('specialized global segment fusion requires multiple segments')
  if global_segments>1 and global_segment_fusion=='concat' and fusion=='interaction':raise ValueError('interaction fusion requires equally sized global and residue representations')
  if global_segment_fusion=='logit_mixture' and fusion!='concat':raise ValueError('logit mixture currently supports concat fusion only')
  if global_dimension%global_segments:raise ValueError('global dimension must be divisible by global segments')
  if global_segments==1:self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,representation_dimension),nn.GELU());self.global_segment_encoders=None
  else:
   segment_dimension=global_dimension//global_segments;self.global_encoder=None;self.global_segment_encoders=nn.ModuleList([nn.Sequential(nn.LayerNorm(segment_dimension),nn.Linear(segment_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) for _ in range(global_segments)]);self.global_segment_logits=nn.Parameter(torch.zeros(global_segments))
  self.residue_encoder=nn.Sequential(nn.LayerNorm(residue_dimension),nn.Linear(residue_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.local=nn.Sequential(nn.Conv1d(representation_dimension,representation_dimension,5,padding=2,groups=representation_dimension),nn.GELU(),nn.Conv1d(representation_dimension,representation_dimension,1),nn.Dropout(dropout));self.multiscale_logits=nn.Parameter(torch.zeros(4)) if pooling=='multiscale_attention' else None;self.attention=nn.Linear(representation_dimension,1);self.multi_attention=nn.Linear(representation_dimension,4);self.statistics_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.statistic_encoders=nn.ModuleList([nn.Sequential(nn.LayerNorm(representation_dimension),nn.Linear(representation_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) for _ in range(4)]) if pooling=='gated_statistics_attention' else None;self.statistic_logits=nn.Parameter(torch.zeros(4)) if pooling=='gated_statistics_attention' else None;self.multi_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));global_output_dimension=representation_dimension*(global_segments if global_segments>1 and global_segment_fusion=='concat' else 1);head_dimension=representation_dimension*4 if fusion=='interaction' else global_output_dimension+representation_dimension;self.head=None if global_segment_fusion=='logit_mixture' else nn.Sequential(nn.LayerNorm(head_dimension),nn.Linear(head_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1));self.expert_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1)) for _ in range(global_segments)]) if global_segment_fusion=='logit_mixture' else None;self.last_expert_logits=None
  self.condition_query=nn.Linear(representation_dimension,representation_dimension,bias=False) if pooling=='conditioned_attention' else None;self.condition_key=nn.Linear(representation_dimension,representation_dimension,bias=False) if pooling=='conditioned_attention' else None;self.condition_scale=representation_dimension**-.5
 def forward(self,global_embedding,residues,mask):
  z=self.residue_encoder(residues)
  global_encoded=self.global_encoder(global_embedding) if self.global_segments==1 else None
  if self.pooling=='conv_attention':z=z+self.local(z.transpose(1,2)).transpose(1,2)*mask[...,None]
  if self.pooling=='local_attention':
   left=torch.roll(z,1,1);right=torch.roll(z,-1,1);left[:,0]=0;right[:,-1]=0;z=z+(left+right)*.5*mask[...,None]
  if self.pooling=='multiscale_attention':
   values=[z];masked=z*mask[...,None];mask_channel=mask[:,None].to(z.dtype)
   for kernel in (3,7,15):
    numerator=F.avg_pool1d(masked.transpose(1,2),kernel,1,kernel//2)*kernel;denominator=F.avg_pool1d(mask_channel,kernel,1,kernel//2).transpose(1,2)*kernel;values.append(numerator.transpose(1,2)/denominator.clamp_min(1))
   z=(torch.stack(values,1)*torch.softmax(self.multiscale_logits,0)[None,:,None,None]).sum(1)*mask[...,None]
  count=mask.sum(1).clamp_min(1)[:,None];mean=(z*mask[...,None]).sum(1)/count
  if self.pooling=='mean':pooled=mean
  elif self.pooling=='multihead_attention':
   logits=self.multi_attention(z).masked_fill(~mask[...,None],-torch.inf);weights=torch.softmax(logits,1);pooled=self.multi_projection(torch.einsum('bnh,bnd->bhd',weights,z).flatten(1))
  elif self.pooling=='conditioned_attention':
   logits=(self.condition_key(z)*self.condition_query(global_encoded)[:,None]).sum(-1)*self.condition_scale;logits=logits.masked_fill(~mask,-torch.inf);pooled=(z*torch.softmax(logits,1)[...,None]).sum(1)
  else:
   logits=self.attention(z).squeeze(-1).masked_fill(~mask,-torch.inf);attention=(z*torch.softmax(logits,1)[...,None]).sum(1)
   if self.pooling in ('statistics_attention','gated_statistics_attention'):
    variance=((z-mean[:,None]).square()*mask[...,None]).sum(1)/count
    maximum=z.masked_fill(~mask[...,None],-torch.inf).amax(1)
    statistics=(mean,variance.clamp_min(0).sqrt(),maximum,attention)
    if self.pooling=='gated_statistics_attention':
     encoded_statistics=torch.stack([encoder(value) for encoder,value in zip(self.statistic_encoders,statistics)],1);pooled=(encoded_statistics*torch.softmax(self.statistic_logits,0)[None,:,None]).sum(1)
    else:pooled=self.statistics_projection(torch.cat(statistics,1))
   else:pooled=attention
  if self.global_segments>1:
   segments=global_embedding.chunk(self.global_segments,1);encoded=torch.stack([encoder(segment) for encoder,segment in zip(self.global_segment_encoders,segments)],1)
   if self.global_segment_fusion=='logit_mixture':
    expert_logits=torch.stack([head(torch.cat((encoded[:,i],pooled),1)).squeeze(-1) for i,head in enumerate(self.expert_heads)],1);self.last_expert_logits=expert_logits;return (expert_logits*torch.softmax(self.global_segment_logits,0)[None]).sum(1)
   global_encoded=encoded.flatten(1) if self.global_segment_fusion=='concat' else (encoded*torch.softmax(self.global_segment_logits,0)[None,:,None]).sum(1)
  features=(global_encoded,pooled,global_encoded*pooled,(global_encoded-pooled).abs()) if self.fusion=='interaction' else (global_encoded,pooled)
  return self.head(torch.cat(features,1)).squeeze(-1)
