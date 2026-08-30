"""Frozen residue-PLM pooling models for quantitative solubility regression."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
class LightweightTCN(nn.Module):
 def __init__(self,dimension,bottleneck=64,dropout=.1):
  super().__init__();self.down=nn.Conv1d(dimension,bottleneck,1);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(bottleneck,bottleneck,3,padding=dilation,dilation=dilation,groups=bottleneck),nn.GELU(),nn.Conv1d(bottleneck,bottleneck,1),nn.Dropout(dropout)) for dilation in (1,2,4)]);self.up=nn.Conv1d(bottleneck,dimension,1)
 def forward(self,z,mask):
  channel_mask=mask[:,None].to(z.dtype);x=self.down((z*mask[...,None]).transpose(1,2))*channel_mask
  for block in self.blocks:x=(x+block(x))*channel_mask
  return z+self.up(x).transpose(1,2)*mask[...,None]
class ShiftTCN(nn.Module):
 def __init__(self,dimension,bottleneck=128,dropout=.1,dilations=(1,2,4)):
  super().__init__();self.down=nn.Linear(dimension,bottleneck);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(bottleneck*3),nn.Linear(bottleneck*3,bottleneck),nn.GELU(),nn.Dropout(dropout)) for _ in dilations]);self.up=nn.Linear(bottleneck,dimension);self.dilations=tuple(dilations)
 def forward(self,z,mask):
  x=self.down(z)*mask[...,None]
  for dilation,block in zip(self.dilations,self.blocks):
   left=torch.roll(x,dilation,1);right=torch.roll(x,-dilation,1);left[:,:dilation]=0;right[:,-dilation:]=0;x=(x+block(torch.cat((left,x,right),-1)))*mask[...,None]
  return z+self.up(x)*mask[...,None]
class GatedShiftTCN(nn.Module):
 def __init__(self,dimension,bottleneck=192,dropout=.1,dilations=(1,2,4,8,16,32,64)):
  super().__init__();self.down=nn.Linear(dimension,bottleneck);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(bottleneck*3),nn.Linear(bottleneck*3,bottleneck*2)) for _ in dilations]);self.residual=nn.ModuleList([nn.Sequential(nn.Linear(bottleneck,bottleneck),nn.Dropout(dropout)) for _ in dilations]);self.up=nn.Linear(bottleneck,dimension);self.dilations=tuple(dilations)
 def forward(self,z,mask):
  x=self.down(z)*mask[...,None]
  for dilation,block,residual in zip(self.dilations,self.blocks,self.residual):
   left=torch.roll(x,dilation,1);right=torch.roll(x,-dilation,1);left[:,:dilation]=0;right[:,-dilation:]=0;filt,gate=block(torch.cat((left,x,right),-1)).chunk(2,-1);x=(x+residual(torch.tanh(filt)*torch.sigmoid(gate)))*mask[...,None]
  return z+self.up(x)*mask[...,None]
class MaskedBiLSTMTextCNN(nn.Module):
 def __init__(self,dimension,dropout=.2):
  super().__init__();hidden=max(64,dimension//2);self.forward_lstm=nn.LSTM(dimension,hidden,batch_first=True);self.reverse_lstm=nn.LSTM(dimension,hidden,batch_first=True);lstm_dimension=hidden*2;channels=dimension;self.convs=nn.ModuleList([nn.Conv1d(lstm_dimension,channels,k,padding=k//2) for k in (3,7,11)]);self.projection=nn.Sequential(nn.LayerNorm(channels*3),nn.Linear(channels*3,dimension),nn.GELU(),nn.Dropout(dropout))
 def forward(self,z,mask):
  lengths=mask.sum(1).to(torch.int64).clamp_min(1);positions=torch.arange(z.shape[1],device=z.device)[None];reverse_indices=(lengths[:,None]-1-positions).clamp_min(0);reverse_input=z.gather(1,reverse_indices[...,None].expand_as(z))*mask[...,None];forward,_=self.forward_lstm(z*mask[...,None]);reverse,_=self.reverse_lstm(reverse_input);reverse=reverse.gather(1,reverse_indices[...,None].expand(-1,-1,reverse.shape[-1]));encoded=torch.cat((forward,reverse),-1)*mask[...,None];channel_mask=mask[:,None];pooled=[]
  for conv in self.convs:
   values=conv(encoded.transpose(1,2)).masked_fill(~channel_mask,-torch.inf);pooled.append(values.amax(2))
  return self.projection(torch.cat(pooled,1))
class SegmentTransformerPooling(nn.Module):
 def __init__(self,dimension,window=32,layers=2,dropout=.1):
  super().__init__();self.window=int(window);self.token=nn.Sequential(nn.LayerNorm(dimension*2),nn.Linear(dimension*2,dimension),nn.GELU(),nn.Dropout(dropout));block=nn.TransformerEncoderLayer(dimension,4,dimension*2,dropout,batch_first=True,norm_first=True,activation='gelu');self.encoder=nn.TransformerEncoder(block,layers);self.query=nn.Linear(dimension,dimension,bias=False);self.key=nn.Linear(dimension,dimension,bias=False);self.scale=dimension**-.5
 def forward(self,z,mask,global_vector):
  batch,length,dimension=z.shape;segments=(length+self.window-1)//self.window;padded=segments*self.window-length;z=F.pad(z,(0,0,0,padded));mask=F.pad(mask,(0,padded));grouped=z.reshape(batch,segments,self.window,dimension);group_mask=mask.reshape(batch,segments,self.window);count=group_mask.sum(2).clamp_min(1)[...,None];mean=(grouped*group_mask[...,None]).sum(2)/count;maximum=grouped.masked_fill(~group_mask[...,None],-torch.inf).amax(2);segment_mask=group_mask.any(2);maximum=maximum.masked_fill(~segment_mask[...,None],0);tokens=self.token(torch.cat((mean,maximum),-1));position=torch.arange(segments,device=z.device,dtype=torch.float32)[:,None];frequency=torch.exp(torch.arange(0,dimension,2,device=z.device,dtype=torch.float32)*(-torch.log(torch.tensor(10000.,device=z.device))/dimension));encoding=torch.zeros(segments,dimension,device=z.device,dtype=torch.float32);encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency[:encoding[:,1::2].shape[1]]);tokens=(tokens+encoding.to(tokens.dtype)[None])*segment_mask[...,None];tokens=self.encoder(tokens,src_key_padding_mask=~segment_mask)*segment_mask[...,None];logits=(self.key(tokens)*self.query(global_vector)[:,None]).sum(-1)*self.scale;logits=logits.masked_fill(~segment_mask,-torch.inf);return (tokens*torch.softmax(logits,1)[...,None]).sum(1)
class ResidueSequenceRegressor(nn.Module):
 def __init__(self,global_dimension=1280,residue_dimension=256,hidden_dimension=256,representation_dimension=256,dropout=.2,pooling='attention',fusion='concat',global_segments=1,global_segment_fusion='weighted_sum',tcn_bottleneck=None,tcn_dilations=(1,2,4),segment_window=32,segment_layers=2,descriptor_dimension=0,physchem_experts=0,physchem_gate_temperature=1.,physchem_top_k=0,physchem_sparse_residual=0.,physchem_gate_context=False):
  super().__init__();self.pooling,self.fusion,self.global_segments,self.global_segment_fusion=pooling,fusion,global_segments,global_segment_fusion;self.descriptor_dimension=int(descriptor_dimension)
  if pooling in ('conditioned_attention','tcn_conditioned_attention','shift_tcn_conditioned_attention','gated_shift_tcn_conditioned_attention','segment_transformer') and global_segments>1 and global_segment_fusion!='weighted_sum':raise ValueError('conditioned pooling requires weighted_sum segment fusion')
  if global_segment_fusion not in ('weighted_sum','concat','logit_mixture'):raise ValueError('global segment fusion must be weighted_sum, concat or logit_mixture')
  if global_segments==1 and global_segment_fusion!='weighted_sum':raise ValueError('specialized global segment fusion requires multiple segments')
  if global_segments>1 and global_segment_fusion=='concat' and fusion=='interaction':raise ValueError('interaction fusion requires equally sized global and residue representations')
  if global_segment_fusion=='logit_mixture' and fusion!='concat':raise ValueError('logit mixture currently supports concat fusion only')
  segmented_dimension=global_dimension-self.descriptor_dimension if global_segments>1 else global_dimension
  if segmented_dimension%global_segments:raise ValueError('PLM global dimension must be divisible by global segments after excluding descriptors')
  if global_segments==1:self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,hidden_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dimension,representation_dimension),nn.GELU());self.global_segment_encoders=None
  else:
   segment_dimension=segmented_dimension//global_segments;self.global_encoder=None;self.global_segment_encoders=nn.ModuleList([nn.Sequential(nn.LayerNorm(segment_dimension),nn.Linear(segment_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) for _ in range(global_segments)]);self.global_segment_logits=nn.Parameter(torch.zeros(global_segments))
  self.residue_encoder=nn.Sequential(nn.LayerNorm(residue_dimension),nn.Linear(residue_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.local=nn.Sequential(nn.Conv1d(representation_dimension,representation_dimension,5,padding=2,groups=representation_dimension),nn.GELU(),nn.Conv1d(representation_dimension,representation_dimension,1),nn.Dropout(dropout));self.multiscale_logits=nn.Parameter(torch.zeros(4)) if pooling=='multiscale_attention' else None;self.attention=nn.Linear(representation_dimension,1);self.multi_attention=nn.Linear(representation_dimension,4);self.statistics_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));self.statistic_encoders=nn.ModuleList([nn.Sequential(nn.LayerNorm(representation_dimension),nn.Linear(representation_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) for _ in range(4)]) if pooling=='gated_statistics_attention' else None;self.statistic_logits=nn.Parameter(torch.zeros(4)) if pooling=='gated_statistics_attention' else None;self.multi_projection=nn.Sequential(nn.LayerNorm(representation_dimension*4),nn.Linear(representation_dimension*4,representation_dimension),nn.GELU(),nn.Dropout(dropout));global_output_dimension=representation_dimension*(global_segments if global_segments>1 and global_segment_fusion=='concat' else 1);head_dimension=representation_dimension*4 if fusion=='interaction' else global_output_dimension+representation_dimension;self.head=None if global_segment_fusion=='logit_mixture' or physchem_experts else nn.Sequential(nn.LayerNorm(head_dimension),nn.Linear(head_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1));self.expert_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1)) for _ in range(global_segments)]) if global_segment_fusion=='logit_mixture' else None;self.physchem_experts=int(physchem_experts);self.physchem_gate_temperature=float(physchem_gate_temperature);self.physchem_top_k=int(physchem_top_k);self.physchem_sparse_residual=float(physchem_sparse_residual);self.physchem_gate_context=bool(physchem_gate_context);gate_dimension=descriptor_dimension+(representation_dimension if self.physchem_gate_context else 0);self.physchem_gate=nn.Sequential(nn.LayerNorm(gate_dimension),nn.Linear(gate_dimension,32),nn.GELU(),nn.Linear(32,physchem_experts)) if physchem_experts else None;self.physchem_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(head_dimension),nn.Linear(head_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1)) for _ in range(physchem_experts)]) if physchem_experts else None;self.last_expert_logits=None;self.last_expert_weights=None
  if self.physchem_gate_temperature<=0:raise ValueError('physchem gate temperature must be positive')
  if self.physchem_top_k<0 or (self.physchem_top_k and self.physchem_top_k>self.physchem_experts):raise ValueError('physchem top-k must be between 0 and the expert count')
  if not 0<=self.physchem_sparse_residual<=1:raise ValueError('physchem sparse residual must be between 0 and 1')
  conditioned=pooling in ('conditioned_attention','tcn_conditioned_attention','shift_tcn_conditioned_attention','gated_shift_tcn_conditioned_attention');self.condition_query=nn.Linear(representation_dimension,representation_dimension,bias=False) if conditioned else None;self.condition_key=nn.Linear(representation_dimension,representation_dimension,bias=False) if conditioned else None;self.condition_scale=representation_dimension**-.5;shift_width=int(tcn_bottleneck or min(128,representation_dimension//2));self.tcn=LightweightTCN(representation_dimension,min(64,representation_dimension//4),dropout) if pooling=='tcn_conditioned_attention' else (ShiftTCN(representation_dimension,shift_width,dropout,tcn_dilations) if pooling=='shift_tcn_conditioned_attention' else (GatedShiftTCN(representation_dimension,shift_width,dropout,tcn_dilations) if pooling=='gated_shift_tcn_conditioned_attention' else None));self.bilstm_textcnn=MaskedBiLSTMTextCNN(representation_dimension,dropout) if pooling=='bilstm_textcnn' else None;self.segment_transformer=SegmentTransformerPooling(representation_dimension,segment_window,segment_layers,dropout) if pooling=='segment_transformer' else None
 def forward(self,global_embedding,residues,mask):
  z=self.residue_encoder(residues)*mask[...,None]
  encoded=None
  if self.global_segments==1:global_encoded=self.global_encoder(global_embedding)
  else:
   plm_embedding=global_embedding[:,:-self.descriptor_dimension] if self.descriptor_dimension else global_embedding;segments=plm_embedding.chunk(self.global_segments,1);encoded=torch.stack([encoder(segment) for encoder,segment in zip(self.global_segment_encoders,segments)],1);global_encoded=(encoded*torch.softmax(self.global_segment_logits,0)[None,:,None]).sum(1) if self.global_segment_fusion=='weighted_sum' else None
  if self.pooling in ('tcn_conditioned_attention','shift_tcn_conditioned_attention','gated_shift_tcn_conditioned_attention'):z=self.tcn(z,mask)
  if self.pooling=='conv_attention':z=z+self.local(z.transpose(1,2)).transpose(1,2)*mask[...,None]
  if self.pooling=='local_attention':
   left=torch.roll(z,1,1);right=torch.roll(z,-1,1);left[:,0]=0;right[:,-1]=0;z=z+(left+right)*.5*mask[...,None]
  if self.pooling=='multiscale_attention':
   values=[z];masked=z*mask[...,None];mask_channel=mask[:,None].to(z.dtype)
   for kernel in (3,7,15):
    numerator=F.avg_pool1d(masked.transpose(1,2),kernel,1,kernel//2)*kernel;denominator=F.avg_pool1d(mask_channel,kernel,1,kernel//2).transpose(1,2)*kernel;values.append(numerator.transpose(1,2)/denominator.clamp_min(1))
   z=(torch.stack(values,1)*torch.softmax(self.multiscale_logits,0)[None,:,None,None]).sum(1)*mask[...,None]
  count=mask.sum(1).clamp_min(1)[:,None];mean=(z*mask[...,None]).sum(1)/count
  if self.pooling=='bilstm_textcnn':pooled=self.bilstm_textcnn(z,mask)
  elif self.pooling=='segment_transformer':pooled=self.segment_transformer(z,mask,global_encoded)
  elif self.pooling=='mean':pooled=mean
  elif self.pooling in ('multihead_attention','multi_query_pooling'):
   logits=self.multi_attention(z).masked_fill(~mask[...,None],-torch.inf);weights=torch.softmax(logits,1);pooled=self.multi_projection(torch.einsum('bnh,bnd->bhd',weights,z).flatten(1))
  elif self.pooling in ('conditioned_attention','tcn_conditioned_attention','shift_tcn_conditioned_attention','gated_shift_tcn_conditioned_attention'):
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
   if self.global_segment_fusion=='logit_mixture':
    expert_logits=torch.stack([head(torch.cat((encoded[:,i],pooled),1)).squeeze(-1) for i,head in enumerate(self.expert_heads)],1);self.last_expert_logits=expert_logits;return (expert_logits*torch.softmax(self.global_segment_logits,0)[None]).sum(1)
   global_encoded=encoded.flatten(1) if self.global_segment_fusion=='concat' else (encoded*torch.softmax(self.global_segment_logits,0)[None,:,None]).sum(1)
  features=(global_encoded,pooled,global_encoded*pooled,(global_encoded-pooled).abs()) if self.fusion=='interaction' else (global_encoded,pooled)
  if self.physchem_experts:
   combined=torch.cat(features,1);self.last_expert_logits=torch.stack([head(combined).squeeze(-1) for head in self.physchem_heads],1);gate_features=torch.cat((global_embedding[:,-self.descriptor_dimension:],global_encoded),1) if self.physchem_gate_context else global_embedding[:,-self.descriptor_dimension:];soft_weights=torch.softmax(self.physchem_gate(gate_features)/self.physchem_gate_temperature,1);weights=soft_weights
   if self.physchem_top_k:
    selected=torch.topk(weights,self.physchem_top_k,dim=1).indices;mask=torch.zeros_like(weights).scatter_(1,selected,1);weights=weights*mask;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8);weights=(1-self.physchem_sparse_residual)*weights+self.physchem_sparse_residual*soft_weights
   self.last_expert_weights=weights;return (self.last_expert_logits*weights).sum(1)
  return self.head(torch.cat(features,1)).squeeze(-1)
