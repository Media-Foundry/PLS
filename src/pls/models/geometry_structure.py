"""Pure-PyTorch CA-kNN message passing for solubility structure encoding."""
from __future__ import annotations
import torch
from torch import nn

class KNNMessageLayer(nn.Module):
 def __init__(self,dimension=256,rbf_dimension=16,dropout=.1,use_sequence_separation=False):
  super().__init__();self.use_sequence_separation=use_sequence_separation;self.register_buffer('centers',torch.linspace(0,20,rbf_dimension));self.scale=20/(rbf_dimension-1);edge_dimension=dimension+rbf_dimension+(3 if use_sequence_separation else 0)
  self.message=nn.Sequential(nn.LayerNorm(edge_dimension),nn.Linear(edge_dimension,dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(dimension,dimension));self.attention=nn.Linear(edge_dimension,1);self.update=nn.Sequential(nn.LayerNorm(dimension*2),nn.Linear(dimension*2,dimension),nn.GELU(),nn.Dropout(dropout))
 def forward(self,z,neighbors,distances,mask,confidence=None):
  batch=torch.arange(len(z),device=z.device)[:,None,None];neighbor_indices=neighbors.long();neighbor=z[batch,neighbor_indices];pairwise=neighbor_indices[..., :,None]==neighbor_indices[...,None,:];duplicate=torch.tril(pairwise,diagonal=-1).any(-1);valid=mask[batch,neighbor_indices]&mask[:,:,None]&(distances>1e-6)&torch.isfinite(distances)&~duplicate;rbf=torch.exp(-((distances[...,None]-self.centers)/self.scale).square())*valid[...,None];parts=[neighbor-z[:,:,None,:],rbf]
  if self.use_sequence_separation:
   center=torch.arange(z.shape[1],device=z.device)[None,:,None];delta=neighbor_indices-center;length=mask.sum(1).clamp_min(1)[:,None,None];absolute=delta.abs();parts.append(torch.stack((delta/length,torch.log1p(absolute)/torch.log1p(length),(absolute<=4).to(z.dtype)),-1))
  edge=torch.cat(parts,-1);edge_product=(confidence[:,:,None] if confidence is not None else torch.ones_like(distances))*(confidence[batch,neighbor_indices] if confidence is not None else torch.ones_like(distances));edge_gate=torch.sqrt(edge_product.clamp_min(1e-8))*valid;message=self.message(edge)*edge_gate[...,None];logits=self.attention(edge).squeeze(-1)+edge_gate.clamp_min(1e-6).log();weights=torch.softmax(logits.masked_fill(~valid,-1e4),2)*valid;weights=weights/weights.sum(2,keepdim=True).clamp_min(1e-8);aggregate=(message*weights[...,None]).sum(2);node_gate=confidence if confidence is not None else mask;return z+self.update(torch.cat((z,aggregate),-1))*node_gate[...,None]

class GeometryLateFusion(nn.Module):
 def __init__(self,seq_dimension=1280,input_dimension=152,hidden_dimension=256,representation_dimension=256,dropout=.15,layers=1,pooling='attention',global_dimension=0,use_sequence_separation=False,fusion='concat',confidence_mode='legacy',residue_sequence_dimension=0,rsa_mean=0.,rsa_std=1.):
  super().__init__();self.pooling=pooling;self.fusion=fusion;self.confidence_mode=confidence_mode;self.residue_sequence_dimension=residue_sequence_dimension;self.rsa_mean=float(rsa_mean);self.rsa_std=float(rsa_std)
  if fusion=='residue_aligned_sparse' and residue_sequence_dimension<=0:raise ValueError('residue_aligned_sparse requires residue_sequence_dimension > 0')
  structure_input_dimension=input_dimension-residue_sequence_dimension if fusion=='residue_aligned_sparse' else input_dimension
  if structure_input_dimension<=0:raise ValueError('structure input dimension must be positive')
  self.sequence=nn.Sequential(nn.LayerNorm(seq_dimension),nn.Linear(seq_dimension,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,representation_dimension),nn.GELU());self.input=nn.Sequential(nn.LayerNorm(structure_input_dimension),nn.Linear(structure_input_dimension,hidden_dimension),nn.GELU());self.layers=nn.ModuleList([KNNMessageLayer(hidden_dimension,16,dropout,use_sequence_separation) for _ in range(layers)]);self.project=nn.Linear(hidden_dimension,representation_dimension);self.attention=nn.Linear(representation_dimension+1,1);self.patch_attention=nn.Sequential(nn.Linear(representation_dimension+5,64),nn.GELU(),nn.Linear(64,1));self.merge=nn.Sequential(nn.Linear(representation_dimension*2,representation_dimension),nn.GELU());self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid());self.global_encoder=nn.Sequential(nn.LayerNorm(global_dimension),nn.Linear(global_dimension,representation_dimension),nn.GELU(),nn.Dropout(dropout)) if global_dimension else None
  if confidence_mode=='propagated_moe':self.residue_confidence=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid());self.protein_confidence=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid());self.sequence_head=nn.Sequential(nn.LayerNorm(representation_dimension),nn.Linear(representation_dimension,representation_dimension//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension//2,1))
  if fusion in ('cross_attention','global_query_pooling'):self.cross_query=nn.Linear(representation_dimension,representation_dimension,bias=False);self.cross_key=nn.Linear(representation_dimension,representation_dimension,bias=False);self.cross_scale=representation_dimension**-.5
  if fusion=='residue_aligned_sparse':
   self.residue_sequence=nn.Sequential(nn.LayerNorm(residue_sequence_dimension),nn.Linear(residue_sequence_dimension,representation_dimension),nn.GELU())
   self.aligned_fusion=nn.Sequential(nn.LayerNorm(representation_dimension*4+1),nn.Linear(representation_dimension*4+1,representation_dimension),nn.GELU(),nn.Dropout(dropout))
   self.local_sequence=nn.Conv1d(representation_dimension,representation_dimension,5,padding=2,groups=representation_dimension)
   self.local_mix=nn.Sequential(nn.LayerNorm(representation_dimension*2),nn.Linear(representation_dimension*2,representation_dimension),nn.GELU())
   self.aligned_graph=KNNMessageLayer(representation_dimension,16,dropout,use_sequence_separation)
   self.aligned_attention=nn.Linear(representation_dimension+1,1)
  branches=(5 if fusion in ('cross_attention','global_query_pooling','residue_aligned_sparse') else 2)+(1 if global_dimension else 0);self.head=nn.Sequential(nn.LayerNorm(representation_dimension*branches),nn.Linear(representation_dimension*branches,representation_dimension),nn.GELU(),nn.Dropout(dropout),nn.Linear(representation_dimension,1))
 def residue_rsa(self,residue):return (residue[...,63]*self.rsa_std+self.rsa_mean).clamp(0,1)
 def forward(self,sequence,residue,plddt,patch,mask,neighbors,distances,global_features=None):
  confidence=None
  if self.confidence_mode=='propagated_moe':
   batch=torch.arange(len(residue),device=residue.device)[:,None,None];valid=mask[batch,neighbors.long()]&mask[:,:,None]&(distances>1e-6);neighbor_conf=(plddt[batch,neighbors.long()]*valid).sum(2)/valid.sum(2).clamp_min(1);rsa=self.residue_rsa(residue);packing=residue[...,141] if residue.shape[-1]>141 else torch.zeros_like(plddt);confidence=self.residue_confidence(torch.stack((plddt,neighbor_conf,rsa,packing),-1)).squeeze(-1)*plddt*mask
  structure_residue=residue[...,:-self.residue_sequence_dimension] if self.fusion=='residue_aligned_sparse' else residue
  z=self.input(structure_residue)*(confidence[...,None] if confidence is not None else mask[...,None])
  for layer in self.layers:z=layer(z,neighbors,distances,mask,confidence)
  z=self.project(z);count=mask.sum(1).clamp_min(1);logits=self.attention(torch.cat((z,plddt[...,None]),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);weight=torch.softmax(logits,1);pooled=(z*weight[...,None]).sum(1)
  if self.pooling=='dual_patch':
   logits=self.patch_attention(torch.cat((z,patch),-1)).squeeze(-1).masked_fill(~mask,-torch.inf);pw=torch.softmax(logits,1);pooled=self.merge(torch.cat((pooled,(z*pw[...,None]).sum(1)),1))
  quality=torch.stack(((plddt*mask).sum(1)/count,((plddt<.7)&mask).sum(1)/count,torch.log1p(count)/10),1);legacy_gate=self.gate(quality);seq=self.sequence(sequence)
  if self.confidence_mode=='propagated_moe':protein_quality=torch.cat((quality,(confidence.sum(1)/count)[:,None],((rsa*mask).sum(1)/count)[:,None]),1);protein_gate=quality[:,0:1]*self.protein_confidence(protein_quality)
  else:protein_gate=legacy_gate;pooled=pooled*legacy_gate
  if self.fusion in ('cross_attention','global_query_pooling'):
   cross_logits=(self.cross_key(z)*self.cross_query(seq)[:,None,:]).sum(-1)*self.cross_scale;cross_logits=cross_logits.masked_fill(~mask,-torch.inf);cross=(z*torch.softmax(cross_logits,1)[...,None]).sum(1);branches=[seq,pooled,cross,seq*cross,(seq-cross).abs()]
  elif self.fusion=='residue_aligned_sparse':
   sequence_tokens=self.residue_sequence(residue[...,-self.residue_sequence_dimension:])*mask[...,None]
   residue_gate=confidence if confidence is not None else mask.to(z.dtype)
   aligned=self.aligned_fusion(torch.cat((sequence_tokens,z,sequence_tokens*z,(sequence_tokens-z).abs(),residue_gate[...,None]),-1))*residue_gate[...,None]
   local=self.local_sequence(aligned.transpose(1,2)).transpose(1,2)*mask[...,None]
   aligned=self.local_mix(torch.cat((aligned,local),-1))*residue_gate[...,None]
   aligned=self.aligned_graph(aligned,neighbors,distances,mask,confidence)
   aligned_logits=self.aligned_attention(torch.cat((aligned,plddt[...,None]),-1)).squeeze(-1).masked_fill(~mask,-torch.inf)
   aligned_pooled=(aligned*torch.softmax(aligned_logits,1)[...,None]).sum(1)
   branches=[seq,pooled,aligned_pooled,seq*aligned_pooled,(seq-aligned_pooled).abs()]
  else:branches=[seq,pooled]
  if self.confidence_mode=='propagated_moe':branches=[branches[0]]+[value*protein_gate for value in branches[1:]]
  if self.global_encoder is not None:branches.append(self.global_encoder(global_features)*protein_gate)
  joint=self.head(torch.cat(branches,1)).squeeze(-1)
  if self.confidence_mode=='propagated_moe':sequence_only=self.sequence_head(seq).squeeze(-1);g=protein_gate.squeeze(-1);return (1-g)*sequence_only+g*joint
  return joint
