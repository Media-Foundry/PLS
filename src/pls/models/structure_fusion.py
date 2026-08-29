"""Fusion alternatives for frozen sequence and pooled structure descriptors."""
from __future__ import annotations
import torch
from torch import nn

def block(inp,hid,out,drop):
 return nn.Sequential(nn.LayerNorm(inp),nn.Linear(inp,hid),nn.GELU(),nn.Dropout(drop),nn.Linear(hid,out),nn.GELU(),nn.Dropout(drop))

class EarlyConcat(nn.Module):
 def __init__(self,seq_dim,struct_dim,hid,rep,drop): super().__init__(); self.net=nn.Sequential(block(seq_dim+struct_dim,hid,rep,drop),nn.Linear(rep,1))
 def forward(self,x): return self.net(x).squeeze(-1)

class LateFusion(nn.Module):
 def __init__(self,seq_dim,struct_dim,hid,rep,drop):
  super().__init__(); self.seq_dim=seq_dim; self.seq=block(seq_dim,hid,rep,drop); self.struct=block(struct_dim,hid,rep,drop); self.head=nn.Sequential(nn.LayerNorm(rep*2),nn.Linear(rep*2,rep),nn.GELU(),nn.Dropout(drop),nn.Linear(rep,1))
 def forward(self,x): return self.head(torch.cat((self.seq(x[:,:self.seq_dim]),self.struct(x[:,self.seq_dim:])),1)).squeeze(-1)

class GatedResidual(nn.Module):
 def __init__(self,seq_dim,struct_dim,hid,rep,drop):
  super().__init__(); self.seq_dim=seq_dim; self.seq=block(seq_dim,hid,rep,drop); self.struct=block(struct_dim,hid,rep,drop); self.gate=nn.Sequential(nn.Linear(rep*2,rep),nn.Sigmoid()); self.head=nn.Linear(rep,1)
 def forward(self,x):
  s=self.seq(x[:,:self.seq_dim]); z=self.struct(x[:,self.seq_dim:]); return self.head(s+self.gate(torch.cat((s,z),1))*z).squeeze(-1)

class FiLMFusion(nn.Module):
 def __init__(self,seq_dim,struct_dim,hid,rep,drop):
  super().__init__(); self.seq_dim=seq_dim; self.seq=block(seq_dim,hid,rep,drop); self.condition=nn.Sequential(block(struct_dim,hid,rep,drop),nn.Linear(rep,rep*2)); self.head=nn.Linear(rep,1)
 def forward(self,x):
  s=self.seq(x[:,:self.seq_dim]); gamma,beta=self.condition(x[:,self.seq_dim:]).chunk(2,1); return self.head(s*(1+.1*torch.tanh(gamma))+beta).squeeze(-1)

def build_fusion(name,seq_dim,struct_dim,hid,rep,drop):
 classes={'early':EarlyConcat,'late':LateFusion,'gated_residual':GatedResidual,'film':FiLMFusion}
 if struct_dim<=0 and name!='early': raise ValueError(f'{name} requires structure features')
 return classes[name](seq_dim,struct_dim,hid,rep,drop)
