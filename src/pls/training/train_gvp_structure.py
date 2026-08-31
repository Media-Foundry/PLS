"""Leakage-safe PDBSol training with equivariant GVP structure messages."""
from __future__ import annotations
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import binary_metrics
from pls.models.gvp_structure import GVPStructureFusion
from pls.training.train_residue_structure import Data,BalancedLengthBatchSampler
from pls.training.train_uesolds_residue import binary_rank_loss

def attach_vectors(base,vectors,coords):return (*base[:8],vectors,coords,base[-1])

class GVPData(Data):
 def __init__(self,*args,vector_dir,**kwargs):super().__init__(*args,**kwargs);self.vector_dir=vector_dir;self._vectors=self._coords=None;self.vector_residues=json.loads((vector_dir/'metadata.json').read_text())['residues']
 def __getitem__(self,i):
  base=super().__getitem__(i);j=self.rows[i][0];lo,hi=int(self.offsets[j]),int(self.offsets[j+1])
  if self._vectors is None:self._vectors=np.memmap(self.vector_dir/'vectors.f16',mode='r',dtype=np.float16,shape=(self.vector_residues,8,3));self._coords=np.memmap(self.vector_dir/'ca_coords.f32',mode='r',dtype=np.float32,shape=(self.vector_residues,3))
  return attach_vectors(base,torch.from_numpy(np.array(self._vectors[lo:hi],dtype=np.float32,copy=True)),torch.from_numpy(np.array(self._coords[lo:hi],copy=True)))
def collate(batch):
 n=max(len(v[1]) for v in batch);b=len(batch);dim=batch[0][1].shape[1];k=batch[0][4].shape[1];categories=batch[0][7].shape[1];residue=torch.zeros(b,n,dim);patch=torch.zeros(b,n,5);mask=torch.zeros(b,n,dtype=torch.bool);neighbors=torch.zeros(b,n,k,dtype=torch.long);distances=torch.zeros(b,n,k);components=torch.full((b,n,categories),-1,dtype=torch.long);vectors=torch.zeros(b,n,8,3);coords=torch.zeros(b,n,3)
 for i,(_,r,_,h,nb,ds,_,pc,v,c,_) in enumerate(batch):m=len(r);residue[i,:m]=r;patch[i,:m]=h;neighbors[i,:m]=nb;distances[i,:m]=ds;components[i,:m]=pc;vectors[i,:m]=v;coords[i,:m]=c;mask[i,:m]=1
 return torch.stack([v[0] for v in batch]),residue,vectors,coords,mask,neighbors,distances,patch,components,torch.stack([v[-1] for v in batch])
def infer(model,loader,device,amp=False):
 model.eval();truth=[];pred=[]
 with torch.inference_mode():
  for seq,res,vectors,coords,mask,neighbors,distances,patch,components,y in loader:
   values=[x.to(device,non_blocking=True) for x in (seq,res,vectors,coords,mask,neighbors,distances,patch,components)]
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):out=model(*values)
   truth.extend(y.tolist());pred.extend(out.float().cpu().tolist())
 return np.asarray(truth,np.float32),np.asarray(pred,np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());d=c['data'];mc=c['model'];tr=c['training']
 if c.get('evaluate_test',False):p.error('test evaluation is permanently disabled')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']):p.error('HIP device mismatch')
 seed=tr['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h:entities=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(entities)}
 rows={s:[] for s in ('train','validation')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']=='PDBSol_ProtSolM' and r['split'] in rows:rows[r['split']].append((index[r['sequence_sha256']],r['sequence_sha256'],float(r['target_value'])))
 status=np.load(d['structure_status'],mmap_mode='r');rows={s:[v for v in values if status[v[0]]==1] for s,values in rows.items()};esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r');stats=json.loads(Path(d['structure_stats']).read_text());mean=torch.tensor(stats['scalar_means']);std=torch.tensor(stats['scalar_stds']);compact=Path(d['compact_structure_dir']);geometry=Path(d['geometry_dir']);residue_esm=Path(d['residue_esm_dir']);vector_dir=Path(d['vector_dir']);surface_patches=Path(d['surface_patch_dir']) if d.get('surface_patch_dir') else None;sets={s:GVPData(v,esm,Path(d['structure_dir']),mean,std,compact,geometry,False,mc['neighbors'],residue_esm,None,None,None,surface_patches,vector_dir=vector_dir) for s,v in rows.items()};labels=np.asarray([v[2] for v in rows['train']]);lengths=[int(entities[v[0]]['length']) for v in rows['train']];sampler=BalancedLengthBatchSampler(labels,lengths,tr['batch_size'],seed,[v[0] for v in rows['train']]);train=DataLoader(sets['train'],batch_sampler=sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=True,pin_memory=True);validation=DataLoader(sets['validation'],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=True,pin_memory=True)
 residue_sequence_dimension=int(json.loads((residue_esm/'pca_metadata.json').read_text())['shape'][1]);residue_dimension=152+residue_sequence_dimension;device=torch.device('cuda:0');model=GVPStructureFusion(esm.shape[1],residue_dimension,mc['scalar_dimension'],mc['vector_dimension'],mc['representation_dimension'],mc['dropout'],mc['layers'],mc.get('fusion','interaction'),residue_sequence_dimension,stats['scalar_means'][1],stats['scalar_stds'][1],mc.get('surface_patches',False),mc.get('patch_spatial_layers',0),mc.get('cross_confidence_power',1.),mc.get('patch_self_edges',False)).to(device);decay=float(tr.get('ema_decay',0));ema=copy.deepcopy(model).eval().requires_grad_(False) if decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay'],fused=tr.get('fused_optimizer',False));writer=SummaryWriter(a.run_dir/'tensorboard');best=-1;best_by_metric={'auprc':-float('inf'),'mcc':-float('inf'),'brier':float('inf')};stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  started=time.monotonic();model.train();total=count=0
  for seq,res,vectors,coords,mask,neighbors,distances,patch,components,y in train:
   seq,res,vectors,coords,mask,neighbors,distances,patch,components,y=[x.to(device,non_blocking=True) for x in (seq,res,vectors,coords,mask,neighbors,distances,patch,components,y)];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)):
    out=model(seq,res,vectors,coords,mask,neighbors,distances,patch,components);loss=nn.functional.binary_cross_entropy_with_logits(out,y);rank_weight=float(tr.get('rank_weight',0))
    if rank_weight:loss=loss+rank_weight*binary_rank_loss(out,y,float(tr.get('rank_temperature',1.)),float(tr.get('rank_hard_fraction',1.)),float(tr.get('rank_margin',0.)))
    patch_aux_weight=float(tr.get('patch_aux_weight',0))
    if patch_aux_weight:
     if model.last_surface_patch_logit is None:raise ValueError('patch auxiliary loss requires surface patch tokens')
     loss=loss+patch_aux_weight*nn.functional.binary_cross_entropy_with_logits(model.last_surface_patch_logit,y)
   if not torch.isfinite(loss):raise FloatingPointError(f'non-finite GVP loss at epoch {epoch}')
   loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(tr.get('max_gradient_norm',5.0)))
   if not torch.isfinite(gradient_norm):raise FloatingPointError(f'non-finite GVP gradient norm at epoch {epoch}')
   opt.step()
   if ema:
    with torch.no_grad():
     for ep,pv in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(pv,alpha=1-decay)
   total+=loss.item()*len(y);count+=len(y)
  seconds=time.monotonic()-started;eval_model=ema or model;truth,logits=infer(eval_model,validation,device,tr.get('amp_bfloat16',False));metrics=binary_metrics(truth,logits);row={'epoch':epoch,'train_loss':total/count,'train_seconds':seconds,'train_samples_per_second':count/seconds,'validation':metrics};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/auroc',metrics['auroc'],epoch);writer.add_scalar('throughput/train_samples_per_second',row['train_samples_per_second'],epoch);state={'model':eval_model.state_dict(),'epoch':epoch,'validation':metrics,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  for metric in ('auprc','mcc','brier'):
   improved=metrics[metric]<best_by_metric[metric] if metric=='brier' else metrics[metric]>best_by_metric[metric]
   if improved:best_by_metric[metric]=metrics[metric];torch.save(state,a.run_dir/'checkpoints'/f'best_{metric}.pt')
  if metrics['auroc']>best:best=metrics['auroc'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);truth,logits=infer(model,validation,device,tr.get('amp_bfloat16',False));(a.run_dir/'validation_metrics.json').write_text(json.dumps({'pdbsol':binary_metrics(truth,logits)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=truth,logits=logits,entity_indices=np.asarray([v[0] for v in rows['validation']],np.int64));print(json.dumps({'best_epoch':state['epoch'],'best_validation_auroc':best,'test_evaluated':False}))
if __name__=='__main__':main()
