"""Leakage-safe residue V4 + frozen ESM training on PDBSol."""
from __future__ import annotations
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import binary_metrics
from pls.models.residue_structure import ResidueLateFusion
from pls.models.geometry_structure import GeometryLateFusion

class Data(Dataset):
 def __init__(self,rows,esm,root,mean,std,compact=None,geometry=None,use_vectors=False,neighbors=16,residue_esm=None,global_structure=None,global_columns=None):
  self.rows,self.esm,self.root,self.mean,self.std,self.compact,self.geometry,self.use_vectors,self.neighbor_count,self.residue_esm,self.global_structure,self.global_columns=rows,esm,root,mean,std,compact,geometry,use_vectors,neighbors,residue_esm,global_structure,global_columns;self._compact_data=None;self._neighbor_data=None;self._residue_esm_data=None
  if compact:
   self.offsets=np.load(compact/'offsets.npy',mmap_mode='r');self.compact_shape=tuple(json.loads((compact/'metadata.json').read_text())['shape'])
  if geometry:
   meta=json.loads((geometry/'metadata.json').read_text());self.geometry_shape=(meta['residues'],meta['neighbors'])
  if residue_esm:self.residue_esm_shape=tuple(json.loads((residue_esm/'pca_metadata.json').read_text())['shape'])
  if global_structure:
   self.global_dimension=int(json.loads((global_structure/'metadata.json').read_text())['dimension']);self.global_data=np.load(global_structure/'descriptors.npy',mmap_mode='r')
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  j,d,y=self.rows[i]
  if self.compact:
   if self._compact_data is None:self._compact_data=np.memmap(self.compact/'residue_features.f16',mode='r',dtype=np.float16,shape=self.compact_shape)
   residue=torch.from_numpy(np.array(self._compact_data[self.offsets[j]:self.offsets[j+1]],dtype=np.float32,copy=True));p=residue[:,-1];patch=torch.cat((residue[:,130:134],p[:,None]),1)
  else:
   x=torch.load(self.root/d[:2]/f'{d}.pt',map_location='cpu',weights_only=False); raw=x['spatial_scalar_raw_features'].float(); scalar=(raw-self.mean)/self.std; p=x['plddt'].float(); residue=torch.cat((x['physchem_features'].float(),scalar,p[:,None]),1); patch=torch.cat((scalar[:,68:72],p[:,None]),1)
  if self.geometry:
   if self._neighbor_data is None:
    self._neighbor_data=np.memmap(self.geometry/'neighbors.i16',mode='r',dtype=np.int16,shape=self.geometry_shape);self._distance_data=np.memmap(self.geometry/'distances.f16',mode='r',dtype=np.float16,shape=self.geometry_shape);self._invariant_data=np.memmap(self.geometry/'vector_invariants.f16',mode='r',dtype=np.float16,shape=(self.geometry_shape[0],44))
   lo,hi=int(self.offsets[j]),int(self.offsets[j+1]);neighbors=torch.from_numpy(np.array(self._neighbor_data[lo:hi,:self.neighbor_count],dtype=np.int64,copy=True));distances=torch.from_numpy(np.array(self._distance_data[lo:hi,:self.neighbor_count],dtype=np.float32,copy=True))
   if self.use_vectors:residue=torch.cat((residue,torch.from_numpy(np.array(self._invariant_data[lo:hi],dtype=np.float32,copy=True))),1)
   if self.residue_esm:
    if self._residue_esm_data is None:self._residue_esm_data=np.memmap(self.residue_esm/'residue_esm2_pca.f16',mode='r',dtype=np.float16,shape=self.residue_esm_shape)
    residue=torch.cat((residue,torch.from_numpy(np.array(self._residue_esm_data[lo:hi],dtype=np.float32,copy=True))),1)
  else:neighbors=torch.zeros(len(residue),1,dtype=torch.long);distances=torch.zeros(len(residue),1)
  global_features=torch.from_numpy(np.array(self.global_data[j,self.global_columns],copy=True)) if self.global_structure else torch.empty(0)
  return torch.from_numpy(np.array(self.esm[j],copy=True)),residue,p,patch,neighbors,distances,global_features,torch.tensor(y,dtype=torch.float32)
def collate(batch):
 n=max(len(v[1]) for v in batch); b=len(batch);dim=batch[0][1].shape[1];k=batch[0][4].shape[1];residue=torch.zeros(b,n,dim); p=torch.zeros(b,n); patch=torch.zeros(b,n,5);mask=torch.zeros(b,n,dtype=torch.bool);neighbors=torch.zeros(b,n,k,dtype=torch.long);distances=torch.zeros(b,n,k)
 for i,(_,r,q,h,nb,ds,_,_) in enumerate(batch): residue[i,:len(r)]=r;p[i,:len(r)]=q;patch[i,:len(r)]=h;neighbors[i,:len(r)]=nb;distances[i,:len(r)]=ds;mask[i,:len(r)]=1
 return torch.stack([v[0] for v in batch]),residue,p,patch,mask,neighbors,distances,torch.stack([v[6] for v in batch]),torch.stack([v[7] for v in batch])
class BalancedLengthBatchSampler(Sampler):
 def __init__(self,labels,lengths,batch_size,seed,pool_factor=20): self.labels,self.lengths,self.batch_size,self.seed,self.pool_factor=labels,lengths,batch_size,seed,pool_factor;self.epoch=0
 def __len__(self): return (len(self.labels)+self.batch_size-1)//self.batch_size
 def __iter__(self):
  counts=np.bincount(self.labels.astype(int),minlength=2);weights=torch.tensor([1/counts[int(v)] for v in self.labels],dtype=torch.double);g=torch.Generator().manual_seed(self.seed+self.epoch);self.epoch+=1;indices=torch.multinomial(weights,len(weights),replacement=True,generator=g).tolist();pool=self.batch_size*self.pool_factor
  for start in range(0,len(indices),pool):
   block=sorted(indices[start:start+pool],key=lambda i:self.lengths[i])
   for j in range(0,len(block),self.batch_size): yield block[j:j+self.batch_size]
def infer(model,loader,device,amp=False,geometry=False):
 model.eval(); pred=[]; truth=[]
 with torch.inference_mode():
  for seq,res,p,patch,mask,neighbors,distances,global_features,y in loader:
   values=[v.to(device) for v in (seq,res,p,patch,mask,neighbors,distances,global_features)]
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): out=model(*values) if geometry else model(*values[:5])[0]
   pred.extend(out.float().cpu().tolist());truth.extend(y.tolist())
 return np.asarray(truth,dtype=np.float32),np.asarray(pred,dtype=np.float32)
def evaluate(model,loader,device,amp=False,geometry=False):
 truth,pred=infer(model,loader,device,amp,geometry);return binary_metrics(truth,pred)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--allow-test-evaluation',action='store_true');a=ap.parse_args();c=json.loads(a.config.read_text());d=c['data'];tr=c['training'];mc=c['model']
 if c.get('evaluate_test',False) and not a.allow_test_evaluation: ap.error('test evaluation requires explicit authorization after freeze')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']): ap.error('HIP device mismatch')
 seed=tr['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h: entity_rows=list(csv.DictReader(h));hashes=[r['sequence_sha256'] for r in entity_rows];length_by_hash={r['sequence_sha256']:int(r['length']) for r in entity_rows};ix={v:i for i,v in enumerate(hashes)}
 esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r');root=Path(d['structure_dir']);stats=json.loads(Path(d['structure_stats']).read_text());mean=torch.tensor(stats['scalar_means']);std=torch.tensor(stats['scalar_stds']);rows={s:[] for s in ('train','validation','test')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']!='PDBSol_ProtSolM':continue
   digest=r['sequence_sha256'];path=root/digest[:2]/f'{digest}.pt'
   if not path.exists():continue
   # Status was audited when pooled cache was built; enforce exact sequence again without loading 21 GB here.
   entity=ix[digest];rows[r['split']].append((entity,digest,float(r['target_value'])))
 status=np.load(Path(d['structure_status']),mmap_mode='r');rows={s:[v for v in values if status[v[0]]==1] for s,values in rows.items()}
 compact=Path(d['compact_structure_dir']) if d.get('compact_structure_dir') else None;geometry=Path(d['geometry_dir']) if d.get('geometry_dir') else None;residue_esm=Path(d['residue_esm_dir']) if d.get('residue_esm_dir') else None;global_structure=Path(d['global_structure_dir']) if d.get('global_structure_dir') else None;global_columns=None
 if global_structure:
  global_meta=json.loads((global_structure/'metadata.json').read_text());groups=mc.get('global_groups',list(global_meta['slices']));global_columns=np.asarray([i for group in groups for i in range(*global_meta['slices'][group])],dtype=np.int64)
 use_vectors=mc.get('use_vector_invariants',False);neighbor_count=mc.get('neighbors',16);sets={s:Data(v,esm,root,mean,std,compact,geometry,use_vectors,neighbor_count,residue_esm,global_structure,global_columns) for s,v in rows.items()};labels=np.array([v[2] for v in rows['train']]);lengths=[length_by_hash[v[1]] for v in rows['train']];batch_sampler=BalancedLengthBatchSampler(labels,lengths,tr['batch_size'],seed)
 train=DataLoader(sets['train'],batch_sampler=batch_sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);loaders={s:DataLoader(sets[s],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True) for s in ('validation','test')}
 device=torch.device('cuda:0');input_dimension=152+(44 if use_vectors else 0)+(int(json.loads((residue_esm/'pca_metadata.json').read_text())['shape'][1]) if residue_esm else 0);global_dimension=len(global_columns) if global_structure else 0;is_geometry=geometry is not None;model=(GeometryLateFusion(esm.shape[1],input_dimension,mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc.get('geometry_layers',1),mc['pooling'],global_dimension,mc.get('use_sequence_separation',False)) if is_geometry else ResidueLateFusion(esm.shape[1],input_dimension,mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc['pooling'])).to(device);ema_decay=float(tr.get('ema_decay',0));ema_model=copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay']);writer=SummaryWriter(a.run_dir/'tensorboard');best=-1;stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  epoch_started=time.monotonic();model.train();total=n=0
  for seq,res,p,patch,mask,neighbors,distances,global_features,y in train:
   seq,res,p,patch,mask,neighbors,distances,global_features,y=[v.to(device,non_blocking=True) for v in (seq,res,p,patch,mask,neighbors,distances,global_features,y)];opt.zero_grad(set_to_none=True)
   smooth=float(tr.get('label_smoothing',0));target=y*(1-smooth)+.5*smooth
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)): out=model(seq,res,p,patch,mask,neighbors,distances,global_features) if is_geometry else model(seq,res,p,patch,mask,mc.get('structure_dropout',0))[0];loss=nn.functional.binary_cross_entropy_with_logits(out,target)
   loss.backward();opt.step()
   if ema_model:
    with torch.no_grad():
     for ema_parameter,parameter in zip(ema_model.parameters(),model.parameters()):ema_parameter.mul_(ema_decay).add_(parameter,alpha=1-ema_decay)
   total+=loss.item()*len(y);n+=len(y)
  train_seconds=time.monotonic()-epoch_started;validation_model=ema_model or model;val=evaluate(validation_model,loaders['validation'],device,tr.get('amp_bfloat16',False),is_geometry);row={'epoch':epoch,'train_loss':total/n,'train_seconds':train_seconds,'train_samples_per_second':n/train_seconds,'epoch_seconds':time.monotonic()-epoch_started,'validation':val};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/auroc',val['auroc'],epoch);writer.add_scalar('throughput/train_samples_per_second',row['train_samples_per_second'],epoch);state={'model':validation_model.state_dict(),'epoch':epoch,'validation':val,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if val['auroc']>best:best=val['auroc'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);amp=tr.get('amp_bfloat16',False);validation_truth,validation_logits=infer(model,loaders['validation'],device,amp,is_geometry);(a.run_dir/'validation_metrics.json').write_text(json.dumps({'pdbsol':binary_metrics(validation_truth,validation_logits)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=validation_truth,logits=validation_logits,entity_indices=np.asarray([v[0] for v in rows['validation']],dtype=np.int64))
 if c.get('evaluate_test',False):(a.run_dir/'test_metrics.json').write_text(json.dumps({'pdbsol':evaluate(model,loaders['test'],device,amp,is_geometry)},indent=2)+'\n')
 print(json.dumps({'best_epoch':state['epoch'],'best_validation_auroc':best,'test_evaluated':c.get('evaluate_test',False)}))
if __name__=='__main__':main()
