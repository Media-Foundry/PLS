"""Leakage-safe residue V4 + frozen ESM training on PDBSol."""
from __future__ import annotations
import argparse,csv,json,os,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import binary_metrics
from pls.models.residue_structure import ResidueLateFusion

class Data(Dataset):
 def __init__(self,rows,esm,root,mean,std,compact=None):
  self.rows,self.esm,self.root,self.mean,self.std,self.compact=rows,esm,root,mean,std,compact;self._compact_data=None
  if compact:
   self.offsets=np.load(compact/'offsets.npy',mmap_mode='r');self.compact_shape=tuple(json.loads((compact/'metadata.json').read_text())['shape'])
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  j,d,y=self.rows[i]
  if self.compact:
   if self._compact_data is None:self._compact_data=np.memmap(self.compact/'residue_features.f16',mode='r',dtype=np.float16,shape=self.compact_shape)
   residue=torch.from_numpy(np.array(self._compact_data[self.offsets[j]:self.offsets[j+1]],dtype=np.float32,copy=True));p=residue[:,-1];patch=torch.cat((residue[:,130:134],p[:,None]),1)
  else:
   x=torch.load(self.root/d[:2]/f'{d}.pt',map_location='cpu',weights_only=False); raw=x['spatial_scalar_raw_features'].float(); scalar=(raw-self.mean)/self.std; p=x['plddt'].float(); residue=torch.cat((x['physchem_features'].float(),scalar,p[:,None]),1); patch=torch.cat((scalar[:,68:72],p[:,None]),1)
  return torch.from_numpy(np.array(self.esm[j],copy=True)),residue,p,patch,torch.tensor(y,dtype=torch.float32)
def collate(batch):
 n=max(len(v[1]) for v in batch); b=len(batch); residue=torch.zeros(b,n,152); p=torch.zeros(b,n); patch=torch.zeros(b,n,5); mask=torch.zeros(b,n,dtype=torch.bool)
 for i,(_,r,q,h,_) in enumerate(batch): residue[i,:len(r)]=r;p[i,:len(r)]=q;patch[i,:len(r)]=h;mask[i,:len(r)]=1
 return torch.stack([v[0] for v in batch]),residue,p,patch,mask,torch.stack([v[4] for v in batch])
class BalancedLengthBatchSampler(Sampler):
 def __init__(self,labels,lengths,batch_size,seed,pool_factor=20): self.labels,self.lengths,self.batch_size,self.seed,self.pool_factor=labels,lengths,batch_size,seed,pool_factor;self.epoch=0
 def __len__(self): return (len(self.labels)+self.batch_size-1)//self.batch_size
 def __iter__(self):
  counts=np.bincount(self.labels.astype(int),minlength=2);weights=torch.tensor([1/counts[int(v)] for v in self.labels],dtype=torch.double);g=torch.Generator().manual_seed(self.seed+self.epoch);self.epoch+=1;indices=torch.multinomial(weights,len(weights),replacement=True,generator=g).tolist();pool=self.batch_size*self.pool_factor
  for start in range(0,len(indices),pool):
   block=sorted(indices[start:start+pool],key=lambda i:self.lengths[i])
   for j in range(0,len(block),self.batch_size): yield block[j:j+self.batch_size]
def evaluate(model,loader,device):
 model.eval(); pred=[]; truth=[]
 with torch.inference_mode():
  for seq,res,p,patch,mask,y in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True): out,_=model(seq.to(device),res.to(device),p.to(device),patch.to(device),mask.to(device))
   pred.extend(out.float().cpu().tolist());truth.extend(y.tolist())
 return binary_metrics(truth,pred)
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
 compact=Path(d['compact_structure_dir']) if d.get('compact_structure_dir') else None;sets={s:Data(v,esm,root,mean,std,compact) for s,v in rows.items()};labels=np.array([v[2] for v in rows['train']]);lengths=[length_by_hash[v[1]] for v in rows['train']];batch_sampler=BalancedLengthBatchSampler(labels,lengths,tr['batch_size'],seed)
 train=DataLoader(sets['train'],batch_sampler=batch_sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);loaders={s:DataLoader(sets[s],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True) for s in ('validation','test')}
 device=torch.device('cuda:0');model=ResidueLateFusion(esm.shape[1],152,mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc['pooling']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay']);writer=SummaryWriter(a.run_dir/'tensorboard');best=-1;stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  model.train();total=n=0
  for seq,res,p,patch,mask,y in train:
   seq,res,p,patch,mask,y=[v.to(device,non_blocking=True) for v in (seq,res,p,patch,mask,y)];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)): out,_=model(seq,res,p,patch,mask,mc.get('structure_dropout',0));loss=nn.functional.binary_cross_entropy_with_logits(out,y)
   loss.backward();opt.step();total+=loss.item()*len(y);n+=len(y)
  val=evaluate(model,loaders['validation'],device);row={'epoch':epoch,'train_loss':total/n,'validation':val};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/auroc',val['auroc'],epoch);state={'model':model.state_dict(),'epoch':epoch,'validation':val,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if val['auroc']>best:best=val['auroc'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);(a.run_dir/'validation_metrics.json').write_text(json.dumps({'pdbsol':evaluate(model,loaders['validation'],device)},indent=2,sort_keys=True)+'\n')
 if c.get('evaluate_test',False):(a.run_dir/'test_metrics.json').write_text(json.dumps({'pdbsol':evaluate(model,loaders['test'],device)},indent=2)+'\n')
 print(json.dumps({'best_epoch':state['epoch'],'best_validation_auroc':best,'test_evaluated':c.get('evaluate_test',False)}))
if __name__=='__main__':main()
