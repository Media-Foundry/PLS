"""Leakage-safe quantitative eSOL training with compact residue ESM2."""
from __future__ import annotations
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import regression_metrics
from pls.models.residue_sequence import ResidueSequenceRegressor
class Data(Dataset):
 def __init__(self,rows,global_esm,offsets,residue_path,shape,descriptors=None):self.rows,self.global_esm,self.offsets,self.residue_path,self.shape,self.descriptors=rows,global_esm,offsets,residue_path,shape,descriptors;self._data=None
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  entity,target=self.rows[i]
  if self._data is None:self._data=np.memmap(self.residue_path,mode='r',dtype=np.float16,shape=self.shape)
  lo,hi=int(self.offsets[entity]),int(self.offsets[entity+1]);res=torch.from_numpy(np.array(self._data[lo:hi],dtype=np.float32,copy=True));global_x=np.array(self.global_esm[entity],copy=True);global_x=np.concatenate((global_x,self.descriptors[entity])) if self.descriptors is not None else global_x;return torch.from_numpy(global_x),res,torch.tensor(target,dtype=torch.float32)
def collate(batch):
 n=max(len(v[1]) for v in batch);res=torch.zeros(len(batch),n,batch[0][1].shape[1]);mask=torch.zeros(len(batch),n,dtype=torch.bool)
 for i,(_,x,_) in enumerate(batch):res[i,:len(x)]=x;mask[i,:len(x)]=1
 return torch.stack([v[0] for v in batch]),res,mask,torch.stack([v[2] for v in batch])
class LengthSampler(Sampler):
 def __init__(self,lengths,batch,seed,pool_factor=20):self.lengths,self.batch,self.seed,self.pool_factor=lengths,batch,seed,pool_factor;self.epoch=0
 def __len__(self):return (len(self.lengths)+self.batch-1)//self.batch
 def __iter__(self):
  g=torch.Generator().manual_seed(self.seed+self.epoch);self.epoch+=1;indices=torch.randperm(len(self.lengths),generator=g).tolist();pool=self.batch*self.pool_factor
  for start in range(0,len(indices),pool):
   block=sorted(indices[start:start+pool],key=lambda i:self.lengths[i])
   for j in range(0,len(block),self.batch):yield block[j:j+self.batch]
def infer(model,loader,device,amp=False):
 model.eval();truth=[];pred=[]
 with torch.inference_mode():
  for global_esm,res,mask,y in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):out=model(global_esm.to(device),res.to(device),mask.to(device))
   truth.extend(y.tolist());pred.extend(out.float().cpu().tolist())
 return np.asarray(truth,np.float32),np.asarray(pred,np.float32)
def rank_loss(pred,target):
 delta_target=target[:,None]-target[None,:];selected=delta_target.abs()>.02
 if not selected.any():return pred.new_zeros(())
 delta_pred=pred[:,None]-pred[None,:];return nn.functional.softplus(-delta_target.sign()[selected]*delta_pred[selected]).mean()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--allow-test-evaluation',action='store_true');a=p.parse_args();c=json.loads(a.config.read_text());d=c['data'];mc=c['model'];tr=c['training']
 if c.get('evaluate_test',False) and not a.allow_test_evaluation:p.error('test evaluation requires explicit authorization after freeze')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']):p.error('HIP device mismatch')
 seed=tr['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h:entities=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(entities)}
 rows={s:[] for s in ('train','validation','test')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']=='eSOL_FGNNSol':rows[r['split']].append((index[r['sequence_sha256']],float(r['target_value'])))
 global_esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r');root=Path(d['residue_esm_dir']);offsets=np.load(Path(d['selection_dir'])/'offsets.npy',mmap_mode='r');shape=tuple(json.loads((root/'pca_metadata.json').read_text())['shape']);sets={s:Data(v,global_esm,offsets,root/'residue_esm2_pca.f16',shape) for s,v in rows.items()};lengths=[int(entities[i]['length']) for i,_ in rows['train']];batch_sampler=LengthSampler(lengths,tr['batch_size'],seed);train=DataLoader(sets['train'],batch_sampler=batch_sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);loaders={s:DataLoader(sets[s],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True) for s in ('validation','test')}
 device=torch.device('cuda:0');model=ResidueSequenceRegressor(global_esm.shape[1],shape[1],mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc['pooling'],mc.get('fusion','concat'),mc.get('global_segments',1),mc.get('global_segment_fusion','weighted_sum')).to(device);decay=float(tr.get('ema_decay',0));ema=copy.deepcopy(model).eval().requires_grad_(False) if decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay'],fused=tr.get('fused_optimizer',False));writer=SummaryWriter(a.run_dir/'tensorboard');best=-2;stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  started=time.monotonic();model.train();total=count=0
  for global_x,res,mask,y in train:
   global_x,res,mask,y=[v.to(device,non_blocking=True) for v in (global_x,res,mask,y)];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)):pred=model(global_x,res,mask);loss=nn.functional.smooth_l1_loss(pred,y,beta=.1)+float(tr.get('rank_weight',0))*rank_loss(pred,y)
   loss.backward();opt.step()
   if ema:
    with torch.no_grad():
     for ep,pv in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(pv,alpha=1-decay)
   total+=loss.item()*len(y);count+=len(y)
  train_seconds=time.monotonic()-started;eval_model=ema or model;truth,pred=infer(eval_model,loaders['validation'],device,tr.get('amp_bfloat16',False));metrics=regression_metrics(truth,pred);row={'epoch':epoch,'train_loss':total/count,'train_seconds':train_seconds,'validation':metrics};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/spearman',metrics['spearman'],epoch);state={'model':eval_model.state_dict(),'epoch':epoch,'validation':metrics,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if metrics['spearman']>best:best=metrics['spearman'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);truth,pred=infer(model,loaders['validation'],device,tr.get('amp_bfloat16',False));(a.run_dir/'validation_metrics.json').write_text(json.dumps({'esol':regression_metrics(truth,pred)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=truth,predictions=pred,entity_indices=np.asarray([v[0] for v in rows['validation']],np.int64))
 if c.get('evaluate_test',False):
  truth,pred=infer(model,loaders['test'],device,tr.get('amp_bfloat16',False));(a.run_dir/'test_metrics.json').write_text(json.dumps({'esol':regression_metrics(truth,pred)},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'best_epoch':state['epoch'],'best_validation_spearman':best,'test_evaluated':c.get('evaluate_test',False)}))
if __name__=='__main__':main()
