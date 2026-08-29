"""Leakage-safe PDBSol screening with frozen PLM and pooled V4 descriptors."""
from __future__ import annotations
import argparse,csv,json,os,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import binary_metrics
from pls.models.structure_fusion import build_fusion

class D(Dataset):
 def __init__(self,rows,esm,struct,cols): self.rows,self.esm,self.struct,self.cols=rows,esm,struct,cols
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  j,y=self.rows[i]; parts=[np.array(self.esm[j],copy=True)]
  if self.cols: parts.append(np.array(self.struct[j,self.cols],copy=True))
  return torch.from_numpy(np.concatenate(parts)),torch.tensor(y,dtype=torch.float32)

def predict(model,loader,device,raw=False):
 model.eval(); p=[]; y=[]
 with torch.inference_mode():
  for x,t in loader: p.extend(model(x.to(device)).cpu().tolist()); y.extend(t.tolist())
 return (np.asarray(y,np.float32),np.asarray(p,np.float32)) if raw else binary_metrics(y,p)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
 p.add_argument('--allow-test-evaluation',action='store_true'); a=p.parse_args(); c=json.loads(a.config.read_text()); d=c['data']; tr=c['training']
 if c.get('evaluate_test',False) and not a.allow_test_evaluation: p.error('test evaluation requires explicit --allow-test-evaluation after model freeze')
 hip=str(tr['hip_device']);
 if os.environ.get('HIP_VISIBLE_DEVICES')!=hip: p.error(f'HIP_VISIBLE_DEVICES must equal configured {hip}')
 seed=int(tr['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h: hashes=[r['sequence_sha256'] for r in csv.DictReader(h)]
 ix={v:i for i,v in enumerate(hashes)}; esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r')
 root=Path(d['structure_dir']); struct=np.load(root/'descriptors.npy',mmap_mode='r'); status=np.load(root/'status.npy',mmap_mode='r'); meta=json.loads((root/'metadata.json').read_text())
 cols=[]
 for group in c['model']['structure_groups']:
  lo,hi=meta['slices'][group]; cols.extend(range(lo,hi))
 rows={s:[] for s in ('train','validation','test')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']!='PDBSol_ProtSolM': continue
   j=ix[r['sequence_sha256']]
   if d.get('require_structure_match',True) and status[j]!=1: continue
   rows[r['split']].append((j,float(r['target_value'])))
 sets={s:D(v,esm,struct,cols) for s,v in rows.items()}; batch=int(tr['batch_size'])
 labels=np.array([y for _,y in rows['train']]); count=np.bincount(labels.astype(int),minlength=2); weights=[1/count[int(y)] for y in labels]
 sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(seed))
 train=DataLoader(sets['train'],batch_size=batch,sampler=sampler); loaders={s:DataLoader(sets[s],batch_size=batch) for s in ('validation','test')}
 device=torch.device('cuda:0'); mconf=c['model']; model=build_fusion(mconf.get('architecture','early'),esm.shape[1],len(cols),mconf['hidden_dimension'],mconf['representation_dimension'],mconf['dropout']).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay']); writer=SummaryWriter(a.run_dir/'tensorboard'); best=-1.; stale=0; history=[]
 for epoch in range(1,tr['epochs']+1):
  model.train(); total=n=0
  for x,y in train:
   x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=nn.functional.binary_cross_entropy_with_logits(model(x),y); loss.backward(); opt.step(); total+=loss.item()*len(y); n+=len(y)
  val=predict(model,loaders['validation'],device); row={'epoch':epoch,'train_loss':total/n,'validation':val}; history.append(row); print(json.dumps(row),flush=True); writer.add_scalar('validation/auroc',val['auroc'],epoch)
  state={'model':model.state_dict(),'epoch':epoch,'config':c,'validation':val}
  if epoch%tr['checkpoint_every']==0: torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if val['auroc']>best: best=val['auroc']; stale=0; torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else: stale+=1
  if stale>=tr['patience']: break
 writer.close(); (a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n'); beststate=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False); model.load_state_dict(beststate['model'])
 validation_truth,validation_logits=predict(model,loaders['validation'],device,True);(a.run_dir/'validation_metrics.json').write_text(json.dumps({'pdbsol':binary_metrics(validation_truth,validation_logits)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=validation_truth,logits=validation_logits,entity_indices=np.asarray([i for i,_ in rows['validation']],np.int64))
 if c.get('evaluate_test',False): (a.run_dir/'test_metrics.json').write_text(json.dumps({'pdbsol':predict(model,loaders['test'],device)},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'best_epoch':beststate['epoch'],'best_validation_auroc':best,'test_evaluated':c.get('evaluate_test',False)}))
if __name__=='__main__': main()
