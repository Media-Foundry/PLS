"""Leakage-safe UESolDS classification with compact residue ESM2 features."""
from __future__ import annotations
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import binary_metrics
from pls.models.residue_sequence import ResidueSequenceRegressor
from pls.training.train_esol_residue import Data,collate
from pls.training.train_residue_structure import BalancedLengthBatchSampler

SOURCE='UESolDS_PLM_Sol_1.1'

def binary_rank_loss(logits,targets):
 positive=logits[targets>.5];negative=logits[targets<=.5]
 if not len(positive) or not len(negative):return logits.new_zeros(())
 return nn.functional.softplus(-(positive[:,None]-negative[None,:])).mean()

def infer(model,loader,device,amp=False):
 model.eval();truth=[];logits=[]
 with torch.inference_mode():
  for global_esm,res,mask,y in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):out=model(global_esm.to(device,non_blocking=True),res.to(device,non_blocking=True),mask.to(device,non_blocking=True))
   truth.extend(y.tolist());logits.extend(out.float().cpu().tolist())
 return np.asarray(truth,np.float32),np.asarray(logits,np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--allow-test-evaluation',action='store_true');a=p.parse_args();c=json.loads(a.config.read_text());d=c['data'];mc=c['model'];tr=c['training']
 if c.get('evaluate_test',False) and not a.allow_test_evaluation:p.error('test evaluation requires explicit authorization after freeze')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']):p.error('HIP device mismatch')
 seed=tr['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h:entities=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(entities)}
 rows={s:[] for s in ('train','validation','test')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']==SOURCE:rows[r['split']].append((index[r['sequence_sha256']],float(r['target_value'])))
 global_esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r');root=Path(d['residue_esm_dir']);offsets=np.load(Path(d['selection_dir'])/'offsets.npy',mmap_mode='r');shape=tuple(json.loads((root/'pca_metadata.json').read_text())['shape']);sets={s:Data(v,global_esm,offsets,root/'residue_esm2_pca.f16',shape) for s,v in rows.items()};labels=np.asarray([y for _,y in rows['train']],np.int64);lengths=np.asarray([int(entities[i]['length']) for i,_ in rows['train']]);batch_sampler=BalancedLengthBatchSampler(labels,lengths,tr['batch_size'],seed);train=DataLoader(sets['train'],batch_sampler=batch_sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);loaders={s:DataLoader(sets[s],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True) for s in ('validation','test')}
 device=torch.device('cuda:0');model=ResidueSequenceRegressor(global_esm.shape[1],shape[1],mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc['pooling']).to(device);decay=float(tr.get('ema_decay',0));ema=copy.deepcopy(model).eval().requires_grad_(False) if decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay']);writer=SummaryWriter(a.run_dir/'tensorboard');best=-1;stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  started=time.monotonic();model.train();total=count=0
  for global_x,res,mask,y in train:
   global_x,res,mask,y=[v.to(device,non_blocking=True) for v in (global_x,res,mask,y)];opt.zero_grad(set_to_none=True);smooth=float(tr.get('label_smoothing',0));target=y*(1-smooth)+.5*smooth
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)):logits=model(global_x,res,mask);loss=nn.functional.binary_cross_entropy_with_logits(logits,target)+float(tr.get('rank_weight',0))*binary_rank_loss(logits,y)
   loss.backward();opt.step()
   if ema:
    with torch.no_grad():
     for ep,pv in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(pv,alpha=1-decay)
   total+=loss.item()*len(y);count+=len(y)
  seconds=time.monotonic()-started;eval_model=ema or model;truth,logits=infer(eval_model,loaders['validation'],device,tr.get('amp_bfloat16',False));metrics=binary_metrics(truth,logits);row={'epoch':epoch,'train_loss':total/count,'train_seconds':seconds,'train_samples_per_second':count/seconds,'validation':metrics};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/auroc',metrics['auroc'],epoch);writer.add_scalar('throughput/train_samples_per_second',row['train_samples_per_second'],epoch);state={'model':eval_model.state_dict(),'epoch':epoch,'validation':metrics,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if metrics['auroc']>best:best=metrics['auroc'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);truth,logits=infer(model,loaders['validation'],device,tr.get('amp_bfloat16',False));(a.run_dir/'validation_metrics.json').write_text(json.dumps({'uesolds':binary_metrics(truth,logits)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=truth,logits=logits,entity_indices=np.asarray([v[0] for v in rows['validation']],np.int64))
 if c.get('evaluate_test',False):
  truth,logits=infer(model,loaders['test'],device,tr.get('amp_bfloat16',False));(a.run_dir/'test_metrics.json').write_text(json.dumps({'uesolds':binary_metrics(truth,logits)},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'best_epoch':state['epoch'],'best_validation_auroc':best,'test_evaluated':c.get('evaluate_test',False)}))
if __name__=='__main__':main()
