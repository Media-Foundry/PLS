"""Leakage-safe eSOL regression with AlphaFold geometry and frozen ESM features."""
from __future__ import annotations
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import regression_metrics
from pls.models.geometry_structure import GeometryLateFusion
from pls.training.train_esol_residue import LengthSampler,rank_loss
from pls.training.train_residue_structure import Data,collate
SOURCE='eSOL_FGNNSol'
def infer(model,loader,device,amp=False):
 model.eval();truth=[];pred=[]
 with torch.inference_mode():
  for seq,res,p,patch,mask,neighbors,distances,global_features,patch_components,y in loader:
   values=[v.to(device,non_blocking=True) for v in (seq,res,p,patch,mask,neighbors,distances,global_features,patch_components)]
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
   if r['source_dataset']==SOURCE and r['split'] in rows:rows[r['split']].append((index[r['sequence_sha256']],r['sequence_sha256'],float(r['target_value'])))
 status=np.load(d['structure_status'],mmap_mode='r');rows={s:[v for v in values if status[v[0]]==1] for s,values in rows.items()};train_entities=np.unique([v[0] for v in rows['train']])
 descriptors=None
 if d.get('sequence_descriptor_dir'):
  raw=np.load(Path(d['sequence_descriptor_dir'])/'descriptors.npy',mmap_mode='r');mean=np.asarray(raw[train_entities],np.float64).mean(0);std=np.maximum(np.asarray(raw[train_entities],np.float64).std(0),1e-6);scale=float(d.get('sequence_descriptor_scale',1));descriptors=(((np.asarray(raw,np.float32)-mean)/std)*scale).astype(np.float32);np.savez(a.run_dir/'descriptor_stats.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),scale=np.float32(scale),train_entity_indices=train_entities)
 esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r');stats=json.loads(Path(d['structure_stats']).read_text());mean=torch.tensor(stats['scalar_means']);std=torch.tensor(stats['scalar_stds']);compact=Path(d['compact_structure_dir']);geometry=Path(d['geometry_dir']);residue_esm=Path(d['residue_esm_dir']);sets={s:Data(v,esm,Path(d['structure_dir']),mean,std,compact,geometry,True,mc.get('neighbors',8),residue_esm,None,None,descriptors) for s,v in rows.items()};lengths=[int(entities[v[0]]['length']) for v in rows['train']];sampler=LengthSampler(lengths,tr['batch_size'],seed);train=DataLoader(sets['train'],batch_sampler=sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);validation=DataLoader(sets['validation'],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True)
 device=torch.device('cuda:0');residue_dimension=152+44+int(json.loads((residue_esm/'pca_metadata.json').read_text())['shape'][1]);global_dimension=descriptors.shape[1] if descriptors is not None else 0;model=GeometryLateFusion(esm.shape[1],residue_dimension,mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc.get('geometry_layers',1),mc['pooling'],global_dimension,mc.get('use_sequence_separation',False),mc.get('fusion','concat'),mc.get('confidence_mode','legacy')).to(device);decay=float(tr.get('ema_decay',0));ema=copy.deepcopy(model).eval().requires_grad_(False) if decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay'],fused=tr.get('fused_optimizer',False));writer=SummaryWriter(a.run_dir/'tensorboard');best=-2;stale=0;history=[]
 for epoch in range(1,tr['epochs']+1):
  started=time.monotonic();model.train();total=count=0
  for seq,res,pconf,patch,mask,neighbors,distances,global_features,patch_components,y in train:
   seq,res,pconf,patch,mask,neighbors,distances,global_features,patch_components,y=[v.to(device,non_blocking=True) for v in (seq,res,pconf,patch,mask,neighbors,distances,global_features,patch_components,y)];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)):pred=model(seq,res,pconf,patch,mask,neighbors,distances,global_features,patch_components);loss=nn.functional.smooth_l1_loss(pred,y,beta=.1)+float(tr.get('rank_weight',0))*rank_loss(pred,y)
   loss.backward();opt.step()
   if ema:
    with torch.no_grad():
     for ep,pv in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(pv,alpha=1-decay)
   total+=loss.item()*len(y);count+=len(y)
  seconds=time.monotonic()-started;eval_model=ema or model;truth,pred=infer(eval_model,validation,device,tr.get('amp_bfloat16',False));metrics=regression_metrics(truth,pred);row={'epoch':epoch,'train_loss':total/count,'train_seconds':seconds,'train_samples_per_second':count/seconds,'validation':metrics};history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/spearman',metrics['spearman'],epoch);writer.add_scalar('throughput/train_samples_per_second',row['train_samples_per_second'],epoch);state={'model':eval_model.state_dict(),'epoch':epoch,'validation':metrics,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if metrics['spearman']>best:best=metrics['spearman'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);truth,pred=infer(model,validation,device,tr.get('amp_bfloat16',False));(a.run_dir/'validation_metrics.json').write_text(json.dumps({'esol':regression_metrics(truth,pred)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=truth,predictions=pred,entity_indices=np.asarray([v[0] for v in rows['validation']],np.int64));print(json.dumps({'best_epoch':state['epoch'],'best_validation_spearman':best,'test_evaluated':False,'validation_structures':len(rows['validation'])}))
if __name__=='__main__':main()
