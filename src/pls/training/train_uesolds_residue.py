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
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());d=c['data'];mc=c['model'];tr=c['training']
 if c.get('evaluate_test',False):p.error('test evaluation is permanently disabled')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']):p.error('HIP device mismatch')
 seed=tr['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 with open(d['entities'],newline='',encoding='utf-8') as h:entities=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(entities)}
 rows={s:[] for s in ('train','validation')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']==SOURCE and r['split'] in rows:rows[r['split']].append((index[r['sequence_sha256']],float(r['target_value'])))
 global_esm=np.load(Path(d['embedding_dir'])/'embeddings.npy',mmap_mode='r')
 if 'embedding_segment' in d:
  metadata=json.loads((Path(d['embedding_dir'])/'metadata.json').read_text());segment=int(d['embedding_segment']);layer_dimension=int(metadata['layer_dimension']);layers=metadata['layers']
  if segment<0 or segment>=len(layers):raise ValueError(f'embedding segment {segment} outside available layers {layers}')
  global_esm=global_esm[:,segment*layer_dimension:(segment+1)*layer_dimension]
 descriptors=None
 if d.get('sequence_descriptor_dir'):
  raw=np.load(Path(d['sequence_descriptor_dir'])/'descriptors.npy',mmap_mode='r');train_entities=np.unique([i for i,_ in rows['train']]);mean=np.asarray(raw[train_entities],np.float64).mean(0);std=np.asarray(raw[train_entities],np.float64).std(0);std=np.maximum(std,1e-6);scale=float(d.get('sequence_descriptor_scale',1));descriptors=(((np.asarray(raw,np.float32)-mean)/std)*scale).astype(np.float32);np.savez(a.run_dir/'descriptor_stats.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),scale=np.float32(scale),train_entity_indices=train_entities)
 root=Path(d['residue_esm_dir']);offsets=np.load(Path(d['selection_dir'])/'offsets.npy',mmap_mode='r');shape=tuple(json.loads((root/'pca_metadata.json').read_text())['shape']);sets={s:Data(v,global_esm,offsets,root/'residue_esm2_pca.f16',shape,descriptors) for s,v in rows.items()};labels=np.asarray([y for _,y in rows['train']],np.int64);lengths=np.asarray([int(entities[i]['length']) for i,_ in rows['train']]);batch_sampler=BalancedLengthBatchSampler(labels,lengths,tr['batch_size'],seed);train=DataLoader(sets['train'],batch_sampler=batch_sampler,num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True);loaders={'validation':DataLoader(sets['validation'],batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate,persistent_workers=tr['workers']>0,pin_memory=True)}
 device=torch.device('cuda:0');global_dimension=global_esm.shape[1]+(descriptors.shape[1] if descriptors is not None else 0);model=ResidueSequenceRegressor(global_dimension,shape[1],mc['hidden_dimension'],mc['representation_dimension'],mc['dropout'],mc['pooling'],mc.get('fusion','concat'),mc.get('global_segments',1),mc.get('global_segment_fusion','weighted_sum'),mc.get('tcn_bottleneck'),mc.get('tcn_dilations',(1,2,4)),mc.get('segment_window',32),mc.get('segment_layers',2),descriptors.shape[1] if descriptors is not None else 0,mc.get('physchem_experts',0),mc.get('physchem_gate_temperature',1.),mc.get('physchem_top_k',0),mc.get('physchem_sparse_residual',0.)).to(device);decay=float(tr.get('ema_decay',0));ema=copy.deepcopy(model).eval().requires_grad_(False) if decay else None;opt=torch.optim.AdamW(model.parameters(),lr=tr['learning_rate'],weight_decay=tr['weight_decay'],fused=tr.get('fused_optimizer',False));writer=SummaryWriter(a.run_dir/'tensorboard');best=-1;stale=0;history=[]
 configured_top_k=model.physchem_top_k;sparse_warmup=int(tr.get('expert_sparse_warmup_epochs',0))
 for epoch in range(1,tr['epochs']+1):
  active_top_k=0 if epoch<=sparse_warmup else configured_top_k;model.physchem_top_k=active_top_k
  if ema:ema.physchem_top_k=active_top_k
  started=time.monotonic();model.train();total=count=0
  for global_x,res,mask,y in train:
   global_x,res,mask,y=[v.to(device,non_blocking=True) for v in (global_x,res,mask,y)];opt.zero_grad(set_to_none=True);smooth=float(tr.get('label_smoothing',0));target=y*(1-smooth)+.5*smooth
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tr.get('amp_bfloat16',True)):
    logits=model(global_x,res,mask);loss=nn.functional.binary_cross_entropy_with_logits(logits,target)+float(tr.get('rank_weight',0))*binary_rank_loss(logits,y)
    if float(tr.get('expert_aux_weight',0)):
     if model.last_expert_logits is None:raise ValueError('expert auxiliary loss requires an expert model')
     auxiliary_target=target[:,None].expand_as(model.last_expert_logits);loss=loss+float(tr['expert_aux_weight'])*nn.functional.binary_cross_entropy_with_logits(model.last_expert_logits,auxiliary_target)
    if float(tr.get('expert_entropy_weight',0)) or float(tr.get('expert_balance_weight',0)):
     if model.last_expert_weights is None:raise ValueError('gate regularization requires a physchem expert model')
     weights=model.last_expert_weights.float();entropy=-(weights*weights.clamp_min(1e-8).log()).sum(1).mean();balance=(weights.mean(0)-1/weights.shape[1]).square().sum()
     loss=loss+float(tr.get('expert_entropy_weight',0))*entropy+float(tr.get('expert_balance_weight',0))*balance
   loss.backward();opt.step()
   if ema:
    with torch.no_grad():
     for ep,pv in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(pv,alpha=1-decay)
   total+=loss.item()*len(y);count+=len(y)
  seconds=time.monotonic()-started;eval_model=ema or model;truth,logits=infer(eval_model,loaders['validation'],device,tr.get('amp_bfloat16',False));metrics=binary_metrics(truth,logits);row={'epoch':epoch,'train_loss':total/count,'train_seconds':seconds,'train_samples_per_second':count/seconds,'validation':metrics,'physchem_top_k':active_top_k}
  if hasattr(eval_model,'global_segment_logits'):row['global_segment_weights']=torch.softmax(eval_model.global_segment_logits.detach().float(),0).cpu().tolist()
  history.append(row);print(json.dumps(row),flush=True);writer.add_scalar('validation/auroc',metrics['auroc'],epoch);writer.add_scalar('throughput/train_samples_per_second',row['train_samples_per_second'],epoch);state={'model':eval_model.state_dict(),'epoch':epoch,'validation':metrics,'config':c}
  if epoch%tr['checkpoint_every']==0:torch.save(state,a.run_dir/'checkpoints'/f'epoch_{epoch:03d}.pt')
  if epoch<=sparse_warmup:stale=0
  elif metrics['auroc']>best:best=metrics['auroc'];stale=0;torch.save(state,a.run_dir/'checkpoints'/'best.pt')
  else:stale+=1
  if stale>=tr['patience']:break
 writer.close();(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');state=torch.load(a.run_dir/'checkpoints'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);model.physchem_top_k=configured_top_k;truth,logits=infer(model,loaders['validation'],device,tr.get('amp_bfloat16',False));(a.run_dir/'validation_metrics.json').write_text(json.dumps({'uesolds':binary_metrics(truth,logits)},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=truth,logits=logits,entity_indices=np.asarray([v[0] for v in rows['validation']],np.int64))
 print(json.dumps({'best_epoch':state['epoch'],'best_validation_auroc':best,'test_evaluated':False}))
if __name__=='__main__':main()
