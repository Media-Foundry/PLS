"""Leakage-safe classical eSOL regression over pooled V4 structure descriptors."""
from __future__ import annotations
import argparse,csv,json,os,pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from torch.utils.tensorboard import SummaryWriter
from pls.evaluation.metrics import regression_metrics

SOURCE='eSOL_FGNNSol'
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--allow-test-evaluation',action='store_true');a=p.parse_args();c=json.loads(a.config.read_text());d=c['data'];mc=c['model'];tr=c['training']
 if c.get('evaluate_test',False) or a.allow_test_evaluation:p.error('this validation-selection trainer never evaluates test')
 if os.environ.get('HIP_VISIBLE_DEVICES')!=str(tr['hip_device']):p.error('HIP device mismatch')
 with open(d['entities'],newline='',encoding='utf-8') as h:entities=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(entities)}
 rows={s:[] for s in ('train','validation')}
 with open(d['observation_split'],newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']==SOURCE and r['split'] in rows:rows[r['split']].append((index[r['sequence_sha256']],float(r['target_value'])))
 pooled=Path(d['pooled_structure_dir']);status=np.load(pooled/'status.npy',mmap_mode='r');structure=np.load(pooled/'descriptors.npy',mmap_mode='r');sequence=np.load(Path(d['sequence_descriptor_dir'])/'descriptors.npy',mmap_mode='r') if d.get('sequence_descriptor_dir') else None
 rows={s:[v for v in values if status[v[0]]==1] for s,values in rows.items()};train_entities=np.asarray([i for i,_ in rows['train']]);columns=np.asarray(mc.get('columns',list(range(structure.shape[1]))),dtype=np.int64)
 mean=np.asarray(structure[train_entities][:,columns],np.float64).mean(0);std=np.maximum(np.asarray(structure[train_entities][:,columns],np.float64).std(0),1e-6)
 def matrix(values):
  ids=np.asarray([i for i,_ in values]);parts=[(np.asarray(structure[ids][:,columns],np.float32)-mean)/std]
  if sequence is not None:
   sm=np.asarray(sequence[train_entities],np.float64).mean(0);ss=np.maximum(np.asarray(sequence[train_entities],np.float64).std(0),1e-6);parts.append((np.asarray(sequence[ids],np.float32)-sm)/ss)
  return np.concatenate(parts,1),np.asarray([y for _,y in values],np.float32),ids
 x,y,_=matrix(rows['train']);vx,vy,vid=matrix(rows['validation']);kind=mc['kind'];common={'random_state':tr['seed']}
 if kind=='extra_trees':model=ExtraTreesRegressor(n_estimators=mc.get('trees',512),max_features=mc.get('max_features',1.0),min_samples_leaf=mc.get('min_samples_leaf',2),n_jobs=tr.get('workers',-1),**common)
 elif kind=='random_forest':model=RandomForestRegressor(n_estimators=mc.get('trees',512),max_features=mc.get('max_features',.5),min_samples_leaf=mc.get('min_samples_leaf',2),n_jobs=tr.get('workers',-1),**common)
 elif kind=='hist_gradient_boosting':model=HistGradientBoostingRegressor(max_iter=mc.get('iterations',300),learning_rate=mc.get('learning_rate',.05),max_leaf_nodes=mc.get('max_leaf_nodes',15),l2_regularization=mc.get('l2_regularization',1.),**common)
 else:raise ValueError(kind)
 started=time.monotonic();model.fit(x,y);seconds=time.monotonic()-started;pred=model.predict(vx);metrics=regression_metrics(vy,pred);history=[{'fit_seconds':seconds,'train_entities':len(y),'validation_entities':len(vy),'validation':metrics}];(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');(a.run_dir/'validation_metrics.json').write_text(json.dumps({'esol':metrics},indent=2,sort_keys=True)+'\n');np.savez_compressed(a.run_dir/'validation_predictions.npz',targets=vy,predictions=pred,entity_indices=vid);pickle.dump(model,(a.run_dir/'checkpoints'/'best.pkl').open('wb'));writer=SummaryWriter(a.run_dir/'tensorboard');writer.add_scalar('validation/spearman',metrics['spearman'],0);writer.close();print(json.dumps({'fit_seconds':seconds,'validation':metrics,'test_evaluated':False}))
if __name__=='__main__':main()
