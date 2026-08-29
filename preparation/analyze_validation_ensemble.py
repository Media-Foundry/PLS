"""Validation-only aligned logit ensembling, temperature scaling and threshold selection."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from pls.evaluation.metrics import binary_metrics
def best_mcc_threshold(targets,probabilities):
 order=np.argsort(-probabilities,kind='stable');scores=probabilities[order];truth=targets[order].astype(np.int64);tp=np.cumsum(truth);fp=np.cumsum(1-truth);fn=tp[-1]-tp;tn=fp[-1]-fp;den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=np.float64),where=den>0);last=np.r_[scores[1:]<scores[:-1],True];candidates=np.flatnonzero(last);best=candidates[np.argmax(mcc[candidates])];return float(scores[best]),float(mcc[best])
def main():
 p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=('pdbsol','uesolds'));a=p.parse_args();loaded=[]
 for run in a.run_dirs:
  task_path=run/f'validation_{a.task}_predictions.npz' if a.task else None;path=task_path if task_path is not None and task_path.exists() else run/'validation_predictions.npz';x=np.load(path);key='logits' if 'logits' in x else 'predictions';loaded.append((run,x['entity_indices'],x['targets'],x[key]))
 entities,targets=loaded[0][1],loaded[0][2]
 for run,e,t,_ in loaded[1:]:
  if not np.array_equal(e,entities) or not np.array_equal(t,targets):raise ValueError(f'unaligned validation predictions: {run}')
 reports=[]
 for size in range(1,len(loaded)+1):
  for chosen in itertools.combinations(range(len(loaded)),size):
   logits=np.mean([loaded[i][3] for i in chosen],axis=0);objective=lambda log_t:np.mean(np.logaddexp(0,logits/np.exp(log_t))-targets*logits/np.exp(log_t));fit=minimize_scalar(objective,bounds=(-3,3),method='bounded');temperature=float(np.exp(fit.x));scaled=logits/temperature;prob=1/(1+np.exp(-np.clip(scaled,-50,50)));threshold,mcc=best_mcc_threshold(targets,prob);report={'runs':[str(loaded[i][0]) for i in chosen],'ensemble_size':size,'uncalibrated':binary_metrics(targets,logits),'temperature':temperature,'calibrated':binary_metrics(targets,scaled),'mcc_threshold':threshold,'mcc_at_selected_threshold':mcc};reports.append(report)
 reports.sort(key=lambda x:(x['calibrated']['auroc'],x['calibrated']['auprc']),reverse=True);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'selection_data':'strict-validation only','reports':reports},indent=2,sort_keys=True)+'\n');print(json.dumps(reports[0],indent=2))
if __name__=='__main__':main()
