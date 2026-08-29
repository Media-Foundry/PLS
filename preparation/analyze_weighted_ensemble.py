"""Validation-only regularized nonnegative logit blending and calibration."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from pls.evaluation.metrics import binary_metrics
from analyze_validation_ensemble import best_mcc_threshold

def main():
 p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+',type=Path);p.add_argument('--task',choices=('pdbsol','uesolds'));p.add_argument('--regularization',type=float,default=.01);p.add_argument('--objective',choices=('bce','rank'),default='bce');p.add_argument('--rank-pairs',type=int,default=200000);p.add_argument('--output',type=Path,required=True);a=p.parse_args();loaded=[]
 for run in a.run_dirs:
  task_path=run/f'validation_{a.task}_predictions.npz' if a.task else None;path=task_path if task_path is not None and task_path.exists() else run/'validation_predictions.npz';x=np.load(path);key='logits' if 'logits' in x else 'predictions';loaded.append((run,x['entity_indices'],x['targets'],x[key]))
 entities,targets=loaded[0][1],loaded[0][2]
 for run,e,t,_ in loaded[1:]:
  if not np.array_equal(e,entities) or not np.array_equal(t,targets):raise ValueError(f'unaligned predictions: {run}')
 matrix=np.stack([x[3] for x in loaded],1).astype(np.float64);n=len(loaded);uniform=np.full(n,1/n)
 if a.objective=='rank':
  rng=np.random.default_rng(20260829);positive=np.flatnonzero(targets>.5);negative=np.flatnonzero(targets<=.5);pair_difference=matrix[rng.choice(positive,a.rank_pairs),:]-matrix[rng.choice(negative,a.rank_pairs),:]
 def objective(weights):
  data_loss=np.mean(np.logaddexp(0,-pair_difference@weights)) if a.objective=='rank' else np.mean(np.logaddexp(0,matrix@weights)-targets*(matrix@weights));return float(data_loss+a.regularization*np.mean((weights-uniform)**2))
 fit=minimize(objective,uniform,method='SLSQP',bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':1000});weights=fit.x;logits=matrix@weights;temperature_fit=minimize_scalar(lambda log_t:np.mean(np.logaddexp(0,logits/np.exp(log_t))-targets*logits/np.exp(log_t)),bounds=(-3,3),method='bounded');temperature=float(np.exp(temperature_fit.x));scaled=logits/temperature;prob=1/(1+np.exp(-np.clip(scaled,-50,50)));threshold,mcc=best_mcc_threshold(targets,prob);report={'selection_data':'strict-validation only','objective':a.objective,'regularization':a.regularization,'rank_pairs':a.rank_pairs if a.objective=='rank' else None,'optimizer_success':bool(fit.success),'runs':[str(x[0]) for x in loaded],'weights':[float(x) for x in weights],'uncalibrated':binary_metrics(targets,logits),'temperature':temperature,'calibrated':binary_metrics(targets,scaled),'mcc_threshold':threshold,'mcc_at_selected_threshold':mcc};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
