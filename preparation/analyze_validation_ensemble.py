"""Validation-only aligned logit ensembling, temperature scaling and threshold selection."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import matthews_corrcoef
from pls.evaluation.metrics import binary_metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=('pdbsol','uesolds'));a=p.parse_args();loaded=[]
 for run in a.run_dirs:
  path=run/(f'validation_{a.task}_predictions.npz' if a.task else 'validation_predictions.npz');x=np.load(path);key='logits' if 'logits' in x else 'predictions';loaded.append((run,x['entity_indices'],x['targets'],x[key]))
 entities,targets=loaded[0][1],loaded[0][2]
 for run,e,t,_ in loaded[1:]:
  if not np.array_equal(e,entities) or not np.array_equal(t,targets):raise ValueError(f'unaligned validation predictions: {run}')
 reports=[]
 for size in range(1,len(loaded)+1):
  for chosen in itertools.combinations(range(len(loaded)),size):
   logits=np.mean([loaded[i][3] for i in chosen],axis=0);objective=lambda log_t:np.mean(np.logaddexp(0,logits/np.exp(log_t))-targets*logits/np.exp(log_t));fit=minimize_scalar(objective,bounds=(-3,3),method='bounded');temperature=float(np.exp(fit.x));scaled=logits/temperature;prob=1/(1+np.exp(-np.clip(scaled,-50,50)));thresholds=np.linspace(.05,.95,901);mcc=np.array([matthews_corrcoef(targets,prob>=v) for v in thresholds]);best=int(mcc.argmax());report={'runs':[str(loaded[i][0]) for i in chosen],'ensemble_size':size,'uncalibrated':binary_metrics(targets,logits),'temperature':temperature,'calibrated':binary_metrics(targets,scaled),'mcc_threshold':float(thresholds[best]),'mcc_at_selected_threshold':float(mcc[best])};reports.append(report)
 reports.sort(key=lambda x:(x['calibrated']['auroc'],x['calibrated']['auprc']),reverse=True);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'selection_data':'strict-validation only','reports':reports},indent=2,sort_keys=True)+'\n');print(json.dumps(reports[0],indent=2))
if __name__=='__main__':main()
