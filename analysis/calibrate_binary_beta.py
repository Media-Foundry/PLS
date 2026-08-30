"""Cross-fitted monotonic beta calibration of a frozen binary ensemble."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from pls.evaluation.metrics import binary_metrics
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_binary_validation import weights

def softplus(value):return np.logaddexp(0,value)
def beta_logits(logits,parameters):
 probability=1/(1+np.exp(-np.clip(logits,-50,50)));probability=np.clip(probability,1e-8,1-1e-8);a=softplus(parameters[0])+1e-6;b=softplus(parameters[1])+1e-6;return a*np.log(probability)-b*np.log1p(-probability)+parameters[2]
def fit_beta(logits,targets):
 def nll(parameters):
  calibrated=beta_logits(logits,parameters);return float(np.mean(np.logaddexp(0,calibrated)-targets*calibrated))
 return minimize(nll,np.asarray([0.,0.,0.]),method='L-BFGS-B',bounds=((-6,6),(-6,6),(-8,8)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--folds',type=int,default=5);args=parser.parse_args();source=json.loads(args.report.read_text());components=weights(source);first=np.load(Path(next(iter(components)))/'validation_predictions.npz');targets=first['targets'];entities=first['entity_indices'];logits=np.zeros_like(first['logits'],dtype=np.float64);mass=sum(components.values())
 for run,weight in components.items():
  data=np.load(Path(run)/'validation_predictions.npz')
  if not np.array_equal(targets,data['targets']) or not np.array_equal(entities,data['entity_indices']):raise ValueError(f'validation alignment mismatch: {run}')
  logits+=weight/mass*data['logits']
 fit=fit_beta(logits,targets);calibrated=beta_logits(logits,fit.x);crossfit=np.zeros_like(logits);fold_reports=[]
 for fold,(train_indices,heldout_indices) in enumerate(StratifiedKFold(args.folds,shuffle=True,random_state=20260830).split(logits,targets)):
  fold_fit=fit_beta(logits[train_indices],targets[train_indices]);heldout=beta_logits(logits[heldout_indices],fold_fit.x);crossfit[heldout_indices]=heldout;fold_reports.append({'fold':fold,'train_entities':int(len(train_indices)),'heldout_entities':int(len(heldout_indices)),'parameters':fold_fit.x.tolist(),'optimization_success':bool(fold_fit.success),'heldout':binary_metrics(targets[heldout_indices],heldout)})
 report={'selection_data':'strict-validation only','test_evaluated':False,'method':'monotonic_beta','source_report':str(args.report),'run_weights':components,'entity_count':int(len(entities)),'uncalibrated':binary_metrics(targets,logits),'parameters':fit.x.tolist(),'optimization_success':bool(fit.success),'calibrated':binary_metrics(targets,calibrated),'crossfit_folds':fold_reports,'crossfit_calibrated':binary_metrics(targets,crossfit)};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main()
