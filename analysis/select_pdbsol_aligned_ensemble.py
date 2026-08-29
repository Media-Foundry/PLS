"""Validation-only coordinate search for adding aligned models to a frozen base ensemble."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import matthews_corrcoef,roc_auc_score
from pls.evaluation.metrics import binary_metrics

def load(run):
 data=np.load(Path(run)/'validation_predictions.npz');return data['targets'],data['logits'],data['entity_indices']

def frozen_components(report):
 """Return run weights from either a legacy sweep or a weighted report."""
 if 'reports' in report:
  entry=max(report['reports'],key=lambda row:row['uncalibrated']['auroc'])
  return {str(run):1/len(entry['runs']) for run in entry['runs']}
 if 'frozen_run_weights' in report:
  components={str(run):float(report['base_weight'])*float(weight) for run,weight in report['frozen_run_weights'].items()}
 else:
  components={str(run):float(report['base_weight'])/len(report['base_runs']) for run in report['base_runs']}
 for run,weight in report.get('candidate_weights',{}).items():components[str(run)]=components.get(str(run),0)+float(weight)
 return components

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-report',type=Path,required=True);parser.add_argument('--candidate',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 prior=json.loads(args.base_report.read_text());components=frozen_components(prior);base_runs=list(components);targets,_,entities=load(base_runs[0]);base=sum(weight*load(run)[1] for run,weight in components.items());base/=sum(components.values());candidates=[]
 for run in args.candidate:
  observed,prediction,observed_entities=load(run)
  if not np.array_equal(targets,observed) or not np.array_equal(entities,observed_entities):raise ValueError(f'validation alignment mismatch: {run}')
  candidates.append(prediction)
 weights=np.zeros(len(candidates));best=roc_auc_score(targets,base);grid=np.linspace(0,.25,26)
 for _ in range(6):
  changed=False
  for index in range(len(weights)):
   winner=(best,weights[index])
   for value in grid:
    proposed=weights.copy();proposed[index]=value
    if proposed.sum()>.5:continue
    logits=(1-proposed.sum())*base+sum(weight*prediction for weight,prediction in zip(proposed,candidates));score=roc_auc_score(targets,logits)
    if score>winner[0]+1e-12:winner=(score,value)
   if winner[1]!=weights[index]:weights[index]=winner[1];best=winner[0];changed=True
  if not changed:break
 logits=(1-weights.sum())*base+sum(weight*prediction for weight,prediction in zip(weights,candidates))
 def nll(log_temperature):
  scaled=logits/np.exp(log_temperature);return float(np.mean(np.logaddexp(0,scaled)-targets*scaled))
 temperature=float(np.exp(minimize_scalar(nll,bounds=(-2,2),method='bounded').x));calibrated=logits/temperature;probability=1/(1+np.exp(-np.clip(calibrated,-50,50)));thresholds=np.unique(np.quantile(probability,np.linspace(0,1,2001)));threshold,mcc=max(((float(value),float(matthews_corrcoef(targets,probability>=value))) for value in thresholds),key=lambda row:row[1])
 report={'selection_data':'strict-validation only','test_evaluated':False,'base_report':str(args.base_report),'base_runs':base_runs,'frozen_run_weights':components,'base_weight':float(1-weights.sum()),'candidate_weights':{str(run):float(weight) for run,weight in zip(args.candidate,weights)},'uncalibrated':binary_metrics(targets,logits),'temperature':temperature,'calibrated':binary_metrics(targets,calibrated),'mcc_threshold':threshold,'mcc_at_selected_threshold':mcc,'entity_count':int(len(entities))}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
