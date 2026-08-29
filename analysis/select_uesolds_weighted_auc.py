"""Validation-only non-negative weighting of UESolDS capacity-diverse models."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import matthews_corrcoef,roc_auc_score
from pls.evaluation.metrics import binary_metrics

def load(run):
 data=np.load(Path(run)/'validation_predictions.npz');return data['targets'],data['logits'],data['entity_indices']

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-report',type=Path,default=Path('outputs/validation_selection/uesolds_shift_tcn_capacity_ensembles.json'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();source=json.loads(args.source_report.read_text())['reports'];base=max(source,key=lambda row:row['uncalibrated']['auroc']);runs=sorted({run for row in source for run in row['runs']});targets,_,entities=load(runs[0]);predictions=[]
 for run in runs:
  observed,prediction,observed_entities=load(run)
  if not np.array_equal(targets,observed) or not np.array_equal(entities,observed_entities):raise ValueError(f'validation alignment mismatch: {run}')
  predictions.append(prediction)
 predictions=np.asarray(predictions);weights=np.asarray([1/len(base['runs']) if run in base['runs'] else 0 for run in runs]);best=roc_auc_score(targets,weights@predictions)
 for _ in range(8):
  changed=False
  for index in range(len(weights)):
   winner=(best,weights.copy())
   for value in np.linspace(0,.4,81):
    proposed=weights.copy();other=proposed.sum()-proposed[index]
    if other>0:proposed*=((1-value)/other);proposed[index]=value
    else:proposed[:]=0;proposed[index]=1
    score=roc_auc_score(targets,proposed@predictions)
    if score>winner[0]+1e-12:winner=(score,proposed.copy())
   if winner[0]>best:best,weights=winner;changed=True
  if not changed:break
 logits=weights@predictions
 def nll(log_temperature):
  scaled=logits/np.exp(log_temperature);return float(np.mean(np.logaddexp(0,scaled)-targets*scaled))
 temperature=float(np.exp(minimize_scalar(nll,bounds=(-2,2),method='bounded').x));calibrated=logits/temperature;probability=1/(1+np.exp(-np.clip(calibrated,-50,50)));thresholds=np.unique(np.quantile(probability,np.linspace(0,1,2001)));threshold,mcc=max(((float(value),float(matthews_corrcoef(targets,probability>=value))) for value in thresholds),key=lambda row:row[1]);report={'selection_data':'strict-validation only','test_evaluated':False,'objective':'auroc','source_report':str(args.source_report),'runs':runs,'weights':weights.tolist(),'uncalibrated':binary_metrics(targets,logits),'temperature':temperature,'calibrated':binary_metrics(targets,calibrated),'mcc_threshold':threshold,'mcc_at_selected_threshold':mcc,'entity_count':int(len(entities))};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
