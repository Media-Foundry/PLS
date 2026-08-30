"""Validation-only incremental candidates selected by exact maximum MCC."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pls.evaluation.metrics import binary_metrics

def load(run):
 data=np.load(Path(run)/'validation_predictions.npz');return data['targets'],data['logits'],data['entity_indices']

def base_components(report):
 if 'reports' in report:
  entry=max(report['reports'],key=lambda row:row['mcc_at_selected_threshold']);return {str(run):1/len(entry['runs']) for run in entry['runs']}
 if 'run_weights' in report:return {str(run):float(weight) for run,weight in report['run_weights'].items() if float(weight)>0}
 if 'base_run_weights' in report:
  components={str(run):float(report['base_weight'])*float(weight) for run,weight in report['base_run_weights'].items()}
  for run,weight in report.get('candidate_weights',{}).items():components[str(run)]=components.get(str(run),0)+float(weight)
  return {run:weight for run,weight in components.items() if weight>0}
 if 'frozen_run_weights' in report:
  components={str(run):float(report['base_weight'])*float(weight) for run,weight in report['frozen_run_weights'].items()}
  for run,weight in report.get('candidate_weights',{}).items():components[str(run)]=components.get(str(run),0)+float(weight)
  return components
 raise ValueError('unsupported MCC base report')

def exact_max_mcc(targets,scores):
 targets=np.asarray(targets,dtype=np.int64);scores=np.asarray(scores,dtype=np.float64);order=np.argsort(-scores,kind='stable');truth=targets[order];ranked=scores[order];ends=np.r_[np.flatnonzero(ranked[:-1]!=ranked[1:]),len(ranked)-1];tp=np.cumsum(truth)[ends].astype(float);predicted=(ends+1).astype(float);fp=predicted-tp;positive=float(truth.sum());negative=float(len(truth)-positive);fn=positive-tp;tn=negative-fp;numerator=tp*tn-fp*fn;denominator=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0);winner=int(np.argmax(mcc));threshold=float(ranked[ends[winner]]);return float(mcc[winner]),threshold

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-report',type=Path,required=True);parser.add_argument('--candidate',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();source=json.loads(args.base_report.read_text());components=base_components(source);runs=list(components);targets,_,entities=load(runs[0]);mass=sum(components.values());base=sum(weight*load(run)[1] for run,weight in components.items())/mass;candidates=[]
 for run in args.candidate:
  observed,prediction,observed_entities=load(run)
  if not np.array_equal(targets,observed) or not np.array_equal(entities,observed_entities):raise ValueError(f'validation alignment mismatch: {run}')
  candidates.append(prediction)
 weights=np.zeros(len(candidates));best,_=exact_max_mcc(targets,base);grid=np.linspace(0,.25,51)
 for _ in range(6):
  changed=False
  for index in range(len(weights)):
   winner=(best,weights[index])
   for value in grid:
    proposed=weights.copy();proposed[index]=value
    if proposed.sum()>.5:continue
    logits=(1-proposed.sum())*base+sum(weight*prediction for weight,prediction in zip(proposed,candidates));score,_=exact_max_mcc(targets,logits)
    if score>winner[0]+1e-12:winner=(score,value)
   if winner[1]!=weights[index]:best,weights=winner[0],weights.copy();weights[index]=winner[1];changed=True
  if not changed:break
 logits=(1-weights.sum())*base+sum(weight*prediction for weight,prediction in zip(weights,candidates));mcc,threshold=exact_max_mcc(targets,logits);report={'selection_data':'strict-validation only','test_evaluated':False,'objective':'mcc','base_report':str(args.base_report),'base_weight':float(1-weights.sum()),'base_run_weights':components,'candidate_weights':{str(run):float(weight) for run,weight in zip(args.candidate,weights)},'uncalibrated':binary_metrics(targets,logits),'mcc_at_selected_threshold':mcc,'logit_threshold':threshold,'probability_threshold':float(1/(1+np.exp(-np.clip(threshold,-50,50)))),'entity_count':int(len(entities))};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main()
