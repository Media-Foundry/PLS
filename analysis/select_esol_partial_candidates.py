"""Add structure-only eSOL candidates while preserving sequence fallback entities."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from pls.evaluation.metrics import regression_metrics
from select_esol_aligned_candidate import load,selected_prediction

def frozen_prediction(report_path):
 source=json.loads(Path(report_path).read_text())
 if 'reports' in source and 'structure_runs' in source:
  entry=source['reports'][0]
  return selected_prediction(source['sequence_runs'],source['structure_runs'][0],source['structure_runs'][1],entry['structure_weights'])
 if 'sequence_runs' in source:
  targets,leaf,entities=selected_prediction(source['sequence_runs'],source['geometry_run'],source['leaf_run'],source['leaf_structure_weights']);observed,full,full_entities=selected_prediction(source['sequence_runs'],source['geometry_run'],source['full_run'],source['full_structure_weights'])
  if not np.array_equal(targets,observed) or not np.array_equal(entities,full_entities):raise ValueError('frozen base arrays do not align')
  return targets,source['leaf_weight']*leaf+source['full_weight']*full,entities
 if 'base_report' not in source or 'candidate_weights' not in source:raise ValueError(f'unsupported eSOL selection report: {report_path}')
 targets,base,entities=frozen_prediction(source['base_report']);positions={entity:index for index,entity in enumerate(entities)};prediction=base.copy()
 for run,weight in source['candidate_weights'].items():
  candidate_targets,values,candidate_entities=load(Path(run));selected=np.asarray([positions[entity] for entity in candidate_entities])
  if not np.array_equal(targets[selected],candidate_targets):raise ValueError(f'validation target mismatch: {run}')
  prediction[selected]+=float(weight)*(values-base[selected])
 return targets,prediction,entities

def objective_score(targets,prediction,objective):
 if objective=='spearman':return float(spearmanr(targets,prediction).statistic)
 if objective=='pearson':return float(np.corrcoef(targets,prediction)[0,1])
 if objective=='rmse':return -float(np.sqrt(np.mean((targets-prediction)**2)))
 if objective=='mae':return -float(np.mean(np.abs(targets-prediction)))
 raise ValueError(objective)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-report',type=Path,required=True);parser.add_argument('--candidate',type=Path,action='append',required=True);parser.add_argument('--objective',choices=('spearman','pearson','rmse','mae'),default='spearman');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();targets,base,entities=frozen_prediction(args.base_report);positions={entity:index for index,entity in enumerate(entities)};candidate_values=[];candidate_positions=[]
 for run in args.candidate:
  candidate_targets,prediction,candidate_entities=load(run);selected=np.asarray([positions[entity] for entity in candidate_entities])
  if not np.array_equal(targets[selected],candidate_targets):raise ValueError(f'validation target mismatch: {run}')
  candidate_values.append(prediction);candidate_positions.append(selected)
 weights=np.zeros(len(candidate_values));best=objective_score(targets,base,args.objective)
 for _ in range(6):
  changed=False
  for index in range(len(weights)):
   winner=(best,weights[index])
   for value in np.linspace(0,.3,301):
    proposed=weights.copy();proposed[index]=value;prediction=base.copy()
    for weight,values,selected in zip(proposed,candidate_values,candidate_positions):prediction[selected]+=weight*(values-base[selected])
    score=objective_score(targets,prediction,args.objective)
    if score>winner[0]+1e-12:winner=(score,value)
   if winner[1]!=weights[index]:weights[index]=winner[1];best=winner[0];changed=True
  if not changed:break
 prediction=base.copy()
 for weight,values,selected in zip(weights,candidate_values,candidate_positions):prediction[selected]+=weight*(values-base[selected])
 slope,intercept=np.polyfit(prediction,targets,1);affine=slope*prediction+intercept;report={'selection_data':'strict-validation only','test_evaluated':False,'objective':args.objective,'base_report':str(args.base_report),'candidate_weights':{str(run):float(weight) for run,weight in zip(args.candidate,weights)},'candidate_entities':{str(run):int(len(selected)) for run,selected in zip(args.candidate,candidate_positions)},'fallback_rule':'Keep the frozen base prediction for entities absent from each partial candidate.','metrics':regression_metrics(targets,prediction),'affine_slope':float(slope),'affine_intercept':float(intercept),'affine_metrics':regression_metrics(targets,affine),'entity_count':int(len(entities))};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
