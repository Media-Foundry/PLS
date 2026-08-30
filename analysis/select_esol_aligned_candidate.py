"""Add full-coverage eSOL candidates to the frozen validation-selected structure blend."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from pls.evaluation.metrics import regression_metrics

def load(run):
 data=np.load(Path(run)/'validation_predictions.npz');return data['targets'],data['predictions'],data['entity_indices']

def selected_prediction(sequence_runs,geometry_run,tree_run,structure_weights):
 targets,_,entities=load(sequence_runs[0]);sequence=np.mean([load(run)[1] for run in sequence_runs],0);gt,gp,ge=load(geometry_run);tt,tp,te=load(tree_run)
 if not np.array_equal(gt,tt) or not np.array_equal(ge,te):raise ValueError('structure validation arrays do not align')
 positions={entity:index for index,entity in enumerate(entities)};selected=np.asarray([positions[entity] for entity in ge]);prediction=sequence.copy();prediction[selected]=(1-sum(structure_weights))*sequence[selected]+structure_weights[0]*gp+structure_weights[1]*tp;return targets,prediction,entities

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-report',type=Path,required=True);parser.add_argument('--candidate',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();source=json.loads(args.base_report.read_text())
 targets,leaf,entities=selected_prediction(source['sequence_runs'],source['geometry_run'],source['leaf_run'],source['leaf_structure_weights']);observed,full,full_entities=selected_prediction(source['sequence_runs'],source['geometry_run'],source['full_run'],source['full_structure_weights'])
 if not np.array_equal(targets,observed) or not np.array_equal(entities,full_entities):raise ValueError('frozen base arrays do not align')
 base=source['leaf_weight']*leaf+source['full_weight']*full;candidates=[]
 for run in args.candidate:
  candidate_targets,prediction,candidate_entities=load(run)
  if not np.array_equal(targets,candidate_targets) or not np.array_equal(entities,candidate_entities):raise ValueError(f'validation alignment mismatch: {run}')
  candidates.append(prediction)
 weights=np.zeros(len(candidates));best=float(spearmanr(targets,base).statistic)
 for _ in range(6):
  changed=False
  for index in range(len(weights)):
   winner=(best,weights[index])
   for value in np.linspace(0,.25,251):
    proposed=weights.copy();proposed[index]=value
    if proposed.sum()>.5:continue
    prediction=(1-proposed.sum())*base+sum(weight*candidate for weight,candidate in zip(proposed,candidates));score=float(spearmanr(targets,prediction).statistic)
    if score>winner[0]+1e-12:winner=(score,value)
   if winner[1]!=weights[index]:weights[index]=winner[1];best=winner[0];changed=True
  if not changed:break
 prediction=(1-weights.sum())*base+sum(weight*candidate for weight,candidate in zip(weights,candidates));slope,intercept=np.polyfit(prediction,targets,1);affine=slope*prediction+intercept
 report={'selection_data':'strict-validation only','test_evaluated':False,'objective':'spearman','base_report':str(args.base_report),'base_weight':float(1-weights.sum()),'candidate_weights':{str(run):float(weight) for run,weight in zip(args.candidate,weights)},'metrics':regression_metrics(targets,prediction),'affine_slope':float(slope),'affine_intercept':float(intercept),'affine_metrics':regression_metrics(targets,affine),'entity_count':int(len(entities))}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
