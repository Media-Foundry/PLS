"""Fit validation-only affine Platt calibration to a frozen binary ensemble."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from pls.evaluation.component_crossfit import (si_component_groups,
                                                stratified_component_folds)
from pls.evaluation.metrics import binary_metrics
from select_pdbsol_aligned_ensemble import frozen_components

def weights(report):
 if 'run_weights' in report:return {str(run):float(weight) for run,weight in report['run_weights'].items()}
 return frozen_components(report)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--folds',type=int,default=5);parser.add_argument('--entities',type=Path,default=Path('benchmark/generated/sequence_entities.csv'));parser.add_argument('--observation-split',type=Path,default=Path('benchmark/generated/strict_si30_observation_split.csv'));args=parser.parse_args();source=json.loads(args.report.read_text());components=weights(source);first=np.load(Path(next(iter(components)))/'validation_predictions.npz');targets=first['targets'];entities=first['entity_indices'];groups=si_component_groups(entities,args.entities,args.observation_split);logits=np.zeros_like(first['logits'],dtype=np.float64);mass=sum(components.values())
 for run,weight in components.items():
  data=np.load(Path(run)/'validation_predictions.npz')
  if not np.array_equal(targets,data['targets']) or not np.array_equal(entities,data['entity_indices']):raise ValueError(f'validation alignment mismatch: {run}')
  logits+=weight/mass*data['logits']
 def fit_platt(train_logits,train_targets):
  def nll(parameters):
   calibrated=np.exp(parameters[0])*train_logits+parameters[1];return float(np.mean(np.logaddexp(0,calibrated)-train_targets*calibrated))
  return minimize(nll,np.zeros(2),method='L-BFGS-B',bounds=((-4,4),(-5,5)))
 fit=fit_platt(logits,targets);slope=float(np.exp(fit.x[0]));intercept=float(fit.x[1]);calibrated=slope*logits+intercept;crossfit=np.zeros_like(logits);folds=[]
 for fold,(train_indices,heldout_indices) in enumerate(stratified_component_folds(targets,groups,args.folds,20260830)):
  fold_fit=fit_platt(logits[train_indices],targets[train_indices]);fold_slope=float(np.exp(fold_fit.x[0]));fold_intercept=float(fold_fit.x[1]);crossfit[heldout_indices]=fold_slope*logits[heldout_indices]+fold_intercept;folds.append({'fold':fold,'train_entities':int(len(train_indices)),'heldout_entities':int(len(heldout_indices)),'train_components':int(len(np.unique(groups[train_indices]))),'heldout_components':int(len(np.unique(groups[heldout_indices]))),'slope':fold_slope,'intercept':fold_intercept,'optimization_success':bool(fold_fit.success)})
 report={'selection_data':'strict-validation only','test_evaluated':False,'source_report':str(args.report),'run_weights':components,'entity_count':int(len(entities)),'si_component_count':int(len(np.unique(groups))),'crossfit_grouping':'strict_si30_component_root_sha256','uncalibrated':binary_metrics(targets,logits),'platt_slope':slope,'platt_intercept':intercept,'calibrated':binary_metrics(targets,calibrated),'crossfit_folds':folds,'crossfit_calibrated':binary_metrics(targets,crossfit),'nll':float(fit.fun),'optimization_success':bool(fit.success)};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
