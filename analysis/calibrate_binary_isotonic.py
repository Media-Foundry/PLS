"""Cross-fitted isotonic calibration of a frozen validation ensemble."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from pls.evaluation.metrics import binary_metrics
from pls.evaluation.component_crossfit import (si_component_groups,
                                                stratified_component_folds)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_binary_validation import weights

def probability_logits(probability):
 probability=np.clip(np.asarray(probability,dtype=np.float64),1e-6,1-1e-6)
 return np.log(probability)-np.log1p(-probability)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--folds',type=int,default=5);parser.add_argument('--entities',type=Path,default=Path('benchmark/generated/sequence_entities.csv'));parser.add_argument('--observation-split',type=Path,default=Path('benchmark/generated/strict_si30_observation_split.csv'));args=parser.parse_args();source=json.loads(args.report.read_text());components=weights(source);first=np.load(Path(next(iter(components)))/'validation_predictions.npz');targets=first['targets'];entities=first['entity_indices'];groups=si_component_groups(entities,args.entities,args.observation_split);logits=np.zeros_like(first['logits'],dtype=np.float64);mass=sum(components.values())
 for run,weight in components.items():
  data=np.load(Path(run)/'validation_predictions.npz')
  if not np.array_equal(targets,data['targets']) or not np.array_equal(entities,data['entity_indices']):raise ValueError(f'validation alignment mismatch: {run}')
  logits+=weight/mass*data['logits']
 full=IsotonicRegression(out_of_bounds='clip').fit(logits,targets);calibrated=probability_logits(full.predict(logits));crossfit=np.zeros_like(logits);fold_reports=[]
 for fold,(train_indices,heldout_indices) in enumerate(stratified_component_folds(targets,groups,args.folds,20260830)):
  model=IsotonicRegression(out_of_bounds='clip').fit(logits[train_indices],targets[train_indices]);heldout=probability_logits(model.predict(logits[heldout_indices]));crossfit[heldout_indices]=heldout;fold_reports.append({'fold':fold,'train_entities':int(len(train_indices)),'heldout_entities':int(len(heldout_indices)),'train_components':int(len(np.unique(groups[train_indices]))),'heldout_components':int(len(np.unique(groups[heldout_indices]))),'knots':int(len(model.X_thresholds_)),'heldout':binary_metrics(targets[heldout_indices],heldout)})
 report={'selection_data':'strict-validation only','test_evaluated':False,'method':'isotonic','source_report':str(args.report),'run_weights':components,'entity_count':int(len(entities)),'si_component_count':int(len(np.unique(groups))),'crossfit_grouping':'strict_si30_component_root_sha256','uncalibrated':binary_metrics(targets,logits),'calibrated':binary_metrics(targets,calibrated),'crossfit_folds':fold_reports,'crossfit_calibrated':binary_metrics(targets,crossfit),'deployment_mapping':{'x_thresholds':full.X_thresholds_.tolist(),'y_thresholds':full.y_thresholds_.tolist()}};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main()
