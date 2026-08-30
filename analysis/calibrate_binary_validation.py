"""Fit validation-only affine Platt calibration to a frozen binary ensemble."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from pls.evaluation.metrics import binary_metrics
from select_pdbsol_aligned_ensemble import frozen_components

def weights(report):
 if 'run_weights' in report:return {str(run):float(weight) for run,weight in report['run_weights'].items()}
 return frozen_components(report)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();source=json.loads(args.report.read_text());components=weights(source);first=np.load(Path(next(iter(components)))/'validation_predictions.npz');targets=first['targets'];entities=first['entity_indices'];logits=np.zeros_like(first['logits'],dtype=np.float64);mass=sum(components.values())
 for run,weight in components.items():
  data=np.load(Path(run)/'validation_predictions.npz')
  if not np.array_equal(targets,data['targets']) or not np.array_equal(entities,data['entity_indices']):raise ValueError(f'validation alignment mismatch: {run}')
  logits+=weight/mass*data['logits']
 def nll(parameters):
  calibrated=np.exp(parameters[0])*logits+parameters[1];return float(np.mean(np.logaddexp(0,calibrated)-targets*calibrated))
 fit=minimize(nll,np.zeros(2),method='L-BFGS-B',bounds=((-4,4),(-5,5)));slope=float(np.exp(fit.x[0]));intercept=float(fit.x[1]);calibrated=slope*logits+intercept;report={'selection_data':'strict-validation only','test_evaluated':False,'source_report':str(args.report),'run_weights':components,'entity_count':int(len(entities)),'uncalibrated':binary_metrics(targets,logits),'platt_slope':slope,'platt_intercept':intercept,'calibrated':binary_metrics(targets,calibrated),'nll':float(fit.fun),'optimization_success':bool(fit.success)};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
