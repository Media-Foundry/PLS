"""Validation-only blend of a full-coverage ensemble and partial structure model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pls.evaluation.metrics import regression_metrics
def load(run):
 path=run/('validation_esol_predictions.npz' if (run/'validation_esol_predictions.npz').exists() else 'validation_predictions.npz');x=np.load(path);return x['entity_indices'],x['targets'],x['predictions']
def main():
 p=argparse.ArgumentParser();p.add_argument('--sequence-runs',nargs='+',type=Path,required=True);p.add_argument('--structure-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=101);a=p.parse_args();loaded=[load(v) for v in a.sequence_runs];entities,targets=loaded[0][:2]
 for e,t,_ in loaded[1:]:
  if not np.array_equal(e,entities) or not np.allclose(t,targets):raise ValueError('sequence predictions are unaligned')
 sequence=np.mean([v[2] for v in loaded],0);se,st,sp=load(a.structure_run);positions={int(v):i for i,v in enumerate(entities)};take=np.asarray([positions[int(v)] for v in se]);
 if not np.allclose(targets[take],st):raise ValueError('structure targets are unaligned')
 reports=[]
 for alpha in np.linspace(0,1,a.steps):
  pred=sequence.copy();pred[take]=(1-alpha)*sequence[take]+alpha*sp;slope,intercept=np.polyfit(pred,targets,1);reports.append({'structure_weight':float(alpha),'metrics':regression_metrics(targets,pred),'affine_slope':float(slope),'affine_intercept':float(intercept),'affine_metrics':regression_metrics(targets,slope*pred+intercept)})
 reports.sort(key=lambda v:(v['metrics']['spearman'],v['metrics']['pearson']),reverse=True);result={'selection_data':'strict-validation only','sequence_runs':[str(v) for v in a.sequence_runs],'structure_run':str(a.structure_run),'full_entities':len(entities),'structure_entities':len(se),'fallback_entities':len(entities)-len(se),'reports':reports};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(reports[0],indent=2))
if __name__=='__main__':main()
