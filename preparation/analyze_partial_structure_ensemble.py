"""Validation-only simplex search over a full-coverage model and partial structure models."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pls.evaluation.metrics import regression_metrics

def load(run:Path):
 path=run/('validation_esol_predictions.npz' if (run/'validation_esol_predictions.npz').exists() else 'validation_predictions.npz');x=np.load(path);return x['entity_indices'],x['targets'],x['predictions']

def main():
 p=argparse.ArgumentParser();p.add_argument('--sequence-runs',nargs='+',type=Path,required=True);p.add_argument('--structure-runs',nargs=2,type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=101);a=p.parse_args();loaded=[load(v) for v in a.sequence_runs];entities,targets=loaded[0][:2]
 for e,t,_ in loaded[1:]:
  if not np.array_equal(e,entities) or not np.allclose(t,targets):raise ValueError('sequence predictions are unaligned')
 sequence=np.mean([v[2] for v in loaded],0);structures=[load(v) for v in a.structure_runs]
 if not np.array_equal(structures[0][0],structures[1][0]) or not np.allclose(structures[0][1],structures[1][1]):raise ValueError('structure predictions are unaligned')
 positions={int(v):i for i,v in enumerate(entities)};take=np.asarray([positions[int(v)] for v in structures[0][0]])
 if not np.allclose(targets[take],structures[0][1]):raise ValueError('structure targets are unaligned')
 reports=[]
 for first_weight in np.linspace(0,1,a.steps):
  for second_weight in np.linspace(0,1-first_weight,a.steps):
   pred=sequence.copy();pred[take]=(1-first_weight-second_weight)*sequence[take]+first_weight*structures[0][2]+second_weight*structures[1][2];slope,intercept=np.polyfit(pred,targets,1);reports.append({'structure_weights':[float(first_weight),float(second_weight)],'metrics':regression_metrics(targets,pred),'affine_slope':float(slope),'affine_intercept':float(intercept),'affine_metrics':regression_metrics(targets,slope*pred+intercept)})
 reports.sort(key=lambda v:(v['metrics']['spearman'],v['metrics']['pearson']),reverse=True);result={'selection_data':'strict-validation only','sequence_runs':[str(v) for v in a.sequence_runs],'structure_runs':[str(v) for v in a.structure_runs],'full_entities':len(entities),'structure_entities':len(take),'fallback_entities':len(entities)-len(take),'reports':reports};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(reports[0],indent=2))

if __name__=='__main__':main()
