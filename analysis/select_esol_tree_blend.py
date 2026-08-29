"""Select a validation-only blend of two already-selected eSOL structure ensembles."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pls.evaluation.metrics import regression_metrics

SEQUENCE_RUNS=(
 'outputs/esol_residue_esm_v1_strict_si30_statistics_noema_s29+08-29-16-50',
 'outputs/esol_residue_esm_physchem_v2_strict_si30_scale01+08-29-19-07',
 'outputs/esol_residue_esm_physchem_v2_strict_si30_scale025+08-29-19-07',
 'outputs/esol_residue_esm_physchem_v2_strict_si30_scale1+08-29-19-07')
GEOMETRY_RUN='outputs/esol_geometry_residue_esm_physchem_v3_strict_si30_dual_seqsep+08-29-19-50'

def load(run):
 data=np.load(Path(run)/'validation_predictions.npz');return data['targets'],data['predictions'],data['entity_indices']

def selected_prediction(tree_run,structure_weights):
 targets,_,entities=load(SEQUENCE_RUNS[0]);sequence=np.mean([load(run)[1] for run in SEQUENCE_RUNS],0);geometry_targets,geometry,geometry_entities=load(GEOMETRY_RUN);tree_targets,tree,tree_entities=load(tree_run)
 if not np.array_equal(geometry_entities,tree_entities) or not np.array_equal(geometry_targets,tree_targets):raise ValueError('structure validation arrays do not align')
 positions={entity:index for index,entity in enumerate(entities)};selected=np.asarray([positions[entity] for entity in geometry_entities]);prediction=sequence.copy();prediction[selected]=(1-sum(structure_weights))*sequence[selected]+structure_weights[0]*geometry+structure_weights[1]*tree;return targets,prediction,entities

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();leaf_run='outputs/esol_pooled_v4_trees_v5_strict_si30_rf_leaf3+08-29-21-10';full_run='outputs/esol_pooled_v4_trees_v5_strict_si30_rf_full+08-29-21-13';targets,leaf,entities=selected_prediction(leaf_run,(.13,.1914));observed,full,full_entities=selected_prediction(full_run,(.14,.1806));assert np.array_equal(targets,observed) and np.array_equal(entities,full_entities)
 rows=[]
 for full_weight in np.linspace(0,1,1001):
  prediction=(1-full_weight)*leaf+full_weight*full;rows.append({'full_weight':float(full_weight),'leaf_weight':float(1-full_weight),'metrics':regression_metrics(targets,prediction)})
 best=max(rows,key=lambda row:row['metrics']['spearman']);report={'selection_data':'strict-validation only','test_evaluated':False,'objective':'spearman','sequence_runs':SEQUENCE_RUNS,'geometry_run':GEOMETRY_RUN,'leaf_run':leaf_run,'full_run':full_run,'leaf_structure_weights':[.13,.1914],'full_structure_weights':[.14,.1806],**best}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
