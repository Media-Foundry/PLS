"""Validation-only subgroup audit for a weighted UESolDS ensemble."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--entities',type=Path,default=Path('benchmark/generated/sequence_entities.csv'));parser.add_argument('--descriptors',type=Path,default=Path('artifacts/features/sequence_descriptors_v1'));args=parser.parse_args();report=json.loads(args.report.read_text());weights=report.get('run_weights') or dict(zip(report['runs'],report['weights']));first=np.load(Path(next(iter(weights)))/'validation_predictions.npz');targets=first['targets'];entity_ids=first['entity_indices'];prediction=np.zeros_like(first['logits'],dtype=np.float64)
 for run,weight in weights.items():
  data=np.load(Path(run)/'validation_predictions.npz')
  if not np.array_equal(targets,data['targets']) or not np.array_equal(entity_ids,data['entity_indices']):raise ValueError(f'validation alignment mismatch: {run}')
  prediction+=float(weight)*data['logits']
 with args.entities.open(newline='',encoding='utf-8') as handle:lengths=np.asarray([int(row['length']) for row in csv.DictReader(handle)])[entity_ids]
 metadata=json.loads((args.descriptors/'metadata.json').read_text());columns=metadata['columns'];descriptors=np.load(args.descriptors/'descriptors.npy',mmap_mode='r')[entity_ids];features={'length':lengths,'hydrophobic_fraction':descriptors[:,columns.index('group_fraction_hydrophobic')],'absolute_net_charge':np.abs(descriptors[:,columns.index('net_charge_fraction')]),'longest_hydrophobic_run':descriptors[:,columns.index('longest_hydrophobic_run_fraction')],'sequence_entropy':descriptors[:,columns.index('shannon_entropy')]};groups={}
 for name,values in features.items():
  edges=np.quantile(values,[0,.25,.5,.75,1]);rows=[]
  for index in range(4):
   selected=(values>=edges[index])&(values<(edges[index+1]) if index<3 else values<=edges[index+1]);rows.append({'quartile':index+1,'lower':float(edges[index]),'upper':float(edges[index+1]),'n':int(selected.sum()),'positive_fraction':float(targets[selected].mean()),'auroc':float(roc_auc_score(targets[selected],prediction[selected]))})
  groups[name]=rows
 result={'selection_data':'strict-validation only','test_evaluated':False,'source_report':str(args.report),'entity_count':int(len(targets)),'overall_auroc':float(roc_auc_score(targets,prediction)),'subgroups':groups};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
