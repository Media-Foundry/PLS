"""Build entity-aligned availability status for extracted structure features."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--observation-split',type=Path,required=True);p.add_argument('--source-dataset',required=True);p.add_argument('--feature-root',type=Path,required=True);p.add_argument('--splits',nargs='+',default=['train','validation']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(rows)}
 requested=set()
 with a.observation_split.open(newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']==a.source_dataset and r['split'] in a.splits:requested.add(r['sequence_sha256'])
 available={d for d in requested if (a.feature_root/d[:2]/f'{d}.pt').is_file()};status=np.zeros(len(rows),np.uint8);status[[index[d] for d in available]]=1;a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'status.npy',status);report={'schema':'PLS_structure_feature_status_v1','source_dataset':a.source_dataset,'splits':a.splits,'requested':len(requested),'available':len(available),'missing':sorted(requested-available),'entity_count':len(rows),'feature_root':str(a.feature_root.resolve())};(a.output/'metadata.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='missing'},indent=2))
if __name__=='__main__':main()
