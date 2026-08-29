"""Build entity-aligned selection status and compact offsets for one endpoint source."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--observation-split',type=Path,required=True);p.add_argument('--source-dataset',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h));index={r['sequence_sha256']:i for i,r in enumerate(rows)}
 selected=set()
 with a.observation_split.open(newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['source_dataset']==a.source_dataset:selected.add(index[r['sequence_sha256']])
 status=np.zeros(len(rows),dtype=np.uint8);status[list(selected)]=1;lengths=np.array([int(r['length']) if status[i] else 0 for i,r in enumerate(rows)],dtype=np.int64);offsets=np.zeros(len(rows)+1,dtype=np.int64);np.cumsum(lengths,out=offsets[1:]);a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'status.npy',status);np.save(a.output/'offsets.npy',offsets);report={'source_dataset':a.source_dataset,'selected_entities':len(selected),'residues':int(offsets[-1]),'entity_count':len(rows)};(a.output/'metadata.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
