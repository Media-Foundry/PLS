"""Fail unless sharded status files exactly cover a selected entity mask."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection-status',type=Path,required=True);p.add_argument('--status-root',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--shard-count',type=int,required=True);a=p.parse_args()
 selected=np.load(a.selection_status,mmap_mode='r')==1;covered=np.zeros(len(selected),dtype=bool);reports=[]
 for shard in range(a.shard_count):
  path=a.status_root/f'{a.prefix}{shard}.npy'
  if not path.exists():raise FileNotFoundError(path)
  status=np.load(path,mmap_mode='r');done=status==1
  if len(status)!=len(selected):raise ValueError(f'length mismatch: {path}')
  indices=np.flatnonzero(done)
  wrong=indices[indices%a.shard_count!=shard]
  invalid=indices[~selected[indices]]
  if len(wrong) or len(invalid):raise ValueError(f'invalid completion markers in shard {shard}: wrong={len(wrong)} unselected={len(invalid)}')
  if np.any(covered&done):raise ValueError(f'overlap in shard {shard}')
  covered|=done;reports.append({'shard':shard,'completed':int(done.sum())})
 missing=int((selected&~covered).sum());extra=int((covered&~selected).sum());report={'selected':int(selected.sum()),'covered':int(covered.sum()),'missing':missing,'extra':extra,'shards':reports};print(json.dumps(report,sort_keys=True))
 if missing or extra:raise SystemExit(1)
if __name__=='__main__':main()
