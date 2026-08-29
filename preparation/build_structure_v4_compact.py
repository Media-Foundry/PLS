"""Build a contiguous float16 residue cache for high-throughput training."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--stats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
 status=np.load(a.status,mmap_mode='r'); lengths=np.array([int(r['length']) if status[i]==1 else 0 for i,r in enumerate(rows)],dtype=np.int64);offsets=np.zeros(len(rows)+1,dtype=np.int64);np.cumsum(lengths,out=offsets[1:]);st=json.loads(a.stats.read_text());mean=torch.tensor(st['scalar_means']);std=torch.tensor(st['scalar_stds']);a.output.mkdir(parents=True,exist_ok=True);data=np.memmap(a.output/'residue_features.f16',mode='w+',dtype=np.float16,shape=(int(offsets[-1]),152));started=time.monotonic()
 for i,row in enumerate(rows):
  if status[i]!=1:continue
  d=row['sequence_sha256'];x=torch.load(a.raw_root/d[:2]/f'{d}.pt',map_location='cpu',weights_only=False);pconf=x['plddt'].float();feat=torch.cat((x['physchem_features'].float(),(x['spatial_scalar_raw_features'].float()-mean)/std,pconf[:,None]),1)
  if len(feat)!=lengths[i] or not torch.isfinite(feat).all():raise ValueError(f'compact mismatch {d}')
  data[offsets[i]:offsets[i+1]]=feat.numpy().astype(np.float16)
  if (i+1)%10000==0:print(json.dumps({'done':i+1,'entities':len(rows),'residues':int(offsets[i+1]),'files_s':round((i+1)/(time.monotonic()-started),2)}),flush=True)
 data.flush();np.save(a.output/'offsets.npy',offsets);meta={'schema':'PLS_V4_compact_f16_v1','shape':[int(offsets[-1]),152],'dtype':'float16','entity_count':len(rows),'valid_entities':int((status==1).sum()),'stats':str(a.stats.resolve()),'source':str(a.raw_root.resolve())};(a.output/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
