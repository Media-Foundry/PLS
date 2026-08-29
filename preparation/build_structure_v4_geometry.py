"""Build compact CA-kNN and V4 vector-invariant caches aligned to compact residues."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--compact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--neighbors',type=int,default=16);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
 status=np.load(a.status,mmap_mode='r');offsets=np.load(a.compact_root/'offsets.npy',mmap_mode='r');total=int(offsets[-1]);k=a.neighbors;a.output.mkdir(parents=True,exist_ok=True)
 neighbors=np.memmap(a.output/'neighbors.i16',mode='w+',dtype=np.int16,shape=(total,k));distances=np.memmap(a.output/'distances.f16',mode='w+',dtype=np.float16,shape=(total,k));invariants=np.memmap(a.output/'vector_invariants.f16',mode='w+',dtype=np.float16,shape=(total,44));started=time.monotonic();used=0
 for i,row in enumerate(rows):
  if status[i]!=1:continue
  d=row['sequence_sha256'];x=torch.load(a.raw_root/d[:2]/f'{d}.pt',map_location='cpu',weights_only=False);coords=x['ca_coords'].numpy();n=len(coords);query=min(k+1,n);dist,idx=cKDTree(coords).query(coords,k=query)
  if query==1:dist=np.zeros((n,1));idx=np.zeros((n,1),dtype=np.int64)
  dist,idx=dist[:,1:],idx[:,1:]
  if idx.shape[1]<k:
   pad=k-idx.shape[1];idx=np.pad(idx,((0,0),(0,pad)),mode='edge');dist=np.pad(dist,((0,0),(0,pad)),mode='edge')
  vec=x['spatial_vector_features'].float();norm=torch.linalg.vector_norm(vec,dim=-1);gram=torch.einsum('nic,njc->nij',vec,vec);tri=torch.triu_indices(8,8);inv=torch.cat((norm,gram[:,tri[0],tri[1]]),1)
  lo,hi=int(offsets[i]),int(offsets[i+1]);neighbors[lo:hi]=idx.astype(np.int16);distances[lo:hi]=dist.astype(np.float16);invariants[lo:hi]=inv.numpy().astype(np.float16);used+=1
  if used%5000==0:print(json.dumps({'proteins':used,'entities_scanned':i+1,'residues':hi,'proteins_s':round(used/(time.monotonic()-started),2)}),flush=True)
 neighbors.flush();distances.flush();invariants.flush();meta={'schema':'PLS_V4_geometry_knn_v1','residues':total,'neighbors':k,'neighbor_dtype':'int16_local_index','distance_dtype':'float16_angstrom','vector_invariant_dimension':44,'valid_entities':used};(a.output/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
