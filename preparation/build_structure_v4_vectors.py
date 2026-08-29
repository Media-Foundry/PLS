"""Build compact raw vector/coordinate caches for equivariant GVP models."""
from __future__ import annotations
import argparse,csv,json,multiprocessing as mp
from pathlib import Path
import numpy as np,torch

VECTORS=COORDS=ROOT=OFFSETS=None
def init_worker(vector_path,coord_path,root,offset_path,residues):
 global VECTORS,COORDS,ROOT,OFFSETS
 VECTORS=np.memmap(vector_path,mode='r+',dtype=np.float16,shape=(residues,8,3));COORDS=np.memmap(coord_path,mode='r+',dtype=np.float32,shape=(residues,3));ROOT=Path(root);OFFSETS=np.load(offset_path,mmap_mode='r')
def copy_one(item):
 i,digest=item;lo,hi=int(OFFSETS[i]),int(OFFSETS[i+1]);path=ROOT/digest[:2]/f'{digest}.pt'
 if hi==lo:return i,0
 x=torch.load(path,map_location='cpu',weights_only=False);vectors=x['spatial_vector_features'].numpy();coords=x['ca_coords'].numpy()
 if vectors.shape!=(hi-lo,8,3) or coords.shape!=(hi-lo,3) or not np.isfinite(vectors).all() or not np.isfinite(coords).all():raise ValueError(f'bad vectors: {digest}')
 VECTORS[lo:hi]=vectors.astype(np.float16);COORDS[lo:hi]=coords.astype(np.float32);return i,hi-lo
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--feature-root',type=Path,required=True);p.add_argument('--compact-dir',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:hashes=[r['sequence_sha256'] for r in csv.DictReader(h)]
 offsets=np.load(a.compact_dir/'offsets.npy',mmap_mode='r');status=np.load(a.status,mmap_mode='r');residues=int(offsets[-1]);a.output.mkdir(parents=True,exist_ok=True);vector_path=a.output/'vectors.f16';coord_path=a.output/'ca_coords.f32';np.memmap(vector_path,mode='w+',dtype=np.float16,shape=(residues,8,3)).flush();np.memmap(coord_path,mode='w+',dtype=np.float32,shape=(residues,3)).flush();items=[(i,d) for i,d in enumerate(hashes) if status[i]==1]
 copied=0
 with mp.get_context('fork').Pool(a.workers,initializer=init_worker,initargs=(vector_path,coord_path,a.feature_root,a.compact_dir/'offsets.npy',residues)) as pool:
  for done,n in enumerate(pool.imap_unordered(copy_one,items,chunksize=8),1):
   copied+=n[1]
   if done%10000==0:print(json.dumps({'entities':done,'total':len(items),'residues':copied}),flush=True)
 meta={'schema':'PLS_V4_equivariant_vector_cache_v1','entities':len(items),'residues':residues,'vector_shape':[residues,8,3],'coordinate_shape':[residues,3],'vector_dtype':'float16','coordinate_dtype':'float32','source':str(a.feature_root.resolve())};(a.output/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
