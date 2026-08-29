"""Project compact 1280D residue ESM2 to train-fitted PCA dimensions on GPU shards."""
from __future__ import annotations
import argparse,csv,json,os,time
from pathlib import Path
import numpy as np,torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--offsets',type=Path,required=True);p.add_argument('--structure-status',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--pca',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard-count',type=int,default=4);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--hip-device',type=int);p.add_argument('--residue-budget',type=int,default=65536);p.add_argument('--initialize-only',action='store_true');a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
 offsets=np.load(a.offsets,mmap_mode='r');valid=np.load(a.structure_status,mmap_mode='r')==1;total=int(offsets[-1]);projection=np.load(a.pca);mean=projection['mean'];components=projection['components'];outdim=components.shape[0];a.output.mkdir(parents=True,exist_ok=True);outpath=a.output/'residue_esm2_pca.f16';meta={'schema':'PLS_ESM2_residue_train_PCA_f16_v1','shape':[total,outdim],'dtype':'float16','pca':str(a.pca.resolve()),'explained_variance_ratio_sum':float(projection['explained_variance_ratio'].sum())}
 if a.initialize_only:
  if not outpath.exists():np.memmap(outpath,mode='w+',dtype=np.float16,shape=(total,outdim)).flush()
  (a.output/'pca_metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta));return
 expected=str(a.hip_device if a.hip_device is not None else a.shard_index)
 if os.environ.get('HIP_VISIBLE_DEVICES')!=expected:raise ValueError('HIP device mismatch')
 source=np.memmap(a.source/'residue_esm2.f16',mode='r',dtype=np.float16,shape=(total,1280));target=np.memmap(outpath,mode='r+',dtype=np.float16,shape=(total,outdim));status_path=a.output/f'pca_status_shard_{a.shard_index}.npy'
 if status_path.exists():status=np.load(status_path,mmap_mode='r+')
 else:status=np.lib.format.open_memmap(status_path,mode='w+',dtype=np.uint8,shape=(len(rows),));status[:]=0;status.flush()
 mean_t=torch.from_numpy(mean).to('cuda:0');components_t=torch.from_numpy(components).to('cuda:0');pending=[i for i in range(len(rows)) if valid[i] and i%a.shard_count==a.shard_index and status[i]!=1];cursor=0;done=int((status==1).sum());started=time.monotonic()
 while cursor<len(pending):
  selected=[];count=0
  while cursor<len(pending):
   i=pending[cursor];n=int(offsets[i+1]-offsets[i])
   if selected and count+n>a.residue_budget:break
   selected.append(i);count+=n;cursor+=1
  batch=np.concatenate([np.array(source[int(offsets[i]):int(offsets[i+1])],dtype=np.float32,copy=True) for i in selected]);x=torch.from_numpy(batch).to('cuda:0',non_blocking=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):y=(x-mean_t)@components_t.T
  y=y.cpu().numpy().astype(np.float16);position=0
  for i in selected:
   n=int(offsets[i+1]-offsets[i]);target[int(offsets[i]):int(offsets[i+1])]=y[position:position+n];position+=n;status[i]=1
  status.flush();done+=len(selected)
  if done%500<len(selected):target.flush();print(json.dumps({'shard':a.shard_index,'done':done,'remaining':len(pending)-cursor,'residues_s':round(sum(int(offsets[i+1]-offsets[i]) for i in pending[:cursor])/(time.monotonic()-started),1)}),flush=True)
 target.flush();print(json.dumps({'shard':a.shard_index,'complete':True,'done':done}))
if __name__=='__main__':main()
