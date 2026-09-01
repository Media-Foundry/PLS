"""Four-way resumable ESM2 residue embedding extraction into one compact memmap."""
from __future__ import annotations
import argparse,csv,json,os,time
from pathlib import Path
import esm,numpy as np,torch
from pls.features.extract_esm2 import MODEL_NAME,REPRESENTATION_LAYER,EMBEDDING_DIMENSION,sha256,chunks

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--offsets',type=Path,required=True);p.add_argument('--structure-status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard-count',type=int,default=4);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--hip-device',type=int);p.add_argument('--cuda-slurm',action='store_true');p.add_argument('--token-budget',type=int,default=4096);p.add_argument('--maximum-residues',type=int,default=1022);p.add_argument('--initialize-only',action='store_true');a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
 offsets=np.load(a.offsets,mmap_mode='r');valid=np.load(a.structure_status,mmap_mode='r')==1;total=int(offsets[-1]);a.output.mkdir(parents=True,exist_ok=True);data_path=a.output/'residue_esm2.f16';meta_path=a.output/'metadata.json'
 meta={'schema':'PLS_ESM2_t33_residue_f16_v1','model':MODEL_NAME,'layer':REPRESENTATION_LAYER,'dimension':EMBEDDING_DIMENSION,'shape':[total,EMBEDDING_DIMENSION],'dtype':'float16','entity_count':len(rows),'valid_entities':int(valid.sum()),'maximum_residues':a.maximum_residues,'entity_manifest_sha256':sha256(a.entities)}
 if a.initialize_only:
  if not data_path.exists():np.memmap(data_path,mode='w+',dtype=np.float16,shape=(total,EMBEDDING_DIMENSION)).flush()
  meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta));return
 if not data_path.exists() or not meta_path.exists():raise FileNotFoundError('run --initialize-only before shards')
 if json.loads(meta_path.read_text())!=meta:raise ValueError('residue ESM2 metadata mismatch')
 if a.cuda_slurm:
  visible=os.environ.get('CUDA_VISIBLE_DEVICES')
  if a.hip_device is not None or not os.environ.get('SLURM_JOB_ID') or not visible or ',' in visible:raise ValueError('cuda-slurm requires exactly one Slurm-assigned GPU')
 else:
  expected=str(a.hip_device if a.hip_device is not None else a.shard_index)
  if os.environ.get('HIP_VISIBLE_DEVICES')!=expected:raise ValueError(f'shard {a.shard_index} requires HIP_VISIBLE_DEVICES={expected}')
 status_path=a.output/f'status_shard_{a.shard_index}.npy'
 if status_path.exists():status=np.load(status_path,mmap_mode='r+')
 else:status=np.lib.format.open_memmap(status_path,mode='w+',dtype=np.uint8,shape=(len(rows),));status[:]=0;status.flush()
 data=np.memmap(data_path,mode='r+',dtype=np.float16,shape=(total,EMBEDDING_DIMENSION));model,alphabet=esm.pretrained.esm2_t33_650M_UR50D();model.eval().requires_grad_(False).half().to('cuda:0');convert=alphabet.get_batch_converter();pending=[i for i,r in enumerate(rows) if valid[i] and i%a.shard_count==a.shard_index and status[i]!=1];pending.sort(key=lambda i:(len(rows[i]['sequence']),rows[i]['sequence_sha256']));cursor=0;done=int((status==1).sum());started=time.monotonic()
 while cursor<len(pending):
  indices=[];items=[]
  while cursor<len(pending):
   i=pending[cursor];parts=list(chunks(rows[i]['sequence'],a.maximum_residues));candidate=items+[(i,x) for x in parts];tokens=(max(len(x) for _,x in candidate)+2)*len(candidate)
   if indices and tokens>a.token_budget:break
   indices.append(i);items=candidate;cursor+=1
  _,_,tokens=convert([(f'{i}:{j}',seq) for j,(i,seq) in enumerate(items)]);tokens=tokens.to('cuda:0',non_blocking=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):rep=model(tokens,repr_layers=[REPRESENTATION_LAYER],return_contacts=False)['representations'][REPRESENTATION_LAYER]
  positions={i:0 for i in indices}
  for row_index,(i,seq) in enumerate(items):
   start=int(offsets[i])+positions[i];data[start:start+len(seq)]=rep[row_index,1:len(seq)+1].cpu().numpy().astype(np.float16);positions[i]+=len(seq)
  for i in indices:
   if positions[i]!=int(offsets[i+1]-offsets[i]):raise ValueError(f'length mismatch {i}')
   status[i]=1
  status.flush();done+=len(indices)
  if done%100< len(indices):data.flush();print(json.dumps({'shard':a.shard_index,'done':done,'remaining':len(pending)-cursor,'proteins_s':round(done/(time.monotonic()-started),2)}),flush=True)
 data.flush();print(json.dumps({'shard':a.shard_index,'complete':True,'completed':done}))
if __name__=='__main__':main()
