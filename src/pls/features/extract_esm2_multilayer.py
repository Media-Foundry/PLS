"""Resumable sharded extraction of exact pooled ESM2 representations from several layers."""
from __future__ import annotations
import argparse,csv,json,os,time
from pathlib import Path
import esm,numpy as np,torch
from pls.features.extract_esm2 import MODEL_NAME,EMBEDDING_DIMENSION,sha256,chunks

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--selection-status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',type=int,nargs='+',default=(12,24,33));p.add_argument('--shard-count',type=int,default=4);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--hip-device',type=int);p.add_argument('--token-budget',type=int,default=4096);p.add_argument('--maximum-residues',type=int,default=1022);p.add_argument('--initialize-only',action='store_true');a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
 selected=np.load(a.selection_status,mmap_mode='r')==1;layers=tuple(a.layers);dimension=len(layers)*EMBEDDING_DIMENSION;a.output.mkdir(parents=True,exist_ok=True);embedding_path=a.output/'embeddings.npy';metadata={'schema':'PLS_ESM2_multilayer_mean_f16_v1','model':MODEL_NAME,'layers':list(layers),'layer_dimension':EMBEDDING_DIMENSION,'dimension':dimension,'shape':[len(rows),dimension],'dtype':'float16','selected_entities':int(selected.sum()),'maximum_residues':a.maximum_residues,'entity_manifest_sha256':sha256(a.entities)}
 if a.initialize_only:
  if not embedding_path.exists():np.lib.format.open_memmap(embedding_path,mode='w+',dtype=np.float16,shape=(len(rows),dimension)).flush()
  (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n');print(json.dumps(metadata));return
 if json.loads((a.output/'metadata.json').read_text())!=metadata:raise ValueError('metadata mismatch')
 expected=str(a.hip_device if a.hip_device is not None else a.shard_index)
 if os.environ.get('HIP_VISIBLE_DEVICES')!=expected:raise ValueError('HIP device mismatch')
 embeddings=np.load(embedding_path,mmap_mode='r+');status_path=a.output/f'status_shard_{a.shard_index}.npy'
 if status_path.exists():status=np.load(status_path,mmap_mode='r+')
 else:status=np.lib.format.open_memmap(status_path,mode='w+',dtype=np.uint8,shape=(len(rows),));status[:]=0;status.flush()
 model,alphabet=esm.pretrained.esm2_t33_650M_UR50D();model.eval().requires_grad_(False).half().to('cuda:0');convert=alphabet.get_batch_converter();pending=[i for i,r in enumerate(rows) if selected[i] and i%a.shard_count==a.shard_index and status[i]!=1];pending.sort(key=lambda i:(len(rows[i]['sequence']),rows[i]['sequence_sha256']));cursor=0;done=int((status==1).sum());started=time.monotonic()
 while cursor<len(pending):
  indices=[];items=[]
  while cursor<len(pending):
   i=pending[cursor];parts=list(chunks(rows[i]['sequence'],a.maximum_residues));candidate=items+[(i,x) for x in parts];tokens=(max(len(x) for _,x in candidate)+2)*len(candidate)
   if indices and tokens>a.token_budget:break
   indices.append(i);items=candidate;cursor+=1
  _,_,tokens=convert([(f'{i}:{j}',seq) for j,(i,seq) in enumerate(items)]);tokens=tokens.to('cuda:0',non_blocking=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):representations=model(tokens,repr_layers=list(layers),return_contacts=False)['representations']
  sums={i:[torch.zeros(EMBEDDING_DIMENSION) for _ in layers] for i in indices};counts={i:0 for i in indices}
  for row_index,(i,sequence) in enumerate(items):
   for layer_index,layer in enumerate(layers):sums[i][layer_index]+=representations[layer][row_index,1:len(sequence)+1].float().sum(0).cpu()
   counts[i]+=len(sequence)
  for i in indices:embeddings[i]=torch.cat([value/counts[i] for value in sums[i]]).numpy().astype(np.float16);status[i]=1
  status.flush();done+=len(indices)
  if done%100<len(indices):embeddings.flush();print(json.dumps({'shard':a.shard_index,'done':done,'remaining':len(pending)-cursor,'proteins_s':round(done/(time.monotonic()-started),2)}),flush=True)
 embeddings.flush();print(json.dumps({'shard':a.shard_index,'complete':True,'completed':done}))
if __name__=='__main__':main()
