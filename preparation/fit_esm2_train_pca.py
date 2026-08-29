"""Fit a leakage-safe ESM2 projection using strict-train PDBSol entities only."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--observation-split',type=Path,required=True);p.add_argument('--source-dataset',default='PDBSol_ProtSolM');p.add_argument('--embedding-dir',type=Path,required=True);p.add_argument('--structure-status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dimensions',type=int,default=256);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h));hashes=[r['sequence_sha256'] for r in rows];index={d:i for i,d in enumerate(hashes)}
 selected=set()
 with a.observation_split.open(newline='',encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if r['split']=='train' and r['source_dataset']==a.source_dataset:selected.add(index[r['sequence_sha256']])
 status=np.load(a.structure_status,mmap_mode='r');indices=np.array(sorted(i for i in selected if status[i]==1));emb=np.load(a.embedding_dir/'embeddings.npy',mmap_mode='r');x=np.array(emb[indices],dtype=np.float32,copy=True);pca=PCA(n_components=a.dimensions,svd_solver='randomized',random_state=20260829,iterated_power=4).fit(x);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32),explained_variance=pca.explained_variance_.astype(np.float32),explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),train_entity_indices=indices);report={'schema':'PLS_ESM2_train_PCA_v1','source_dataset':a.source_dataset,'strict_train_entities':len(indices),'input_dimension':x.shape[1],'output_dimension':a.dimensions,'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'observation_split':str(a.observation_split.resolve())};a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
