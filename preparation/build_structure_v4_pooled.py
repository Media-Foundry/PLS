"""Build fixed protein-level V4 descriptors aligned to the entity manifest."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
import torch


SLICES = {"physchem": [0, 124], "scalar": [124, 480], "vector": [480, 568], "patch": [568, 604]}


def stats4(x):
    return torch.cat((x.mean(0), x.std(0, unbiased=False), x.amax(0), x.amin(0)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--entities',type=Path,required=True); p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--stats',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with a.entities.open(newline='',encoding='utf-8') as h: hashes=[r['sequence_sha256'] for r in csv.DictReader(h)]
    st=json.loads(a.stats.read_text()); mean=torch.tensor(st['scalar_means']); std=torch.tensor(st['scalar_stds'])
    a.output.mkdir(parents=True,exist_ok=True); out=np.lib.format.open_memmap(a.output/'descriptors.npy',mode='w+',dtype='float32',shape=(len(hashes),604))
    status=np.lib.format.open_memmap(a.output/'status.npy',mode='w+',dtype='uint8',shape=(len(hashes),)); status[:]=0
    for i,d in enumerate(hashes):
        path=a.feature_root/d[:2]/f'{d}.pt'
        if not path.exists(): continue
        x=torch.load(path,map_location='cpu',weights_only=False)
        if not x['sequence_exact_match']: status[i]=2; continue
        phys=x['physchem_features'].float(); scal=(x['spatial_scalar_raw_features'].float()-mean)/std
        vec=x['spatial_vector_features'].float(); norms=torch.linalg.vector_norm(vec,dim=-1)
        gram=torch.einsum('nic,njc->nij',vec,vec); tri=torch.triu_indices(8,8); inv=torch.cat((norms,gram[:,tri[0],tri[1]]),1)
        # Explicit confidence-aware surface/hydrophobic summaries. Indices are frozen V4 schema positions.
        surf=torch.cat((scal[:,6:20],scal[:,68:72]),1); confidence=x['plddt'].float()
        high=confidence>=.7; selected=surf[high] if high.any() else surf
        patch=torch.cat((selected.mean(0),selected.amax(0)))
        descriptor=torch.cat((phys.mean(0),phys.std(0,unbiased=False),stats4(scal),inv.mean(0),inv.std(0,unbiased=False),patch))
        if descriptor.shape!=(604,) or not torch.isfinite(descriptor).all(): raise ValueError(f'bad descriptor {d}')
        out[i]=descriptor.numpy(); status[i]=1
        if (i+1)%10000==0: print(json.dumps({'done':i+1,'total':len(hashes),'available':int((status[:i+1]==1).sum())}),flush=True)
    out.flush(); status.flush(); meta={'dimension':604,'slices':SLICES,'status':{'missing':0,'valid':1,'sequence_mismatch':2},
        'stats':str(a.stats.resolve()),'entities':str(a.entities.resolve())}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'entities':len(hashes),'valid':int((status==1).sum()),'mismatch':int((status==2).sum()),'missing':int((status==0).sum())}))
if __name__=='__main__': main()
