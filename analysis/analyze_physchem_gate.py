"""Validation-only utilization audit for a trained physicochemical MoE gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch import nn

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();config=json.loads((args.run/'config.json').read_text());experts=int(config['model']['physchem_experts']);stats=np.load(args.run/'descriptor_stats.npz');raw=np.load(Path(config['data']['sequence_descriptor_dir'])/'descriptors.npy',mmap_mode='r');entities=np.load(args.run/'validation_predictions.npz')['entity_indices'];descriptors=(np.asarray(raw[entities],np.float32)-stats['mean'])/stats['std']*float(stats['scale']);dimension=descriptors.shape[1];gate=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,32),nn.GELU(),nn.Linear(32,experts));state=torch.load(args.run/'checkpoints'/'best.pt',map_location='cpu',weights_only=False)['model'];gate.load_state_dict({key.removeprefix('physchem_gate.'):value for key,value in state.items() if key.startswith('physchem_gate.')});gate.eval()
 temperature=float(config['model'].get('physchem_gate_temperature',1.))
 if temperature<=0:raise ValueError('physchem gate temperature must be positive')
 with torch.inference_mode():weights=torch.softmax(gate(torch.from_numpy(descriptors))/temperature,1)
 soft_weights=weights
 top_k=int(config['model'].get('physchem_top_k',0))
 if top_k:
  selected=torch.topk(weights,top_k,dim=1).indices;mask=torch.zeros_like(weights).scatter_(1,selected,1);weights=weights*mask;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8);residual=float(config['model'].get('physchem_sparse_residual',0.));weights=(1-residual)*weights+residual*soft_weights
 else:residual=0.
 weights=weights.numpy()
 entropy=-(weights*np.log(np.clip(weights,1e-12,None))).sum(1);winners=weights.argmax(1);report={'selection_data':'strict-validation only','test_evaluated':False,'run':str(args.run),'entity_count':int(len(entities)),'experts':experts,'gate_temperature':temperature,'top_k':top_k,'sparse_residual':residual,'mean_weights':weights.mean(0).tolist(),'std_weights':weights.std(0).tolist(),'argmax_fraction':np.bincount(winners,minlength=experts).astype(float).tolist(),'mean_entropy':float(entropy.mean()),'maximum_entropy':float(np.log(experts)),'effective_experts':float(np.exp(entropy.mean()))};report['argmax_fraction']=(np.asarray(report['argmax_fraction'])/len(winners)).tolist();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
