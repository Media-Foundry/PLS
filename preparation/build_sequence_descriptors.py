"""Build label-independent global sequence physicochemical descriptors."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

AA='ACDEFGHIKLMNPQRSTVWY'
GROUPS={'hydrophobic':'AVILMFWY','positive':'KRH','negative':'DE','charged':'DEKRH','aromatic':'FWY','polar':'STNQCY','aliphatic':'AILV','turn':'GPN','tiny':'ACGST'}

def longest_fraction(sequence,alphabet):
 best=current=0
 for residue in sequence:
  current=current+1 if residue in alphabet else 0;best=max(best,current)
 return best/max(len(sequence),1)

def describe(sequence):
 n=max(len(sequence),1);fractions=np.asarray([sequence.count(a)/n for a in AA],np.float32);entropy=-sum(v*math.log(v+1e-12) for v in fractions);groups=[sum(sequence.count(a) for a in residues)/n for residues in GROUPS.values()]
 extra=[math.log1p(len(sequence)),entropy,(sequence.count('K')+sequence.count('R')-sequence.count('D')-sequence.count('E'))/n,longest_fraction(sequence,GROUPS['hydrophobic']),longest_fraction(sequence,GROUPS['charged']),sum(residue not in AA for residue in sequence)/n]
 return np.asarray([*fractions,*groups,*extra],np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
 values=np.stack([describe(row['sequence']) for row in rows]);a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'descriptors.npy',values);columns=[f'aa_fraction_{x}' for x in AA]+[f'group_fraction_{x}' for x in GROUPS]+['log1p_length','shannon_entropy','net_charge_fraction','longest_hydrophobic_run_fraction','longest_charged_run_fraction','noncanonical_fraction'];(a.output/'metadata.json').write_text(json.dumps({'schema':'PLS_sequence_descriptors_v1','shape':list(values.shape),'dtype':'float32','columns':columns},indent=2,sort_keys=True)+'\n');print(json.dumps({'shape':list(values.shape),'columns':len(columns)}))
if __name__=='__main__':main()
