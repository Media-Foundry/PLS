"""Prepare audited AlphaFold eSOL structures for the reusable V4 extractor."""
from __future__ import annotations
import argparse,csv,os
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--mapping',type=Path,required=True);p.add_argument('--entities',type=Path,required=True);p.add_argument('--structure-dir',type=Path,required=True);p.add_argument('--version',type=int,default=6);p.add_argument('--csv-root',type=Path,required=True);p.add_argument('--pdb-root',type=Path,required=True);a=p.parse_args()
 with a.entities.open(newline='',encoding='utf-8') as h:sequences={r['sequence_sha256']:r['sequence'] for r in csv.DictReader(h)}
 with a.mapping.open(newline='',encoding='utf-8') as h:rows=[r for r in csv.DictReader(h) if r['strict_split'] in ('train','validation') and r['uniprot_accession']]
 a.csv_root.mkdir(parents=True,exist_ok=True);a.pdb_root.mkdir(parents=True,exist_ok=True);grouped={'train':[],'validation':[],'test':[]}
 for row in rows:
  accession=row['uniprot_accession'];source=(a.structure_dir/accession[:2]/f'AF-{accession}-F1-model_v{a.version}.pdb').resolve();target=a.pdb_root/f'{accession}.ef.pdb'
  if not source.is_file():raise FileNotFoundError(source)
  if target.is_symlink() and target.resolve()!=source:target.unlink()
  if not target.exists():os.symlink(source,target)
  grouped[row['strict_split']].append({'name':accession,'aa_seq':sequences[row['sequence_sha256']]})
 for split,filename in (('train','train.csv'),('validation','valid.csv'),('test','test.csv')):
  with (a.csv_root/filename).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=('name','aa_seq'));w.writeheader();w.writerows(sorted(grouped[split],key=lambda v:v['name']))
 print({key:len(value) for key,value in grouped.items()})
if __name__=='__main__':main()
