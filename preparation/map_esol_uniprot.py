"""Map eSOL sequences to UniProt accessions by exact full-sequence identity."""
from __future__ import annotations
import argparse,csv,gzip,json
from collections import Counter,defaultdict
from pathlib import Path

def fasta_records(path:Path):
 opener=gzip.open if path.suffix in ('.gz','.gzip') else open
 with opener(path,'rt',encoding='utf-8') as handle:
  header=None;parts=[]
  for line in handle:
   line=line.strip()
   if line.startswith('>'):
    if header is not None:yield header,''.join(parts)
    header=line[1:];parts=[]
   elif line:parts.append(line)
  if header is not None:yield header,''.join(parts)

def main():
 p=argparse.ArgumentParser();p.add_argument('--esol-manifest',type=Path,required=True);p.add_argument('--strict-split',type=Path,required=True);p.add_argument('--uniprot-fasta',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 strict={}
 with a.strict_split.open(newline='',encoding='utf-8') as h:
  for row in csv.DictReader(h):
   if row['source_dataset']=='eSOL_FGNNSol':strict[row['sequence_sha256']]=row['split']
 sequence_accessions=defaultdict(list)
 for header,sequence in fasta_records(a.uniprot_fasta):
  token=header.split()[0];parts=token.split('|');accession=parts[1] if len(parts)>=3 else token;reviewed=token.startswith('sp|');sequence_accessions[sequence].append((not reviewed,accession))
 rows=[]
 with a.esol_manifest.open(newline='',encoding='utf-8') as h:
  for row in csv.DictReader(h):
   matches=sorted(sequence_accessions.get(row['sequence'],[]));accessions=[v for _,v in matches];rows.append({'protein_id':row['protein_id'],'sequence_sha256':row['sequence_sha256'],'strict_split':strict[row['sequence_sha256']],'uniprot_accession':accessions[0] if accessions else '','all_exact_accessions':';'.join(accessions),'exact_match_count':len(accessions)})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 split_total=Counter(v['strict_split'] for v in rows);split_mapped=Counter(v['strict_split'] for v in rows if v['uniprot_accession']);report={'method':'exact full-sequence identity against UniProt proteome FASTA','total':len(rows),'mapped':sum(bool(v['uniprot_accession']) for v in rows),'ambiguous':sum(v['exact_match_count']>1 for v in rows),'by_strict_split':{s:{'total':split_total[s],'mapped':split_mapped[s]} for s in ('train','validation','test')}};a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
