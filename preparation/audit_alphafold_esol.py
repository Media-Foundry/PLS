"""Audit AlphaFold eSOL structures against exact source sequence hashes."""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','SEC':'U','PYL':'O'}
PATTERN=re.compile(r'^AF-(.+)-F1-model_v(\d+)\.pdb$')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mapping',type=Path,required=True);p.add_argument('--structure-dir',type=Path,required=True);p.add_argument('--splits',nargs='+',default=['train','validation']);p.add_argument('--version',type=int,default=6);p.add_argument('--workers',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.mapping.open(newline='',encoding='utf-8') as h:expected={r['uniprot_accession']:r for r in csv.DictReader(h) if r['strict_split'] in a.splits and r['uniprot_accession']}
 found={}
 for path in a.structure_dir.rglob('*.pdb'):
  match=PATTERN.match(path.name)
  if match and int(match.group(2))==a.version:found[match.group(1)]=path
 def audit(item):
  accession,row=item;path=found.get(accession)
  if path is None:return {'accession':accession,'status':'missing'}
  sequence=[];plddt=[];last=None
  with path.open(encoding='ascii') as handle:
   for line in handle:
    if not line.startswith('ATOM') or line[12:16].strip()!='CA' or line[16] not in (' ','A'):continue
    residue=(line[21],line[22:26],line[26]);
    if residue==last:continue
    last=residue;sequence.append(AA.get(line[17:20].strip(),'X'));plddt.append(float(line[60:66]))
  digest=hashlib.sha256(''.join(sequence).encode('ascii')).hexdigest();status='ok' if digest==row['sequence_sha256'] else 'sequence_mismatch';return {'accession':accession,'status':status,'residues':len(sequence),'bytes':path.stat().st_size,'mean_plddt':sum(plddt)/len(plddt) if plddt else None,'observed_sha256':digest}
 with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(audit,expected.items()))
 failures=[r for r in rows if r['status']!='ok'];report={'schema':'PLS_AlphaFoldDB_audit_v1','version':a.version,'splits':a.splits,'expected':len(expected),'found_expected':sum(v in found for v in expected),'extra':sorted(set(found)-set(expected)),'ok':sum(r['status']=='ok' for r in rows),'failures':failures,'total_bytes':sum(r.get('bytes',0) for r in rows),'mean_plddt':sum(r['mean_plddt'] for r in rows if r.get('mean_plddt') is not None)/sum(r.get('mean_plddt') is not None for r in rows)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='extra'},indent=2));
 if failures or report['extra']:raise SystemExit(1)
if __name__=='__main__':main()
