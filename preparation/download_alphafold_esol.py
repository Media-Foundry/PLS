"""Resumable sharded AlphaFold DB downloads for exactly mapped eSOL proteins."""
from __future__ import annotations
import argparse,csv,json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--mapping',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--splits',nargs='+',default=['train','validation']);p.add_argument('--version',type=int,default=6);p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--retries',type=int,default=4);a=p.parse_args()
 if not 0<=a.shard_index<a.shard_count:p.error('shard-index must be in [0, shard-count)')
 with a.mapping.open(newline='',encoding='utf-8') as h:accessions=sorted({r['uniprot_accession'] for r in csv.DictReader(h) if r['strict_split'] in a.splits and r['uniprot_accession']})
 selected=accessions[a.shard_index::a.shard_count];a.output_dir.mkdir(parents=True,exist_ok=True)
 def download(accession):
  target=a.output_dir/accession[:2]/f'AF-{accession}-F1-model_v{a.version}.pdb';target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and target.stat().st_size>1000:return accession,'existing',target.stat().st_size
  url=f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{a.version}.pdb';part=target.with_suffix('.pdb.part');error=''
  for attempt in range(a.retries):
   try:
    request=urllib.request.Request(url,headers={'User-Agent':'PLS-research/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response,part.open('wb') as out:
     while chunk:=response.read(1<<20):out.write(chunk)
    if part.stat().st_size<=1000:raise ValueError('downloaded structure is unexpectedly small')
    part.replace(target);return accession,'downloaded',target.stat().st_size
   except Exception as exc:error=repr(exc);part.unlink(missing_ok=True);time.sleep(2**attempt)
  return accession,'failed',error
 counts={'downloaded':0,'existing':0,'failed':0};failures=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(download,v) for v in selected]
  for done,future in enumerate(as_completed(futures),1):
   accession,status,detail=future.result();counts[status]+=1
   if status=='failed':failures.append({'accession':accession,'error':detail})
   if done%100==0 or done==len(futures):print(json.dumps({'done':done,'total':len(futures),**counts}),flush=True)
 report={'schema':'PLS_AlphaFoldDB_download_v1','version':a.version,'splits':a.splits,'shard_index':a.shard_index,'shard_count':a.shard_count,'selected':len(selected),**counts,'failures':failures};(a.output_dir/f'download_shard_{a.shard_index:03d}_of_{a.shard_count:03d}.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
