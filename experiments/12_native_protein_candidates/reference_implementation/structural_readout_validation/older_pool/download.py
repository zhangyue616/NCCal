from pathlib import Path
import time,json,subprocess,sys,argparse,os
from concurrent.futures import ThreadPoolExecutor,as_completed
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def source(pid):
    pid=pid.lower();dest=P/f'sources/mmcif/{pid}.cif.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    for d in [P/'sources/mmcif',ROOT/'structure/cache/mmcif',ROOT/'cross_architecture/external/cache/mmcif',ROOT/'nccal_evidence_extension/cohort/public_sources/mmcif']:
        f=d/f'{pid}.cif.gz'
        if f.exists() and f.stat().st_size>0:return str(f)
    for url in (f'https://files.rcsb.org/download/{pid}.cif.gz',f'https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}.cif.gz'):
        part=Path(str(dest)+'.part.'+str(os.getpid()));r=subprocess.run(['curl.exe','--silent','--show-error','--fail','--location','--max-time','50','--retry','1',url,'-o',str(part)],capture_output=True)
        if r.returncode==0:part.replace(dest);return str(dest)
    raise RuntimeError('source download failed '+pid)
def main():
    a=argparse.ArgumentParser();a.add_argument('--limit',type=int,default=600);args=a.parse_args()
    for i in range(120):
        try:freeze=json.loads((P/'ACQUISITION_FREEZE.json').read_text());break
        except:time.sleep(2)
    else:raise RuntimeError('front not ready')
    ids=sorted({r['entities'][0].split('_')[0] for r in freeze['groups'][:args.limit]});out=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(source,p):p for p in ids}
        for i,f in enumerate(as_completed(jobs),1):
            try:out.append({'pdb_id':jobs[f],'path':f.result()})
            except Exception as e:out.append({'pdb_id':jobs[f],'error':str(e)})
            if i%25==0:print('DOWNLOADED',i,len(ids),'ok',sum('path'in x for x in out),flush=True)
    (P/f'download_{args.limit}.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in out),encoding='utf-8');print('DOWNLOAD_DONE',len(out),flush=True)
if __name__=='__main__':main()
