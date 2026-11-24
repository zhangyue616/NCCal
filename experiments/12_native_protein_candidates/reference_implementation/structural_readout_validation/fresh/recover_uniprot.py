from pathlib import Path
import json,time,subprocess
P=Path(__file__).parent
errs=[json.loads(l)for l in(P/'uniprot_errors.jsonl').read_text(encoding='utf8').splitlines()if l]
log=[]
print('COOLDOWN_60S',len(errs),flush=True);time.sleep(60)
for i,r in enumerate(errs,1):
 acc=r['accession'];path=P/f'sources/uniprot/{acc}.json'
 if path.exists():continue
 url='https://rest.uniprot.org/uniprotkb/'+acc+'.json'
 result=subprocess.run(['curl.exe','--silent','--show-error','--fail','--location','--max-time','65','--dump-header',str(path)+'.recovery_headers','--write-out','%{http_code}',url,'-o',str(path)+'.part'],capture_output=True)
 code=result.stdout.decode(errors='replace').strip();ok=result.returncode==0
 if ok:Path(str(path)+'.part').replace(path)
 log.append(dict(accession=acc,status=code,success=ok,error=result.stderr.decode(errors='replace')[-200:]))
 (P/'uniprot_recovery_log.json').write_text(json.dumps(log,indent=2))
 print('RECOVERED',i,len(errs),acc,code,flush=True)
 # Respect any renewed rate limit; do not flood retries.
 if code=='429':
  print('RATE_LIMIT_PERSISTED_STOPPING_FOR_BACKOFF',flush=True);break
 time.sleep(2)
print('RECOVERY_DONE',sum(r['success']for r in log),len(errs),flush=True)
