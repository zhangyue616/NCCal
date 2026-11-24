from pathlib import Path
import subprocess,sys,time,json,hashlib
P=Path(__file__).resolve().parent;H=P.parent/'sheet';py=sys.executable
def run(script,*args):
    r=subprocess.run([py,str(script),*map(str,args)])
    if r.returncode:raise RuntimeError(str(script)+' failed')
for i in range(180):
    try:acq=json.loads((P/'ACQUISITION_FREEZE.json').read_text());break
    except:time.sleep(2)
else:raise RuntimeError('front unavailable')
for limit in (600,1200,1800):
    if (P/f'ISOLATION_SUMMARY_{limit}.json').exists():
        print('REUSE_COMPLETED_ISOLATION',limit,flush=True)
    else:
        run(P/'physical.py','--limit',limit,'--workers',6)
        while not(P.parent/'fresh/COHORT_FREEZE.json').exists():print('WAITING_F_FREEZE_FOR_FINAL_ISOLATION',flush=True);time.sleep(30)
        run(P/'isolate.py','--limit',limit)
    summary=json.loads((P/f'ISOLATION_SUMMARY_{limit}.json').read_text());counts=summary['counts_by_split'];passed=counts['D']['groups']>=30 and counts['H']['groups']>=18
    print('BATCH_STOP_CHECK',limit,'D',counts['D'],'H',counts['H'],'threshold',passed,flush=True)
    if passed or limit==1800:
        run(H/'spotcheck.py','--tag','D_extra','--input',P/'chains.jsonl');run(H/'finalize.py','--tag','D_extra')
        freeze=H/'COHORT_FREEZE_D_extra.json';d=json.loads(freeze.read_text());d['split_roles']={'D':'independent_development_augmentation','H':'new_homology_isolated_confirmation'};d['acquisition_evidence']={str(x):hashlib.sha256(x.read_bytes()).hexdigest()for x in [P/'PLAN.md',P/'front.py',P/'download.py',P/'physical.py',P/'isolate.py',P/'run_stages.py',P/'ACQUISITION_FREEZE.json',P/f'ISOLATION_SUMMARY_{limit}.json',P/'group_roster.jsonl',P/f'full_native_prior_F_audit_{limit}.tsv.gz',P/f'new_components_audit_{limit}.tsv']};d['blind_stopping']={'batch':limit,'threshold_met':passed,'requested_D_groups':30,'requested_H_groups':18,'hard_ceiling_clusters':1800};freeze.write_text(json.dumps(d,indent=2)+'\n')
        ready=H/'READY_D_extra.json'
        if ready.exists():
            r=json.loads(ready.read_text());r['freeze_sha256']=hashlib.sha256(freeze.read_bytes()).hexdigest();r['split_roles']=d['split_roles'];r['blind_stopping']=d['blind_stopping'];ready.write_text(json.dumps(r,indent=2)+'\n')
        print('OLDER_POOL_COMPLETE',limit,counts,'threshold',passed,flush=True);break
