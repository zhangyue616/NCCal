"""Resumable serial GPU owner for Exp2; one model is released per job."""
from prepare import *
from extract import trained
from references import one
from masked_1246 import trained_masked,random_masked
import traceback,time,gc
MODELS=['esm2_8m','esm2_35m','esm2_650m','esmc_600','e1_150m']
def stopped():
    return any(p.exists() for p in [OUT/'STOP_SECTION6.json',OUT.parent/'STOP_SECTION6.json'])
def main():
    freeze=js(OUT/'FREEZE.json')
    for name,h in freeze['GPU_code_files'].items():assert sha(OUT/name)==h,name
    jobs=[('trained',m,None) for m in MODELS]+[('trained_masked','esm2_8m',None)]
    # Each completed network produces its chosen D identity and all groups.
    # The two masked/unmasked jobs of 8M use the identical seed population.
    for i in range(1,200):
        jobs.extend(('random',m,i) for m in MODELS);jobs.append(('random_masked','esm2_8m',i))
    for i in range(1,10):
        jobs.extend(('rotation',m,i) for m in MODELS);jobs.append(('rotation_masked','esm2_8m',i))
    start=time.time();done=0;timings=[]
    for kind,model,index in jobs:
        if stopped():
            write('GPU_QUEUE_STATUS.json',dict(status='STOP_SECTION6_AT_NETWORK_BOUNDARY',utc=datetime.now(timezone.utc).isoformat(),done_jobs=done,next_job=[kind,model,index],elapsed_seconds=time.time()-start));return
        t=time.time()
        try:
            if kind=='trained':trained(model)
            elif kind=='trained_masked':trained_masked()
            elif kind=='random_masked':random_masked(index,'random')
            elif kind=='rotation_masked':random_masked(index,'rotation')
            else:one(model,index,kind)
        except Exception as e:
            write('GPU_QUEUE_FAILURE.json',dict(status='TECHNICAL_FAILURE_NO_AUTOMATIC_RETRY',utc=datetime.now(timezone.utc).isoformat(),kind=kind,model=model,index=index,error=str(e),traceback=traceback.format_exc(),done_jobs=done,next_step='Diagnose one concrete cause; log implementation-only correction and affected scope before resume; never change scientific inputs'))
            raise
        elapsed=time.time()-t;done+=1;timings.append(dict(kind=kind,model=model,index=index,seconds=elapsed))
        write('GPU_QUEUE_STATUS.json',dict(status='RUNNING',utc=datetime.now(timezone.utc).isoformat(),done_jobs=done,total_jobs=len(jobs),last_job=[kind,model,index],last_seconds=elapsed,elapsed_seconds=time.time()-start,completed_job_timings=timings,input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json')))
        gc.collect()
    write('GPU_QUEUE_COMPLETE.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),jobs=done,elapsed_seconds=time.time()-start,timings=timings,input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json')))
if __name__=='__main__':main()
