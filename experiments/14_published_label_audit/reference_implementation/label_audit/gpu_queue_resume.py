from prepare import *
from extract import trained
from references import one
from masked_1246 import trained_masked,random_masked
import argparse,time,traceback,gc,os,torch
from model_runtime import Runtime
MODELS=['esm2_8m','esm2_35m','esm2_650m','esmc_600','e1_150m']
INITIALIZATIONS=[]
def allocator_boundary_policy(run):
    """Release only inactive cached blocks after the original forward returns."""
    original=Runtime.forward;original_init=Runtime.__init__;count=[0]
    torch.cuda.set_per_process_memory_fraction(.45)
    def initialize(self,name,seed=None):
        start=time.monotonic();original_init(self,name,seed);torch.cuda.synchronize()
        event=dict(utc=datetime.now(timezone.utc).isoformat(),model=name,seed=seed,seconds=time.monotonic()-start,allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved(),runtime_model=self.metadata)
        INITIALIZATIONS.append(event)
        with (run/'INITIALIZATIONS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def forward(self,records,mask_positions=None):
        try:return original(self,records,mask_positions)
        finally:
            count[0]+=1;allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
            if reserved-allocated>2*1024**3:
                start=time.monotonic();torch.cuda.empty_cache()
                event=dict(utc=datetime.now(timezone.utc).isoformat(),forward=count[0],model=self.name,chains=len(records),native_lengths=[len(r['sequence']) for r in records],allocated_before_bytes=allocated,reserved_before_bytes=reserved,allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved(),release_seconds=time.monotonic()-start,scientific_operator_parameters_changed=False)
                with (run/'CACHE_RELEASES.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    Runtime.__init__=initialize;Runtime.forward=forward
def stopped():return any(p.exists() for p in [OUT/'STOP_SECTION6.json',OUT.parent/'STOP_SECTION6.json'])
def path_for(kind,model,index):
    name=kind if index is None else f'random_{index:03d}' if kind=='random' else f'rotation_{index:02d}' if kind=='rotation' else f'random_masked_{index:03d}' if kind=='random_masked' else f'rotation_masked_{index:02d}'
    return OUT/'readouts'/model/name
def main(max_seed=199,rotations=True,max_rotation=9):
    frozen=js(OUT/'FREEZE.json')
    for name,h in frozen['GPU_code_files'].items():assert sha(OUT/name)==h,name
    assert (OUT/'readouts/esmc_600/trained/COMPLETE.json').exists(),'Finish protected C600 continuation before queue resume'
    pause=js(OUT/'PAUSE_CHECKPOINT.json')
    for path,h in pause['completed_receipts'].items():assert sha(Path(path))==h,'Accepted native receipt drift'
    run_name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=OUT/'gpu_resume_runs'/run_name;run.mkdir(parents=True,exist_ok=False)
    (run/'DRIVER_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    allocator_boundary_policy(run)
    jobs=[('trained',m,None) for m in MODELS]+[('trained_masked','esm2_8m',None)]
    for i in range(1,max_seed+1):
        jobs.extend(('random',m,i) for m in MODELS);jobs.append(('random_masked','esm2_8m',i))
    if rotations:
        for i in range(1,max_rotation+1):
            jobs.extend(('rotation',m,i) for m in MODELS);jobs.append(('rotation_masked','esm2_8m',i))
    write(str((run/'START.json').relative_to(OUT)),dict(status='RESUMING_FROZEN_199_9_SCHEDULE',utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),driver_sha256=sha(Path(__file__)),input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json'),joint_freeze_sha256=sha(OUT/'FREEZE.json'),max_seed_this_invocation=max_seed,final_frozen_seed_budget=199,rotations_this_invocation=rotations,max_rotation_this_invocation=max_rotation,final_frozen_rotation_budget=9,accepted_directories_not_recomputed=True,scientific_changes=False,allocator_execution_policy='Only inactive CUDA cache released at original-forward return if reserved-minus-allocated exceeds2GiB;45% process allocator budget backstop, above measured3.5GB active peak; no batch/precision/weights/inputs/operator changes'))
    start=time.time();completed=[];skipped=[]
    for kind,model,index in jobs:
        folder=path_for(kind,model,index)
        if(folder/'COMPLETE.json').exists():skipped.append(str(folder));continue
        if stopped():
            write(str((run/'STATUS.json').relative_to(OUT)),dict(status='STOP_AT_COMPLETE_JOB_BOUNDARY',utc=datetime.now(timezone.utc).isoformat(),next_job=[kind,model,index],completed=completed,skipped=skipped,elapsed_seconds=time.time()-start));return
        if (run/'YIELD_AFTER_JOB.json').exists():
            write(str((run/'STATUS.json').relative_to(OUT)),dict(status='SCHEDULER_YIELD_AT_COMPLETE_JOB_BOUNDARY',utc=datetime.now(timezone.utc).isoformat(),next_job=[kind,model,index],completed=completed,skipped=skipped,elapsed_seconds=time.time()-start));return
        torch.cuda.reset_peak_memory_stats();before=torch.cuda.mem_get_info();t=time.time();init_start=len(INITIALIZATIONS)
        try:
            if kind=='trained':trained(model)
            elif kind=='trained_masked':trained_masked()
            elif kind=='random_masked':random_masked(index,'random')
            elif kind=='rotation_masked':random_masked(index,'rotation')
            else:one(model,index,kind)
        except Exception as e:
            write(str((run/'FAILURE.json').relative_to(OUT)),dict(status='NO_AUTOMATIC_RETRY',utc=datetime.now(timezone.utc).isoformat(),job=[kind,model,index],error=str(e),traceback=traceback.format_exc(),completed=completed));raise
        torch.cuda.synchronize();seconds=time.time()-t
        record=dict(kind=kind,model=model,index=index,seconds=seconds,initializations=INITIALIZATIONS[init_start:],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved(),device_free_before_bytes=before[0],device_free_after_bytes=torch.cuda.mem_get_info()[0],complete_receipt=str(folder/'COMPLETE.json'))
        completed.append(record)
        write(str((run/'STATUS.json').relative_to(OUT)),dict(status='RUNNING',utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),completed=completed,skipped=skipped,elapsed_seconds=time.time()-start))
        print('RESUMED JOB COMPLETE',record,flush=True);gc.collect()
    # Both instantiated observations of an 8M initialization must correspond
    # to one seed, not two independent additions to the reference population.
    seed_checks=[];initialization_archive=[]
    for log in (OUT/'gpu_resume_runs').glob('*/INITIALIZATIONS.jsonl'):
        with log.open(encoding='utf-8') as f:initialization_archive.extend(json.loads(line) for line in f if line.strip())
    for i in range(1,max_seed+1):
        unmasked=OUT/f'readouts/esm2_8m/random_{i:03d}/SELECTION.json';masked=OUT/f'readouts/esm2_8m/random_masked_{i:03d}/SELECTION.json'
        if unmasked.exists() and masked.exists():
            a,b=js(unmasked),js(masked);assert a['seed']==b['seed']==202611230000+i
            observed=[r['runtime_model'] for r in initialization_archive if r['model']=='esm2_8m' and r['seed']==a['seed']]
            assert len(observed)>=2,'Missing actual paired initialization observations'
            fingerprints={r['first_FFN_weight_sample_sha256'] for r in observed};assert len(fingerprints)==1
            seed_checks.append(dict(index=i,seed=a['seed'],same_initializer_and_config=True,observed_initializations=len(observed),first_FFN_1024_float_sample_sha256=next(iter(fingerprints)),sample_identity_check=True,full_weight_hash_comparison=False,independent_network_count=1))
    write(str((run/'COMPLETE.json').relative_to(OUT)),dict(status='COMPLETE_REQUESTED_RESUME_TRANCHE',utc=datetime.now(timezone.utc).isoformat(),completed=completed,skipped=skipped,elapsed_seconds=time.time()-start,same_seed_observation_checks=seed_checks,final_frozen_budget=199))
    if max_seed==199 and rotations and max_rotation==9:
        assert len(jobs)==1254 and all((path_for(*job)/'COMPLETE.json').exists() for job in jobs)
        write(str((run/'FULL_QUEUE_COMPLETE.json').relative_to(OUT)),dict(status='ALL_1254_FROZEN_READOUT_JOBS_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),native_jobs=6,random_observation_jobs=1194,rotation_observation_jobs=54,random_network_budget_per_architecture=199,physical_architectures=5,same_8M_masked_unmasked_seed_checks=len(seed_checks),rotation_budget=9,input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json'),joint_freeze_sha256=sha(OUT/'FREEZE.json'),completed_readout_paths=[str(path_for(*job)) for job in jobs]))
    print('RESUME TRANCHE COMPLETE',str(run),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-seed',type=int,default=199);p.add_argument('--no-rotations',action='store_true');p.add_argument('--max-rotation',type=int,default=9);a=p.parse_args();assert 1<=a.max_seed<=199 and 1<=a.max_rotation<=9;main(a.max_seed,not a.no_rotations,a.max_rotation)
