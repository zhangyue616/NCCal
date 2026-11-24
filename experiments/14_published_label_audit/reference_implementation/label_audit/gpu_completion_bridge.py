"""One-shot, metadata-only completion bridge for the unchanged CPU watcher."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
from datetime import datetime,timezone
OUT=Path('label_audit')
FROZEN={
 'INPUT_FREEZE.json':'b3f2e0d3e46615929758a65c50136f4f6bff4092f008d757ed5616b1bfabe4a3',
 'FREEZE.json':'97999fb643e7ce2286dc2935f033ef7acc450091f240e7665f48cd911768390b',
 'CPU_PROTOCOL_FREEZE.json':'6648f18264c490231ec4134c3a67d39f32354e90800ee858c5452ce587f0c753',
 'CPU_IMPLEMENTATION_CORRECTION_001.json':'c4b171c492f740870b6b6756c673c6b82540101f1abc824fc13cd26cb23667b8'}
MODELS=['esm2_8m','esm2_35m','esm2_650m','esmc_600','e1_150m']
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def stopped():return (OUT/'STOP_SECTION6.json').exists() or (OUT.parent/'STOP_SECTION6.json').exists()
class AuthorStop(Exception):pass
def expected_jobs():
 jobs=[('trained',m,None) for m in MODELS]+[('trained_masked','esm2_8m',None)]
 for kind,budget in [('random',199),('rotation',9)]:
  for i in range(1,budget+1):
   jobs.extend((kind,m,i) for m in MODELS);jobs.append((kind+'_masked','esm2_8m',i))
 return jobs
def folder(kind,model,index):
 name=kind if index is None else kind+'_'+(f'{index:03d}' if kind.startswith('random') else f'{index:02d}')
 return OUT/'readouts'/model/name
def verify(run,attempt):
 assert sha(Path(__file__))==sha(attempt/'BRIDGE_SOURCE.py'),'Live bridge source differs from archived execution source'
 for rel,digest in FROZEN.items():assert sha(OUT/rel)==digest,'Freeze drift: '+rel
 for rel,digest in read(OUT/'INPUT_FREEZE.json')['input_files'].items():assert sha(OUT/rel)==digest,'Input drift: '+rel
 for rel,digest in read(OUT/'FREEZE.json')['GPU_code_files'].items():assert sha(OUT/rel)==digest,'Frozen GPU code drift: '+rel
 start=read(run/'START.json');assert sha(run/'DRIVER_SOURCE.py')==start['driver_sha256']
 receipt=run/'FULL_QUEUE_COMPLETE.json';full=read(receipt)
 assert full['status']=='ALL_1254_FROZEN_READOUT_JOBS_COMPLETE'
 assert full['input_freeze_sha256']==FROZEN['INPUT_FREEZE.json'] and full['joint_freeze_sha256']==FROZEN['FREEZE.json']
 assert full['random_network_budget_per_architecture']==199 and full['rotation_budget']==9 and full['same_8M_masked_unmasked_seed_checks']==199
 jobs=expected_jobs();assert len(jobs)==1254
 reported=[str(Path(p).resolve()).casefold() for p in full['completed_readout_paths']]
 required=[str(folder(*job).resolve()).casefold() for job in jobs]
 assert len(reported)==len(set(reported))==1254 and set(reported)==set(required)
 rows=[]
 for kind,model,index in jobs:
  if stopped():raise AuthorStop()
  path=folder(kind,model,index);meta=read(path/'COMPLETE.json');assert meta['status']=='COMPLETE' and meta['model']==model
  assert meta['input_freeze_sha256']==FROZEN['INPUT_FREEZE.json']
  assert meta['reference']==('random' if kind.startswith('random') else 'rotation' if kind.startswith('rotation') else 'trained')
  if index is not None:
   assert meta['index']==index and meta['seed']==(202611230000+index if kind.startswith('random') else None)
   assert not(path/'values.npy').exists(),'Completed reference retained unexpected temporary full array'
  assert sha(path/'pairs.tsv')==meta['pairs_sha256'],'Pairs drift: '+str(path)
  for field,filename in [('auc_sha256','auc.tsv'),('value_schema_sha256','VALUE_SCHEMA.json')]:
   if meta.get(field):assert sha(path/filename)==meta[field],filename+' drift: '+str(path)
  rows.append(dict(kind=kind,model=model,index=index,complete_path=str(path/'COMPLETE.json'),complete_sha256=sha(path/'COMPLETE.json'),pairs_sha256=meta['pairs_sha256']))
 initializations=[]
 for log in (OUT/'gpu_resume_runs').glob('*/INITIALIZATIONS.jsonl'):
  with log.open(encoding='utf-8') as f:initializations.extend(json.loads(line) for line in f if line.strip())
 checks=[]
 for i in range(1,200):
  a=read(folder('random','esm2_8m',i)/'SELECTION.json');b=read(folder('random_masked','esm2_8m',i)/'SELECTION.json');seed=202611230000+i
  assert a['seed']==b['seed']==seed
  obs=[r['runtime_model'] for r in initializations if r['model']=='esm2_8m' and r['seed']==seed]
  assert len(obs)>=2 and all(r['parameters']==7512474 and r['unit_width']==1280 and r['kind']=='esm2' for r in obs)
  fingerprints={r['first_FFN_weight_sample_sha256'] for r in obs};assert len(fingerprints)==1
  fingerprint=next(iter(fingerprints));assert a['runtime_model']['first_FFN_weight_sample_sha256']==fingerprint
  checks.append(dict(index=i,seed=seed,actual_initialization_observations=len(obs),FFN_1024_float_sample_sha256=fingerprint,independent_network_budget=1,full_weight_hash_comparison=False))
 manifest=attempt/'VERIFIED_MANIFEST.json';dump(manifest,dict(status='VERIFIED_1254_COMPLETE_RECEIPTS_AND_199_SHARED_8M_SEEDS',utc=now(),freeze_hashes=FROZEN,readout_rows=rows,shared_8M_seed_checks=checks))
 return dict(status='COMPLETE',utc=now(),readout_jobs=1254,native_jobs=6,random_observation_jobs=1194,rotation_observation_jobs=54,random_networks_per_model=199,rotations=9,shared_8M_seed_identity_checks=199,full_queue_receipt=str(receipt),full_queue_receipt_sha256=sha(receipt),verified_manifest=str(manifest),verified_manifest_sha256=sha(manifest),freeze_hashes=FROZEN,bridge_source_sha256=sha(Path(__file__)),CPU_watcher_completion_signal_only=True)
def main(run_name):
 assert os.name=='nt','Non-overwriting atomic os.rename semantics are required on this Windows host'
 run=OUT/'gpu_resume_runs'/run_name;assert run.is_dir() and run.resolve().is_relative_to((OUT/'gpu_resume_runs').resolve())
 attempt=run/'completion_bridge'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');attempt.mkdir(parents=True,exist_ok=False)
 (attempt/'BRIDGE_SOURCE.py').write_bytes(Path(__file__).read_bytes())
 dump(attempt/'START.json',dict(status='WAITING_FOR_FULL_RECEIPT_NO_EARLY_GLOBAL_MARKER',utc=now(),PID=os.getpid(),target=str(run/'FULL_QUEUE_COMPLETE.json'),later_run_completion_allowed_after_scheduler_yield=True,global_signal=str(OUT/'GPU_QUEUE_COMPLETE.json'),source_sha256=sha(Path(__file__)),poll_seconds=30,requires='1254 COMPLETE artifacts and pair hashes,199 actual shared8M seed observations,4 freeze hashes and frozen inputs/GPU code'))
 print('COMPLETION BRIDGE START',os.getpid(),str(attempt),flush=True)
 try:
  while True:
   if stopped():raise AuthorStop()
   if (OUT/'GPU_QUEUE_COMPLETE.json').exists():
    dump(attempt/'COMPLETE.json',dict(status='EXISTING_GLOBAL_MARKER_PRESERVED',utc=now()));return
   candidates=sorted((OUT/'gpu_resume_runs').glob('*/FULL_QUEUE_COMPLETE.json'))
   candidates=[p for p in candidates if p.parent.name>=run_name]
   if candidates:
    # A normal scheduler yield may finish the same frozen queue in a later
    # invocation. The full canonical1254/199/freeze verification is unchanged.
    run=candidates[0].parent;break
   time.sleep(30)
  result=verify(run,attempt)
  if stopped():raise AuthorStop()
  temporary=OUT/('GPU_QUEUE_COMPLETE.json.pending.'+str(os.getpid()))
  dump(temporary,result)
  # os.rename on Windows atomically exposes a whole file and refuses overwrite.
  os.rename(temporary,OUT/'GPU_QUEUE_COMPLETE.json')
  dump(attempt/'COMPLETE.json',dict(status='GLOBAL_COMPLETION_SIGNAL_PUBLISHED_AFTER_VERIFICATION',utc=now(),global_signal=str(OUT/'GPU_QUEUE_COMPLETE.json')))
  print('GPU_QUEUE_COMPLETE verified and published',flush=True)
 except AuthorStop:
  dump(attempt/'STATUS.json',dict(status='AUTHOR_STOP_SAFE_EXIT_NO_GLOBAL_SIGNAL_WRITTEN',utc=now()));print('BRIDGE AUTHOR STOP',flush=True)
 except Exception as exc:
  dump(attempt/'FAILURE.json',dict(status='NO_AUTOMATIC_RETRY_NO_GLOBAL_MARKER',utc=now(),error=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_name');main(p.parse_args().run_name)
