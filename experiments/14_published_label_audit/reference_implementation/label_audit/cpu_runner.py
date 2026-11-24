"""CPU sequence baselines and locked-unit scores; no GPU/model extraction."""
from __future__ import annotations
import os,sys
for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse,collections,csv,gzip,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path
import joblib,numpy as np
import cpu_statistics_core as core
HERE=Path(__file__).resolve().parent
CPU=HERE/'cpu'
def now():return datetime.now(timezone.utc).isoformat()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text('',encoding='utf-8');return
 with p.open('w',encoding='utf-8',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def readtable(p):
 with Path(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f,delimiter='\t'))
def input_gate():
 # The GPU/input owner issues this receipt only after the whole cohort lock.
 p=HERE/'INPUT_FREEZE.json'
 if not p.exists():raise RuntimeError('joint_final_INPUT_FREEZE_required')
 freeze=read(p)
 if freeze.get('status')in('DRAFT','PRELIMINARY'):raise RuntimeError('final_input_freeze_is_only_draft')
 cp=read(HERE/'CPU_PROTOCOL_FREEZE.json');expected=dict(cp['implementation_files'])
 corrections=list(HERE.glob('CPU_IMPLEMENTATION_CORRECTION_*.json'))+list(CPU.glob('CPU_IMPLEMENTATION_CORRECTION_*.json'))
 for correction in sorted(corrections,key=lambda p:p.name):expected.update(read(correction)['corrected_implementation_files'])
 for name,h in expected.items():
  if sha(HERE/name)!=h:raise RuntimeError('CPU_frozen_implementation_changed:'+name)
 return freeze
def prepare():
 input_gate();seq={r['accession']:r for r in map(json.loads,(HERE/'inputs/sequences.jsonl').read_text(encoding='utf-8').splitlines())};pairs=readtable(HERE/'inputs/pairs.tsv');by=collections.defaultdict(list)
 for p in pairs:by[p['concept'],p['split']].append(p)
 manifest=[]
 for concept in sorted({p['concept']for p in pairs}):
  folder=CPU/'features'/digest(concept)[:20];folder.mkdir(parents=True,exist_ok=True);records=[]
  for split in sorted({s for c,s in by if c==concept}):
   rr=sorted(by[concept,split],key=lambda r:(r['group'],r['accession'],int(r['early0']),int(r['late0'])));x=[];strong=[];y=[]
   assert len({r['group']for r in rr})==len(rr),'one_pair_per_group'
   for r in rr:
    s=seq[r['accession']]['sequence'];a,b=int(r['early0']),int(r['late0']);assert s[a]==s[b]==r['aa']and 15<=a<b<len(s)-15
    xx,ss=core.encode_pair(s,a,b);x.append(xx);strong.append(ss);y.append(float(r['target']))
   np.savez_compressed(folder/(split+'.npz'),linear=np.array(x),strong=np.array(strong),y=np.array(y),groups=np.array([r['group']for r in rr]),accessions=np.array([r['accession']for r in rr]),early0=np.array([int(r['early0'])for r in rr]),late0=np.array([int(r['late0'])for r in rr]))
   table(folder/(split+'_pairs.tsv'),rr);records.append(dict(split=split,groups=len(rr)))
  drows=by.get((concept,'D'),[]);dsig=digest(concept+'|'+json.dumps(sorted([(r['group'],r['accession'],r['early0'],r['late0'],r['target'])for r in drows]),separators=(',',':')))
  manifest.append(dict(concept=concept,folder=str(folder),D_groups=len(drows),D_ordered_pair_identity=dsig,splits=records,baseline_eligible=len(drows)>=20))
 dump(CPU/'FEATURE_MANIFEST.json',dict(status='COMPLETE_INPUT_FEATURES_NO_CONFIRMATION_SCORING',utc=now(),input_freeze_sha256=sha(HERE/'INPUT_FREEZE.json'),sequences_sha256=sha(HERE/'inputs/sequences.jsonl'),pairs_sha256=sha(HERE/'inputs/pairs.tsv'),concepts=manifest,linear_features=686,strong_features=1346,confirmation_activations_read=0))
 print('CPU_FEATURES_PREPARED',len(manifest),flush=True)
@lru_cache(maxsize=24)
def pack(path):
 with np.load(path,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def baseline_one(entry):
 folder=CPU/'baselines'/entry['D_ordered_pair_identity'];folder.mkdir(parents=True,exist_ok=True)
 if(folder/'COMPLETE.json').exists():return read(folder/'COMPLETE.json')
 d=pack(str(Path(entry['folder'])/'D.npz'));start=time.monotonic();penalty,linear_scores=core.select_linear(d['linear'],d['y'],d['groups'])
 def progress(i,j):dump(folder/'PROGRESS.json',dict(utc=now(),concept=entry['concept'],grid_index=i,fold_index=j,elapsed_s=time.monotonic()-start))
 params,scores=core.select_strong(d['strong'],d['y'],d['groups'],progress);model=core.strong_fit(d['strong'],d['y'],d['groups'],params);joblib.dump(model,folder/'strong_base.joblib')
 result=dict(status='COMPLETE_D_ONLY_BASELINE',utc=now(),concept=entry['concept'],D_groups=len(d['y']),D_ordered_pair_identity=entry['D_ordered_pair_identity'],linear_penalty=penalty,linear_CV_Brier=linear_scores,linear_penalty_grid=list(core.PENALTIES),strong_params=params,strong_CV_Brier=scores,strong_grid=core.STRONG_GRID,elapsed_s=time.monotonic()-start,D_features_path=str(Path(entry['folder'])/'D.npz'),confirmation_outputs_read=0)
 dump(folder/'COMPLETE.json',result);return result
def baselines(workers):
 input_gate();manifest=read(CPU/'FEATURE_MANIFEST.json');entries=[e for e in manifest['concepts']if e['baseline_eligible']];results=[]
 with ProcessPoolExecutor(max_workers=workers)as pool:
  futures={pool.submit(baseline_one,e):e for e in entries}
  for f in as_completed(futures):r=f.result();results.append(r);print('D_BASELINE_COMPLETE',r['concept'],r['D_groups'],round(r['elapsed_s'],1),flush=True)
 dump(CPU/'BASELINE_FIT_RECEIPT.json',dict(status='COMPLETE_D_ONLY_BASELINES',utc=now(),workers=workers,threads_per_worker=1,results=results,confirmation_scores_produced=0))
def split_frame(split):return'T'if split=='T'else'NEW'if split in('RESERVE','FRESH')else None
def feature_entry(concept):return next((e for e in read(CPU/'FEATURE_MANIFEST.json')['concepts']if e['concept']==concept),dict(concept=concept,D_groups=0,baseline_eligible=False,splits=[],folder=str(CPU/'no_pair_features')))
def ordered_z(rr,features):
 lookup={(r['group'],r['accession'],int(r['early0']),int(r['late0'])):r for r in rr}
 result=[]
 for group,acc,a,b,y in zip(features['groups'],features['accessions'],features['early0'],features['late0'],features['y']):
  key=(str(group),str(acc),int(a),int(b))
  if key not in lookup:raise ValueError('model_readout_missing_frozen_pair:'+str(key))
  r=lookup[key];assert int(float(r['target']))==int(y),'readout_target_mismatch';result.append(float(r['oriented_late_minus_early']))
 return np.array(result)
def load_frame(entry,frame):
 splits=['T']if frame=='T'else['RESERVE','FRESH'];pieces=[]
 for split in splits:
  p=Path(entry['folder'])/(split+'.npz')
  if p.exists():pieces.append(pack(str(p)))
 if not pieces:return None
 return{k:np.concatenate([p[k]for p in pieces])for k in pieces[0]}
def auc_domain_rows(auc_rows,sequences,family,frame):
 # AUC eligibility is independent of whether a same-AA pair exists.
 fresh_groups={r['group']for r in sequences if str(r.get('own_fresh_eligible','False')).lower()in('true','1')}
 return[r for r in auc_rows if split_frame(r['split'])==frame and r.get('auroc','')not in('','nan','NaN','None')and np.isfinite(float(r['auroc']))and(not(family=='OWN'and frame=='NEW')or r['group']in fresh_groups)]
def descriptive_without_D(rr,auc_rows,entry,meta,out,frames=('T','NEW'),reason='DESCRIPTIVE_ONLY_INSUFFICIENT_D_GROUPS'):
 sequences=[json.loads(x)for x in(HERE/'inputs/sequences.jsonl').read_text(encoding='utf-8').splitlines()];results=[]
 for frame in frames:
  pairs=[r for r in rr if split_frame(r['split'])==frame];aa=auc_domain_rows(auc_rows,sequences,meta['family'],frame)
  if meta['family']=='OWN'and frame=='NEW':
   pairs=[r for r in pairs if str(r.get('own_fresh_eligible','False')).lower()in('true','1')]
  universe=sorted({r['group']for r in sequences if split_frame(r['split'])==frame});ag=np.array([r['group']for r in aa]);av=np.array([float(r['auroc'])for r in aa]);pg=np.array([r['group']for r in pairs]);credit=core.paired_credit([float(r['oriented_late_minus_early'])for r in pairs],[float(r['target'])for r in pairs])
  empty=dict(n=0,mean=None,ci95=None,p_one_sided=None,undefined_bootstrap=2000);auc_inf=core.inference(av,ag,universe,compute_p=False)if len(aa)else empty.copy();pair_inf=core.inference(credit-.5,pg,universe,compute_p=False)if len(pairs)else empty.copy()
  availability='ALPHA_ESTIMABLE_BETA_GAMMA_INSUFFICIENT'if len(aa)>=20 else'THREE_QUESTIONS_MISSING'if not aa and not pairs else'PARTIAL_DESCRIPTIVE_DATA_ONLY'
  result=dict(status=reason,availability=availability,formal_beta_gamma_eligible=False,frame=frame,D_groups=entry['D_groups'],groups=len(pairs),auc_groups=len(aa),pair_accuracy=float(credit.mean())if len(pairs)else None,pair_centered_inference=pair_inf,linear_Brier_gain=empty.copy(),strong_Brier_gain=empty.copy(),annotation_vs_unannotated_AUROC=auc_inf,**meta)
  dump(out/(frame+'_RESULT.json'),result)
  if aa:table(out/(frame+'_group_AUC.tsv'),aa)
  if pairs:table(out/(frame+'_paired_description.tsv'),[dict(group=r['group'],target=r['target'],oriented_late_minus_early=r['oriented_late_minus_early'],paired_credit=float(c))for r,c in zip(pairs,credit)])
  results.append(result)
 return results
def score_one(task):
 directory=Path(task['directory']);unit=task['unit'];concept=task['concept'];entry=feature_entry(concept);job=digest(str(directory)+'|'+unit+'|'+concept)[:24];out=CPU/'unit_results'/job
 if(out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
 out.mkdir(parents=True,exist_ok=True);rr=[r for r in readtable(directory/'pairs.tsv')if r['unit_id']==unit and r['concept']==concept]if(directory/'pairs.tsv').exists()else[]
 auc_rows=[r for r in readtable(directory/'auc.tsv')if r['unit_id']==unit and r['concept']==concept]if(directory/'auc.tsv').exists()else[]
 first=rr[0]if rr else auc_rows[0]if auc_rows else{};reference_kind=read(directory/'COMPLETE.json').get('reference',directory.name);native=reference_kind=='trained'or directory.name in('trained','trained_masked');reference='trained'if native else first.get('reference_id',directory.name);family=('PUBLIC'if first.get('source','').lower()=='hallee'else'OWN')if native else('OWN'if concept.startswith('OWN_')else'PUBLIC');meta={k:first.get(k,'')for k in('source','model','block0','unit0','pole','sign','readpoint')};meta.update(unit_id=unit,concept=concept,reference=reference,family=family,readout_family=first.get('family'),original_family=first.get('original_family'),masked=str(first.get('masked','False')).lower()in('true','1'))
 catalog={u['unit_id']:u for u in read(HERE/'UNIT_MANIFEST.json')['units']}
 if unit in catalog:
  for key in('source','model','block0','unit0','sign','pole','annotation_category','masked','status'):meta['identity_status'if key=='status'else key]=catalog[unit].get(key)
  meta['readout_family']=catalog[unit].get('family');meta['original_family']=catalog[unit].get('family');meta['family']='PUBLIC'if meta['source']=='Hallee'else'OWN'
 if not entry['baseline_eligible']:
  frames=descriptive_without_D(rr,auc_rows,entry,meta,out);result=dict(status='COMPLETE_DESCRIPTIVE_ONLY_NO_SEQUENCE_FIT',D_groups=entry['D_groups'],frames=frames,**meta);dump(out/'COMPLETE.json',result);return result
 if not rr:raise ValueError('eligible_unit_missing_frozen_pair_readouts')
 baseline_dir=CPU/'baselines'/entry['D_ordered_pair_identity'];base=read(baseline_dir/'COMPLETE.json');d=pack(str(Path(entry['folder'])/'D.npz'));dz=ordered_z([r for r in rr if r['split']=='D'],d);scale=core.scalar_scale(dz);start=time.monotonic()
 # Linear augmented penalty follows archived paired_rsa: D-CV for this scalar.
 linear_aug_penalty,linear_aug_scores=core.select_linear(d['linear'],d['y'],d['groups'],dz)
 base_model=joblib.load(baseline_dir/'strong_base.joblib');aug_model=None
 if scale>1e-12:aug_model=core.strong_fit(d['strong'],d['y'],d['groups'],base['strong_params'],dz/scale);joblib.dump(aug_model,out/'strong_plus_scalar.joblib')
 sequences=[json.loads(x)for x in(HERE/'inputs/sequences.jsonl').read_text(encoding='utf-8').splitlines()];results=[]
 for frame in('T','NEW'):
  t=load_frame(entry,frame)
  if t is None:
   results.extend(descriptive_without_D(rr,auc_rows,entry,meta,out,frames=(frame,),reason='DESCRIPTIVE_ONLY_NO_EVALUATION_PAIRS'));continue
  tr=[r for r in rr if split_frame(r['split'])==frame]
  if not tr:
   results.extend(descriptive_without_D(rr,auc_rows,entry,meta,out,frames=(frame,),reason='DESCRIPTIVE_ONLY_NO_READOUT_PAIRS'));continue
  full_frame_groups=len(t['y'])
  if family=='OWN'and frame=='NEW':
   if any('own_fresh_eligible'not in r for r in tr):raise ValueError('own_NEW_fresh_eligibility_required')
   eligible={r['group']for r in tr if str(r['own_fresh_eligible']).lower()in('true','1')};keep=np.array([str(g)in eligible for g in t['groups']]);t={k:v[keep]for k,v in t.items()};tr=[r for r in tr if r['group']in eligible]
   if not len(t['y']):
    results.extend(descriptive_without_D(rr,auc_rows,entry,meta,out,frames=(frame,),reason='DESCRIPTIVE_ONLY_NO_FRESH_ELIGIBLE_PAIRS'));continue
  tz=ordered_z(tr,t);target=t['y'];groups=t['groups'];n=len(target);universe=sorted({r['group']for r in sequences if split_frame(r['split'])==frame})
  p_linear=core.linear_predict(d['linear'],t['linear'],d['y'],base['linear_penalty']);p_aug=core.linear_predict(d['linear'],t['linear'],d['y'],linear_aug_penalty,dz,tz)
  p_strong=core.strong_predict(base_model,t['strong']);p_strong_aug=p_strong.copy()if aug_model is None else core.strong_predict(aug_model,t['strong'],tz/scale)
  if scale<=1e-12:p_aug=p_linear.copy()
  lin0=(p_linear-target)**2;lin1=(p_aug-target)**2;strong0=(p_strong-target)**2;strong1=(p_strong_aug-target)**2;credit=core.paired_credit(tz,target)
  valid_auc=auc_domain_rows(auc_rows,sequences,family,frame)
  auc_values=np.array([float(r['auroc'])for r in valid_auc]);auc_groups=np.array([r['group']for r in valid_auc]);assert len(set(auc_groups))==len(auc_groups),'repeated_AUC_group'
  assessable=n>=20;pair_inf=core.inference(credit-.5,groups,universe,compute_p=native and assessable);linear_inf=core.inference(lin0-lin1,groups,universe,compute_p=native and assessable);strong_inf=core.inference(strong0-strong1,groups,universe,compute_p=native and assessable);auc_inf=core.inference(auc_values,auc_groups,universe,compute_p=False)if len(auc_values)else dict(n=0,mean=None,ci95=None,p_one_sided=None)
  result=dict(status='EVALUABLE'if assessable else'INSUFFICIENT_GROUPS',frame=frame,D_groups=len(dz),groups=n,auc_groups=len(auc_values),pair_accuracy=float(credit.mean()),pair_centered_inference=pair_inf,linear_Brier_gain=linear_inf,strong_Brier_gain=strong_inf,annotation_vs_unannotated_AUROC=auc_inf,D_scalar_augmented_mean=0.,D_scalar_augmented_SD_RMS=scale,zero_D_scalar_scale=scale<=1e-12,linear_penalty_base=base['linear_penalty'],linear_penalty_augmented=linear_aug_penalty,strong_params=base['strong_params'],**meta)
  result['own_T_status']='RETROSPECTIVE_NOT_NEW_PROTEINS' if family=='OWN' and frame=='T' else None
  result['frame_before_own_fresh_filter_groups']=full_frame_groups;result['own_NEW_excluded_previously_used_groups']=full_frame_groups-n if family=='OWN'and frame=='NEW'else 0
  result['formal_beta_gamma_eligible']=assessable;result['availability']='ALL_THREE_ESTIMABLE'if assessable and len(auc_values)>=20 else'BETA_GAMMA_ESTIMABLE_ALPHA_MISSING_OR_SMALL'if assessable else'PARTIAL_DESCRIPTIVE_DATA_ONLY'
  # Preserve every group for independent recomputation and all unfavorable rows.
  table(out/(frame+'_group_effects.tsv'),[dict(group=str(g),target=int(y),oriented_late_minus_early=float(z),paired_credit=float(c),linear_probability=float(a),linear_plus_scalar_probability=float(b),strong_probability=float(s),strong_plus_scalar_probability=float(tprob),linear_Brier_gain=float(lg),strong_Brier_gain=float(sg))for g,y,z,c,a,b,s,tprob,lg,sg in zip(groups,target,tz,credit,p_linear,p_aug,p_strong,p_strong_aug,lin0-lin1,strong0-strong1)])
  if valid_auc:table(out/(frame+'_group_AUC.tsv'),valid_auc)
  dump(out/(frame+'_RESULT.json'),result);results.append(result)
 receipt=dict(status='COMPLETE_UNIT_CPU_ANALYSIS',utc=now(),job=job,directory=str(directory),unit_id=unit,concept=concept,frames=results,D_scalar_groups=len(dz),linear_augmented_CV_scores=linear_aug_scores,elapsed_s=time.monotonic()-start,readout_complete_sha256=sha(directory/'COMPLETE.json'),**{k:v for k,v in meta.items()if k not in('unit_id','concept')});dump(out/'COMPLETE.json',receipt);return receipt
def score_readout(directory,workers):
 input_gate();directory=Path(directory)
 if not(directory/'COMPLETE.json').exists():raise RuntimeError('GPU_readout_COMPLETE_required')
 # Joint evaluation authority is separate from the result-blind D feature stage.
 if not(HERE/'FREEZE.json').exists():raise RuntimeError('joint_final_FREEZE_required_before_confirmation_scores')
 rr=(readtable(directory/'pairs.tsv')if(directory/'pairs.tsv').exists()else[])+(readtable(directory/'auc.tsv')if(directory/'auc.tsv').exists()else[]);keys={(r['unit_id'],r['concept'])for r in rr}
 if read(directory/'COMPLETE.json').get('reference')=='trained'or directory.name in('trained','trained_masked'):
  for u in read(HERE/'UNIT_MANIFEST.json')['units']:
   if u['model']==directory.parent.name and bool(u.get('masked'))==('masked'in directory.name):keys.add((u['unit_id'],u['concept']))
 tasks=[dict(directory=str(directory),unit=u,concept=c)for u,c in sorted(keys)]
 with ProcessPoolExecutor(max_workers=workers)as pool:
  fut=[pool.submit(score_one,t)for t in tasks]
  for f in as_completed(fut):r=f.result();print('CPU_UNIT_COMPLETE',r['unit_id'],r['concept'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','baselines','score-readout']);p.add_argument('--workers',type=int,default=8);p.add_argument('--directory');a=p.parse_args()
 if a.stage=='prepare':prepare()
 elif a.stage=='baselines':baselines(a.workers)
 else:score_readout(a.directory,a.workers)
