"""Preselected raw-group and residue-value audit; no fits or model forwards."""
import os,sys
sys.dont_write_bytecode=True
for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import csv,itertools,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;CPU=HERE/'cpu';RESULTS=HERE/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p):
 with Path(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f,delimiter='\t'))
def number(x):return float(x)if x not in(None,'','None','nan','NaN')else None
def assertclose(a,b,tol=1e-11):
 if a is None and b is None:return 0.
 assert a is not None and b is not None
 error=abs(float(a)-float(b));assert error<=tol,(a,b,error);return error
plan=read(HERE/'CPU_INDEPENDENT_AUDIT_PLAN.json');master=table(RESULTS/'per_unit_results.tsv');lookup={(r['unit_id'],r['frame']):r for r in master};jobs=[(p.parent,read(p))for p in(CPU/'unit_results').glob('*/COMPLETE.json')];seq=[json.loads(x)for x in(HERE/'inputs/sequences.jsonl').read_text(encoding='utf-8').splitlines()];labels=read(HERE/'inputs/LABELS.json');checks=[];auc_checks=[];maxerr=0.
fresh_groups={r['group']for r in seq if str(r.get('own_fresh_eligible','False')).lower()in('true','1')};alpha_denominator_checks=[]
for model in('esm2_8m','esm2_35m','esm2_650m'):
 source=table(HERE/'readouts'/model/'trained'/'auc.tsv');eligible=[r for r in source if r['concept']=='OWN_EXPOSURE'and r['split']in('RESERVE','FRESH')and r['group']in fresh_groups and np.isfinite(float(r['auroc']))]
 target=next(r for r in master if r['model']==model and r['concept']=='OWN_EXPOSURE'and r['frame']=='NEW')
 assert len({r['group']for r in eligible})==int(target['auc_groups'])==154 and int(target['groups'])==136
 maxerr=max(maxerr,assertclose(np.mean([float(r['auroc'])for r in eligible]),number(target['auc_mean'])))
 alpha_denominator_checks.append(dict(model=model,own_NEW_alpha_groups=154,own_NEW_beta_gamma_groups=136,source='raw native AUC rows filtered by frozen sequence eligibility, independently of pair availability',status='PASS'))
for unit in plan['unit_ids']:
 candidates=[(folder,job)for folder,job in jobs if job['unit_id']==unit and job.get('reference')=='trained']
 if not candidates:checks.append(dict(unit_id=unit,status='NO_COMPLETED_NATIVE_JOB'));continue
 folder,job=candidates[0]
 for frame in('T','NEW'):
  if(unit,frame)not in lookup:continue
  row=lookup[unit,frame];p=folder/(frame+'_group_effects.tsv');desc=folder/(frame+'_paired_description.tsv');effects=table(p)if p.exists()else table(desc)if desc.exists()else[]
  if effects:
   credit=np.array([1. if float(r['oriented_late_minus_early'])*(2*float(r['target'])-1)>0 else .5 if float(r['oriented_late_minus_early'])==0 else 0. for r in effects]);maxerr=max(maxerr,assertclose(credit.mean(),number(row['pair_accuracy'])))
   if p.exists():
    target=np.array([float(r['target'])for r in effects]);base=np.array([float(r['strong_probability'])for r in effects]);plus=np.array([float(r['strong_plus_scalar_probability'])for r in effects]);linear=np.array([float(r['linear_probability'])for r in effects]);linearplus=np.array([float(r['linear_plus_scalar_probability'])for r in effects]);gain=(base-target)**2-(plus-target)**2;lgain=(linear-target)**2-(linearplus-target)**2
    maxerr=max(maxerr,assertclose(gain.mean(),number(row['strong_mean'])),assertclose(lgain.mean(),number(row['linear_mean'])))
    for r,g,l in zip(effects,gain,lgain):maxerr=max(maxerr,assertclose(g,float(r['strong_Brier_gain'])),assertclose(l,float(r['linear_Brier_gain'])))
    # Independent direct resampling of a layer-wide vector with explicit NaNs.
    universe=sorted({r['group']for r in seq if(r['split']=='T'if frame=='T'else r['split']in('RESERVE','FRESH'))});mapping={g:i for i,g in enumerate(universe)};draw=np.random.default_rng(2026112304).integers(len(universe),size=(2000,len(universe)))
    for stem,values in[('pair',credit-.5),('strong',gain),('linear',lgain)]:
     wide=np.full(len(universe),np.nan)
     for r,v in zip(effects,values):wide[mapping[r['group']]]=v
     sampled=wide[draw];counts=np.isfinite(sampled).sum(1);boot=np.nansum(sampled,1)/np.where(counts>0,counts,np.nan);lo,hi=np.nanpercentile(boot,[2.5,97.5]);shift=.5 if stem=='pair'else 0
     maxerr=max(maxerr,assertclose(lo+shift,number(row[stem+'_CI_lo'])),assertclose(hi+shift,number(row[stem+'_CI_hi'])))
   checks.append(dict(unit_id=unit,frame=frame,groups=len(effects),probability_loss_recomputed=p.exists(),signed_pair_credit_recomputed=True,status='PASS'))
  elif row.get('availability')=='THREE_QUESTIONS_MISSING':checks.append(dict(unit_id=unit,frame=frame,status='DECLARED_MISSING_NO_ZERO_FILL'))
  # Rerank only the two prospective public audit identities, first 3 groups.
  ap=folder/(frame+'_group_AUC.tsv')
  if unit.startswith('Hallee|')and ap.exists():
   directory=Path(job['directory'])if job.get('directory')else HERE/'readouts'/job['model']/('trained_masked'if job.get('masked')else'trained');schema=read(directory/'VALUE_SCHEMA.json');columns={u['unit_id']:i for i,u in enumerate(schema['units'])};values=np.load(directory/'values.npy',mmap_mode='r');col=columns[unit]
   for r in sorted(table(ap),key=lambda x:x['group'])[:3]:
    acc=r['accession'];lab=labels[acc][row['concept']];index=schema['index'][acc];positive=values[index['offset']+np.array(lab['positive0']),col];negative=values[index['offset']+np.array(lab['negative0']),col];assert np.isfinite(positive).all()and np.isfinite(negative).all()
    statistic=float(np.mean((positive[:,None]>negative[None,:]).astype(float)+.5*(positive[:,None]==negative[None,:])));error=assertclose(statistic,float(r['auroc']));maxerr=max(maxerr,error);auc_checks.append(dict(unit_id=unit,frame=frame,group=r['group'],n_positive=len(positive),n_unannotated=len(negative),pairwise_Mann_Whitney_AUROC=statistic,saved_group_AUROC=float(r['auroc']),abs_error=error))
# BH with a scalar backwards loop, independent of the main vectorized code.
bh_checks=[]
for frame in('T','NEW'):
 for family in('PUBLIC','OWN'):
  rr=[r for r in master if r['frame']==frame and r['family']==family and r['status']=='EVALUABLE']
  for secondary in(False,True):
   hypotheses=[]
   for i,r in enumerate(rr):
    for name in(['linear']if secondary else['pair','strong']):
     p=number(r.get(name+'_p'))
     if p is not None:hypotheses.append((p,i,name))
   order=sorted(hypotheses);running=1.
   for rank in range(len(order),0,-1):
    p,i,name=order[rank-1];running=min(running,p*len(order)/rank);maxerr=max(maxerr,assertclose(running,number(rr[i].get(name+'_q'))))
   bh_checks.append(dict(frame=frame,family=family,secondary_linear=secondary,hypotheses=len(order),status='PASS'))
out=dict(status='PASS_PRESELECTED_INDEPENDENT_RECOMPUTATION',group_checks=checks,raw_residue_AUC_checks=auc_checks,BH_checks=bh_checks,alpha_denominator_002_independent_checks=alpha_denominator_checks,maximum_absolute_discrepancy=maxerr,selected_units=plan['unit_ids'],new_fits=0,new_model_forwards=0,scope='native fixed-unit statistics only; K199/9 family coverage is a separate final acceptance')
(RESULTS/'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=out['status'],group_checks=len(checks),raw_AUC_checks=len(auc_checks),BH_families=len(bh_checks),maximum_absolute_discrepancy=maxerr)))
