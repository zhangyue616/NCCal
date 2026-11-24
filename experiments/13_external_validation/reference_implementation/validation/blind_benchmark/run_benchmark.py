"""Frozen native-activation label benchmark. Source data are read only."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,hashlib,argparse,time
from datetime import datetime,timezone
from collections import Counter
import numpy as np
from scipy.stats import binom,binomtest
from scipy.special import expit
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
C=ROOT/'nccal_evidence_extension/cohort';M=ROOT/'assets/esm2_8m'
SEEDS={'extract':18472001,'calibration':28472003,'null_evaluation':28472009,'truth':38472011,'single_0.25':48472001,'single_1':48472003,'single_4':48472007,'distributed_1':58472003,'distributed_4':58472007,'real_bootstrap':68472001}
METHODS=['D_unadjusted','D_max_calibrated','D_then_E_locked','E_locked','D_effect75','D_E_effect75']
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,v):
 (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n',encoding='utf8')
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig'),delimiter='\t'))
def table(n,rr):
 with (OUT/n).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
def jl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]
def verify():
 f=read(OUT/'FREEZE.json')
 for p,h in f['bindings'].items():assert sha(p)==h,'Frozen file changed: '+p
 return f
def prepare():
 assert not (OUT/'FREEZE.json').exists(),'Already frozen; never overwrite a scored freeze'
 chains=jl(C/'chains.jsonl');ch={x['chain_id']:x for x in chains}
 sf={x['site_id']:x for x in rows(C/'structure_features.tsv')};obs=rows(C/'observed_residues.tsv')
 by={}
 for r in obs:
  if r['aa']=='C':continue
  by.setdefault((r['homology_group'],r['chain_id'],r['aa']),[]).append((int(r['seq_index0']),float(sf[r['site_id']]['sf_nonCys_remote_CA_count_10A']),r['site_id']))
 selected={};candidate_counts=Counter()
 for (g,cid,aa),sites in by.items():
  sites.sort()
  for i,(a,da,ida) in enumerate(sites):
   for b,db,idb in sites[i+1:]:
    if b-a<=30 or abs(da-db)<4:continue
    candidate_counts[g]+=1
    key=hashlib.sha256(f'density-pair-v1|{g}|{cid}|{a}|{b}'.encode()).hexdigest()
    r={'group':g,'chain_id':cid,'split':ch[cid]['split'],'early_index0':a,'late_index0':b,'early_site_id':ida,'late_site_id':idb,'aa':aa,'early_density':da,'late_density':db,'target':int(db>da),'pair_selection_sha256':key,'accession':ch[cid]['accession'],'sequence_sha256':ch[cid]['sequence_sha256'],'native_length':len(ch[cid]['sequence']),'taxid':ch[cid]['taxid'],'organism':ch[cid]['organism']}
    if g not in selected or key<selected[g]['pair_selection_sha256']:selected[g]=r
 pairs=[selected[g] for g in sorted(selected)]
 assert len(pairs)>=20 and sum(x['split']=='A' for x in pairs)>=8
 table('density_pairs.tsv',pairs)
 rng=np.random.default_rng(SEEDS['truth']);axes=rng.choice(1280,9,replace=False).tolist();signs=rng.choice([-1,1],8).tolist()
 files=[OUT/'PROTOCOL.md',Path(__file__),OUT/'density_pairs.tsv',C/'chains.jsonl',C/'observed_residues.tsv',C/'structure_features.tsv',C/'COHORT_FREEZE.json',C/'sequence_similarity_audit.tsv',M/'model.safetensors',M/'config.json',M/'vocab.txt']
 freeze={'utc':utc(),'status':'FROZEN_BEFORE_NEW_FORWARD_OR_ASSOCIATION','bindings':{str(p):sha(p) for p in files},'seeds':SEEDS,'single_truth_axis_zero_based':axes[0],'distributed_truth_axes_zero_based':axes[1:],'distributed_signs':signs,'n_calibration':1999,'n_null':1000,'n_signal_per_condition':500,'conditions':[['single',.25],['single',1],['single',4],['distributed',1],['distributed',4]],'width':1280,'original_groups':47,'eligible_pairs':len(pairs),'split_counts':dict(Counter(x['split'] for x in pairs)),'excluded_groups':sorted(set(x['homology_group'] for x in chains)-set(selected)),'candidate_counts':dict(candidate_counts),'activation_scoring_performed':False}
 dump('FREEZE.json',freeze);print(json.dumps({k:v for k,v in freeze.items() if k not in ('bindings','candidate_counts')}),flush=True)
def extract():
 f=verify();pairs=rows(OUT/'density_pairs.tsv');chains={x['chain_id']:x for x in jl(C/'chains.jsonl')}
 import torch
 from transformers import EsmForMaskedLM
 torch.manual_seed(SEEDS['extract']);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 model=EsmForMaskedLM.from_pretrained(M,local_files_only=True).float().eval()
 assert (model.config.hidden_size,model.config.intermediate_size,model.config.num_hidden_layers)==(320,1280,6)
 vocab={t:i for i,t in enumerate((M/'vocab.txt').read_text().splitlines())}
 seqs={chains[p['chain_id']]['sequence_sha256']:chains[p['chain_id']]['sequence'] for p in pairs}
 cap={}
 class AtLayer(Exception):pass
 def hook(m,i,o):cap['a']=o.detach();raise AtLayer()
 h=model.esm.encoder.layer[1].intermediate.dense.register_forward_hook(hook)
 dest=OUT/'pair_activations';dest.mkdir(exist_ok=True);start=time.monotonic();details=[]
 for n,(key,s) in enumerate(sorted(seqs.items())):
  indices=sorted({int(p[k]) for p in pairs if p['sequence_sha256']==key for k in ('early_index0','late_index0')})
  ids=torch.tensor([[vocab['<cls>'],*[vocab[a] for a in s],vocab['<eos>']]])
  assert ids.shape[1]==len(s)+2 and not (ids==vocab['<mask>']).any()
  with torch.inference_mode():
   try:model.esm(input_ids=ids,attention_mask=torch.ones_like(ids))
   except AtLayer:pass
  a=cap.pop('a')[0,1:-1].float().cpu().numpy();assert a.shape==(len(s),1280) and np.isfinite(a).all()
  np.savez_compressed(dest/f'{key}.npz',positions_index0=np.array(indices),values=a[indices])
  # Only an extraction check against the already saved signed coordinate, not a new endpoint.
  errors=[]
  for p in pairs:
   if p['sequence_sha256']!=key:continue
   old=ROOT/'disulfide/activations'/f"{p['chain_id']}.npz"
   if old.exists():
    with np.load(old,allow_pickle=False) as z:errors.append(float(np.max(np.abs(a[:,361]-z['signed361']))))
  details.append({'sequence_sha256':key,'length':len(s),'saved_sites':len(indices),'source_signed361_max_abs_error':max(errors) if errors else None})
  if n%10==0:print('extracted',n+1,'of',len(seqs),'elapsed',round(time.monotonic()-start,2),flush=True)
 h.remove()
 dump('EXTRACTION.json',{'utc':utc(),'torch':torch.__version__,'device':'cpu','n_unique_full_native_forwards':len(seqs),'width':1280,'block_1indexed':2,'location':'FFN_dense_pre_GELU','unmasked':True,'all_source_sequences_unchanged':True,'saved_full_width_pair_sites':sum(r['saved_sites'] for r in details),'seconds':time.monotonic()-start,'details':details,'freeze_sha256':sha(OUT/'FREEZE.json')})
def matrix():
 pairs=rows(OUT/'density_pairs.tsv');lookup={};delta=[]
 for p in pairs:
  key=p['sequence_sha256']
  if key not in lookup:
   with np.load(OUT/'pair_activations'/f'{key}.npz',allow_pickle=False) as z:lookup[key]={int(i):v.copy() for i,v in zip(z['positions_index0'],z['values'])}
  delta.append(lookup[key][int(p['late_index0'])].astype(float)-lookup[key][int(p['early_index0'])].astype(float))
 z=np.stack(delta);assert np.isfinite(z).all()
 d=np.array([p['split']=='A' for p in pairs]);y=np.array([2*int(p['target'])-1 for p in pairs],dtype=np.int32)
 return pairs,z,d,y
def wilson(k,n):
 if not n:return [None,None]
 zz=1.959963984540054;ph=k/n;den=1+zz*zz/n;c=(ph+zz*zz/(2*n))/den;h=zz*np.sqrt(ph*(1-ph)/n+zz*zz/(4*n*n))/den
 return [float(c-h),float(c+h)]
def evaluate(sd,se,yd,ye,cal):
 # Y has rounds x groups, s groups x axes. Truth never enters this selector.
 corr=yd@sd;axis=np.argmax(np.abs(corr),axis=1);k=np.arange(len(yd));direction=np.where(corr[k,axis]>=0,1,-1)
 signed=np.abs(corr[k,axis]);nonzero=(sd[:,axis]!=0).sum(0);correct=(nonzero+signed)//2
 dp=binom.sf(correct-1,nonzero,.5);dw=.5+signed/(2*len(sd))
 epred=se[:,axis].T*direction[:,None];ecorr=(ye*epred).sum(1);enonzero=(epred!=0).sum(1);ecorrect=(enonzero+ecorr)//2
 ep=binom.sf(ecorrect-1,enonzero,.5);ew=.5+ecorr/(2*len(se))
 # Maxima are comparable count scores on the identical development matrix.
 cp=(1+(cal[:,None]>=signed[None,:]).sum(0))/(len(cal)+1)
 flags=np.stack([dp<=.05,cp<=.05,(dp<=.05)&(ep<=.05),ep<=.05,dw>=.75,(dw>=.75)&(ew>=.75)],axis=1)
 return {'axis':axis,'direction':direction,'D_win':dw,'E_win':ew,'D_p_unadjusted':dp,'D_p_search':cp,'E_p_locked':ep,'D_nonzero':nonzero,'E_nonzero':enonzero,'flags':flags}
def analyze():
 f=verify();pairs,z,d,y=matrix();sd=np.sign(z[d]).astype(np.int32);se=np.sign(z[~d]).astype(np.int32)
 nD,nE=len(sd),len(se);w=z.shape[1];assert len(set(p['group'] for p in pairs))==len(pairs)
 rg=np.random.default_rng(SEEDS['calibration']);cy=rg.choice([-1,1],(f['n_calibration'],nD)).astype(np.int32);cal=np.max(np.abs(cy@sd),axis=1)
 np.save(OUT/'independent_calibration_maxima.npy',cal)
 np.savez_compressed(OUT/'fixed_pair_background.npz',delta=z,development=d,real_signed_target=y,groups=np.array([p['group'] for p in pairs]))
 detail=[];summaries=[]
 def one_condition(name,yd,ye,truth_axes=None,truth_signs=None,oracle=None):
  e=evaluate(sd,se,yd,ye,cal);N=len(yd)
  rec=np.zeros(N,dtype=bool)
  if truth_axes is not None:
   for a,sg in zip(truth_axes,truth_signs):rec|=(e['axis']==a)&(e['direction']==sg)
  for j,m in enumerate(METHODS):
   hit=e['flags'][:,j];nr=int(hit.sum());joint=int((hit&rec).sum())
   summaries.append({'condition':name,'method':m,'rounds':N,'retained':nr,'retention_rate':nr/N,'retention_mc95_low':wilson(nr,N)[0],'retention_mc95_high':wilson(nr,N)[1],'recovered_generator_member_and_sign':int(rec.sum()) if truth_axes is not None else None,'recovery_rate':float(rec.mean()) if truth_axes is not None else None,'retained_and_recovered':joint if truth_axes is not None else None,'retained_recovered_rate':joint/N if truth_axes is not None else None,'mean_D_win':float(e['D_win'].mean()),'mean_E_win':float(e['E_win'].mean())})
  for i in range(N):
   detail.append({'condition':name,'round':i,'selected_axis_zero_based':int(e['axis'][i]),'selected_sign':int(e['direction'][i]),'D_win':float(e['D_win'][i]),'E_win':float(e['E_win'][i]),'D_unadjusted_p':float(e['D_p_unadjusted'][i]),'D_search_p':float(e['D_p_search'][i]),'E_locked_p':float(e['E_p_locked'][i]),'recovered_generator_member_and_sign':int(rec[i]) if truth_axes is not None else '',**{m:int(e['flags'][i,j]) for j,m in enumerate(METHODS)}})
  out={'condition':name,'n_rounds':N,'selection_original_index_recovery':int(rec.sum()) if truth_axes is not None else None,'index_recovery_mc95':wilson(int(rec.sum()),N) if truth_axes is not None else None,'selection_top_axes':dict(Counter(map(int,e['axis'])).most_common(10))}
  if oracle is not None:
   oc=(ye*oracle[None,:]).sum(1);onz=int((oracle!=0).sum());op=binom.sf((onz+oc)//2-1,onz,.5);keep=op<=.05
   out.update(oracle_confirmation_retained=int(keep.sum()),oracle_confirmation_rate=float(keep.mean()),oracle_mc95=wilson(int(keep.sum()),N))
  return out
 rng=np.random.default_rng(SEEDS['null_evaluation']);flips=rng.choice([-1,1],(f['n_null'],len(z))).astype(np.int32);ny=flips*y
 conditions=[one_condition('null_pair_flips',ny[:,d],ny[:,~d])]
 rms=np.sqrt(np.mean(z[d]**2,axis=0));rms=np.where(rms>0,rms,1.);zn=z/rms
 for kind,beta in f['conditions']:
  if kind=='single':axes=[f['single_truth_axis_zero_based']];signs=[1];latent=zn[:,axes[0]]
  else:
   axes=f['distributed_truth_axes_zero_based'];signs=f['distributed_signs'];latent=(zn[:,axes]@signs)/np.sqrt(8);latent/=max(1e-12,float(np.sqrt(np.mean(latent[d]**2))))
  p=expit(beta*latent);name=kind+'_'+str(beta);rng=np.random.default_rng(SEEDS[name]);sy=np.where(rng.random((f['n_signal_per_condition'],len(z)))<p,1,-1).astype(np.int32)
  c=one_condition(name,sy[:,d],sy[:,~d],axes,signs,np.sign(latent[~d]).astype(np.int32))
  c.update(expected_oracle_E_accuracy=float(np.mean(np.maximum(p[~d],1-p[~d]))),expected_E_label_entropy_bits=float(np.mean(-(p[~d]*np.log2(np.clip(p[~d],1e-30,1))+(1-p[~d])*np.log2(np.clip(1-p[~d],1e-30,1))))),development_latent_rms=float(np.sqrt(np.mean(latent[d]**2))),confirmation_latent_rms=float(np.sqrt(np.mean(latent[~d]**2))))
  conditions.append(c)
 real=evaluate(sd,se,y[d][None,:],y[~d][None,:],cal);a=int(real['axis'][0]);direction=int(real['direction'][0]);wins=.5+.5*y[~d]*se[:,a]*direction
 rng=np.random.default_rng(SEEDS['real_bootstrap']);boot=wins[rng.integers(0,nE,(10000,nE))].mean(1)
 realinfo={'selected_axis_zero_based':a,'selected_sign':direction,'D_groups':nD,'E_groups':nE,'D_win':float(real['D_win'][0]),'E_win':float(real['E_win'][0]),'D_p_unadjusted':float(real['D_p_unadjusted'][0]),'D_p_search':float(real['D_p_search'][0]),'E_p_locked':float(real['E_p_locked'][0]),'E_win_group_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'E_ties':int((se[:,a]==0).sum()),'E_wilson95_nontied':wilson(int((wins==1).sum()),int((wins!=.5).sum())),'retained_methods':{m:bool(real['flags'][0,j]) for j,m in enumerate(METHODS)}}
 rrows=[]
 for p,ww,dd in zip([p for p,b in zip(pairs,d) if not b],wins,z[~d,a]):rrows.append({**p,'locked_axis_delta':float(dd),'locked_axis_sign':direction,'win':float(ww)})
 table('real_confirmation_predictions.tsv',rrows);table('simulation_rounds.tsv',detail);table('retention_summary.tsv',summaries)
 # All levels are explicit; null MC intervals are conditional, never biological N.
 result={'utc':utc(),'freeze_sha256':sha(OUT/'FREEZE.json'),'data':{'protein_homology_groups':len(z),'development_groups':nD,'confirmation_groups':nE,'unique_sequences_used':len(set(p['sequence_sha256'] for p in pairs)),'width':w,'coordinate_sign_candidates':2*w,'host_taxa':dict(Counter(p['organism'] for p in pairs)),'length_min_max':[min(int(p['native_length']) for p in pairs),max(int(p['native_length']) for p in pairs)],'D_zero_differences':int((sd==0).sum()),'E_zero_differences':int((se==0).sum()),'independent_units':'original homology groups; one natural structural pair each','confirmation_composition':dict(Counter(p['split'] for p,b in zip(pairs,d) if not b))},'calibration':{'n':len(cal),'independent_seed':SEEDS['calibration'],'max_signed_correctness_quantiles':{str(q):float(np.quantile(cal,q)) for q in [.5,.9,.95,.99,1]},'p_resolution':1/(len(cal)+1),'realization_frozen_before_evaluation_null_labels':True},'actual_tests':{'calibration_full_searches':len(cal),'evaluated_generated_label_searches':len(detail),'real_full_searches':1,'original_coordinates_per_search':w,'signs_per_coordinate':2,'development_selected_binomial_tests_per_round':1,'independent_confirmation_tests_per_round':1,'search_empirical_rank_tests_per_round':1,'oracle_confirmation_tests_per_signal_round':1,'effect_threshold_comparisons_per_round':2,'full_search_coordinate_evaluations':(len(cal)+len(detail)+1)*w,'separate_nominal_methods_are_comparators_not_six_claims_adjusted_as_a_family':True},'conditions':conditions,'real_structural_test':realinfo,'limitations':['Conditional Monte Carlo on fixed natural activation background, not additional proteins','Generated practice labels are semi-synthetic and not biological functional truth','One screened axis and limited homology groups, not all NCCal gates','Same-level search calibration uses additional reference searches; confirmation uses additional independent groups','Historical source proteins reused; original B+T is benchmark confirmation, not an untouched original biological assay test','Cross-split homology cutoff does not rule out remote evolutionary relationships','Planted coordinate recovery differs from detecting a correlated predictive association'],'summary':summaries}
 dump('RESULT.json',result);print(json.dumps({'data':result['data'],'real':realinfo,'null':[x for x in summaries if x['condition']=='null_pair_flips'],'conditions':conditions},ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze']);args=p.parse_args();globals()[args.stage]()
