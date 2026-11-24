"""Independent structural endpoint assays, deterministic selection and output fidelity."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,csv,hashlib,time,copy,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,f1_score
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer,EsmForMaskedLM
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;ROOT=O.parents[1];C=O.parent/'cohort';M=ROOT/'assets/esm2_8m'
SEED=20261122;AA='ACDEFGHIKLMNPQRSTVWY';DEV=('A','B');RNG=np.random.default_rng(SEED)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE=='cuda': torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,d):
 p=O/n;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n',encoding='utf8');q.replace(p)
def table(n,d):pd.DataFrame(d).to_csv(O/n,sep='\t',index=False)
def read(n):return json.loads((O/n).read_text())
def groupmean(v,g):
 d=pd.DataFrame({'v':np.asarray(v),'g':np.asarray(g)});return d.groupby('g',sort=True).v.mean()
def ci(v,null=0):
 v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
 if not len(v):return {'n_groups':0,'mean':None,'ci95':None,'p':None}
 if len(v)<3:return {'n_groups':len(v),'mean':float(v.mean()),'ci95':None,'p':None,'null':null,'inference_status':'TOO_FEW_GROUPS_NO_INFERENCE'}
 r=np.random.default_rng(SEED);b=np.mean(v[r.integers(0,len(v),(5000,len(v)))],axis=1);m=float(v.mean())
 return {'n_groups':len(v),'mean':m,'ci95':np.quantile(b,[.025,.975]).tolist(),'p':float((1+np.sum(np.abs(b-m)>=abs(m-null)))/5001),'null':null,'inference_status':'FRAGILE_LT10_GROUPS' if len(v)<10 else 'GROUP_BOOTSTRAP'}
def holm(vals):
 ordered=sorted(vals,key=lambda k:vals[k]['p'] if vals[k]['p'] is not None else 1);last=0
 for rank,k in enumerate(ordered):
  if vals[k]['p'] is None:vals[k]['p_holm']=None;continue
  last=max(last,min(1,(len(vals)-rank)*vals[k]['p']));vals[k]['p_holm']=last
 return vals
def load():
 chains=[json.loads(l) for l in (C/'chains.jsonl').read_text().splitlines() if l.strip()]
 sites=pd.read_csv(C/'sites.tsv',sep='\t',dtype={'site_id':str,'chain_id':str,'homology_group':str},keep_default_na=False)
 pairs=pd.read_csv(C/'pairs.tsv',sep='\t',dtype={'chain_id':str,'homology_group':str})
 obs=pd.read_csv(C/'observed_residues.tsv',sep='\t',keep_default_na=False)
 sf=pd.read_csv(C/'structure_features.tsv',sep='\t',keep_default_na=False)
 return chains,sites,pairs,obs,sf
def bind():
 files=[O/'PROTOCOL.md',Path(__file__),M/'model.safetensors',M/'config.json',M/'vocab.txt']+[C/x for x in ('chains.jsonl','sites.tsv','pairs.tsv','observed_residues.tsv','structure_features.tsv')]+[O/'author/sae_layer2_normalized.pt']
 d={'utc':utc(),'status':'FROZEN_BEFORE_FORWARD','files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files], 'device':DEVICE,'torch':torch.__version__,'seed':SEED,'new_forward_count':0}
 if (O/'BINDINGS.json').exists():
  old=read('BINDINGS.json');assert old['files']==d['files'],'Binding drift'
 else:dump('BINDINGS.json',d)
 return d
def models():
 torch.manual_seed(SEED)
 trained=EsmForMaskedLM.from_pretrained(M,local_files_only=True).float().eval()
 assert (trained.config.hidden_size,trained.config.intermediate_size,trained.config.num_hidden_layers)==(320,1280,6)
 torch.manual_seed(SEED);untrained=EsmForMaskedLM(trained.config).float().eval()
 shuffled=copy.deepcopy(trained);gen=torch.Generator().manual_seed(SEED)
 with torch.no_grad():
  for p in shuffled.parameters():p.copy_(p.flatten()[torch.randperm(p.numel(),generator=gen)].reshape(p.shape))
 for m in (untrained,shuffled):m.esm.encoder.layer=torch.nn.ModuleList(list(m.esm.encoder.layer)[:2])
 trained.to(DEVICE);untrained.to(DEVICE);shuffled.to(DEVICE)
 tok=AutoTokenizer.from_pretrained(M,local_files_only=True)
 state=torch.load(O/'author/sae_layer2_normalized.pt',map_location=DEVICE,weights_only=True)
 assert state['encoder.weight'].shape==(10240,320)
 sw=state['encoder.weight'][4181];sb=state['encoder.bias'][4181];smean=state['bias']
 g=np.random.default_rng(SEED).normal(size=(1280,1280));q,r=qr(g,mode='economic');rot=(q*np.where(np.diag(r)<0,-1.,1.)).astype('float32')
 np.save(O/'rotation.npy',rot)
 def forward(chain,model,sae=False):
  seq=chain['sequence'];assert set(seq)<=set(AA) and len(seq)<=1022
  enc=tok(seq,return_tensors='pt',padding=False,truncation=False).to(DEVICE)
  assert enc['input_ids'].shape==(1,len(seq)+2) and not (enc['input_ids']==tok.mask_token_id).any()
  assert tok.convert_ids_to_tokens(enc['input_ids'][0,1:-1].tolist())==list(seq)
  cap={};h=model.esm.encoder.layer[1].intermediate.dense.register_forward_hook(lambda m,i,o:cap.update(pre=o.detach()))
  h2=model.esm.encoder.layer[1].register_forward_hook(lambda m,i,o:cap.update(res=(o[0] if isinstance(o,tuple) else o).detach()))
  with torch.inference_mode():model.esm(**enc)
  h.remove();h2.remove()
  pre=cap['pre'][0,1:-1].float().cpu().numpy()
  latent=torch.relu((cap['res'][0,1:-1]-smean)@sw+sb).float().cpu().numpy() if sae else None
  assert np.isfinite(pre).all()
  return pre,latent
 return trained,untrained,shuffled,tok,rot,forward
def wins(d):return (d>0).astype(float)+.5*(d==0)
def pick(arms,pairs,chains):
 gs=[];rows={a:[] for a in arms}
 for p in pairs.itertuples():
  gs.append(p.homology_group)
  for a,lookup in arms.items():
   z=lookup[p.chain_id];x=np.r_[np.maximum(z[p.positive_index0],0),np.maximum(-z[p.positive_index0],0)];y=np.r_[np.maximum(z[p.negative_index0],0),np.maximum(-z[p.negative_index0],0)]
   rows[a].append(wins(x-y))
 locks={};detail=[]
 for a,v in rows.items():
  w=pd.DataFrame(np.stack(v)).assign(group=gs).groupby('group').mean().mean().to_numpy(copy=True)
  w[[361,1280+361]]=-np.inf
  ix=int(np.argmax(w));locks[a]={'pole_index':ix,'unit':ix%1280,'polarity':1 if ix<1280 else -1,'A_group_win':float(w[ix]),'searched_poles':2558,'A_pairs':len(gs),'A_groups':len(set(gs))}
  for j in range(2560):
   detail.append({'arm':a,'pole_index':j,'unit':j%1280,'polarity':1 if j<1280 else -1,'A_group_win':float(w[j]) if np.isfinite(w[j]) else None})
 table('selection_scan_A.tsv',detail);dump('SELECTION_LOCK.json',{'utc':utc(),'T_forward_count':0,'selection_split':'A','tie_rule':'smallest pole index','arms':locks})
 return locks
def extract():
 bind();chains,sites,pairs,obs,sf=load();trained,untrained,shuffled,tok,rot,forward=models()
 O.joinpath('activations').mkdir(exist_ok=True)
 A=[c for c in chains if c['split']=='A'];arms={x:{} for x in ('native','rotation','untrained','shuffled')};saved={}
 print('A extraction',len(A),flush=True)
 for n,c in enumerate(A):
  z,sae=forward(c,trained,True);u,_=forward(c,untrained);s,_=forward(c,shuffled)
  arms['native'][c['chain_id']]=z;arms['rotation'][c['chain_id']]=z@rot;arms['untrained'][c['chain_id']]=u;arms['shuffled'][c['chain_id']]=s;saved[c['chain_id']]=(z,sae)
  if n%10==0:print('A',n,flush=True)
 pa=pairs[pairs.split=='A'];assert len(pa)>0
 locks=pick(arms,pa,chains)
 def save(c,z,sae,arm):
  vals={'fixed361':np.maximum(z[:,361],0),'sae4181':sae,'signed361':z[:,361]}
  for a,l in locks.items(): vals[a]=np.maximum(l['polarity']*arm[a][:,l['unit']],0)
  vals['signed_candidate']=z[:,locks['native']['unit']]
  np.savez_compressed(O/'activations'/f"{c['chain_id']}.npz",**vals)
 for c in A:save(c,*saved[c['chain_id']],{a:v[c['chain_id']] for a,v in arms.items()})
 del arms,saved
 for split in ('B','T'):
  current=[c for c in chains if c['split']==split];print(split,'extraction',len(current),flush=True)
  for n,c in enumerate(current):
   z,sae=forward(c,trained,True);u,_=forward(c,untrained);s,_=forward(c,shuffled)
   save(c,z,sae,{'native':z,'rotation':z@rot,'untrained':u,'shuffled':s})
   if n%10==0:print(split,n,flush=True)
 dump('EXTRACTION.json',{'utc':utc(),'chains':len(chains),'device':DEVICE,'unmasked':True,'full_sequences':True,'pre_GELU':True,'layer_1indexed':2,'sae_read':'ReLU(encoder.weight[4181] @ (layer2_residual-bias)+encoder.bias[4181])','controls':['Haar seed20261122','random initialized seed20261122','each trained parameter shuffled seed20261122'],'selection_lock_sha':sha(O/'SELECTION_LOCK.json')})
def feature(seq,pos):
 x=np.zeros(10*21+3)
 for k,d in enumerate([d for d in range(-5,6) if d]):
  j=pos+d;c=seq[j] if 0<=j<len(seq) else '_';x[k*21+(AA.index(c) if c in AA else 20)]=1
 x[-3:]=[np.log1p(len(seq)),pos/max(1,len(seq)-1),seq.count('C')/len(seq)]
 return x
def gweights(g):
 u,c=np.unique(g,return_counts=True);d=dict(zip(u,c));return np.array([1/d[x] for x in g])*len(g)/len(u)
def ridgefit(x,y,g,alpha):
 med=np.nanmedian(x,axis=0);med=np.where(np.isfinite(med),med,0);x=np.where(np.isfinite(x),x,med)
 w=gweights(g);sc=StandardScaler().fit(x,sample_weight=w);m=Ridge(alpha=alpha).fit(sc.transform(x),y,sample_weight=w)
 return lambda xx:m.predict(sc.transform(np.where(np.isfinite(xx),xx,med))),{'alpha':alpha,'median':med.tolist(),'scale':sc.scale_.tolist(),'mean':sc.mean_.tolist(),'coef':m.coef_.tolist(),'intercept':float(m.intercept_)}
def logfit(x,y,g,Cval):
 # Pair covariates have no offset; preserve difference antisymmetry and scale only.
 sc=np.sqrt(np.average(x*x,axis=0,weights=gweights(g)));sc=np.where(sc>1e-8,sc,1)
 xx=np.r_[x/sc,-x/sc];yy=np.r_[y,1-y];ww=np.r_[gweights(g),gweights(g)]
 m=LogisticRegression(C=Cval,fit_intercept=False,solver='lbfgs',max_iter=3000).fit(xx,yy,sample_weight=ww)
 return lambda z:expit((z/sc)@m.coef_[0]),{'C':Cval,'scale':sc.tolist(),'coef':m.coef_[0].tolist()}
def metrics(y,v,threshold=None):
 y=np.asarray(y,dtype=int);v=np.asarray(v);d={'n':len(y),'positive':int(y.sum()),'negative':int((1-y).sum())}
 if len(np.unique(y))>1:d.update(auroc=float(roc_auc_score(y,v)),ap=float(average_precision_score(y,v)))
 if threshold is not None:
  q=v>=threshold;tp=int((q & (y==1)).sum());fp=int((q & (y==0)).sum());fn=int((~q & (y==1)).sum());d.update(threshold=float(threshold),tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
 return d
def analyze():
 chains,sites,pairs,obs,sf=load();cl={c['chain_id']:c for c in chains};acts={c['chain_id']:dict(np.load(O/'activations'/f"{c['chain_id']}.npz")) for c in chains}
 groups={c['chain_id']:c['homology_group'] for c in chains};splits={c['chain_id']:c['split'] for c in chains}
 # Recover observed truth from eligibility-aware Cys table, never interval labels.
 lab={(r.chain_id,int(r.seq_index0)):int(r.endpoint) for r in sites.itertuples() if str(r.endpoint) in ('0','1') and str(r.eligible).lower() in ('true','1')}
 sr=[]
 for r in obs.itertuples():
  cid=r.chain_id;i=int(r.seq_index0);aa=cl[cid]['sequence'][i]
  if aa=='C' and (cid,i) not in lab:continue
  row={'chain_id':cid,'seq_index0':i,'group':groups[cid],'split':splits[cid],'aa':aa,'endpoint':lab.get((cid,i),0)}
  row.update({a:float(acts[cid][a][i]) for a in ('fixed361','native','rotation','untrained','shuffled','sae4181')});sr.append(row)
 allsites=pd.DataFrame(sr);table('all_observed_readouts.tsv',allsites)
 discr={};thresholds={}
 for arm in ('fixed361','native','rotation','untrained','shuffled','sae4181'):
  d=allsites[allsites.split.isin(DEV)];pr,re,th=precision_recall_curve(d.endpoint,d[arm]);f=2*pr[:-1]*re[:-1]/np.maximum(1e-20,pr[:-1]+re[:-1]);mx=f.max();thr=float(th[np.flatnonzero(f==mx)[-1]]);thresholds[arm]=thr
  t=allsites[allsites.split=='T'];discr[arm]={'all_observed':metrics(t.endpoint,t[arm],thr),'Cys_only':metrics(t[t.aa=='C'].endpoint,t[t.aa=='C'][arm],thr)}
 dump('THRESHOLD_LOCK_DEV.json',{'utc':utc(),'thresholds':thresholds,'test_threshold_selection':False});dump('DISCRIMINATION.json',discr)
 # All eligible Cys sites, with structure-only sf_ covariates.
 ss=sites[sites.apply(lambda r:str(r.endpoint) in ('0','1') and str(r.eligible).lower() in ('true','1'),axis=1)].copy().reset_index(drop=True)
 for f in ('homology_group','split'):
  ss[f]=ss.chain_id.map({c['chain_id']:c[f] for c in chains})
 sx=np.stack([feature(cl[r.chain_id]['sequence'],int(r.seq_index0)) for r in ss.itertuples()])
 sfcols=[x for x in sf.columns if x.startswith('sf_')];assert sfcols
 assert not any(any(q in k.lower() for q in ('disulf','bond','sulfur','sulphur','partner','sg_distance')) for k in sfcols)
 sf=sf.set_index('site_id');extra=np.array([[pd.to_numeric(sf.loc[r.site_id,k],errors='coerce') for k in sfcols] for r in ss.itertuples()],dtype=float);fx=np.c_[sx,extra]
 ai=np.flatnonzero(ss.split=='A');bi=np.flatnonzero(ss.split=='B');di=np.flatnonzero(ss.split.isin(DEV));ti=np.flatnonzero(ss.split=='T');gs=ss.homology_group.to_numpy()
 ix={r.site_id:i for i,r in enumerate(ss.itertuples())};pp=[];px=[]
 for p in pairs.itertuples():
  a=ix[p.positive_site_id];b=ix[p.negative_site_id];assert ss.iloc[a].endpoint in (1,'1') and ss.iloc[b].endpoint in (0,'0')
  if int(ss.iloc[a].seq_index0)>int(ss.iloc[b].seq_index0):a,b=b,a
  pp.append({'pair_id':p.pair_id,'chain_id':p.chain_id,'group':p.homology_group,'split':p.split,'a_site':ss.iloc[a].site_id,'b_site':ss.iloc[b].site_id,'a_index':a,'b_index':b,'y':int(str(ss.iloc[a].endpoint)=='1')});px.append(sx[a]-sx[b])
 pf=pd.DataFrame(pp);px=np.stack(px);pg=pf.group.to_numpy();py=pf.y.to_numpy();pa=np.flatnonzero(pf.split=='A');pb=np.flatnonzero(pf.split=='B');pdv=np.flatnonzero(pf.split.isin(DEV));pt=np.flatnonzero(pf.split=='T')
 predrows=ss[['site_id','chain_id','seq_index0','homology_group','split','endpoint']].copy();fits={};recon={};pairres={};pairfamily={};reconfamily={};groupdata=[]
 # The prediction increment uses the rectified readout, reconstruction uses signed preactivation.
 for arm in ('fixed361','native','rotation','untrained','shuffled','sae4181'):
  vals=np.array([acts[r.chain_id][arm][int(r.seq_index0)] for r in ss.itertuples()]);pv=vals[pf.a_index]-vals[pf.b_index];sgn=np.where(py==1,1,-1);pf[arm+'_delta_bonded']=pv*sgn;pf[arm+'_win']=wins(pv*sgn)
  best={};ps={}
  for model,x in [('local',px),('plus',np.c_[px,pv])]:
   scores=[]
   for cval in (.01,.1,1.,10.):
    fn,meta=logfit(x[pa],py[pa],pg[pa],cval);scores.append((float(groupmean((fn(x[pb])-py[pb])**2,pg[pb]).mean()),cval))
   loss,cval=min(scores);fn,meta=logfit(x[pdv],py[pdv],pg[pdv],cval);best[model]={'B_brier':loss,'fit':meta};ps[model]=fn(x);pf[arm+'_'+model+'_p_earlier']=ps[model]
  b0=(ps['local'][pt]-py[pt])**2;b1=(ps['plus'][pt]-py[pt])**2;gain=groupmean(b0-b1,pg[pt]);ciobj=ci(gain)
  tdelta=pf.iloc[pt][arm+'_delta_bonded'].to_numpy();res={'pairs':len(pt),'groups':len(gain),'correct':int((tdelta>0).sum()),'ties':int((tdelta==0).sum()),'incorrect':int((tdelta<0).sum()),'win':ci(groupmean(wins(tdelta),pg[pt]),null=.5),'local_brier':float(groupmean(b0,pg[pt]).mean()),'plus_brier':float(groupmean(b1,pg[pt]).mean()),'paired_brier_gain':ciobj,'fit_selection':best};pairres[arm]=res
  for g,v in gain.items():groupdata.append({'family':'pair_brier_gain','readout':arm,'group':g,'value':v})
  if arm in ('fixed361','native'):pairfamily[arm]=ciobj
 for arm,col in [('fixed361','signed361'),('native','signed_candidate')]:
  yy=np.array([acts[r.chain_id][col][int(r.seq_index0)] for r in ss.itertuples()]);preds={};fits[arm]={}
  mean=float(np.average(yy[di],weights=gweights(gs[di])));preds['mean']=np.full(len(yy),mean);fits[arm]['mean']=mean
  for model,x in [('seq',sx),('struct',fx)]:
   trial=[]
   for alpha in (.1,1.,10.,100.):
    fn,meta=ridgefit(x[ai],yy[ai],gs[ai],alpha);trial.append((float(groupmean((fn(x[bi])-yy[bi])**2,gs[bi]).mean()),alpha))
   loss,alpha=min(trial);fn,meta=ridgefit(x[di],yy[di],gs[di],alpha);preds[model]=fn(x);fits[arm][model]={'B_mse':loss,'fit':meta}
  predrows[arm+'_signed_actual']=yy
  report={}
  for model,p in preds.items():
   predrows[arm+'_'+model]=p;err=(p[ti]-yy[ti])**2;gv=groupmean(err,gs[ti]);report[model]={'MSE':ci(gv),'pearson':float(np.corrcoef(p[ti],yy[ti])[0,1]) if np.std(p[ti])>1e-12 else None,'abs_residual_quantiles':np.quantile(np.abs(p[ti]-yy[ti]),[.5,.9,.95,.99,1]).tolist()}
   for g,v in gv.items():groupdata.append({'family':'reconstruction_mse','readout':arm+'_'+model,'group':g,'value':v})
  for before,after in [('mean','seq'),('seq','struct')]:
   gv=groupmean((preds[before][ti]-yy[ti])**2-(preds[after][ti]-yy[ti])**2,gs[ti]);o=ci(gv);report[before+'_minus_'+after]=o;reconfamily[arm+'_'+before+'_minus_'+after]=o
  recon[arm]=report
 fixed_ranking={};frgroups=[]
 for split in ('pooled_ABT','A','B','T'):
  ff=pf if split=='pooled_ABT' else pf[pf.split==split]
  rr={}
  for arm in ('fixed361','sae4181'):
   dd=ff[arm+'_delta_bonded'];gv=groupmean(ff[arm+'_win'],ff.group);rr[arm]={'chains':len(ff),'groups':len(gv),'correct':int((dd>0).sum()),'ties':int((dd==0).sum()),'incorrect':int((dd<0).sum()),'success':ci(gv,null=.5)}
   for gg,vv in gv.items():frgroups.append({'scope':split,'readout':arm,'group':gg,'win':vv})
  rr['paired_361_minus_SAE']=ci(groupmean(ff.fixed361_win-ff.sae4181_win,ff.group));fixed_ranking[split]=rr
 table('externally_nominated_rank_per_group.tsv',frgroups);dump('EXTERNALLY_NOMINATED_RANKING.json',{'timing':'user request before any new cohort forward/scoring','rule':'fixed positive361=max(z,0), SAE4181 official ReLU latent; no fit or threshold; higher ranks bonded; tie0.5','same_dataset_not_another_replication':True,'candidate_pooled':False,'results':fixed_ranking})
 dump('FIT_LOCK.json',{'utc':utc(),'structure_columns':sfcols,'fits':fits,'T_refits':0});table('cys_predictions.tsv',predrows);table('paired_predictions.tsv',pf);table('per_group.tsv',groupdata)
 dump('ANALYSIS.json',{'counts':{'chains':len(chains),'homology_groups':len(set(groups.values())),'all_observed':len(allsites),'eligible_Cys':len(ss),'test_Cys':len(ti),'test_pairs':len(pt),'test_pair_groups':len(set(pg[pt]))},'discrimination':discr,'paired':pairres,'reconstruction':recon,'external_fixed_ranking':fixed_ranking,'paired_family_holm':holm(pairfamily),'reconstruction_family_holm':holm(reconfamily)})
def fidelity():
 chains,sites,pairs,obs,sf=load();cl={c['chain_id']:c for c in chains};pred=pd.read_csv(O/'cys_predictions.tsv',sep='\t').set_index('site_id');lock=read('SELECTION_LOCK.json')['arms']['native'];trained=EsmForMaskedLM.from_pretrained(M,local_files_only=True).float().eval().to(DEVICE);tok=AutoTokenizer.from_pretrained(M,local_files_only=True)
 rows=[];noops=[]
 for n,p in enumerate(pairs[pairs.split=='T'].itertuples()):
  c=cl[p.chain_id];seq=c['sequence'];enc=tok(seq,return_tensors='pt',truncation=False).to(DEVICE);idx=np.array([p.positive_index0,p.negative_index0]);tt=torch.as_tensor(idx+1,device=DEVICE);ids=enc['input_ids'][0,1:-1]
  cap={};h=trained.esm.encoder.layer[1].intermediate.dense.register_forward_hook(lambda m,i,o:cap.update(z=o.detach().clone()))
  with torch.inference_mode():base=trained(**enc).logits[0,1:-1].double()
  h.remove();base_log=torch.log_softmax(base,dim=-1);base_p=base_log.exp();base_nll=-base_log[torch.arange(len(seq),device=DEVICE),ids];orig=cap['z']
  for arm,unit in [('fixed361',361),('native',lock['unit'])]:
   actual=orig[0,tt,unit].cpu().numpy();pp=pred.loc[[p.positive_site_id,p.negative_site_id]];assert np.max(np.abs(actual-pp[arm+'_signed_actual'].to_numpy()))<1e-4
   deltas={k:pp[arm+'_'+k].to_numpy()-actual for k in ('mean','seq','struct')};raw=actual[::-1]-actual;norm=float(np.linalg.norm(deltas['seq']));rnorm=float(np.linalg.norm(raw));deltas['swap_raw']=raw;deltas['swap_equal']=raw*norm/rnorm if rnorm>0 else np.zeros(2);deltas['struct_displacement_swap']=deltas['struct'][::-1].copy()
   assert rnorm==0 or abs(np.linalg.norm(deltas['swap_equal'])-norm)<1e-5
   runs=[('mean',1.),('seq',.5),('seq',1.),('struct',.5),('struct',1.),('swap_raw',1.),('swap_equal',1.),('struct_displacement_swap',1.)]
   if n==0:runs=[('noop',0.)]+runs
   for method,dose in runs:
    delta=deltas[method]*dose if method!='noop' else np.zeros(2);target=torch.as_tensor(actual+delta,dtype=orig.dtype,device=DEVICE)
    def replace(m,i,o):
     out=o.clone();out[0,tt,unit]=target;return out
    h=trained.esm.encoder.layer[1].intermediate.dense.register_forward_hook(replace)
    with torch.inference_mode():logits=trained(**enc).logits[0,1:-1].double()
    h.remove();logs=torch.log_softmax(logits,dim=-1);kl=(base_p*(base_log-logs)).sum(dim=-1);change=-logs[torch.arange(len(seq),device=DEVICE),ids]-base_nll
    if method=='noop':
     err=float((logits-base).abs().max().item());noops.append({'arm':arm,'chain_id':p.chain_id,'logit_max_abs_error':err});assert err<1e-5;continue
    rr={'chain_id':p.chain_id,'group':p.homology_group,'pair_id':p.pair_id,'readout':arm,'unit':unit,'method':method,'dose':dose,'target_count':2,'all_residues':len(seq),'target_nll_delta':float(change[idx].mean().item()),'target_KL':float(kl[idx].mean().item()),'all_nll_delta':float(change.mean().item()),'all_KL':float(kl.mean().item()),'displacement_site1':float(delta[0]),'displacement_site2':float(delta[1]),'displacement_l1':float(np.abs(delta).sum()),'displacement_l2':float(np.linalg.norm(delta)),'displacement_max':float(np.abs(delta).max()),'swap_degenerate':rnorm==0}
    rows.append(rr)
  if n%5==0:print('fidelity',n,flush=True)
 table('output_fidelity_per_chain.tsv',rows);dump('NOOP_VALIDATION.json',noops);output_report()
def output_report():
 df=pd.read_csv(O/'output_fidelity_per_chain.tsv',sep='\t');rep={};family={};swapfamily={};structswapfamily={};grows=[];outliers=[]
 for (arm,method,dose),f in df.groupby(['readout','method','dose']):
  key=f'{arm}:{method}:{dose}';rep[key]={'chains':len(f),'groups':f.group.nunique(),'targets':int(f.target_count.sum()),'sequence_residues':int(f.all_residues.sum())}
  for m in ('target_nll_delta','target_KL','all_nll_delta','all_KL','displacement_l2'):
   g=groupmean(f[m],f.group);rep[key][m]=ci(g);rep[key][m]['quantiles']=np.quantile(f[m],[0,.5,.9,.95,.99,1]).tolist()
   for gg,v in g.items():grows.append({'readout':arm,'method':method,'dose':dose,'endpoint':m,'group':gg,'value':v})
  largest=f.sort_values('target_KL',ascending=False).head(10)
  outliers.extend(largest.assign(rank=np.arange(1,len(largest)+1)).to_dict('records'))
  g=f.groupby('group').target_KL.mean().sort_values(ascending=False);rep[key]['target_KL_leave_top1_group']=float(g.iloc[1:].mean()) if len(g)>1 else None;rep[key]['target_KL_leave_top5_groups']=float(g.iloc[5:].mean()) if len(g)>5 else None
 for arm in ('fixed361','native'):
  for before,after in [('mean','seq'),('seq','struct'),('swap_equal','seq'),('struct_displacement_swap','struct')]:
   a=df[(df.readout==arm)&(df.method==before)&(df.dose==1)].set_index('chain_id');b=df[(df.readout==arm)&(df.method==after)&(df.dose==1)].set_index('chain_id');assert set(a.index)==set(b.index);b=b.loc[a.index]
   for m in ('target_nll_delta','target_KL'):
    target=swapfamily if before=='swap_equal' else structswapfamily if before=='struct_displacement_swap' else family
    target[f'{arm}:{before}_minus_{after}:{m}']=ci(groupmean(a[m]-b[m],a.group))
 table('output_fidelity_per_group.tsv',grows);table('output_outliers_top10.tsv',outliers);dump('OUTPUT_FIDELITY.json',{'arms':rep,'paired_family_holm':holm(family),'exploratory_swap_family_holm':holm(swapfamily),'exploratory_struct_placement_family_holm':holm(structswapfamily)})
def validate():
 chains,sites,pairs,obs,sf=load();cg={c['chain_id']:c for c in chains};group_splits={}
 for c in chains:
  assert hashlib.sha256(c['sequence'].encode()).hexdigest()==c['sequence_sha256'];group_splits.setdefault(c['homology_group'],set()).add(c['split'])
 assert all(len(x)==1 for x in group_splits.values())
 for p in pairs.itertuples():
  assert cg[p.chain_id]['sequence'][p.positive_index0]==cg[p.chain_id]['sequence'][p.negative_index0]=='C'
 df=pd.read_csv(O/'output_fidelity_per_chain.tsv',sep='\t');eq=df[(df.method=='swap_equal')].set_index(['chain_id','readout']);se=df[(df.method=='seq')&(df.dose==1)].set_index(['chain_id','readout']);err=float(np.max(np.abs(eq.displacement_l2-se.loc[eq.index].displacement_l2)))
 assert err<1e-5
 st=df[(df.method=='struct')&(df.dose==1)].set_index(['chain_id','readout']);sw=df[df.method=='struct_displacement_swap'].set_index(['chain_id','readout']).loc[st.index]
 serr=float(np.max(np.abs(st.displacement_l2-sw.displacement_l2)));assert serr<1e-10
 assert np.max(np.abs(st.displacement_site1-sw.displacement_site2))<1e-10 and np.max(np.abs(st.displacement_site2-sw.displacement_site1))<1e-10
 assert len(df)==int((pairs.split=='T').sum())*16
 for p in (O/'activations').glob('*.npz'):
  a=np.load(p);assert all(np.isfinite(a[k]).all() for k in a.files)
 dump('VALIDATION.json',{'status':'PASS','utc':utc(),'chains':len(chains),'homology_groups':len(group_splits),'test_pairs':int((pairs.split=='T').sum()),'output_rows':len(df),'group_split_disjoint':True,'pair_identity_and_endpoint_mapping':True,'natural_sequence_hashes':True,'all_finite':True,'equal_displacement_max_error':err,'structural_own_displacement_swap_l2_error':serr,'structural_own_displacements_exactly_swapped':True,'noop':read('NOOP_VALIDATION.json'),'known_limitations':['independent observed-structure assay, not original Swiss-Prot F1 replication','unmasked target NLL is output fidelity, not predictive masked NLL','homology groups small or incomplete affect inference','structural reconstruction can contain correlated endpoint proxies despite excluding direct bond features']})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['extract','analyze','fidelity','validate','output_report']);a=ap.parse_args();globals()[a.phase]()
