"""D-only sequence fits and group inference; no PLM imports or extraction."""
from __future__ import annotations
import os
for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import functools
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from threadpoolctl import threadpool_limits
AA='ACDEFGHIKLMNPQRSTVWY'
LINEAR_ALPHABET=AA+'X^'
PENALTIES=(.1,.01,.001,.0001)
STRONG_GRID=[dict(learning_rate=lr,max_leaf_nodes=leaf,l2_regularization=l2)for lr in(.05,.1)for leaf in(7,15)for l2 in(1.,10.)]
def encode_site(seq,pos0):
 onehot=np.zeros((31,21),dtype=np.float32)
 for j,off in enumerate(range(-15,16)):
  p=pos0+off;onehot[j,AA.index(seq[p])if 0<=p<len(seq)and seq[p]in AA else 20]=1
 window=seq[max(0,pos0-50):min(len(seq),pos0+51)]
 comp=np.array([window.count(a)/len(window)for a in AA],dtype=np.float32)
 return np.r_[onehot.ravel(),comp,pos0/max(1,len(seq)-1),np.log1p(len(seq))].astype(np.float32)
def linear_site(seq,pos0):
 # Exact original model_inputs.design_matrix vocabulary and covariates.
 v=np.zeros(31*22+4,dtype=np.float64)
 for k,off in enumerate(range(-15,16)):
  p=pos0+off;s=seq[p]if 0<=p<len(seq)else'^';j=LINEAR_ALPHABET.index(s)if s in LINEAR_ALPHABET else 20;v[k*22+j]=1
 rel=pos0/max(1,len(seq)-1);v[-4:]=[rel,rel*rel,len(seq)/256,1.]
 return v
def encode_pair(seq,early0,late0):return linear_site(seq,early0)-linear_site(seq,late0),np.r_[encode_site(seq,early0),encode_site(seq,late0)]
def group_weights(groups):
 _,inv,cnt=np.unique(groups,return_inverse=True,return_counts=True);w=1/cnt[inv];return w/w.mean()
def folds(groups):
 n=len(np.unique(groups))
 if n<2:raise ValueError('fewer_than_two_D_groups')
 splitter=GroupKFold(n_splits=5)if n>=5 else LeaveOneGroupOut()
 return list(splitter.split(np.empty((len(groups),1)),groups=groups))
def kernel_fit(k,eval_k,train_ix,test_ix,z,eval_z,labs,penalty,include_axis):
 # Copied exactly from archived paired_rsa.py (fold count belongs to caller).
 a=k[np.ix_(train_ix,train_ix)];b=eval_k[np.ix_(test_ix,train_ix)]
 local_trace=np.mean(np.diag(a))
 if local_trace<=0:raise ValueError('Degenerate sequence kernel')
 a=a/local_trace;b=b/local_trace
 if include_axis:
  zz=z[train_ix];zz_scale=np.mean(zz**2)
  if zz_scale>1e-12:a=(a+np.outer(zz,zz)/zz_scale)*.5;b=(b+np.outer(eval_z[test_ix],zz)/zz_scale)*.5
 a.flat[::a.shape[0]+1]+=len(train_ix)*penalty
 coeff=cho_solve(cho_factor(a,lower=True,check_finite=False),labs[train_ix],check_finite=False)
 return np.clip((1+b@coeff)/2,0,1)
def select_linear(x,y,g,z=None):
 k=x@x.T;z=np.zeros(len(y))if z is None else np.asarray(z,dtype=float);include=bool(np.mean(z*z)>1e-12);scores=[]
 for penalty in PENALTIES:
  p=np.empty(len(y))
  for tr,va in folds(g):p[va]=kernel_fit(k,k,tr,va,z,z,2*y-1,penalty,include)
  scores.append(float(np.average((p-y)**2,weights=group_weights(g))))
 return PENALTIES[int(np.argmin(scores))],scores
def linear_predict(x,xt,y,penalty,z=None,zt=None):
 z=np.zeros(len(y))if z is None else np.asarray(z,dtype=float);zt=np.zeros(len(xt))if zt is None else np.asarray(zt,dtype=float)
 return kernel_fit(x@x.T,xt@x.T,np.arange(len(x)),np.arange(len(xt)),z,zt,2*y-1,penalty,bool(np.mean(z*z)>1e-12))
def symmetry_data(x,y,g):return np.r_[x,-x],np.r_[y,1-y],np.r_[g,g]
def swap_strong(x):
 f=x.shape[1]//2;return np.c_[x[:,f:],x[:,:f]]
def strong_augment(x,y,g,z=None):
 reverse=swap_strong(x)
 if z is not None:x=np.c_[x,z];reverse=np.c_[reverse,-z]
 return np.r_[x,reverse],np.r_[y,1-y],np.r_[g,g]
def strong_fit(x,y,g,params,z=None):
 xx,yy,gg=strong_augment(x,y,g,z)
 model=HistGradientBoostingClassifier(max_iter=200,min_samples_leaf=10,random_state=2026112303,early_stopping=False,**params)
 with threadpool_limits(limits=1):model.fit(xx,yy,sample_weight=group_weights(gg))
 return model
def strong_predict(model,x,z=None):
 reverse=swap_strong(x)
 if z is not None:x=np.c_[x,z];reverse=np.c_[reverse,-z]
 with threadpool_limits(limits=1):return .5*(model.predict_proba(x)[:,1]+1-model.predict_proba(reverse)[:,1])
def select_strong(x,y,g,progress=None):
 scores=[];ff=folds(g)
 for i,params in enumerate(STRONG_GRID):
  pp=np.empty(len(y))
  for j,(tr,va)in enumerate(ff):
   model=strong_fit(x[tr],y[tr],g[tr],params);pp[va]=strong_predict(model,x[va])
   if progress is not None:progress(i,j)
  scores.append(float(np.average((pp-y)**2,weights=group_weights(g))))
 return STRONG_GRID[int(np.argmin(scores))],scores
def scalar_scale(z):
 # Scale is fitted on augmented D: mean(z,-z)=0, variance=mean(z**2).
 z=np.asarray(z,dtype=float)
 if not np.isfinite(z).all():raise ValueError('nonfinite_D_scalar')
 return float(np.sqrt(np.mean(z*z)))
def auc(y,z):
 y=np.asarray(y,bool);z=np.asarray(z,float);pos=y.sum();neg=(~y).sum()
 if not pos or not neg:return np.nan
 ranks=rankdata(z,method='average');return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))
def paired_credit(oriented_late_minus_early,target=None):
 d=np.asarray(oriented_late_minus_early,float)
 if target is not None:d=d*(2*np.asarray(target,float)-1)
 return(np.sign(d)+1)/2
def bh(pvalues):
 p=np.asarray(pvalues,float);out=np.full(p.shape,np.nan);keep=np.flatnonzero(np.isfinite(p));order=keep[np.argsort(p[keep],kind='stable')]
 if len(order):out[order]=np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1].clip(0,1)
 return out
@functools.lru_cache(maxsize=4)
def group_bootstrap_counts(universe,seed=2026112304):
 n=len(universe);rng=np.random.default_rng(seed);draw=rng.integers(n,size=(2000,n));counts=np.empty((2000,n),dtype=np.int16)
 for i,d in enumerate(draw):counts[i]=np.bincount(d,minlength=n)
 return counts
def signflip_p(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)];nonzero=v[v!=0];m=len(nonzero);observed=float(v.sum());hits=0
 if m<=18:
  total=2**m
  for start in range(0,total,2048):
   signs=2*((np.arange(start,min(start+2048,total))[:,None]>>np.arange(m))&1)-1;hits+=int((signs@nonzero>=observed-1e-15).sum())
  p=hits/total
 else:
  total=99999;rr=np.random.default_rng(2026112305)
  for start in range(0,total,1024):
   signs=rr.integers(0,2,size=(min(1024,total-start),m),dtype=np.int8)*2-1;hits+=int((signs@nonzero>=observed-1e-15).sum())
  p=(hits+1)/(total+1)
 return float(p)
def inference(values,groups=None,universe=None,seed=2026112304,compute_p=True):
 values=np.asarray(values,float);keep=np.isfinite(values);groups=np.arange(len(values)).astype(str)if groups is None else np.asarray(groups,dtype=str)
 assert values.shape==groups.shape and len(set(groups))==len(groups),'one_group_one_effect_required'
 values=values[keep];groups=groups[keep];n=len(values)
 if not n:return dict(n=0,mean=None,ci95=None,p_one_sided=None,undefined_bootstrap=2000)
 universe=tuple(sorted(groups.tolist()))if universe is None else tuple(sorted(universe));lookup={g:i for i,g in enumerate(universe)}
 if any(g not in lookup for g in groups):raise ValueError('group_missing_from_shared_bootstrap_universe')
 counts=group_bootstrap_counts(universe,seed)[:,[lookup[g]for g in groups]].astype(float);den=counts.sum(axis=1);boot=(counts@values)/np.where(den>0,den,np.nan)
 return dict(n=n,mean=float(values.mean()),ci95=np.nanquantile(boot,[.025,.975]).tolist(),p_one_sided=signflip_p(values)if compute_p else None,undefined_bootstrap=int(np.sum(den==0)),null='sign-exchangeable group-centered effects',bootstrap_universe_groups=len(universe))
def classify(auc_all,pair_accuracy,pair_q,strong_gain,strong_q,n_groups):
 if n_groups<20:return'INSUFFICIENT_GROUPS'
 if strong_q is not None and np.isfinite(strong_q)and strong_q<.05 and strong_gain>0:return'INCREMENT_OVER_SPECIFIED_SEQUENCE_BASELINE'
 if pair_q is not None and np.isfinite(pair_q)and pair_q<.05 and pair_accuracy>.5:return'SAME_AA_ANNOTATION_DISCRIMINATION'
 if auc_all is not None and np.isfinite(auc_all)and auc_all>=.70:return'AA_LEVEL_COMPATIBLE_NOT_PROVEN_EXCLUSIVE'
 return'UNRESOLVED'
