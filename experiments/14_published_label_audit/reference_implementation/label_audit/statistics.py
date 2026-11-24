"""Frozen numerical core; callers supply group-level annotations and readouts.
No model extraction occurs here. Confirmation scoring must be called only after
the controller's common gate and final cohort/input freeze have passed.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
AA='ACDEFGHIKLMNPQRSTVWY'
STRONG_GRID=[dict(learning_rate=lr,max_leaf_nodes=leaf,l2_regularization=l2) for lr in [.05,.1] for leaf in [7,15] for l2 in [1.,10.]]
def encode_site(seq,pos0):
    onehot=np.zeros((31,21),dtype=np.float32)
    for j,offset in enumerate(range(-15,16)):
        p=pos0+offset;onehot[j,AA.index(seq[p]) if 0<=p<len(seq) else 20]=1
    window=seq[max(0,pos0-50):min(len(seq),pos0+51)]
    comp=np.array([window.count(a)/len(window) for a in AA],dtype=np.float32)
    return np.r_[onehot.ravel(),comp,pos0/max(1,len(seq)-1),np.log1p(len(seq))].astype(np.float32)
def group_weights(groups):
    _,inv,cnt=np.unique(groups,return_inverse=True,return_counts=True);w=1/cnt[inv];return w/w.mean()
def symmetry_data(x,y,g):return np.concatenate([x,-x]),np.r_[y,1-y],np.r_[g,g]
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
def select_strong(x,y,g):
    unique=len(np.unique(g));assert unique>=5
    folds=list(GroupKFold(n_splits=5).split(x,y,g));scores=[]
    for params in STRONG_GRID:
        loss=[]
        for tr,va in folds:
            model=strong_fit(x[tr],y[tr],g[tr],params);p=strong_predict(model,x[va]);loss.extend(((p-y[va])**2).tolist())
        scores.append(np.mean(loss))
    best=int(np.argmin(scores));return STRONG_GRID[best],scores
def auc(y,z):
    y=np.asarray(y,bool);z=np.asarray(z,float);pos=y.sum();neg=(~y).sum()
    if not pos or not neg:return np.nan
    ranks=rankdata(z,method='average');return (ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)
def paired_credit(difference):return (np.sign(difference)+1)/2
def bh(pvalues):
    p=np.asarray(pvalues,float);out=np.full(p.shape,np.nan);keep=np.flatnonzero(np.isfinite(p));order=keep[np.argsort(p[keep],kind='stable')]
    if len(order):out[order]=np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1].clip(0,1)
    return out
def inference(values,seed=2026112304):
    values=np.asarray(values,float);values=values[np.isfinite(values)];n=len(values)
    if not n:return dict(n=0,mean=None,ci95=None,p_one_sided=None)
    rng=np.random.default_rng(seed);boots=[]
    for _ in range(20):boots.extend(values[rng.integers(n,size=(100,n))].mean(1))
    nonzero=values[values!=0];m=len(nonzero);observed=values.sum();hits=0
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
    return dict(n=n,mean=float(values.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),p_one_sided=float(p),null='sign-exchangeable group-centered effects')
def classify(auc_all,pair_accuracy,pair_q,strong_gain,strong_q,n_groups):
    if n_groups<20:return 'INSUFFICIENT_GROUPS'
    if strong_q<.05 and strong_gain>0:return 'INCREMENT_OVER_SPECIFIED_SEQUENCE_BASELINE'
    if pair_q<.05 and pair_accuracy>.5:return 'SAME_AA_ANNOTATION_DISCRIMINATION'
    if auc_all>=.70:return 'AA_LEVEL_COMPATIBLE_NOT_PROVEN_EXCLUSIVE'
    return 'UNRESOLVED'

# Formal implementation supersedes the initial timing draft above.  No caller
# should substitute the draft LogisticRegression for the archived linear kernel.
from cpu_statistics_core import *
