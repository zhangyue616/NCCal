"""Statistical definitions shared by the molecular experiments."""
import numpy as np

def rank_test(values,two_sided=False):
    values=np.asarray(values,dtype=float)
    n=len(values)
    ranks=np.stack([((values<x).sum(0)+.5*((values==x).sum(0)-1))/(n-1) for x in values]).mean(1)
    upper=np.mean(ranks>=ranks[0]);lower=np.mean(ranks<=ranks[0])
    return dict(p=float(min(1,2*min(upper,lower)) if two_sided else upper),fold=float(np.exp(np.median(values[0]-np.median(values[1:],axis=0)))))

def holm(pvalues):
    pvalues=np.asarray(pvalues,float);order=np.argsort(pvalues);result=np.empty_like(pvalues)
    result[order]=np.minimum(1,np.maximum.accumulate(pvalues[order]*(len(pvalues)-np.arange(len(pvalues)))))
    return result

def pooled_r2(actual,predicted,weights=None):
    actual=np.asarray(actual,float);predicted=np.asarray(predicted,float)
    if weights is None:weights=np.ones(len(actual))/len(actual)
    mean=np.sum(weights[:,None]*actual,axis=0)
    denominator=np.sum(weights[:,None]*(actual-mean)**2,axis=0)
    error=np.sum(weights[:,None]*(actual-predicted)**2,axis=0)
    return 1-np.divide(error,denominator,out=np.full_like(error,np.nan),where=denominator>0)

def transfer(ell):
    ell=np.asarray(ell)
    preference=ell[...,0,:]-np.median(ell[...,1:,:],axis=-2)
    theta=preference[...,0,:]-np.median(preference[...,1:,:],axis=-2)
    gain=ell[...,0,0,:]-np.median(ell[...,1:,0,:],axis=-2)
    return {'Theta':theta.mean(-1),'G_init':gain.mean(-1)}

def bootstrap_counts(n,replicates=2000,seed=20261123):
    rng=np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(replicates)]).astype(float)/n

def census_from_molecules(advantages,canonical_r2):
    """Input shape: molecule, representation group, channel, contrast."""
    advantages=np.asarray(advantages,float);n,_,units,_=advantages.shape
    draws=bootstrap_counts(n);mean=advantages.mean(0)
    ci=np.quantile(draws@advantages.reshape(n,-1),[.025,.975],axis=0).reshape(2,2,units,2)
    usable=(canonical_r2>=.5)&(ci[0,0]>0).all(-1)
    retained=(ci[0,1]>0).all(-1);failed=ci[1,1,:,0]<0
    status=np.where(retained,'retained',np.where(failed,'failed','undetermined'))
    return mean,ci,usable,status
