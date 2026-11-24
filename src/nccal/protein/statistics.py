"""Group-balanced statistics used by the protein analyses."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from functools import lru_cache

def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if Path(path).suffix==".tsv" else ","))

def bh(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p,kind="stable")
    adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p));result[order]=np.minimum(adjusted,1);return result

def group_equal_r2(group_rows, mse_column):
    grand=np.mean([float(r["mean_z"]) for r in group_rows])
    variance=np.mean([float(r["mean_z2"]) for r in group_rows])-grand*grand
    return 1-np.mean([float(r[mse_column]) for r in group_rows])/variance

def group_means(records, value):
    groups={}
    for record in records:groups.setdefault(record["group"],[]).append(float(record[value]))
    return {key:float(np.mean(values)) for key,values in groups.items()}

def paired_bootstrap(values, seed, draws=2000):
    a=np.asarray(values,dtype=float);index=np.random.default_rng(seed).integers(0,len(a),(draws,len(a)),dtype=np.int32)
    return np.mean(a[index],axis=1)

@lru_cache(maxsize=4)
def shared_bootstrap_counts(universe):
    n=len(universe);draw=np.random.default_rng(2026112304).integers(n,size=(2000,n));counts=np.empty((2000,n),dtype=np.int16)
    for i,indices in enumerate(draw):counts[i]=np.bincount(indices,minlength=n)
    return counts

def shared_intervals(values,groups,universe):
    universe=tuple(sorted(universe));lookup={g:i for i,g in enumerate(universe)}
    counts=shared_bootstrap_counts(universe)[:,[lookup[g]for g in groups]].astype(float);den=counts.sum(axis=1)
    boot=(counts@np.asarray(values,float))/np.where(den>0,den,np.nan)[:,None]
    return np.nanquantile(boot,[.025,.975],axis=0)
