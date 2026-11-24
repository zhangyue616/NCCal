"""Select one RSA axis in development, then estimate held-out absolute errors."""
import argparse
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from model_inputs import HERE, STRUCTURE, activations, design_matrix, grouped_folds, group_means, interval, read_rows, weights

PENALTIES = (0.1, 0.01, 0.001, 0.0001)

def development_signature(rows):
    digest=hashlib.sha256()
    for row in rows:
        digest.update((repr(row)+"\n").encode("utf8"))
    return digest.hexdigest()

def sequence_fit(x, y, w, penalty):
    normal=(x.T @ x.multiply(w[:,None])).toarray()
    normal.flat[::len(normal)+1] += penalty
    normal[-1,-1] -= penalty
    rhs = x.T @ (w*y)
    return cho_solve(cho_factor(normal, lower=True,check_finite=False),rhs,check_finite=False)

def residual_slopes(z, residual, w):
    """Weighted moments in float64 without an array-wide float64 conversion."""
    sums=np.zeros(z.shape[1]); squares=np.zeros(z.shape[1]); cross=np.zeros(z.shape[1])
    for start in range(0,len(w),512):
        block=z[start:start+512].astype(np.float64)
        wt=w[start:start+512]
        sums += wt@block
        squares += wt@(block*block)
        cross += (wt*residual[start:start+512])@block
    var=np.maximum(0, squares-sums*sums)
    cov=cross-sums*np.dot(w,residual)
    beta=np.divide(cov,var,out=np.zeros_like(var),where=var>1.e-12)
    return sums,beta

def dev_fit(model, rows, cohort, out):
    x=design_matrix(rows,cohort)
    y=np.array([r[5] for r in rows]); groups=np.array([r[2] for r in rows])
    folds=grouped_folds(groups)
    z=activations(rows,model,"D",all_axes=True)
    by_fold=[(folds!=k,folds==k) for k in range(4)]
    losses={}
    for penalty in PENALTIES:
        pred=np.empty_like(y)
        for train,test in by_fold:
            coef=sequence_fit(x[train],y[train],weights(groups[train]),penalty)
            pred[test]=x[test]@coef
        losses[penalty]=float(np.mean(group_means(np.abs(y-pred),groups)))
    penalty=min(PENALTIES,key=losses.get)
    axis_loss=np.zeros(z.shape[1])
    all_w=weights(groups)
    for train,test in by_fold:
        w=weights(groups[train]); coef=sequence_fit(x[train],y[train],w,penalty)
        mu,beta=residual_slopes(z[train],y[train]-x[train]@coef,w)
        error=y[test]-x[test]@coef
        test_ids=np.flatnonzero(test)
        for start in range(0,len(error),512):
            idx=test_ids[start:start+512]
            e=error[start:start+512,None]-(z[idx]-mu)*beta
            axis_loss += all_w[idx]@np.abs(e)
    axis=int(np.argmin(axis_loss))
    w=weights(groups); coef=sequence_fit(x,y,w,penalty)
    mu,beta=residual_slopes(z[:,[axis]],y-x@coef,w)
    np.savez(out/"sequence_model.npz",coef=coef,mu=mu,beta=beta)
    obj={"model":model,"axis_zero_based":axis,"axis_candidates":z.shape[1],"rows":len(y),
         "groups":int(len(np.unique(groups))),"development_data_signature":development_signature(rows),
         "penalty":penalty,"cv_baseline_mae":losses[penalty],
         "cv_augmented_mae":float(axis_loss[axis]),"dev_median_remote_contacts":float(np.median([r[6] for r in rows])),
         "selection":"only D structural labels, group-held-out folds"}
    (out/"selection.json").write_text(json.dumps(obj,indent=2)+"\n",encoding="utf8")
    return obj

def contact_result(rows, error_gain, cutoff):
    strata=defaultdict(lambda:[[],[]])
    for r,gain in zip(rows,error_gain):
        which=int(r[6]>cutoff)
        strata[r[2],r[3],r[4],min(10,int(np.floor(10*r[5])))][which].append(float(gain))
    group_data=defaultdict(list)
    for (group,aa,ss,rsa_bin),(low,high) in strata.items():
        if low and high: group_data[group].append((min(len(low),len(high)),np.mean(low),np.mean(high)))
    estimates=[]
    for group,vals in sorted(group_data.items()):
        n,lo,hi=np.asarray(vals).T
        estimates.append((group,np.average(lo,weights=n),np.average(hi,weights=n)))
    if not estimates: return {"support_groups":0},[]
    _,lo,hi=zip(*estimates); lo=np.asarray(lo);hi=np.asarray(hi)
    return {"support_groups":len(lo),"common_aa_secondary_rsa_strata":sum(map(len,group_data.values())),
            "cutoff_from_D":cutoff,"low_gain_mae":float(lo.mean()),"high_gain_mae":float(hi.mean()),
            "high_minus_low":float(np.mean(hi-lo)),"paired_95ci":interval(hi-lo)},estimates

def holdout(model,drows,rows,cohort,out,structures):
    s=json.loads((out/"selection.json").read_text(encoding="utf8"))
    if s["development_data_signature"]!=development_signature(drows):
        raise ValueError("Development input differs from locked fit")
    axis=s["axis_zero_based"]
    z=activations(rows,model,"T",axis=axis)
    fit=np.load(out/"sequence_model.npz",allow_pickle=False)
    x=design_matrix(rows,cohort);y=np.array([r[5] for r in rows]); groups=np.array([r[2] for r in rows])
    base=x@fit["coef"]
    aug=base+(z.astype(float)-fit["mu"][0])*fit["beta"][0]
    b=np.abs(y-base); a=np.abs(y-aug)
    bgroup=group_means(b,groups); agroup=group_means(a,groups)
    contact,gstats=contact_result(rows,b-a,s["dev_median_remote_contacts"])
    with (out/"group_errors.tsv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["group","sequence_mae","augmented_mae","sequence_minus_augmented"])
        w.writerows(zip(np.unique(groups),bgroup,agroup,bgroup-agroup))
    with (out/"contact_groups.tsv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["group","low_contacts_gain","high_contacts_gain"]);w.writerows(gstats)
    from paired_rsa import paired_analysis
    paired=paired_analysis(drows,rows,cohort,model,axis,structures,out)
    obj={"selection":s,"T_groups":len(bgroup),"T_residues":len(y),
         "sequence_mae":float(bgroup.mean()),"augmented_mae":float(agroup.mean()),
         "group_mae_difference":float(np.mean(bgroup-agroup)),"paired_95ci":interval(bgroup-agroup),
         "long_range_sensitivity":contact,"buried_exposed_pairs":paired}
    (out/"results.json").write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return obj

def main():
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument("command",choices=("fit","score"))
    cli.add_argument("model",choices=("8m","35m"))
    cli.add_argument("--structural-dir",type=Path,default=STRUCTURE)
    cli.add_argument("--out",type=Path)
    opt=cli.parse_args()
    out=opt.out or (HERE/"results"/opt.model);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        d,cohort=read_rows("D",opt.structural_dir)
        if opt.command=="fit": obj=dev_fit(opt.model,d,cohort,out)
        else:
            t,_=read_rows("T",opt.structural_dir)
            obj=holdout(opt.model,d,t,cohort,out,opt.structural_dir)
    print(json.dumps(obj,indent=2))
if __name__=="__main__":
    main()
