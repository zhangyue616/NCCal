"""Secondary matched comparison, with both sites' order determined by sequence position."""
import csv
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from model_inputs import activations,design_matrix,grouped_folds,interval

def get_pairs(rows,cohort,model,axis,split,structures):
    by_pos={(r[0],r[1]):i for i,r in enumerate(rows)}
    pairs=[]; labels=[]; groups=[]
    with (structures/"matched_pairs.tsv").open(encoding="utf-8-sig",newline="") as source:
        for r in csv.DictReader(source,delimiter="\t"):
            if r["split"]!=split: continue
            i=by_pos[r["accession"],int(r["pos1"])];j=by_pos[r["accession"],int(r["pos2"])]
            buried=rows[i];exposed=rows[j]
            if (buried[2]!=r["group"] or buried[3:5]!=exposed[3:5] or not (buried[5]<=0.10 and exposed[5]>=0.40)):
                raise ValueError("Pair violates its structural definition")
            # The ordered target is positive when the later residue is exposed.
            if buried[1]<exposed[1]:
                pairs.append((i,j)); labels.append(1.)
            else:
                pairs.append((j,i)); labels.append(-1.)
            groups.append(r["group"])
    if len(groups)!=len(set(groups)): raise ValueError("Repeated group in pair analysis")
    if not pairs: return None
    early,late=np.asarray(pairs,dtype=int).T
    x=design_matrix(rows,cohort)
    z=activations(rows,model,split,axis=axis)
    return x[early]-x[late],z[early].astype(float)-z[late],np.array(labels),np.array(groups)

def kernel_fit(k,eval_k,train_ix,test_ix,z,eval_z,labs,penalty,include_axis):
    # k is the descriptor-difference Gram matrix. eval_k has test rows and train columns.
    a=k[np.ix_(train_ix,train_ix)]
    b=eval_k[np.ix_(test_ix,train_ix)]
    local_trace=np.mean(np.diag(a))
    if local_trace<=0: raise ValueError("Degenerate sequence kernel")
    a=a/local_trace; b=b/local_trace
    if include_axis:
        zz=z[train_ix]; zz_scale=np.mean(zz**2)
        if zz_scale>1e-12:
            a=(a+np.outer(zz,zz)/zz_scale)*.5
            b=(b+np.outer(eval_z[test_ix],zz)/zz_scale)*.5
    a.flat[::a.shape[0]+1] += len(train_ix)*penalty
    coeff=cho_solve(cho_factor(a,lower=True,check_finite=False),labs[train_ix],check_finite=False)
    return np.clip((1+b@coeff)/2,0,1)

def paired_analysis(dev,test,cohort,model,axis,structures,out):
    d=get_pairs(dev,cohort,model,axis,"D",structures); t=get_pairs(test,cohort,model,axis,"T",structures)
    if d is None or t is None or len(d[2])<4:
        return {"D_groups":0 if d is None else len(d[2]),"T_groups":0 if t is None else len(t[2]),"status":"insufficient matched groups"}
    x,z,yt,groups=d;xt,zt,ys,tgroups=t
    f=grouped_folds(groups); train_kernel=(x@x.T).toarray(); test_kernel=(xt@x.T).toarray()
    target_d=(yt+1)/2; target_t=(ys+1)/2
    errors=[]; penalties=[]; predictions=[]
    for augment in (False,True):
        candidates=(.1,.01,.001,.0001)
        scores=[]
        for strength in candidates:
            pred=np.empty(len(yt))
            for i in range(4):
                ixs=np.flatnonzero(f!=i); ids=np.flatnonzero(f==i)
                pred[ids]=kernel_fit(train_kernel,train_kernel,ixs,ids,z,z,yt,strength,augment)
            scores.append(np.mean((target_d-pred)**2))
        best=candidates[int(np.argmin(scores))]
        p=kernel_fit(train_kernel,test_kernel,np.arange(len(yt)),np.arange(len(ys)),z,zt,yt,best,augment)
        predictions.append(p)
        errors.append((target_t-p)**2);penalties.append(best)
    diff=errors[0]-errors[1]
    with (out/"buried_exposed_group_errors.tsv").open("w",encoding="utf8",newline="") as file:
        writer=csv.writer(file,delimiter="\t")
        writer.writerow(["group","later_residue_exposed","sequence_probability",
                         "augmented_probability","sequence_brier","augmented_brier","improvement"])
        writer.writerows(zip(tgroups,target_t.astype(int),*predictions,*errors,diff))
    return {"D_groups":len(yt),"T_groups":len(ys),"penalties":penalties,"sequence_brier":float(errors[0].mean()),
            "augmented_brier":float(errors[1].mean()),"improvement":float(diff.mean()),"paired_95ci":interval(diff)}
