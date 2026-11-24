"""Report every eligible group's prediction, reference ranks and paired uncertainty."""
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
from cohorts import ROOT,read_tsv,write_tsv
from experiment import PRED,FITS,saved

def grouped(path,task,eligible=None):
    rows=[r for r in read_tsv(path) if r["task"]==task and (eligible is None or r["group"] in eligible)]
    if len(rows)!=len({r["group"] for r in rows}):
        raise ValueError(f"Repeated group in {path.name}")
    if not rows: raise ValueError(f"No external groups for {task}")
    y,b,p=(np.array([float(r[k]) for r in rows]) for k in ("target","local_probability","augmented_probability"))
    delta=(y-b)**2-(y-p)**2
    if not np.allclose(delta,[float(r["gain"]) for r in rows],rtol=0,atol=1e-14):
        raise ValueError("Stored paired differences do not reconstruct")
    return rows,y,b,p,delta

def bootstrap(d):
    if len(d)<2:return [[None,None],[None,None]]
    rng=np.random.default_rng(20261122)
    means=np.concatenate([d[rng.integers(0,len(d),(100,len(d)))].mean(axis=1) for _ in range(100)])
    return [np.quantile(means,[.025,.975]).tolist(),np.quantile(means,[.025/3,1-.025/3]).tolist()]

def correct(prob,y):
    return int(np.sum((prob>.5)&(y==1))+np.sum((prob<.5)&(y==0)))

def evaluate(eligible=None,subdirectory="results"):
    summary={};refs=[];selected=[];out=ROOT/subdirectory;out.mkdir(exist_ok=True)
    for arch,t in (("esm2_35m","contact"),("carp_38m","contact"),("carp_38m","exposure"),("source_814","exposure")):
        base=arch if arch=="source_814" else arch+"_trained"
        rs,y,b,p,delta=grouped(PRED/(base+".tsv"),t,eligible)
        selected.extend(dict(architecture=arch,**r) for r in rs)
        group_ids=[r["group"] for r in rs]
        low,sim=bootstrap(delta)
        item=dict(n=len(rs),gain=float(delta.mean()),ci95=low,simultaneous_interval=sim,local_brier=float(np.mean((y-b)**2)),
                  augmented_brier=float(np.mean((y-p)**2)),local_correct=correct(b,y),augmented_correct=correct(p,y),
                  local_ties=int(np.sum(b==.5)),augmented_ties=int(np.sum(p==.5)),
                  lower_loss_groups=int(np.sum(delta>0)),higher_loss_groups=int(np.sum(delta<0)),median_gain=float(np.median(delta)))
        if arch!="source_814":
            fit=saved(FITS/(base+".npz"))
            item["axis_zero_based"]=int(fit[t+"_axis"])
            if t=="exposure": item["dev_oof_teacher_corr"]=float(fit["teacher_corr"]);item["readout_scale"]=float(fit["scale"])
            for stage,n in (("random",199),("rotation",9)):
                gains=[]
                for j in range(n):
                    name=f"{arch}_{stage}_{j:03d}"
                    r,yr,br,pr,ds=grouped(PRED/(name+".tsv"),t,eligible)
                    if [x["group"] for x in r]!=group_ids or not np.array_equal(yr,y) or not np.array_equal(br,b):
                        raise ValueError("Incomparable reference cohort or local baseline")
                    v=float(ds.mean());gains.append(v)
                    f=saved(FITS/(name+".npz"))
                    refs.append(dict(architecture=arch,endpoint=t,reference=stage,index=j,
                                     selected_axis=int(f[t+"_axis"]),brier_gain=v,correct=correct(pr,yr)))
                arr=np.array(gains)
                ge=int(np.sum(arr>=item["gain"]))
                item[stage]=dict(n=n,ge_native=ge,rank=ge+1,upper_tail_fraction=(ge+1)/(n+1),
                                 min=float(arr.min()),median=float(np.median(arr)),max=float(arr.max()))
            item["strong_frozen_criteria"]=bool(sim[0] is not None and sim[0]>0
                    and item["rotation"]["ge_native"]==0 and (199-item["random"]["ge_native"])/199>=.95)
        summary[arch+"_"+t]=item
    write_tsv(out/"reference_scores.tsv",tuple(refs[0].keys()),refs)
    write_tsv(out/"native_pair_predictions.tsv",tuple(selected[0].keys()),selected)
    (out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    evaluate()
    homology=ROOT/"validation"/"HOMOLOGY.json"
    if homology.exists():
        report=json.loads(homology.read_text())
        if report["excluded_groups"]:
            evaluate(set(report["retained_groups"]),"results_sequence_screened")
    coverage=ROOT/"validation"/"COVERAGE.json"
    if coverage.exists():
        high=set(json.loads(coverage.read_text())["high_groups"])
        summary=[]
        for arch in ("esm2_35m","carp_38m"):
            rows,y,b,p,d=grouped(PRED/(arch+"_trained.tsv"),"contact",high)
            ci,_=bootstrap(d)
            summary.append(dict(architecture=arch,n=len(rows),local_brier=float(np.mean((y-b)**2)),
                                augmented_brier=float(np.mean((y-p)**2)),gain=float(d.mean()),
                                ci95_low=ci[0],ci95_high=ci[1]))
        write_tsv(ROOT/"results"/"high_structure_coverage_contacts.tsv",tuple(summary[0].keys()),summary)
