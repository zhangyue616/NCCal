"""Measure what the published channel adds to existing sequence references on the sheet task."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/"strand_reference"))
sys.path.insert(0,str(ROOT/"scan"))
from run_reference import describe, fit_model, probability
from analyse import window_features


def table(path):
    return pd.read_csv(path,sep="\t",dtype={"pair_id":str,"record_id":str,"group":str})


def whole_strands(rows,sequences):
    vectors=[]
    for r in rows.to_dict("records"):
        seq=sequences[r["record_id"]]
        ends=[]
        for tag in ("positive","negative"):
            i,j=int(r[tag+"_strand_start_index0"]),int(r[tag+"_strand_end_index0"])
            assert 0<=i<j<len(seq) and j-i+1>=3
            ends.append(describe(seq[i:j+1]))
        vectors.append(ends[0]-ends[1])
    return np.array(vectors)


def main():
    old=ROOT/"scan/merged_cohorts"
    fresh=ROOT/"sheet"
    inputs=pd.concat([table(old/"pairs.tsv"),table(fresh/"pairs_F.tsv")],ignore_index=True)
    signed=table(HERE/"ESMC634_topology_pairs.tsv")
    keys=["split","group","record_id","pair_id","aa"]
    rows=inputs.merge(signed[keys+["delta"]],on=keys,validate="one_to_one")
    assert len(rows)==len(inputs)==len(signed)
    sequences={}
    for path in [old/"chains.jsonl",fresh/"chains_F.jsonl"]:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            c=json.loads(line); sequences[c["record_id"]]=c["sequence"]
    dev=rows.split.eq("D").to_numpy()
    grp=rows.loc[dev,"group"]
    weights=1/grp.map(grp.value_counts()).to_numpy();weights/=weights.mean()
    delta=rows.delta.to_numpy()
    fit={"note":"Both references fitted on the original D groups only. Published pole and orientation fixed in the earlier test."}
    records=[]
    for name,x in [("local_window",window_features(rows,sequences)),("whole_strand",whole_strands(rows,sequences))]:
        base=x
        scale=float(np.sqrt(np.average(delta[dev]**2,weights=weights))) if name=="local_window" else 1.
        assert scale>0 and np.isfinite(x).all()
        design=np.column_stack([base,delta/scale])
        probs={}
        for suffix,X in [("base",base),("plus",design)]:
            if name=="whole_strand":
                coef=fit_model(X[dev],grp.to_numpy());pred=probability(coef,X)
            else:
                Xd=X[dev]
                model=LogisticRegression(C=1,fit_intercept=False,solver="lbfgs",max_iter=1000)
                model.fit(np.vstack([Xd,-Xd]),np.r_[np.ones(len(Xd)),np.zeros(len(Xd))],sample_weight=np.tile(weights,2))
                pred=model.predict_proba(X)[:,1]
                coef=dict(beta=model.coef_[0].tolist(),published_rms=scale)
            probs[suffix]=pred
            fit[name+"_"+suffix]=coef
        scores=rows[keys].copy()
        scores["reference"]=name
        scores["base_brier"]=(1-probs["base"])**2
        scores["plus_brier"]=(1-probs["plus"])**2
        scores["brier_gain"]=scores.base_brier-scores.plus_brier
        records.append(scores)
    all_scores=pd.concat(records,ignore_index=True)
    all_scores.to_csv(HERE/"ESMC634_increment_pairs.tsv",sep="\t",index=False)
    summaries=[]
    for (ref,split),part in all_scores.groupby(["reference","split"]):
        groups=part.groupby("group")[["base_brier","plus_brier","brier_gain"]].mean()
        rng=np.random.default_rng(77179+sum(map(ord,split)))
        ix=rng.integers(len(groups),size=(9999,len(groups)))
        lo,hi=np.quantile(groups.brier_gain.to_numpy()[ix].mean(axis=1),[.025,.975])
        summaries.append(dict(reference=ref,split=split,groups=len(groups),pairs=len(part),
                              base_brier=groups.base_brier.mean(),plus_brier=groups.plus_brier.mean(),
                              brier_gain=groups.brier_gain.mean(),low=lo,high=hi))
    out=pd.DataFrame(summaries);out.to_csv(HERE/"ESMC634_increment_summary.tsv",sep="\t",index=False)
    (HERE/"ESMC634_increment_fits.json").write_text(json.dumps(fit,indent=2),encoding="utf8")
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
