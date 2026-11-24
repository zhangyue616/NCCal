"""Fit and evaluate the prespecified ESM-2 35M neuron 814 reconstruction experiment."""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax, rel_entr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PARENT = HERE.parent / "protein_extension"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT / "rsa"))
from model_inputs import ALPHABET, activations, design_matrix, grouped_folds, read_rows, weights

PENALTIES = (0.1, 0.01, 0.001, 0.0001, 0.00001)
MODELS = ("seq", "near", "full")
BOOT_SEED = 20261121
DRAWS = 2000

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)

def save_npz(path, fields):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    temp.replace(path)

def save_tsv(path, columns):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        n = len(next(iter(columns.values())))
        if any(len(values) != n for values in columns.values()):
            raise ValueError("Per-group columns have inconsistent sizes")
        writer.writerows(zip(*columns.values()))
    temp.replace(path)

def check_plan():
    lock = json.loads((HERE / "PROTOCOL_FREEZE.json").read_text(encoding="utf-8-sig"))
    actual = digest(HERE / "PROTOCOL.md")
    if actual.lower() != lock["protocol_sha256"].lower():
        raise ValueError("The preregistered local protocol changed")
    return actual

def indices_for_b(rows, split):
    groups = np.asarray([row[2] for row in rows])
    if split == "D":
        _, first, counts = np.unique(groups, return_index=True, return_counts=True)
        indices = [start + np.linspace(0, n-1, min(n, 32), dtype=int) for start,n in zip(first,counts)]
        return np.concatenate(indices)
    pairs = set()
    with (PARENT / "structure" / "matched_pairs.tsv").open(encoding="utf-8-sig",newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            if item["split"] == split:
                for field in ("pos1", "pos2"):
                    key = (item["group"], item["accession"], int(item[field]))
                    if key in pairs:
                        raise ValueError("Duplicate paired site")
                    pairs.add(key)
    indices = np.array([i for i,row in enumerate(rows) if (row[2],row[0],row[1]) in pairs])
    if len(indices) != 1934 or len(indices) != len(pairs) or not np.all(np.unique(groups[indices],return_counts=True)[1] == 2):
        raise ValueError("Matched holdout count or group composition failed")
    return indices

def site_keys(acc, pos, group):
    return [(str(a),int(p),str(g)) for a,p,g in zip(acc,pos,group)]

def alignment(source_keys, target_keys):
    order = {key:i for i,key in enumerate(source_keys)}
    if len(order) != len(source_keys) or len(target_keys) != len(source_keys) or set(target_keys) != set(order):
        raise ValueError("The two data sources contain different sites")
    return np.asarray([order[k] for k in target_keys],dtype=int)

def dataset(stage, split):
    rows,cohort = read_rows(split)
    expected = {"D": (20974,325), "T": (173859,1099)}[split]
    if (len(rows),len({r[2] for r in rows})) != expected:
        raise ValueError("Original clean cohort changed")
    keys = [(r[0],r[1],r[2]) for r in rows]
    path = OUT / f"geometry_{split}.npz"
    with np.load(path,allow_pickle=False) as f:
        order = alignment(site_keys(f["acc"],f["pos"],f["group"]), keys)
        counts = f["counts"][order]
    if counts.shape != (len(rows),80) or not np.array_equal(counts[:,40:].sum(axis=1),np.array([r[6] for r in rows])):
        raise ValueError("Contact features disagree with original remote-contact tallies")
    if stage == "B":
        keep = indices_for_b(rows,split)
        counts = counts[keep]
        rows = [rows[i] for i in keep]
        keys = [keys[i] for i in keep]
        with np.load(OUT / f"masked_native_{split}.npz",allow_pickle=False) as f:
            order = alignment(site_keys(f["acc"],f["pos"],f["group"]), keys)
            y = f["z"][order].astype(np.float64).reshape(-1)
    else:
        y = activations(rows,"35m",split,axis=814).astype(np.float64)
    if not np.isfinite(y).all() or y.shape != (len(rows),):
        raise ValueError("Invalid native activation targets")
    x = design_matrix(rows,cohort)
    if stage == "B":
        # Offset +1 is precisely the single masked readout residue. Exclude its identity.
        offset = 16 * len(ALPHABET)
        keep_columns = np.r_[np.arange(offset),np.arange(offset+len(ALPHABET),x.shape[1])]
        x = x[:,keep_columns]
    geo = sparse.csr_matrix(np.log1p(counts.astype(np.float64)))
    fields = {"group":np.array([r[2] for r in rows]),"acc":np.array([r[0] for r in rows]),
              "pos":np.array([r[1] for r in rows],dtype=int)}
    return fields,x,geo,y

def designs(x,geo):
    for name in MODELS:
        if name == "seq":
            yield name,x
        else:
            end = 40 if name == "near" else 80
            yield name,sparse.hstack((x[:,:-1],geo[:,:end],x[:,-1]),format="csr")

def weighted_normal(x, y, w):
    return (x.T @ x.multiply(w[:,None])).toarray(), np.asarray(x.T @ (w*y)).reshape(-1)

def solve_ridge(normal, rhs, penalty):
    regularized = normal.copy()
    regularized.flat[::len(regularized)+1] += penalty
    regularized[-1,-1] -= penalty
    return cho_solve(cho_factor(regularized,lower=True,check_finite=False),rhs,check_finite=False)

def fit(stage):
    protocol = check_plan()
    model_file = OUT / f"fit_{stage}.npz"
    receipt = OUT / f"fit_{stage}_FREEZE.json"
    if receipt.exists():
        verify_fit(stage)
        print(f"{stage} development fit already sealed",flush=True)
        return
    fields,x,geo,y = dataset(stage,"D")
    groups = fields["group"]
    folds = grouped_folds(groups)
    w = weights(groups)
    coefficients = {"mean":np.asarray(np.dot(w,y))}
    fitlog = {}
    for name,design in designs(x,geo):
        squared_errors = np.zeros((len(PENALTIES),len(y)))
        for fold in range(4):
            train = folds != fold
            val = ~train
            normal,rhs = weighted_normal(design[train], y[train], weights(groups[train]))
            for j,penalty in enumerate(PENALTIES):
                coef = solve_ridge(normal,rhs,penalty)
                squared_errors[j,val] = (y[val] - design[val] @ coef)**2
        cv = squared_errors @ w
        best = int(np.argmin(cv))
        normal,rhs = weighted_normal(design, y, w)
        coefficients["coef_"+name] = solve_ridge(normal,rhs,PENALTIES[best])
        fitlog[name] = {"penalty":PENALTIES[best],"D_OOF_MSE":dict(zip(map(str,PENALTIES),cv.tolist()))}
    save_npz(model_file,coefficients)
    save_json(receipt,{"stage":stage,"sealed_utc":utc(),"protocol_sha256":protocol,"fit_sha256":digest(model_file),
                       "groups":int(len(np.unique(groups))),"sites":len(y),"fits":fitlog,
                       "source":"unmasked sequence" if stage=="A" else "source visible, +1 site masked"})
    print(f"{stage} development fit sealed, {len(y)} sites", flush=True)

def verify_fit(stage):
    obj=json.loads((OUT/f"fit_{stage}_FREEZE.json").read_text(encoding="utf8"))
    if obj["protocol_sha256"]!=check_plan() or obj["fit_sha256"]!=digest(OUT/f"fit_{stage}.npz"):
        raise ValueError("Fit receipt mismatch")
    return obj

def group_average(values,index,n):
    return np.bincount(index,weights=values,minlength=len(n))/n

def draws_for(n):
    return np.random.default_rng(BOOT_SEED).integers(0,n,size=(DRAWS,n),dtype=np.int32)

def intervals(samples):
    confidence = 1 - .05/3
    return {"ci95":np.quantile(samples,[.025,.975]).tolist(),
            "ci_simultaneous_98_3333":np.quantile(samples,[(1-confidence)/2,(1+confidence)/2]).tolist()}

def estimate(point,samples):
    return {"estimate":float(point),**intervals(samples)}

def score(stage):
    freeze = verify_fit(stage)
    fields,x,geo,y = dataset(stage,"T")
    with np.load(OUT/f"fit_{stage}.npz", allow_pickle=False) as f:
        predictions = {"mean":np.full(len(y),float(f["mean"]))}
        for name,design in designs(x,geo):
            predictions[name] = np.asarray(design@f["coef_"+name]).reshape(-1)
    levels,index,n = np.unique(fields["group"],return_inverse=True,return_counts=True)
    ng=len(levels)
    yz=group_average(y,index,n)
    yz2=group_average(y*y,index,n)
    var=float(yz2.mean()-yz.mean()**2)
    if var<=0 or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Nonpositive target variance or invalid predicted activation")
    draws=draws_for(ng)
    bvar=yz2[draws].mean(axis=1)-yz[draws].mean(axis=1)**2
    if np.any(bvar<=0):
        raise ValueError("Zero variance in a bootstrap draw")
    all_mse={};outmodels={};cols={"group":levels,"n":n,"mean_z":yz,"mean_z2":yz2}
    for name,pred in predictions.items():
        gm=group_average((y-pred)**2,index,n)
        ga=group_average(np.abs(y-pred),index,n)
        all_mse[name]=gm
        cols[name+"_mse"]=gm;cols[name+"_mae"]=ga
        outmodels[name]={"R2":estimate(1-gm.mean()/var,1-gm[draws].mean(axis=1)/bvar),
                         "MSE":float(gm.mean()),"MAE":float(ga.mean())}
    contrasts={}
    for a,b in (("full","seq"),("full","near"),("full","mean")):
        delta=all_mse[b]-all_mse[a]
        contrasts[a+"_minus_"+b+"_R2"] = estimate(delta.mean()/var,delta[draws].mean(axis=1)/bvar)
    result={"stage":stage,"scored_utc":utc(),"D_fit_sealed_utc":freeze["sealed_utc"],
            "T_groups":ng,"T_sites":len(y),"group_equal_variance":var,"models":outmodels,"contrasts":contrasts}
    save_npz(OUT/f"site_predictions_{stage}_T.npz",{**fields,"z":y,**predictions})
    save_tsv(OUT/f"reconstruction_groups_{stage}_T.tsv",cols)
    if stage=="B":
        if not np.all(n==2):
            raise ValueError("Expected one two-member pair per group")
        ordered=np.argsort(index,kind="stable").reshape(-1,2)
        mate=np.empty(len(y),dtype=int);mate[ordered[:,0]]=ordered[:,1];mate[ordered[:,1]]=ordered[:,0]
        actual_full=predictions["full"].astype(np.float32).astype(np.float64)
        swapped=(y+(actual_full-y)[mate]).astype(np.float32)
        save_npz(OUT/"B_replacements_T.npz",{**fields,"z":y.astype(np.float32),
                  "mean":predictions["mean"].astype(np.float32),"seq":predictions["seq"].astype(np.float32),
                  "full":actual_full.astype(np.float32),"swapped":swapped})
    save_json(OUT/f"score_{stage}.json",result)
    print(f"{stage} reconstruction scored, {ng} groups",flush=True)

def js_distance(native, changed):
    p=softmax(np.asarray(native,dtype=np.float64),axis=1)
    q=softmax(np.asarray(changed,dtype=np.float64),axis=1)
    mid=(p+q)/2
    return np.maximum(0,0.5*(rel_entr(p,mid)+rel_entr(q,mid)).sum(axis=1))

def writeback_score():
    verify_fit("B")
    with np.load(OUT/"B_replacements_T.npz",allow_pickle=False) as f:
        fields={k:f[k] for k in ("acc","pos","group")}
        replacement={k:f[k].astype(np.float64) for k in ("z","mean","seq","full","swapped")}
    keys=site_keys(fields["acc"],fields["pos"],fields["group"])
    with np.load(OUT/"writeback_T.npz",allow_pickle=False) as f:
        oi=alignment(site_keys(f["acc"],f["pos"],f["group"]),keys)
        logits={k:f["logits_"+k][oi].astype(np.float64) for k in ("native","mean","seq","full","swapped")}
    with np.load(OUT/"masked_native_T.npz",allow_pickle=False) as f:
        oi=alignment(site_keys(f["acc"],f["pos"],f["group"]),keys)
        prev=f["logits"][oi]
        prevz=f["z"][oi].reshape(-1).astype(np.float64)
    if not np.array_equal(prevz,replacement["z"]):
        raise ValueError("Writeback dose uses native values from other contexts")
    nlogitdiff=float(np.max(np.abs(prev-logits["native"])))
    if nlogitdiff>1e-4 or any(v.shape!=(1934,20) or not np.isfinite(v).all() for v in logits.values()):
        raise ValueError(f"Native logits fail same-input replay: {nlogitdiff}")
    groups,index,n=np.unique(fields["group"],return_inverse=True,return_counts=True)
    if len(groups)!=967 or np.any(n!=2):
        raise ValueError("Invalid writeback paired design")
    draws=draws_for(len(groups));gj={};dose={};means={};cols={"group":groups}
    for arm in ("mean","seq","full","swapped"):
        gj[arm]=group_average(js_distance(logits["native"],logits[arm]),index,n)
        dose[arm]=group_average((replacement[arm]-prevz)**2,index,n)
        g=gj[arm];cols[arm+"_JSD"]=g;cols[arm+"_dose_MSE"]=dose[arm]
        means[arm]={"JSD_nats":estimate(g.mean(),g[draws].mean(axis=1)),
                    "mean_squared_displacement":float(dose[arm].mean())}
    technical_dose=float(np.max(np.abs(2*(dose["swapped"]-dose["full"]))))
    scale=np.maximum(1.0,2*dose["full"])
    if np.max(np.abs(2*(dose["swapped"]-dose["full"]))/scale)>2e-6:
        raise ValueError("Float32 pairwise matched dose failed")
    contrasts={}
    for arm in ("seq","swapped","mean"):
        d=gj[arm]-gj["full"];contrasts[arm+"_minus_full_JSD"]=estimate(d.mean(),d[draws].mean(axis=1))
    ac=json.loads((OUT/"score_A.json").read_text())["contrasts"]["full_minus_seq_R2"]
    gates={k:bool(v["ci_simultaneous_98_3333"][0]>0) for k,v in
           (("A_structure_beyond_sequence",ac),("B_structure_vs_sequence",contrasts["seq_minus_full_JSD"]),
            ("B_structure_vs_placement",contrasts["swapped_minus_full_JSD"]))}
    save_tsv(OUT/"writeback_groups_T.tsv",cols)
    save_json(OUT/"score_writeback.json",{"scored_utc":utc(),"groups":len(groups),"sites":len(index),
              "native_replay_max_logit_diff":nlogitdiff,"matched_dose_pair_sum_max_abs_diff":technical_dose,
              "arms":means,"contrasts":contrasts,"predeclared_joint_checks":gates,"all_three_positive":all(gates.values())})
    print(f"Masked-output writeback scored on {len(groups)} pairs",flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("fit","score","writeback"))
    parser.add_argument("stage",nargs="?",choices=("A","B"))
    args=parser.parse_args()
    check_plan()
    with threadpool_limits(limits=3):
        if args.action=="writeback":
            writeback_score()
        elif args.stage is None:
            parser.error("fit and score require A or B")
        elif args.action=="fit":
            fit(args.stage)
        else:
            score(args.stage)

if __name__ == "__main__":
    main()
