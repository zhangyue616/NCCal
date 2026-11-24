"""Shared held-out scoring, sequence comparators and equal-budget reference bases."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
TASKS = ["helix_vs_strand", "helix_start_vs_internal"]
SPLITS = ["D", "T", "E"]
ALPH = list("ACDEFGHIKLMNPQRSTVWY") + ["?", "."]

def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype={"group":str, "record_id":str})

def win(x):
    return (x > 0) + 0.5 * (x == 0)

def group_uncertainty(values, null):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, p=None, n_informative=0)
    rng = np.random.default_rng(41624)
    means = x[rng.integers(len(x), size=(9999, len(x)))].mean(axis=1)
    z = x - null
    z = z[np.abs(z) > 1e-12]
    if len(z) == 0:
        p = 1.
    elif len(z) <= 18:
        integers = np.arange(2 ** len(z))[:, None]
        signs = 1 - 2 * ((integers >> np.arange(len(z))) & 1)
        p = float(np.mean(signs @ z >= z.sum() - 1e-12))
    else:
        rng = np.random.default_rng(41623)
        over = 0
        for start in range(0, 99999, 2000):
            s = 2 * rng.integers(2, size=(min(2000, 99999-start), len(z))) - 1
            over += (s @ z >= z.sum() - 1e-12).sum()
        p = float((over + 1) / 100000)
    return {"mean":float(x.mean()), "lo":float(np.quantile(means,.025)),
            "hi":float(np.quantile(means,.975)), "p":p, "n_informative":len(z)}

def by_group(y, rows):
    return pd.Series(np.asarray(y)).groupby(rows.group.to_numpy()).mean()

class Differences:
    def __init__(self, model):
        self.model = model
        self.tables, self.layers_by_split = {}, {}
        if model == "AMPLIFY":
            p = ROOT / "amplify/results/trained"
            self.sels = json.loads((p / "locked.json").read_text())
            for split in SPLITS:
                path = p / f"{split}_pairs.tsv"
                if path.exists():
                    self.tables[split] = read_tsv(path).reset_index(drop=True)
                    self.layers_by_split[split] = np.load(p / f"{split}_delta.npy", mmap_mode="r")
            self.depth = self.layers_by_split["D"].shape[1]
        else:
            p = ROOT / "e1/trained"
            refs = json.loads((p / "selection_lock.json").read_text(encoding="utf-8-sig"))["selections"]
            self.sels = [{"task":k["task"],"block_1":k["layer0"]+1,"channel_0":k["unit0"],
                          "sign":int(k["sign"])} for k in refs]
            pairs = read_tsv(p / "pair_index.tsv")
            self.layer_files = sorted((p/"pair_deltas").glob("layer_*.npy"))
            self.depth = len(self.layer_files)
            assert self.depth == 20
            self.row_indices = {}
            for split in SPLITS:
                sub = pairs[pairs.split == split]
                if not sub.empty:
                    self.tables[split] = sub.reset_index(drop=True)
                    self.row_indices[split] = sub.matrix_row.to_numpy()

    def layer(self, split, index):
        if self.model == "AMPLIFY":
            return np.ascontiguousarray(self.layers_by_split[split][:,index,:])
        raw = np.load(self.layer_files[index], mmap_mode="r")
        return np.ascontiguousarray(raw[self.row_indices[split]])

def get_native(lib):
    tables = []
    for split, rs in lib.tables.items():
        for k in lib.sels:
            rows = rs[rs.task == k["task"]].copy()
            vals = lib.layer(split, k["block_1"]-1)[rows.index,k["channel_0"]]
            rows["delta"] = k["sign"] * vals
            rows["model"] = lib.model
            rows["channel_0"], rows["block_1"], rows["sign"] = k["channel_0"], k["block_1"], k["sign"]
            tables.append(rows)
    return pd.concat(tables, ignore_index=True)

def count_windows(sequences, rows):
    features = np.zeros((len(rows), 8 * len(ALPH)), dtype=np.float64)
    for r, row in enumerate(rows.itertuples()):
        seq = sequences[row.record_id]
        assert seq[row.index_positive] == seq[row.index_negative] == row.aa
        for j, off in enumerate([-4,-3,-2,-1,1,2,3,4]):
            for i, sgn in [(row.index_positive,1),(row.index_negative,-1)]:
                aa = seq[i+off] if 0 <= i+off < len(seq) else "."
                col = ALPH.index(aa) if aa in ALPH else ALPH.index("?")
                features[r, j*len(ALPH)+col] += sgn
    return features

def probe(task_rows, sequences):
    rows = task_rows.reset_index(drop=True).copy()
    X = count_windows(sequences, rows)
    disc = rows.split == "D"
    d = rows.delta.to_numpy()
    groups = rows.loc[disc].group
    weights = 1/groups.map(groups.value_counts()).to_numpy()
    weights /= weights.mean()
    scale = np.sqrt(np.average(d[disc]**2, weights=weights))
    assert scale > 0
    for suffix, design in [("seq",X),("plus",np.column_stack((X,d/scale)))]:
        train = design[disc]
        clf = LogisticRegression(C=1.,fit_intercept=False,solver="lbfgs",max_iter=1000)
        clf.fit(np.vstack((train,-train)),np.r_[np.ones(len(train)),np.zeros(len(train))],
                sample_weight=np.tile(weights,2))
        rows[f"prob_{suffix}"] = clf.predict_proba(design)[:,1]
        rows[f"brier_{suffix}"] = (1-rows[f"prob_{suffix}"])**2
        rows[f"correct_{suffix}"] = win(rows[f"prob_{suffix}"] - .5)
    return rows

def make_orthogonal(X, seed, layer):
    rng = np.random.default_rng([77203, seed, layer])
    n = X.shape[1]
    r = X
    for _ in range(2):
        perm = rng.permutation(n)
        signs = (2*rng.integers(0,2,n)-1).astype(np.float32)
        r = dct(np.ascontiguousarray(r[:,perm]*signs),type=2,norm="ortho",axis=-1,workers=2)
    return r

def group_integer_ranks(rs, task):
    sub = rs[rs.task == task]
    member = [part.index.to_numpy() for _,part in sub.groupby("group")]
    common = math.lcm(*(len(ids) for ids in member))
    return member, common, 2*len(member)*common

def rotations(lib, folder):
    locks = []
    tests = []
    for seed in range(1,10):
        trainrows = lib.tables["D"]
        sets = {t:group_integer_ranks(trainrows,t) for t in TASKS}
        best = {t:(-1,-1,-1,1) for t in TASKS}
        for layer in range(lib.depth):
            mat = lib.layer("D",layer)
            rot = make_orthogonal(mat, seed, layer)
            if layer == 0:
                assert np.allclose((mat[:6]**2).sum(axis=1),(rot[:6]**2).sum(axis=1),rtol=2e-5,atol=1e-7)
            for task, (members, common, denominator) in sets.items():
                points = np.zeros(rot.shape[1],dtype=np.int64)
                for ids in members:
                    r = rot[ids]
                    points += (common//len(ids))*(2*(r>0).sum(axis=0)+(r==0).sum(axis=0))
                signed = np.stack((points,denominator-points),axis=-1)
                unit, side = np.unravel_index(np.argmax(signed), signed.shape)
                rank = int(signed[unit,side])
                if rank > best[task][0]:
                    best[task]=(rank,layer,int(unit),1 if side==0 else -1)
        run=[]
        for task,(rank,l,u,sgn) in best.items():
            run.append(dict(model=lib.model,seed=seed,task=task,block_1=l+1,channel_0=u,sign=sgn,
                            development_score=rank/sets[task][2]))
        locks.extend(run)
        dump(folder/f"rotation_{seed}_lock.json",run)
        for split,rs in lib.tables.items():
            for k in run:
                part=rs[rs.task==k["task"]]
                rotated=make_orthogonal(lib.layer(split,k["block_1"]-1), seed, k["block_1"]-1)
                val=k["sign"]*rotated[part.index,k["channel_0"]]
                grouped=by_group(win(val),part)
                for g,acc in grouped.items():
                    tests.append(dict(model=lib.model,seed=seed,split=split,task=k["task"],group=g,accuracy=acc))
        print(lib.model,"orthogonal reference",seed,flush=True)
    pd.DataFrame(tests).to_csv(folder/"rotation_groups.tsv",sep="\t",index=False)
    pd.DataFrame(locks).to_csv(folder/"rotation_locks.tsv",sep="\t",index=False)

def analyse(model):
    lib=Differences(model)
    folder=OUT/model
    folder.mkdir(exist_ok=True)
    chains={r["record_id"]:r["sequence"] for r in map(json.loads,(ROOT/"cohort/chains.jsonl").read_text(encoding="utf-8-sig").splitlines())}
    all_rows=get_native(lib)
    probe_rows=pd.concat([probe(rs,chains) for _,rs in all_rows.groupby("task")],ignore_index=True)
    if "same_helix_segment" not in probe_rows:
        probe_rows["same_helix_segment"]=(probe_rows.positive_helix_segment_id==probe_rows.negative_helix_segment_id)
    probe_rows["same_helix_segment"]=probe_rows.same_helix_segment.astype(str).str.lower().isin(["true","1","yes"])
    probe_rows.to_csv(folder/"paired_predictions.tsv",sep="\t",index=False)
    result=[]
    for (task,split),rows in probe_rows.groupby(["task","split"]):
        divisions=[("all",rows)]
        if task == "helix_start_vs_internal":
            same = rows[rows.same_helix_segment]
            divisions.append(("same_helix", same))
            divisions.append(("same_helix_deep",same[same.negative_helix_distance_to_end >= 4]))
        for scope,sub in divisions:
            if sub.empty:
                continue
            acc=win(sub.delta)
            metrics={"single_channel":(acc,.5),
                     "accuracy_sequence":(sub.correct_seq,.5),"accuracy_plus":(sub.correct_plus,.5),
                     "brier_sequence":(sub.brier_seq,0),"brier_plus":(sub.brier_plus,0),
                     "brier_gain":(sub.brier_seq-sub.brier_plus,0)}
            for title,(v,null) in metrics.items():
                est=group_uncertainty(by_group(v,sub),null)
                if title in ["brier_sequence","brier_plus","accuracy_sequence","accuracy_plus"]:
                    est["p"]=None
                result.append(dict(model=model,task=task,split=split,scope=scope,metric=title,
                    groups=sub.group.nunique(),pairs=len(sub),pooled_mean=float(np.mean(v)),**est))
    pd.DataFrame(result).to_csv(folder/"summary.tsv",sep="\t",index=False)
    rotations(lib,folder)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("model",choices=["AMPLIFY","E1"])
    analyse(ap.parse_args().model)
