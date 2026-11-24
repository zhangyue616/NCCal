"""Targeted independent arithmetic: D-only ESM contact and structural pairs.

No production modules are imported; no external model result or null result is read.
"""
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
OLD = PROJECT / "protein_extension"


def tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def feature(seq, pos):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX^"
    v = np.zeros(31*len(alphabet)+4)
    for offset in range(-15, 16):
        index = pos-1+offset
        aa = seq[index] if 0 <= index < len(seq) else "^"
        v[(offset+15)*len(alphabet) + alphabet.index(aa if aa in alphabet else "X")] = 1
    relative = (pos-1)/max(1, len(seq)-1)
    v[-4:] = relative, relative**2, len(seq)/256, 1
    return v


def main():
    pairs = tsv(ROOT/"cohorts/D_contact.tsv")
    with (ROOT/"cohorts/D_sequences.jsonl").open(encoding="utf8") as f:
        seq = {r["accession"]: r["sequence"] for r in map(json.loads, f)}
    groups = np.array([r["group"] for r in pairs])
    assert len(pairs) == len(set(groups)) == 77
    x = np.array([feature(seq[r["accession"]],int(r["late"])) -
                  feature(seq[r["accession"]],int(r["early"])) for r in pairs])
    kernel = x@x.T
    target = np.array([int(r["target"]) for r in pairs])
    mapping = {g:i%4 for i,g in enumerate(np.random.default_rng(20261118).permutation(np.unique(groups)))}
    fold = np.array([mapping[g] for g in groups])
    indices = np.arange(len(pairs))

    def predict(train, test, penalty):
        gram = kernel[np.ix_(train,train)]
        normalizer = np.trace(gram)/len(train)
        alpha = np.linalg.solve(gram/normalizer + np.eye(len(train))*len(train)*penalty,
                                2*target[train]-1)
        return np.clip((1+(kernel[np.ix_(test,train)]/normalizer)@alpha)/2, 0, 1)

    def oof(rows, penalty):
        output = np.zeros(len(rows))
        for held in sorted(set(fold[rows])):
            testmask = fold[rows] == held
            output[testmask] = predict(rows[~testmask], rows[testmask], penalty)
        return output

    candidates = [(float(np.mean((target-oof(indices,p))**2)), p) for p in (.1,.01,.001,.0001)]
    selected = min(range(4), key=lambda i:candidates[i][0])
    local_brier, penalty = candidates[selected]
    base = oof(indices, penalty)
    delta = []
    for r in pairs:
        a = np.load(OLD/"rsa/activations/35m/D-all"/(r["accession"]+".npy"), mmap_mode="r", allow_pickle=False)
        # Production takes float32 differences then promotes to float64.
        delta.append(a[int(r["late"])-1] - a[int(r["early"])-1])
    delta = np.array(delta,dtype=np.float64)
    losses = np.zeros(delta.shape[1])
    for held in range(4):
        train, test = indices[fold!=held],indices[fold==held]
        residual = target[train]-oof(train,penalty)
        coeff = np.sum(delta[train]*residual[:,None],axis=0)/(1.01*np.maximum(np.sum(delta[train]**2,axis=0),1e-20))
        augmented = np.clip(base[test,None]+delta[test]*coeff,0,1)
        losses += np.sum((target[test,None]-augmented)**2,axis=0)
    losses /= len(pairs)
    axis = int(np.argmin(losses))
    coefficient = float(np.dot(delta[:,axis],target-base)/(1.01*np.sum(delta[:,axis]**2)))
    with np.load(ROOT/"fits/esm2_35m_trained.npz",allow_pickle=False) as stored:
        assert axis == int(stored["contact_axis"])
        differences = {"local_brier":abs(local_brier-float(stored["contact_oof_local"])),
                       "selected_brier":abs(float(losses[axis])-float(stored["contact_oof_brier"])),
                       "source_coefficient":abs(coefficient-float(stored["contact_slope"])),
                       "all_axes_max":float(np.max(np.abs(losses-stored["contact_allaxis_oof_brier"])))}
    assert max(differences.values()) < 1e-12

    sites = [r for r in tsv(OLD/"structure/D_frozen/residue_features.tsv")
             if r["main_clean"]=="True" and r["interface"]=="False"]
    with np.load(PROJECT/"neuron_814/results/geometry_D.npz",allow_pickle=False) as g:
        geom = {(str(a),int(p)):(int(c.sum()),int(c[40:].sum())) for a,p,c in zip(g["acc"],g["pos"],g["counts"])}
    strata = {}
    detail = {}
    for r in sites:
        accession,pos = r["accession"],int(r["pos"])
        total,remote = geom[accession,pos]
        assert remote == int(r["remote_contacts"])
        bin_ = min(10,int(float(r["rsa"])*10))
        key=(r["group"],accession,r["aa"],r["secondary"],bin_,total)
        strata.setdefault(key,[[],[]])[int(remote>0)].append(pos)
        detail[accession,pos]={"aa":r["aa"],"secondary":r["secondary"],"rsa_bin":bin_,"total":total,"remote":remote}
    best = {}
    count = 0
    for (group,acc,*_), (negative,positive) in strata.items():
        for n,p in product(negative,positive):
            if abs(n-p)<=30:
                continue
            count += 1
            early,late=sorted((n,p))
            digest=hashlib.sha256(f"contact|{group}|{acc}|{early}|{late}".encode("utf8")).digest()
            record={"group":group,"accession":acc,"early":str(early),"late":str(late),"target":str(int(p==late))}
            if group not in best or digest<best[group][0]:
                best[group]=(digest,record)
    rebuilt=[best[g][1] for g in sorted(best)]
    assert rebuilt==pairs and count==117
    sample=[]
    for pair in pairs[:2]:
        sample.append({**pair,"early_structure":detail[pair["accession"],int(pair["early"])],
                       "late_structure":detail[pair["accession"],int(pair["late"])]})
    result={"status":"PASS","scope":"D-only targeted independent arithmetic, no external predictions/results read",
            "independent_implementation":"numpy direct solve and independent dense features; no production module imports",
            "groups":77,"fold_sizes":[int(np.sum(fold==i)) for i in range(4)],
            "penalty":penalty,"local_oof_brier":local_brier,"axis_zero_based":axis,
            "selected_oof_brier":float(losses[axis]),"refit_coefficient":coefficient,
            "absolute_differences_from_stored":differences,
            "rebuilt_structural_candidate_pairs":count,"rebuilt_structural_groups":len(best),
            "pair_manifest_exact_match":True,"two_pair_examples":sample}
    output=ROOT/"validation/DEV_CONTACT_INDEPENDENT.json"
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
