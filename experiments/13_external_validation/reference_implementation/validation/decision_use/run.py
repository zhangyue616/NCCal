"""Risk-based allocation using paired, previously archived predictions."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RSA = PROJECT / "exposure/results"
EXTERNAL = PROJECT / "cross_architecture"
SCALE = PROJECT / "scale/trained_650m/b16/native"
BOOTSTRAPS = 5000
SEED = 2026112207


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_rows(path, data):
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, list(data[0]), delimiter="\t")
        wr.writeheader()
        wr.writerows(data)


def read_predictions(path, expected, keep=None):
    data = [r for r in rows(path) if "task" not in r or r["task"] == "exposure"]
    if keep is not None:
        data = [r for r in data if r["group"] in keep]
    data.sort(key=lambda r: r["group"])
    ids = [r["group"] for r in data]
    if not (len(ids) == len(set(ids)) == expected):
        raise ValueError(f"{path}: {len(ids)} groups, expected {expected}")
    actual = np.array([float(r.get("later_residue_exposed", r.get("target"))) for r in data])
    base = np.array([float(r.get("sequence_probability", r.get("local_probability"))) for r in data])
    augmented = np.array([float(r["augmented_probability"]) for r in data])
    if not (np.isin(actual, [0, 1]).all() and np.isfinite([base, augmented]).all()
            and np.min([base, augmented]) >= 0 and np.max([base, augmented]) <= 1):
        raise ValueError("Invalid score or target")
    archived = np.array([float(r.get("improvement", r.get("gain"))) for r in data])
    np.testing.assert_allclose((base - actual)**2 - (augmented - actual)**2, archived, atol=1e-11, rtol=0)
    return ids, actual, base, augmented


def correctness(scores, targets):
    return np.where(scores == .5, .5, ((scores > .5) == targets).astype(float))


def evaluate(frame, path, expected, keep=None):
    groups, y, base, augmented = read_predictions(path, expected, keep)
    n = len(groups)
    count = n // 3
    # Presort once by confidence and hash so bootstrap copies of a group stay together.
    tie_key = [hashlib.sha256(("nccal-route-v1|" + group).encode()).hexdigest() for group in groups]
    order = np.lexsort((tie_key, np.abs(base - .5)))
    groups = np.array(groups)[order]
    y, base, augmented = y[order], base[order], augmented[order]
    brier_base, brier_aug = (base-y)**2, (augmented-y)**2
    gain = brier_base-brier_aug
    correct_base, correct_aug = correctness(base, y), correctness(augmented, y)
    routed = np.arange(n) < count
    hybrid = np.where(routed, augmented, base)
    delta = correct_aug-correct_base
    fraction = count/n
    entries = [dict(frame=frame, group=g, later_exposed=int(label), sequence=s0, plus_neuron=s1,
                    selected=int(sel), local_correct_credit=c0, augmented_correct_credit=c1)
               for g, label, s0, s1, sel, c0, c1 in zip(groups, y, base, augmented, routed, correct_base, correct_aug)]
    stats = dict(frame=frame, n=n, selected=count, boundary_abs_score_displacement=float(np.abs(base[count-1]-.5)),
                 brier_local=float(brier_base.mean()), brier_all=float(brier_aug.mean()),
                 brier_routed=float(np.mean((hybrid-y)**2)),
                 brier_uniform=float(brier_base.mean()-fraction*gain.mean()),
                 correct_local=float(correct_base.sum()), correct_all=float(correct_aug.sum()),
                 correct_routed=float(correctness(hybrid,y).sum()),
                 expected_correct_uniform=float(correct_base.sum()+fraction*delta.sum()),
                 routed_fixed_incorrect=int(np.sum(routed & (delta==1))),
                 routed_spoiled_correct=int(np.sum(routed & (delta==-1))),
                 routed_half_changes=int(np.sum(routed & (np.abs(delta)==.5))),
                 fixed_incorrect_all=int(np.sum(delta==1)), spoiled_correct_all=int(np.sum(delta==-1)))
    obs = (gain[routed].sum()/n, float(gain.mean()))
    stats["gain_routed"],stats["gain_all"] = obs
    stats["gain_routed_above_equal_budget"] = obs[0]-fraction*obs[1]
    rng = np.random.default_rng(SEED)
    reps = []
    for start in range(0, BOOTSTRAPS, 100):
        draw = np.sort(rng.integers(0,n,size=(min(100,BOOTSTRAPS-start),n)),axis=1)
        all_gain = gain[draw].mean(axis=1)
        routed_gain = gain[draw[:,:count]].sum(axis=1)/n
        reps.append(np.column_stack((routed_gain, all_gain, routed_gain-fraction*all_gain)))
    reps = np.concatenate(reps)
    for i,(key,tail) in enumerate([("gain_routed",.025),("gain_all",.025),("gain_routed_above_equal_budget",.0125)]):
        stats[key+"_lower"],stats[key+"_upper"] = np.quantile(reps[:,i],[tail,1-tail]).tolist()
    assert np.isclose(stats["gain_routed_above_equal_budget"], stats["brier_uniform"]-stats["brier_routed"])
    return stats, entries


def main():
    strict=set(json.loads((EXTERNAL/"validation/HOMOLOGY.json").read_text(encoding="utf-8"))["retained_groups"])
    original=RSA/"35m/buried_exposed_group_errors.tsv"
    more=EXTERNAL/"predictions/source_814.tsv"
    frames=[("814 original",original,967,None),("814 external strict",more,50,strict),("814 external full",more,108,None),
            ("801 original",RSA/"8m/buried_exposed_group_errors.tsv",967,None),
            ("4566 original",SCALE/"buried_exposed_group_errors.tsv",967,None)]
    summary, groups = [], []
    for label,path,n,keep in frames:
        s,g=evaluate(label,path,n,keep)
        summary.append(s)
        groups.extend(g)
    write_rows(HERE/"policies.tsv",summary)
    write_rows(HERE/"group_decisions.tsv",groups)
    print(json.dumps([{k:r[k] for k in ["frame","n","selected","correct_local","correct_all","correct_routed",
                  "routed_fixed_incorrect","routed_spoiled_correct","gain_routed","gain_routed_above_equal_budget",
                  "gain_routed_above_equal_budget_lower","gain_routed_above_equal_budget_upper"]} for r in summary],indent=2))


if __name__ == "__main__":
    main()
