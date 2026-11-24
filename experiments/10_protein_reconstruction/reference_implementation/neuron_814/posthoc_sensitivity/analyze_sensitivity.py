"""Post hoc robustness and exposure-stratified analysis of the archived 814 intervention."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr, softmax
from scipy.stats import binomtest, spearmanr, trim_mean, wilcoxon

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
OLD = EXP / "results"
PAIR_FILE = EXP.parent / "structure/matched_pairs.tsv"
SEED = 20261121
BOOT_N = 2000
ARMS = ("mean", "seq", "full", "swapped", "seq_swapped")
CLASSES = ("buried", "exposed")


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(name, rows):
    with (ROOT / name).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, values):
    (ROOT / name).write_text(json.dumps(values, allow_nan=False, indent=2) + "\n", encoding="utf8")


def draws(n):
    return np.random.default_rng(SEED).integers(0, n, size=(BOOT_N,n), dtype=np.int32)


def ci(values):
    return np.quantile(values, [.025,.975]).tolist()


def mean_interval(values, samples):
    return ci(values[samples].mean(axis=1))


def jsd(native_logits, patch_logits):
    native = softmax(np.asarray(native_logits, dtype=np.float64), axis=1)
    patch = softmax(np.asarray(patch_logits, dtype=np.float64), axis=1)
    mid = (native + patch) / 2
    return np.maximum(0, .5 * (rel_entr(native, mid) + rel_entr(patch, mid)).sum(axis=1))


def summarize_difference(diff, samples):
    ordered = np.sort(diff)[::-1]
    nz = np.count_nonzero(diff)
    count = int((diff > 0).sum())
    interval = mean_interval(diff, samples)
    stats = {"n":len(diff), "mean":float(diff.mean()), "ci95_lo":interval[0], "ci95_hi":interval[1],
             "median":float(np.median(diff)), "n_positive":count, "n_ties":int(len(diff)-nz),
             "sign_p_one_sided":float(binomtest(count, nz, .5, alternative="greater").pvalue),
             "signed_rank_p_one_sided":float(wilcoxon(diff, alternative="greater", method="approx",
                                                     zero_method="wilcox", correction=False).pvalue),
             "trimmed_mean_symmetric_5pct":float(trim_mean(diff, .05))}
    for n in (5,10,20):
        stats[f"largest_{n}_signed_share"] = float(ordered[:n].sum()/diff.sum()) if diff.sum() != 0 else None
        stats[f"mean_after_removing_{n}_largest"] = float(ordered[n:].mean())
    return stats


def robustness_row(label, reference, target, samples):
    good = (reference > 0) & (target > 0)
    lref, ltar = np.log(reference[good]), np.log(target[good])
    logratio = lref - ltar
    use_draws = samples if good.all() else draws(len(logratio))
    ratio_ci = np.exp(mean_interval(logratio, use_draws))
    return {"contrast":label, **summarize_difference(reference-target, samples),
            "gm_reference":float(np.exp(lref.mean())), "gm_target":float(np.exp(ltar.mean())),
            "gm_ratio":float(np.exp(logratio.mean())), "gm_ratio_ci95_lo":float(ratio_ci[0]),
            "gm_ratio_ci95_hi":float(ratio_ci[1]), "gm_common_positive_pairs":int(good.sum())}


def reconstruct_check():
    rows = read_tsv(OLD / "reconstruction_groups_A_T.tsv")
    cols = {k:np.asarray([float(r[k]) for r in rows]) for k in ("mean_z","mean_z2","seq_mse","near_mse","full_mse")}
    variance = cols["mean_z2"].mean()-cols["mean_z"].mean()**2
    sample = draws(len(rows))
    sampled_variance = cols["mean_z2"][sample].mean(axis=1)-cols["mean_z"][sample].mean(axis=1)**2
    results = {"groups":len(rows), "sites":sum(int(r["n"]) for r in rows),
               "r2":{k:float(1-cols[k+"_mse"].mean()/variance) for k in ("seq","near","full")}}
    for base in ("near","seq"):
        d = cols[base+"_mse"]-cols["full_mse"]
        bootstrap=d[sample].mean(axis=1)/sampled_variance
        results["full_minus_"+base] = {"r2_gain":float(d.mean()/variance), "ci95":ci(bootstrap),
                                      "ci_98_3333":np.quantile(bootstrap,[.05/6,1-.05/6]).tolist(),
                                      "n_groups_with_lower_MSE":int((d>0).sum()),"ties":int((d==0).sum())}
    archived = json.loads((OLD/"score_A.json").read_text())
    if any(abs(v-archived["models"][k]["R2"]["estimate"]) > 5e-14 for k,v in results["r2"].items()):
        raise ValueError("Recomputed R² differs from original")
    return results


def load_sites():
    with np.load(OLD/"writeback_T.npz", allow_pickle=False) as f:
        acc, pos, group, z = (f[k] for k in ("acc","pos","group","z"))
        native = f["logits_native"].astype(np.float64)
        if not np.array_equal(f["logits_restored"], f["logits_native"]):
            raise ValueError("Original restoration is not bitwise exact")
        actual = {a:f["actual_"+a].astype(np.float64) for a in ARMS[:-1]}
        divergences = {a:jsd(native,f["logits_"+a]) for a in ARMS[:-1]}
    with np.load(ROOT/"sequence_swap_T.npz", allow_pickle=False) as f:
        second_keys = [(str(a),int(p),str(g)) for a,p,g in zip(f["acc"], f["pos"], f["group"])]
        original_keys = [(str(a),int(p),str(g)) for a,p,g in zip(acc,pos,group)]
        ix = {k:i for i,k in enumerate(second_keys)}
        if len(ix)!=1934 or set(ix)!=set(original_keys):
            raise ValueError("New forward and original file have different sites")
        align = np.array([ix[k] for k in original_keys])
        actual["seq_swapped"] = f["actual_seq_swapped"][align].astype(np.float64)
        divergences["seq_swapped"] = jsd(native, f["logits_seq_swapped"][align])
    lookup = {(str(g),str(a),int(p)):i for i,(a,p,g) in enumerate(zip(acc,pos,group))}
    pairs = sorted((r for r in read_tsv(PAIR_FILE) if r["split"]=="T"), key=lambda r:r["group"])
    order, site_metadata = [], []
    for r in pairs:
        sides = sorted(((float(r["rsa"+str(j)]),int(r["pos"+str(j)])) for j in (1,2)))
        if not (sides[0][0]<=.10 and sides[1][0]>=.40):
            raise ValueError("Pair violates the original RSA categories")
        index = [lookup[(r["group"],r["accession"],p)] for rsa,p in sides]
        order.append(index)
        for side,(rsa,p) in zip(CLASSES,sides):
            site_metadata.append({"group":r["group"],"acc":r["accession"],"class":side,"pos":p,
                                  "aa":r["aa"],"rsa":rsa,"secondary":r["secondary"]})
    order = np.asarray(order)
    if order.shape!=(967,2) or len(set(order.ravel()))!=1934:
        raise ValueError("Paired ordering is incomplete")
    z = z[order].astype(np.float64)
    js = {k:v[order] for k,v in divergences.items()}
    delta = {k:v[order]-z for k,v in actual.items()}
    for fitted,swapped in (("seq","seq_swapped"),("full","swapped")):
        expected=(z+delta[fitted][:,::-1]).astype(np.float32).astype(np.float64)
        if not np.array_equal(expected,actual[swapped][order]):
            raise ValueError("Matched switch did not swap actual deltas")
        base_dose=(delta[fitted]**2).sum(axis=1)
        switch_dose=(delta[swapped]**2).sum(axis=1)
        if np.max(np.abs(base_dose-switch_dose)/np.maximum(1.,base_dose))>2e-6:
            raise ValueError("Matched switch differs in total dose")
    for i,s in enumerate(site_metadata):
        s.update({"native_z":z.ravel()[i]})
        for k in ARMS:
            val=delta[k].ravel()[i]
            s.update({f"{k}_delta":val,f"{k}_dose2":val*val,f"{k}_JSD":js[k].ravel()[i]})
    write_tsv("site_level.tsv", site_metadata)
    return pairs, js, delta


def class_summaries(js,dose, sample):
    rows, comparisons = [], {}
    for k in ARMS:
        for side,name in enumerate(CLASSES):
            d,j = dose[k][:,side], js[k][:,side]
            ok=d>0
            rows.append({"arm":k,"class":name,"sites":len(j),"mean_dose2":d.mean(),"median_dose2":np.median(d),
                         "mean_JSD":j.mean(),"median_JSD":np.median(j),"sum_JSD_per_sum_dose2":j.sum()/d.sum(),
                         "median_site_JSD_per_dose2":np.median(j[ok]/d[ok]),"nonzero_dose_sites":int(ok.sum())})
        d=dose[k][:,1]-dose[k][:,0]
        boot_ratio = []
        for side in (0,1):
            boot_ratio.append(js[k][:,side][sample].sum(axis=1)/dose[k][:,side][sample].sum(axis=1))
        ratios = js[k].sum(axis=0)/dose[k].sum(axis=0)
        comparisons[k] = {"exposed_minus_buried_mean_dose2":float(d.mean()),"median_paired_difference":float(np.median(d)),
                          "dose_difference_ci95":mean_interval(d,sample), "exposed_larger_count":int((d>0).sum()),
                          "sensitivity_exposed_minus_buried":float(ratios[1]-ratios[0]),
                          "sensitivity_difference_ci95":ci(boot_ratio[1]-boot_ratio[0])}
    write_tsv("exposure_classes.tsv",rows)
    return comparisons


def dose_decomposition(pairs,js,dose,sample):
    oriented = js["swapped"]-js["full"]
    observed = oriented.mean(axis=1)
    diff_squared = dose["swapped"]-dose["full"]
    class_k = (js["mean"].sum(axis=0)-js["mean"])/(dose["mean"].sum(axis=0)-dose["mean"])
    denominator = dose["mean"]+dose["seq"]
    site_k=class_k.copy()
    ok=denominator>1e-6
    site_k[ok]=(js["mean"]+js["seq"])[ok]/denominator[ok]
    result={"observed_mean":float(observed.mean()),"sides":{}, "site_proxy_fallback_sites":int((~ok).sum()),
            "fallback_by_class":dict(zip(CLASSES,(~ok).sum(axis=0).astype(int).tolist()))}
    for side,name in enumerate(CLASSES):
        vals=oriented[:,side]
        result["sides"][name]={"mean_site_swap_minus_original":float(vals.mean()),"ci95":mean_interval(vals,sample),
                               "positive_sites":int((vals>0).sum()),"contribution_to_pair_mean":float(.5*vals.mean()),
                               "share_of_pair_contrast":float(.5*vals.mean()/observed.mean())}
    predictions={}
    for key,coef in (("other_pairs_class_coefficients",class_k),("independent_arm_site_coefficients",site_k)):
        pred=(coef*diff_squared).mean(axis=1)
        predictions[key]=pred
        result[key]={"predicted_mean":float(pred.mean()),"observed_mean":float(observed.mean()),
                     "ratio_to_observed":float(pred.mean()/observed.mean()),
                     "mean_residual":float((observed-pred).mean()),
                     "spearman_predicted_observed":float(spearmanr(pred,observed).statistic),
                     "same_sign_pairs":int((np.sign(pred)==np.sign(observed)).sum())}
    out=[]
    for i,r in enumerate(pairs):
        out.append({"group":r["group"],"acc":r["accession"],"observed_difference":observed[i],
                    "buried_difference":oriented[i,0],"exposed_difference":oriented[i,1],
                    "class_expected":predictions["other_pairs_class_coefficients"][i],
                    "site_expected":predictions["independent_arm_site_coefficients"][i]})
    write_tsv("pair_dose_decomposition.tsv",out)
    return result


def main():
    pairs,js,changes = load_sites()
    dose={k:val**2 for k,val in changes.items()}
    original_rows=read_tsv(OLD/"writeback_groups_T.tsv")
    original={r["group"]:r for r in original_rows}
    ordered_groups=[r["group"] for r in pairs]
    if len(original)!=967 or set(original)!=set(ordered_groups):
        raise ValueError("Original group table keys do not match paired sample")
    pair_js={}
    max_error=0.
    for k,val in js.items():
        pair_js[k]=val.mean(axis=1)
        if k in ARMS[:-1]:
            archived=np.array([float(original[g][k+"_JSD"]) for g in ordered_groups])
            max_error=max(max_error,float(np.max(np.abs(pair_js[k]-archived))))
            if max_error>1e-14:
                raise ValueError("Site logits disagree with the original paired table")
            pair_js[k]=archived
    sample=draws(len(pairs))
    rows=[robustness_row(f"{a}_minus_{b}",pair_js[a],pair_js[b],sample)
          for a,b in (("swapped","full"),("seq","full"),("mean","full"),("seq_swapped","seq"))]
    write_tsv("robustness.tsv",rows)
    absolute=[]
    for k,vals in pair_js.items():
        q=mean_interval(vals,sample)
        absolute.append({"arm":k,"arithmetic_mean":vals.mean(),"mean_ci95_lo":q[0],"mean_ci95_hi":q[1],
                         "median":np.median(vals),"maximum":vals.max(),"max_over_median":vals.max()/np.median(vals),
                         "top10_share_of_absolute_jsd":np.sort(vals)[-10:].sum()/vals.sum(),"n_zero":int((vals==0).sum())})
    write_tsv("absolute_output_deviation.tsv",absolute)
    placement_full=pair_js["swapped"]-pair_js["full"]
    placement_seq=pair_js["seq_swapped"]-pair_js["seq"]
    contrast=placement_full-placement_seq
    placement_rows=[{"group":g,"acc":r["accession"],"structure_placement":f,"sequence_placement":s,
                     "structure_minus_sequence_placement":f-s,"sequence_swapped_JSD":j}
                     for g,r,f,s,j in zip(ordered_groups,pairs,placement_full,placement_seq,pair_js["seq_swapped"])]
    write_tsv("sequence_placement_pairs.tsv",placement_rows)
    details={"reconstruction":reconstruct_check(),"largest_archive_jsd_disagreement":max_error,
             "class_comparison":class_summaries(js,dose,sample),
             "decomposition":dose_decomposition(pairs,js,dose,sample),
             "paired_structure_minus_sequence_placement":summarize_difference(contrast,sample),
             "placement_ranks_spearman":float(spearmanr(placement_full,placement_seq).statistic),
             "same_top20_in_both_placements":len(set(np.argsort(placement_full)[-20:]) & set(np.argsort(placement_seq)[-20:]))}
    write_json("sensitivity_summary.json",details)
    print("Post hoc summaries complete for",len(pairs),"pairs and",len(pairs)*2,"sites.")


if __name__ == "__main__":
    main()
