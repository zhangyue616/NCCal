"""Post hoc description of the 20 largest signed swapped-minus-full T pairs.

Reads frozen writeback/logit/structural/cohort files only; runs no model and
does not change the prespecified analysis or its denominator.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.special import rel_entr, softmax

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
RESULTS = BASE / "results"
STRUCTURE = ROOT / "protein_extension" / "structure"
PROTEINS = ROOT / "data" / "revision" / "protein_fresh" / "proteins.jsonl"
FROZEN_ARMS = ("mean", "seq", "full", "swapped")
ARMS = FROZEN_ARMS + ("seq_swapped",)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows):
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def jsd(native, changed):
    # Frozen scoring definition: natural-log JSD on renormalized canonical-AA
    # logits, with float64 softmax/relative entropy and numerical floor at 0.
    p, q = softmax(native.astype(np.float64), axis=1), softmax(changed.astype(np.float64), axis=1)
    m = (p + q) / 2
    return np.maximum(0, .5 * (rel_entr(p, m) + rel_entr(q, m)).sum(axis=1))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "q05": float(np.quantile(a, .05)), "q25": float(np.quantile(a, .25)),
            "q75": float(np.quantile(a, .75)), "q95": float(np.quantile(a, .95))}


def percentile(value, reference):
    a = np.asarray(reference)
    return float(100 * ((a < value).sum() + .5 * (a == value).sum()) / len(a))


def main():
    table = read_tsv(RESULTS / "writeback_groups_T.tsv")
    group_table = {r["group"]: r for r in table}
    if len(table) != 967 or len(group_table) != 967:
        raise ValueError("Expected 967 unique frozen T groups")
    signed = {g: float(r["swapped_JSD"]) - float(r["full_JSD"]) for g, r in group_table.items()}
    ranked = sorted(group_table, key=lambda g: (-signed[g], g))
    top_groups, rest_groups = ranked[:20], ranked[20:]
    pairs = {}
    for r in read_tsv(STRUCTURE / "matched_pairs.tsv"):
        if r["split"] == "T":
            if r["group"] in pairs:
                raise ValueError("Duplicate frozen matched pair")
            pairs[r["group"]] = r
    if set(pairs) != set(group_table):
        raise ValueError("Frozen pair mapping and group readouts differ")
    cohort = {}
    with PROTEINS.open(encoding="utf-8-sig") as stream:
        for line in stream:
            r = json.loads(line)
            if r["id"] in cohort:
                raise ValueError("Repeated archived accession")
            cohort[r["id"]] = r
    with np.load(RESULTS / "writeback_T.npz", allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    keys = list(zip(data["group"].tolist(), data["acc"].tolist(), data["pos"].tolist()))
    indices = {k: i for i, k in enumerate(keys)}
    if len(indices) != 1934 or len(keys) != 1934:
        raise ValueError("Expected 1934 unique writeback source sites")
    with np.load(HERE / "sequence_swap_T.npz", allow_pickle=False) as saved:
        new_keys = list(zip(saved["group"].tolist(), saved["acc"].tolist(), saved["pos"].tolist()))
        new_index = {k: i for i, k in enumerate(new_keys)}
        if len(new_keys) != len(new_index) or set(new_index) != set(indices):
            raise ValueError("New sequence-swapped source keys differ")
        align = [new_index[k] for k in keys]
        for name in ("actual_seq_swapped", "logits_seq_swapped"):
            data[name] = saved[name][align].copy()
            if data[name].dtype != np.float32 or not np.isfinite(data[name]).all():
                raise ValueError("New sequence-swapped output must be finite float32")
    native_z = data["z"].astype(np.float64)
    delta = {arm: data["actual_" + arm].astype(np.float64) - native_z for arm in ARMS}
    distances = {arm: jsd(data["logits_native"], data["logits_" + arm]) for arm in ARMS}
    restored = jsd(data["logits_native"], data["logits_restored"])
    if not np.array_equal(data["logits_native"], data["logits_restored"]) or np.any(restored != 0):
        raise ValueError("Frozen native restoration differs")
    pair_records, site_records = {}, {}
    max_jsd_error = max_dose_error = 0.0
    for g in ranked:
        p = pairs[g]
        acc = p["accession"]
        protein = cohort[acc]
        if protein["known_group_id"] != g:
            raise ValueError(f"Archived known-group mismatch {acc}")
        seq = protein["sequence"]
        pfam = sorted(set(protein.get("pfam", [])))
        no_pfam = protein.get("no_Pfam")
        if no_pfam is not None and bool(no_pfam) != (len(pfam) == 0):
            raise ValueError(f"Archived Pfam/no_Pfam disagree {acc}")
        row = {"group": g, "accession": acc, "pfam": ";".join(pfam) or "NO_ANNOTATION",
               "no_pfam_annotation": len(pfam) == 0, "sequence_length": len(seq),
               "pdb_id": p["pdb_id"], "assembly_id": p["assembly_id"],
               "author_chain": p["chain"], "assembly_chain": int(p["assembly_chain"])}
        site_rows, ii = [], []
        for number, role in ((1, "buried"), (2, "exposed")):
            pos = int(p["pos" + str(number)])
            aa = seq[pos - 1]
            if aa != p["aa"]:
                raise ValueError(f"Frozen paired AA/sequence mismatch {acc}:{pos}")
            index = indices[(g, acc, pos)]
            ii.append(index)
            rsa = float(p["rsa" + str(number)])
            if (role == "buried" and rsa > .10) or (role == "exposed" and rsa < .40):
                raise ValueError("Frozen pair role/RSA mismatch")
            nterm, cterm = pos - 1, len(seq) - pos
            nearest = min(nterm, cterm)
            site = {"group": g, "accession": acc, "pfam": row["pfam"],
                    "no_pfam_annotation": row["no_pfam_annotation"], "sequence_length": len(seq),
                    "pair_member": number, "role": role, "source_position": pos,
                    "source_aa": aa, "masked_readout_position": pos + 1, "rsa": rsa,
                    "nterminal_distance_aa": nterm, "cterminal_distance_aa": cterm,
                    "nearest_terminal_distance_aa": nearest, "within15_terminal": nearest <= 15,
                    "native_z": float(native_z[index]), "native_JSD_nats": 0.0,
                    "restored_actual_delta": 0.0, "restored_JSD_nats": float(restored[index])}
            for arm in ARMS:
                site[arm + "_actual_value"] = float(data["actual_" + arm][index])
                site[arm + "_actual_delta"] = float(delta[arm][index])
                site[arm + "_delta_squared"] = float(delta[arm][index] ** 2)
                site[arm + "_JSD_nats"] = float(distances[arm][index])
            site["swapped_minus_full_JSD_nats"] = float(distances["swapped"][index] - distances["full"][index])
            site_rows.append(site)
            for field in ("source_position", "source_aa", "rsa", "nterminal_distance_aa", "cterminal_distance_aa",
                          "nearest_terminal_distance_aa", "within15_terminal", "native_z",
                          "native_JSD_nats", "restored_actual_delta", "restored_JSD_nats"):
                row[role + "_" + field] = site[field]
            for arm in ARMS:
                row[role + "_" + arm + "_actual_delta"] = site[arm + "_actual_delta"]
                row[role + "_" + arm + "_JSD_nats"] = site[arm + "_JSD_nats"]
        row["any_source_within15_terminal"] = any(s["within15_terminal"] for s in site_rows)
        row["pair_nearest_terminal_distance_aa"] = min(s["nearest_terminal_distance_aa"] for s in site_rows)
        row["native_pair_JSD_nats"] = 0.0
        row["restored_pair_JSD_nats"] = 0.0
        row["restored_pair_dose_MSE"] = 0.0
        for arm in ARMS:
            j = float(distances[arm][ii].mean())
            d = float((delta[arm][ii] ** 2).mean())
            if arm in FROZEN_ARMS:
                jerr = abs(j - float(group_table[g][arm + "_JSD"]))
                derr = abs(d - float(group_table[g][arm + "_dose_MSE"]))
                max_jsd_error, max_dose_error = max(max_jsd_error, jerr), max(max_dose_error, derr)
                if jerr > 1e-18 or derr > 1e-12:
                    raise ValueError(f"Frozen group readout does not reproduce {g}/{arm}: {jerr} {derr}")
            row[arm + "_pair_JSD_nats"] = j
            row[arm + "_pair_dose_MSE"] = d
        row["signed_swapped_minus_full_JSD_nats"] = signed[g]
        pair_records[g], site_records[g] = row, site_rows
    full_dose_all = [pair_records[g]["full_pair_dose_MSE"] for g in ranked]
    for g in ranked:
        pair_records[g]["full_dose_percentile_all967_midrank"] = percentile(pair_records[g]["full_pair_dose_MSE"], full_dose_all)
    top_rows = [{"signed_contrast_rank": rank, **pair_records[g]} for rank, g in enumerate(top_groups, 1)]
    top_sites = [{"signed_contrast_rank": rank, **s} for rank, g in enumerate(top_groups, 1) for s in site_records[g]]
    write_tsv(HERE / "top20_pairs.tsv", top_rows)
    write_tsv(HERE / "top20_sites.tsv", top_sites)

    def describe(groups):
        pp = [pair_records[g] for g in groups]
        ss = [s for g in groups for s in site_records[g]]
        members = [cohort[r["accession"]].get("pfam", []) for r in pp]
        family_counts = Counter(f for entry in members for f in sorted(set(entry)))
        accessions = Counter(r["accession"] for r in pp)
        annotation_sets = Counter(r["pfam"] for r in pp)
        return {
            "pairs": len(pp), "sites": len(ss), "unique_proteins": len(accessions),
            "max_pairs_one_protein": max(accessions.values()),
            "no_pfam_pairs": sum(r["no_pfam_annotation"] for r in pp),
            "annotated_pairs": sum(not r["no_pfam_annotation"] for r in pp),
            "unique_pfam_ids": len(family_counts), "pfam_membership_counts": dict(family_counts.most_common()),
            "pfam_annotation_set_counts": dict(annotation_sets.most_common()),
            "max_one_pfam_membership_pairs": max(family_counts.values(), default=0),
            "sequence_length": stats([r["sequence_length"] for r in pp]),
            "source_nearest_terminal_distance_aa": stats([s["nearest_terminal_distance_aa"] for s in ss]),
            "pair_min_terminal_distance_aa": stats([r["pair_nearest_terminal_distance_aa"] for r in pp]),
            "source_within15_terminal_count": sum(s["within15_terminal"] for s in ss),
            "pairs_any_source_within15_terminal_count": sum(r["any_source_within15_terminal"] for r in pp),
            "full_dose_percentile_all967_midrank": stats([r["full_dose_percentile_all967_midrank"] for r in pp]),
            "pairs_full_dose_at_or_above_all967_95th_percentile": int(sum(r["full_pair_dose_MSE"] >= np.quantile(full_dose_all, .95) for r in pp)),
            "native_z": stats([s["native_z"] for s in ss]),
            "signed_swapped_minus_full_JSD_nats": stats([signed[g] for g in groups]),
            "arms": {arm: {"pair_JSD_nats": stats([r[arm + "_pair_JSD_nats"] for r in pp]),
                           "pair_dose_MSE": stats([r[arm + "_pair_dose_MSE"] for r in pp]),
                           "site_actual_delta": stats([s[arm + "_actual_delta"] for s in ss])} for arm in ARMS},
        }

    summaries = {"top20": describe(top_groups), "remaining947": describe(rest_groups), "all967": describe(ranked)}
    family_ids = sorted(set(summaries["all967"]["pfam_membership_counts"]))
    family_comparison = []
    for family in family_ids:
        a = summaries["top20"]["pfam_membership_counts"].get(family, 0)
        b = summaries["remaining947"]["pfam_membership_counts"].get(family, 0)
        family_comparison.append({"pfam": family, "top20_pairs": a, "top20_denominator": 20,
                                  "remaining947_pairs": b, "remaining947_denominator": 947,
                                  "all967_pairs": a + b, "all967_denominator": 967})
    family_comparison.sort(key=lambda r: (-r["top20_pairs"], -r["all967_pairs"], r["pfam"]))
    all_sum, top_sum = sum(signed.values()), sum(signed[g] for g in top_groups)
    tail = {"all967_signed_contrast_sum": all_sum, "top20_signed_contrast_sum": top_sum,
            "remaining947_signed_contrast_sum": all_sum - top_sum,
            "top20_fraction_of_all967_signed_sum": top_sum / all_sum if all_sum else None,
            "top20_fraction_of_all967_pairs": 20 / 967,
            "all967_positive_contrast_pairs": sum(v > 0 for v in signed.values()),
            "top20_arms": {arm: {"fraction_of_all967_JSD_sum": sum(pair_records[g][arm + "_pair_JSD_nats"] for g in top_groups) / sum(pair_records[g][arm + "_pair_JSD_nats"] for g in ranked),
                                 "fraction_of_all967_dose_sum": sum(pair_records[g][arm + "_pair_dose_MSE"] for g in top_groups) / sum(pair_records[g][arm + "_pair_dose_MSE"] for g in ranked)} for arm in ARMS}}
    output = {
        "analysis": "Post hoc sensitivity item 2; signed swapped-minus-full descending, no absolute-value selection",
        "selection": {"top_n": 20, "original_pair_denominator": 967, "remaining_pair_denominator": 947,
                      "tie_break": "group lexical ascending", "selected_groups_in_rank_order": top_groups},
        "definitions": {"native_z": "masked-context source post-GELU coordinate 814, block 6; visible source and masked source+1",
                        "actual_delta": "float64(actual float32 patched value) minus float64(native float32 z)",
                        "JSD": "canonical-20 softmax, natural log, frozen scoring formula; pair = arithmetic mean of two site JSDs",
                        "pair_dose_MSE": "arithmetic mean of actual squared displacement at the two sources",
                        "terminal_distance": "min(pos-1, sequence_length-pos); canonical full-sequence termini are distance 0; <=15 counts 15 as near",
                        "dose_percentile": "100*(number less + 0.5*number equal)/967; full pair dose MSE",
                        "pfam": "archived protein-level membership; multiple Pfam IDs count in each membership denominator; missing annotations separately reported"},
        "verification": {"status": "PASS", "pairs_joined": 967, "sites_joined": 1934,
                         "sequence_amino_acid_and_known_group_matches": True,
                         "buried_exposed_roles_match_frozen_RSA": True,
                         "max_abs_pair_JSD_difference_vs_frozen_table": max_jsd_error,
                         "max_abs_pair_dose_MSE_difference_vs_frozen_table": max_dose_error,
                         "restoration_bitwise_native": True,
                         "new_sequence_swapped_arm_key_matched_sites": len(keys)},
        "summaries": summaries, "tail_contribution": tail, "pfam_membership_comparison": family_comparison,
        "all967_full_dose_MSE_95th_percentile": float(np.quantile(full_dose_all, .95)),
        "limitations": [
            "Selected post hoc by the same signed output contrast being described; these descriptions are not independent confirmation or a replacement primary endpoint.",
            "One pair per known homology group and one representative protein means protein uniqueness is largely imposed by the original design.",
            "Archived Pfam annotations are protein-level, may be multiple or missing, and do not establish a shared site domain or functional mechanism; no functional interpretation was attempted.",
            "Terminal distances use the full canonical sequence, not the ends of resolved experimental-coordinate fragments. Strict main-clean selection already requires an observed +/-15 window, so this cohort is depleted for terminal sites by design.",
            "All magnitudes are frozen output-fidelity JSD in nats and actual intervention doses; this analysis does not measure physical structural change.",
            "The new sequence-swapped arm is included by the immutable group/accession/site keys; top20 membership is still selected exclusively by the original signed structure-swapped-minus-full contrast."
        ],
        "inputs": {str(path): sha(path) for path in (RESULTS / "writeback_groups_T.tsv", RESULTS / "writeback_T.npz",
                   STRUCTURE / "matched_pairs.tsv", PROTEINS, HERE / "sequence_swap_T.npz")},
        "script_sha256": sha(Path(__file__).resolve()),
        "outputs": [str(HERE / name) for name in ("top20_pairs.tsv", "top20_sites.tsv", "top20_profile.json")],
    }
    temp = HERE / "top20_profile.json.partial"
    temp.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(HERE / "top20_profile.json")
    print(json.dumps({"verification": output["verification"], "tail_contribution": tail,
                      "summary": {k: {field: v[field] for field in ("pairs", "sites", "unique_proteins", "no_pfam_pairs",
                         "unique_pfam_ids", "max_one_pfam_membership_pairs", "source_nearest_terminal_distance_aa",
                         "source_within15_terminal_count", "pairs_any_source_within15_terminal_count",
                         "full_dose_percentile_all967_midrank", "pairs_full_dose_at_or_above_all967_95th_percentile")}
                                  for k, v in summaries.items()},
                      "top20_pfam_membership_counts": summaries["top20"]["pfam_membership_counts"]}, indent=2))


if __name__ == "__main__":
    main()
