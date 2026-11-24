"""Protein quantities independently anchored to pair data or frozen statistics.

Selectors name the task, frame, readout and statistic. No worksheet value or
numeric-nearest lookup participates in selecting evidence. Returned values use
the manuscript's display units, including percentages where indicated.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy.stats import rankdata


def bindings(root):
    root = Path(root)
    base = "results/protein/"
    out = {}

    def rows(relative):
        with (root / relative).open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file, delimiter="\t"))

    def js(relative):
        return json.loads((root / relative).read_text("utf8"))

    def put(number, value, source, operation, frozen=False):
        key = number if isinstance(number, str) else "SI-" + str(number)
        method = "FROZEN_STATISTIC" if frozen else "RECOMPUTED"
        out[key] = {"value": float(value), "source": source,
                    "operation": method + ": " + operation}

    def one(records, **criteria):
        matched = [row for row in records
                   if all(row.get(key) == value for key, value in criteria.items())]
        if len(matched) != 1:
            raise ValueError("Expected one semantic record: " + repr(criteria))
        return matched[0]

    def group_mean(records, value):
        groups = defaultdict(list)
        for row in records:
            groups[row["group"]].append(value(row))
        if not groups:
            raise ValueError("Empty group-level estimand")
        return np.mean([np.mean(v) for v in groups.values()])

    def credit(delta):
        return 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5

    # S-G16: frozen alanine-fitted transfer; one pair per group and identity.
    rel = base + "external_validation/alanine_transfer/PAIRED_SCORES.tsv"
    transfer = rows(rel)
    for identity, numbers in [("L", (1426, 1427, 1428)),
                              ("V", (1432, 1433, 1434))]:
        selected = [r for r in transfer
                    if r["identity"] == identity and r["strict_retained"] == "True"]
        for number, arm in zip(numbers[:2], ["L", "Lj"]):
            put(number, group_mean(selected, lambda r: (float(r[arm + "_p_a"]) - int(r["y_a"])) ** 2),
                rel, "direct-sequence-screened identity=" + identity +
                "; mean per-group squared error of " + arm + "_p_a against y_a; no refitting")
        put(numbers[2], group_mean(selected, lambda r: (float(r["L_p_a"]) - int(r["y_a"])) ** 2 -
                                 (float(r["Lj_p_a"]) - int(r["y_a"])) ** 2),
            rel, "direct-sequence-screened identity=" + identity +
            "; mean paired Brier(L)-Brier(Lj), preserving the frozen interaction predictor")
    rel = base + "external_validation/alanine_transfer/SUMMARY.json"
    for identity, numbers in [("L", (1430, 1431)), ("V", (1435, 1436))]:
        result = one(js(rel)["results"], frame="direct_sequence_screen", identity=identity)
        interval = result["comparisons"]["L_minus_Lj"]["CI97_5_simultaneous_percentile"]
        for number, side in zip(numbers, [0, 1]):
            put(number, interval[side], rel, "results[frame=direct_sequence_screen, identity=" +
                identity + "].comparisons.L_minus_Lj.CI97_5_simultaneous_percentile[" +
                str(side) + "]; archived simultaneous two-comparison interval", frozen=True)

    # Source814 uses the original linear reference; keep the two populations.
    for directory, point, low, high in [("results", 1446, 1449, 1450),
                                        ("results_sequence_screened", 1452, 1453, 1454)]:
        stem = base + "native_protein_candidates/cross_architecture/" + directory + "/"
        rel = stem + "native_pair_predictions.tsv"
        selected = [r for r in rows(rel) if r["architecture"] == "source_814" and r["task"] == "exposure"]
        put(point, group_mean(selected, lambda r: (float(r["local_probability"]) - int(r["target"])) ** 2 -
                              (float(r["augmented_probability"]) - int(r["target"])) ** 2),
            rel, "architecture=source_814, task=exposure; group-equal paired Brier reduction, population=" + directory)
        rel = stem + "SUMMARY.json"
        interval = js(rel)["source_814_exposure"]["ci95"]
        for number, side in [(low, 0), (high, 1)]:
            put(number, interval[side], rel, "source_814_exposure.ci95[" + str(side) +
                "], population=" + directory + "; frozen percentile interval", frozen=True)

    # S-G17: all observed T residues and the cysteine-only restriction differ.
    rel = base + "external_validation/disulfide/all_observed_readouts.tsv"
    observed = [r for r in rows(rel) if r["split"] == "T"]

    def auc(records, score):
        labels = np.array([int(r["endpoint"]) for r in records])
        ranks = rankdata([score(r) for r in records], method="average")
        positive = int(labels.sum())
        negative = len(labels) - positive
        if not positive or not negative:
            raise ValueError("AUROC requires both classes")
        return (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)

    put("MS-0585", auc(observed, lambda r: float(r["fixed361"])), rel,
        "split=T, all observed residues; pooled Mann-Whitney AUROC of fixed positive pole361, half credit for tied scores")
    put("MS-0586", auc([r for r in observed if r["aa"] == "C"], lambda r: float(r["fixed361"])),
        rel, "split=T, aa=C; pooled Mann-Whitney AUROC of fixed positive pole361, half credit for tied scores")
    put(1468, auc(observed, lambda r: float(r["aa"] == "C")), rel,
        "split=T, all observed residues; AUROC of residue-letter-C indicator against disulfide endpoint, distinct from fixed361")

    # Whole-protein conditional rank correlations use the original five columns.
    rel = base + "external_validation/protein_level_label/per_group.tsv"
    proteins = rows(rel)
    matrix = np.array([[float(r[k]) for k in ["act", "charge74", "sheet_fraction", "length", "propensity"]]
                       for r in proteins])
    ranked = rankdata(matrix, axis=0)
    for number, column in [(1505, 1), (1509, 2)]:
        controls = np.column_stack([np.ones(len(ranked)), np.delete(ranked, [0, column], axis=1)])
        target_pair = ranked[:, [0, column]]
        residual = target_pair - controls @ np.linalg.lstsq(controls, target_pair, rcond=None)[0]
        put(number, np.corrcoef(residual.T)[0, 1], rel,
            "Partial Spearman: rank act and " + ["act", "charge74", "sheet_fraction"][column] +
            "; residualize both on intercept and the other three ranked covariates; correlate residuals")
    rel = base + "external_validation/protein_level_label/RESULT.json"
    primary = js(rel)["primary"]
    for number, measure, side in [(1507, "charge74", 0), (1508, "charge74", 1), (1510, "sheet_fraction", 0)]:
        put(number, one(primary, measure=measure)["simultaneous_97_5"][side], rel,
            "primary[measure=" + measure + "].simultaneous_97_5[" + str(side) +
            "]; frozen paired-covariate bootstrap interval", frozen=True)
    rel = base + "external_validation/protein_level_label/application_predictions.tsv"
    predictions = rows(rel)
    for number, column in [(1516, "sequence_prediction"), (1517, "plus_204")]:
        put(number, group_mean(predictions, lambda r: abs(float(r["observed_sheet"]) - float(r[column]))),
            rel, "Mean held-out per-group absolute error of " + column + " against observed_sheet; no refitting")

    # S-G18: locked one-sided binomial result and group-weighted reconstruction.
    rel = base + "external_validation/validation/blind_benchmark/RESULT.json"
    real = js(rel)["real_structural_test"]
    if real["E_ties"] != 0:
        raise ValueError("The locked non-tied binomial denominator changed")
    n = int(real["E_groups"])
    successes = round(real["E_win"] * n)
    put(1547, sum(math.comb(n, k) for k in range(successes, n + 1)) / 2 ** n,
        rel, "Locked real-structural E test: exact one-sided Binomial(n=E_groups,p=0.5) upper tail at E_win*n; zero archived ties")
    rel = base + "external_validation/validation/known_basis/T_predictions.tsv"
    selected = [r for r in rows(rel) if r["arm"] == "hadamard16"]
    counts = Counter(r["group"] for r in selected)
    weights = np.array([1 / counts[r["group"]] for r in selected]); weights /= weights.sum()
    truth = np.array([float(r["source_original814"]) for r in selected])
    variance = np.dot(weights, (truth - np.dot(weights, truth)) ** 2)
    for number, column in [(1559, "selected_single_prediction"), (1561, "OMP_prediction")]:
        prediction = np.array([float(r[column]) for r in selected])
        put(number, 1 - np.dot(weights, (truth - prediction) ** 2) / variance, rel,
            "arm=hadamard16; 1-group-equal MSE(" + column + ")/group-equal global target variance; each group's total weight equal")
    rel = base + "external_validation/validation/decision_use/group_decisions.tsv"
    decisions = rows(rel)
    for number, frame in [(1593, "814 original"), (1595, "814 external strict")]:
        selected = [r for r in decisions if r["frame"] == frame]
        gain = np.array([(int(r["later_exposed"]) - float(r["sequence"])) ** 2 -
                         (int(r["later_exposed"]) - float(r["plus_neuron"])) ** 2 for r in selected])
        chosen = np.array([int(r["selected"]) for r in selected])
        put(number, np.mean(chosen * gain) - np.mean(chosen) * np.mean(gain), rel,
            "frame=" + frame + "; routed Brier gain minus equal-budget uniformly allocated expected gain, from frozen pair probabilities and selected mask")
    rel = base + "external_validation/validation/decision_use/policies.tsv"
    policies = rows(rel)
    for number, frame, side in [(1600, "814 original", "lower"), (1601, "814 original", "upper"),
                                (1602, "814 external strict", "lower"), (1603, "814 external strict", "upper")]:
        put(number, one(policies, frame=frame)["gain_routed_above_equal_budget_" + side], rel,
            "frame=" + frame + "; gain_routed_above_equal_budget_" + side + "; frozen conditional bootstrap interval", frozen=True)

    # S-G19: fixed E1 scope filters and genuinely paired Brier increments.
    stem = base + "native_protein_candidates/native_channels/"
    rel = stem + "analysis/inference_all.tsv"
    inference = rows(rel)
    for number, split, side in [(1632, "T", "lo"), (1633, "T", "hi"), (1636, "E", "lo"), (1637, "E", "hi")]:
        record = one(inference, model="E1", task="helix_start_vs_internal", split=split, scope="all", metric="single_channel")
        put(number, 100 * float(record[side]), rel, "model=E1, task=helix_start_vs_internal, split=" + split +
            ", scope=all, metric=single_channel; 100*" + side + "; frozen group-bootstrap accuracy interval", frozen=True)
    metadata = {r["pair_id"]: r for r in rows(stem + "cohort/pairs.tsv")}
    rel = stem + "analysis/locked_pair_scores.tsv"
    pairs = rows(rel)
    for number, split in [(1639, "T"), (1642, "E")]:
        selected = [r for r in pairs if r["model"] == "E1" and r["task"] == "helix_start_vs_internal"
                    and r["split"] == split and metadata[r["pair_id"]]["same_helix_segment"] == "True"]
        put(number, 100 * group_mean(selected, lambda r: credit(float(r["delta"]))), rel,
            "model=E1, task=helix_start_vs_internal, split=" + split +
            "; join pair_id to cohort/pairs.tsv, restrict same_helix_segment=True; mean pair tie-credit within groups, then equal group mean, percent")
    for number, split in [(1651, "T"), (1655, "E")]:
        selected = [r for r in pairs if r["model"] == "E1" and r["task"] == "helix_start_vs_internal" and r["split"] == split]
        put(number, group_mean(selected, lambda r: (1 - float(r["prob_seq"])) ** 2 - (1 - float(r["prob_plus"])) ** 2),
            rel, "model=E1, task=helix_start_vs_internal, split=" + split +
            "; paired Brier reduction from sequence and sequence-plus-channel probabilities, positive-minus-negative orientation; equal group mean")
    rel = stem + "analysis/inference_all.tsv"
    for number, split, side in [(1653, "T", "lo"), (1654, "T", "hi"), (1656, "E", "lo"), (1657, "E", "hi")]:
        put(number, one(inference, model="E1", task="helix_start_vs_internal", split=split, scope="all", metric="brier_gain")[side],
            rel, "model=E1, task=helix_start_vs_internal, split=" + split +
            ", scope=all, metric=brier_gain; " + side + "; frozen paired group-bootstrap gain interval", frozen=True)
    for filename, numbers in [("pair_results.tsv", [(1665, "T"), (1666, "E")]),
                               ("hallee786_pair_results.tsv", [(1667, "T"), (1668, "E")])]:
        rel = stem + "published/" + filename
        public_pairs = rows(rel)
        for number, split in numbers:
            selected = [r for r in public_pairs if r["task"] == "helix_vs_strand" and r["split"] == split]
            put(number, 100 * group_mean(selected, lambda r: credit(float(r["delta"]))), rel,
                "Externally fixed " + ("UniRep h341" if filename == "pair_results.tsv" else "ESM2 positive pole786") +
                ", task=helix_vs_strand, split=" + split + "; positive-minus-negative activation difference; half tie credit, mean within group then percent group mean")
    rel = stem + "terminal_specificity/pair_results.tsv"
    selected = [r for r in rows(rel) if r["task"] == "C_vs_interior4" and r["split"] == "E"]
    put(1677, 100 * group_mean(selected, lambda r: credit(float(r["signed_delta"]))), rel,
        "Post hoc task=C_vs_interior4, split=E; frozen signed_delta is interior-minus-C; half tie credit then equal group mean, percent")

    # S-G20: paired question-specific advantages and localization on fresh data.
    stem = base + "native_protein_candidates/structural_readout_validation/"
    rel = stem + "fixed_axes/F_groups.tsv"
    common = rows(rel)
    for number, question, first, second in [(1690, "interior_vs_start", "unit2192", "unit1900"),
                                            (1697, "helix_vs_strand", "unit1900", "unit2192")]:
        put(number, 100 * np.mean([float(r[first + "_" + question]) - float(r[second + "_" + question]) for r in common]),
            rel, "F common-group paired accuracy contrast, question=" + question +
            ";100*mean(" + first + "-" + second + "), preserving the same13 groups")
    rel = stem + "fixed_axes/F_summary.tsv"
    summary = rows(rel)
    for number, question, side in [(1693, "interior_vs_start", "ci_low"), (1694, "interior_vs_start", "ci_high"),
                                   (1699, "helix_vs_strand", "ci_low"), (1700, "helix_vs_strand", "ci_high")]:
        record = one(summary, metric="preassigned_advantage", question=question)
        put(number, 100 * float(record[side]), rel, "metric=preassigned_advantage, question=" + question +
            ";100*" + side + "; frozen paired common-group bootstrap interval in percentage points", frozen=True)
    rel = stem + "localization/fresh_by_window.tsv"
    windows = rows(rel)
    put(1707, 100 * group_mean(windows, lambda r: float(r["neuron_correct"])), rel,
        "F five-position exact localization by the fixed neuron rule; mean neuron_correct within group, then equal group mean, percent")
    put(1712, group_mean(windows, lambda r: float(r["sequence_nll"]) - float(r["combined_nll"])), rel,
        "F localization: paired sequence_nll-combined_nll within each frozen window; mean within group, then equal group mean")
    rel = stem + "localization/fresh_summary.tsv"
    record = one(rows(rel), measure="nll_gain")
    for number, side in [(1714, "ci_low"), (1715, "ci_high")]:
        put(number, record[side], rel, "measure=nll_gain; " + side + "; frozen paired group-bootstrap localization interval", frozen=True)
    return out


get_bindings = bindings
