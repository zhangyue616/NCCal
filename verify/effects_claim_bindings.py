"""Distribution summaries for the complete estimable public protein labels."""
from pathlib import Path
import csv
import numpy as np


def bindings(root):
    root = Path(root)
    source = "results/protein/published_label_audit/label_audit/results/per_unit_results.tsv"
    with (root / source).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    public = [r for r in rows if r["family"] == "PUBLIC" and r["reference"] == "trained"
              and r["frame"] == "T" and r["formal_beta_gamma_eligible"] == "True"]
    if not public or any(not r["pair_accuracy"] or not r["auc_mean"] for r in public):
        raise ValueError("The complete estimable public-label distribution is required.")
    if len({r["unit_id"] for r in public}) != len(public):
        raise ValueError("Duplicate unit-label identities in the public distribution.")
    accuracy = np.array([float(r["pair_accuracy"]) for r in public])
    auc = np.array([float(r["auc_mean"]) for r in public])
    quartiles = np.quantile(accuracy, [.25, .75], method="linear")
    result = {}

    def put(ids, value, path, operation):
        for identifier in ids.split():
            result[identifier] = dict(value=float(value), source=path, operation=operation)

    prefix = "family=PUBLIC, reference=trained, frame=T, formal_beta_gamma_eligible=True; all eligible labels, irrespective of pass flags; "
    put("MS-2237 SI-2242", len(public), source, prefix + "count distinct unit-label identities.")
    put("MS-2238 SI-2243", np.median(accuracy), source, prefix + "median pair_accuracy; source accuracy already averages same-amino-acid pair credit equally across homology groups.")
    put("MS-2239", quartiles[0], source, prefix + "25th percentile of pair_accuracy, NumPy quantile method=linear.")
    put("MS-2240", quartiles[1], source, prefix + "75th percentile of pair_accuracy, NumPy quantile method=linear.")
    put("MS-2241", np.median(auc), source, prefix + "median auc_mean; each source value is group-equal annotation-versus-unannotated residue AUROC, a different estimand from pair_accuracy.")

    return result
