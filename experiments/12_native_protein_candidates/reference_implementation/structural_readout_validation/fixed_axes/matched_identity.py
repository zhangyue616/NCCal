"""Sensitivity of the crossover to the amino-acid mix within each protein."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QUESTIONS = ("interior_vs_start", "helix_vs_strand")
UNITS = ("unit2192", "unit1900")


def main():
    scores = pd.read_csv(HERE/"F_pairs.tsv", sep="\t")
    annotations = pd.read_csv(HERE.parent/"fresh"/"pair_results.tsv", sep="\t")
    join = ["record_id", "group", "pair_id"]
    rows = scores.merge(annotations[join + ["aa"]], on=join, validate="one_to_one")
    assert len(rows) == len(scores)
    cell_keys = ["group", "aa"]
    shared = rows.groupby(cell_keys)["question"].nunique().eq(2)
    shared_cells = shared.loc[shared].reset_index()[cell_keys]
    keep = rows.merge(shared_cells, on=cell_keys, validate="many_to_one")
    assert len(keep) and set(keep["question"]) == set(QUESTIONS)
    cells = keep.groupby(cell_keys + ["question"]).agg(
        pairs=("pair_id", "size"), unit2192=("unit2192", "mean"), unit1900=("unit1900", "mean")
    ).reset_index()
    cells.to_csv(HERE/"F_matched_identity_cells.tsv", sep="\t", index=False)
    groups = cells.groupby(["group", "question"])[list(UNITS)].mean().unstack("question")
    rng = np.random.default_rng(91220)
    draws = rng.integers(len(groups), size=(9999, len(groups)))
    result = []
    for question, chosen, alternate in (
        (QUESTIONS[0], UNITS[0], UNITS[1]), (QUESTIONS[1], UNITS[1], UNITS[0])
    ):
        a, b = (groups[(unit, question)].to_numpy() for unit in (chosen, alternate))
        diff = a - b
        n = len(diff)
        leave_one_out = (diff.sum() - diff)/(n - 1) if n > 1 else np.array([np.nan])
        low, high = np.quantile(diff[draws].mean(axis=1), [.025, .975])
        result.append(dict(
            question=question,groups=n,shared_group_aa_cells=len(shared_cells),
            amino_acid_types=keep.aa.nunique(),pairs=int((keep.question == question).sum()),
            original_pairs=int((rows.question == question).sum()),selected=chosen,alternate=alternate,
            accuracy_selected=a.mean(),accuracy_alternate=b.mean(),difference=diff.mean(),
            ci_low=low,ci_high=high,loo_min=leave_one_out.min(),loo_max=leave_one_out.max()
        ))
    table = pd.DataFrame(result)
    table.to_csv(HERE/"F_matched_identity_summary.tsv", sep="\t", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
