"""Apply the fixed two-by-two F comparison to the already evaluated T and E groups."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "native_channels"
TRAINED = OLD / "e1" / "trained"


def main():
    lock = json.loads((TRAINED / "selection_lock.json").read_text())
    original = OLD / "cohort" / "pairs.tsv"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == lock["pairs_sha256"]
    df = pd.read_csv(TRAINED / "pair_index.tsv", sep="\t", keep_default_na=False)
    assert df.matrix_row.to_list() == list(range(len(df)))
    assert df.pair_id.to_list() == pd.read_csv(original, sep="\t").pair_id.to_list()
    recorded = pd.read_csv(TRAINED / "selected_pair_scores.tsv", sep="\t").set_index("pair_id")
    strict = df.same_helix_segment.astype(str).str.lower().eq("true")
    helix_strand = df.task.eq("helix_vs_strand")
    keep = df.split.isin(["T", "E"]) & (helix_strand | (df.task.eq("helix_start_vs_internal") & strict))
    use = df.loc[keep].copy()
    use["question"] = use.task.map({"helix_vs_strand": "helix_vs_strand", "helix_start_vs_internal": "interior_vs_start"})
    sign = np.where(use.question.eq("interior_vs_start"), -1, 1)
    for layer, unit, task in [(14, 2192, "helix_start_vs_internal"), (19, 1900, "helix_vs_strand")]:
        match = next(s for s in lock["selections"] if s["task"] == task)
        assert (match["layer0"], match["unit0"]) == (layer, unit)
        matrix = np.load(TRAINED / "pair_deltas" / f"layer_{layer:02d}.npy", mmap_mode="r")
        assert matrix.shape == (len(df), 2304)
        values = np.asarray(matrix[use.matrix_row,unit], dtype=float)
        assert np.isfinite(values).all()
        delta = sign * values
        use[f"unit{unit}"] = (delta > 0).astype(float) + .5 * (delta == 0)
        diag = use.loc[use.task == task]
        np.testing.assert_array_equal(diag[f"unit{unit}"], recorded.loc[diag.pair_id,"correct"])
    # A protein group must contribute at least one pair to each task in its own split.
    common = use.groupby(["split","group"]).question.transform("nunique").eq(2)
    pairs = use.loc[common, ["split","group","record_id","pair_id","question","aa","unit2192","unit1900"]]
    pairs.to_csv(HERE / "TE_fixed_pairs.tsv",sep="\t",index=False)
    groups = pairs.groupby(["split","question","group"]).agg(
        pairs=("pair_id","size"),unit2192=("unit2192","mean"),unit1900=("unit1900","mean")).reset_index()
    groups.to_csv(HERE / "TE_fixed_groups.tsv",sep="\t",index=False)
    output = []
    for (split, question), frame in groups.groupby(["split","question"]):
        first, other = ("unit2192","unit1900") if question=="interior_vs_start" else ("unit1900","unit2192")
        diff = (frame[first]-frame[other]).to_numpy()
        rng = np.random.default_rng(91220)
        draws = rng.integers(len(frame),size=(9999,len(frame)))
        lo,hi = np.quantile(diff[draws].mean(axis=1),[.025,.975])
        output.append(dict(split=split,question=question,groups=len(frame),pairs=int(frame.pairs.sum()),
                           unit2192=frame.unit2192.mean(),unit1900=frame.unit1900.mean(),
                           preassigned=first,difference=diff.mean(),low=lo,high=hi,
                           groups_favouring=int((diff>0).sum()),groups_tied=int((diff==0).sum())))
    pd.DataFrame(output).to_csv(HERE / "TE_fixed_summary.tsv",sep="\t",index=False)
    print(pd.DataFrame(output).to_string(index=False))


if __name__ == "__main__":
    main()
