"""Post hoc description of when the four previously selected readouts make a distinction."""
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
UNITS=["E1_1900","Alley_UniRep341","Hallee_ESM2_786","Hallee_ESMC_634"]


def main():
    old=pd.read_csv(HERE/"F_pairs.tsv",sep="\t")
    strand=pd.read_csv(HERE/"ESMC634_F_helix_pairs.tsv",sep="\t")
    keys=["pair_id","group","record_id","aa"]
    table=old[keys+UNITS[:3]].merge(strand[keys+["correct"]],on=keys,validate="one_to_one").rename(
        columns={"correct":UNITS[3]})
    assert len(table)==len(old)==len(strand)==228
    assert set(table[UNITS].stack().unique())<={0.,.5,1.}
    table["all_distinct"]=(table[UNITS]!=.5).all(axis=1)
    rows=[]
    for unit in UNITS:
        for scope,mask in [("this readout distinguishes",table[unit]!=.5),("all four distinguish",table.all_distinct)]:
            kept=table.loc[mask]
            by_group=kept.groupby("group")[unit].mean()
            rows.append(dict(scope=scope,unit=unit,total_pairs=len(table),covered_pairs=len(kept),
                         total_groups=table.group.nunique(),covered_groups=len(by_group),
                         wins=int((kept[unit]==1).sum()),losses=int((kept[unit]==0).sum()),
                         coverage_fraction=len(kept)/len(table),pooled_accuracy=kept[unit].mean(),
                         covered_group_accuracy=by_group.mean()))
    table.to_csv(HERE/"F_tie_coverage_pairs.tsv",sep="\t",index=False)
    out=pd.DataFrame(rows);out.to_csv(HERE/"F_tie_coverage.tsv",sep="\t",index=False)
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
