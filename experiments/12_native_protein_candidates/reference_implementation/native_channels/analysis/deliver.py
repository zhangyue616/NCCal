import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from inference import ROOT, OUT, TASKS, SPLITS, by_group, group_uncertainty, read_tsv, win

def holm(p):
    vals=np.asarray(p,dtype=float)
    order=np.argsort(vals)
    result=np.empty_like(vals)
    result[order]=np.maximum.accumulate(np.minimum(1., vals[order]*np.arange(len(vals),0,-1)))
    return result

def collect_results():
    raw=pd.concat([read_tsv(OUT/m/"summary.tsv") for m in ["AMPLIFY","E1"]],ignore_index=True)
    raw["p_Holm"]=np.nan
    for split in ["T","E"]:
        for metric in ["single_channel","brier_gain"]:
            use=(raw.split==split)&(raw.scope=="all")&(raw.metric==metric)
            assert use.sum()==4
            raw.loc[use,"p_Holm"]=holm(raw.loc[use,"p"])
    raw.to_csv(OUT/"inference_all.tsv",sep="\t",index=False)
    native=raw[(raw.scope=="all")&(raw.metric=="single_channel")].copy()
    native["kind"]="new_screen"
    locks={}
    pair_files=[]
    for m in ["AMPLIFY","E1"]:
        pairs=read_tsv(OUT/m/"paired_predictions.tsv")
        pair_files.append(pairs)
        for (_,t),g in pairs.groupby(["model","task"]):
            lk=g[["block_1","channel_0","sign"]].drop_duplicates()
            assert len(lk)==1
            locks[m,t]=lk.iloc[0].to_dict()
    for col in ["block_1","channel_0","sign"]:
        native[col]=[locks[m,t][col] for m,t in zip(native.model,native.task)]
    records=[]
    for m, unit, b, file in [("UniRep-1900",341,1,"group_results.tsv"),("ESM-2 8M",786,5,"hallee786_group_results.tsv")]:
        base=read_tsv(ROOT/"published"/file)
        for (sp,t),d in base.groupby(["split","task"]):
            uu=group_uncertainty(d.fraction_positive,.5)
            if t!=TASKS[0]:
                uu["p"]=None
            records.append(dict(kind="published_fixed" if t==TASKS[0] else "historical_secondary",
                 model=m,task=t,split=sp,scope="all",metric="single_channel",groups=len(d),pairs=d.n_pairs.sum(),
                 pooled_mean=np.average(d.fraction_positive,weights=d.n_pairs),block_1=b,channel_0=unit,
                 sign=1,p_Holm=None,**uu))
    hist=pd.DataFrame(records)
    for sp in ["T","E"]:
        ix=(hist.split==sp)&(hist.kind=="published_fixed")
        hist.loc[ix,"p_Holm"]=holm(hist.loc[ix,"p"])
    stats=pd.concat([hist,native],ignore_index=True)
    stats.to_csv(OUT/"coordinate_confirmations.tsv",sep="\t",index=False)
    joined=pd.concat(pair_files,ignore_index=True)
    joined[["pair_id","group","model","task","split","block_1","channel_0","sign","delta",
            "prob_seq","prob_plus","brier_seq","brier_plus"]].to_csv(OUT/"locked_pair_scores.tsv",sep="\t",index=False)
    return raw,stats

def reference_summary(stats):
    refs=[]
    for m in ["AMPLIFY","E1"]:
        rot=read_tsv(OUT/m/"rotation_groups.tsv")
        rot=rot.groupby(["seed","split","task"],as_index=False).accuracy.mean()
        rot["family"]="orthogonal basis"
        rot["model"]=m
        refs.append(rot)
        for seed in range(88001,88010):
            if m=="AMPLIFY":
                q=read_tsv(ROOT/f"amplify/results/{seed}/signed_pairs.tsv")
                q["accuracy"]=win(q.delta)
                groups=q.groupby(["split","task","group"],as_index=False).accuracy.mean()
            else:
                groups=read_tsv(ROOT/f"e1/random_{seed}/selected_group_scores.tsv")
            rr=groups.groupby(["split","task"],as_index=False).accuracy.mean()
            rr["seed"],rr["family"],rr["model"]=seed,"untrained initialization",m
            refs.append(rr)
    refs=pd.concat(refs,ignore_index=True)
    refs.to_csv(OUT/"reference_scores.tsv",sep="\t",index=False)
    rows=[]
    for (m,t,sp,fam),g in refs.groupby(["model","task","split","family"]):
        acc=stats.loc[(stats.model==m)&(stats.task==t)&(stats.split==sp),"mean"].item()
        assert len(g)==9
        rows.append(dict(model=m,task=t,split=sp,reference=fam,trained_native=acc,n=9,
                 reference_min=g.accuracy.min(),reference_median=g.accuracy.median(),reference_max=g.accuracy.max(),
                 n_strictly_below=int((g.accuracy<acc-1e-12).sum()),
                 n_tied=int((np.abs(g.accuracy-acc)<=1e-12).sum())))
    return pd.DataFrame(rows)

def plot(stats,ref):
    models=["UniRep-1900","ESM-2 8M","AMPLIFY","E1"]
    labels={"UniRep-1900":"UniRep 1900","ESM-2 8M":"ESM-2 8M","AMPLIFY":"AMPLIFY 120M","E1":"Profluent E1-150M"}
    plt.rcParams.update({"font.family":"Arial","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,2,figsize=(9.65,3.33),gridspec_kw={"width_ratios":[1.42,1]},layout="constrained")
    for ax,t,title,mods in zip(axs,TASKS,["Helix versus strand","Helix start versus interior"],[models,models[2:]]):
        ys=np.arange(len(mods))[::-1]
        ylabels=[]
        for m,y in zip(mods,ys):
            level=stats[(stats.model==m)&(stats.task==t)]
            best=level.iloc[0]
            desc=f"h[{int(best.channel_0)}]" if m=="UniRep-1900" else f"B{int(best.block_1)} · {int(best.channel_0)}"
            ylabels.append(f"{labels[m]}\n{desc}")
            for sp,offset,color,mark in [("T",.11,"#267992","o"),("E",-.11,"#BD6534","s")]:
                row=level[level.split==sp].iloc[0]
                ax.errorbar(row["mean"],y+offset,xerr=[[row["mean"]-row.lo],[row.hi-row["mean"]]],
                         lw=1.1,marker=mark,ms=5,color=color,capsize=2,zorder=3)
        ax.set_yticks(ys,ylabels);ax.tick_params(axis="y",length=0)
        ax.set_ylim(-.43,len(mods)-.55)
        ax.set_xlim(.34,.99);ax.set_xticks([.4,.5,.6,.7,.8,.9])
        ax.axvline(.5,ls="--",lw=1,color="#999999")
        ax.set_xlabel("Group-averaged pair accuracy")
        ax.set_title(title,loc="left",fontweight="bold")
        for edge in ["top","right","left"]:
            ax.spines[edge].set_visible(False)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker="o",ls="",color="#267992",label="Confirmation"),
             Line2D([0],[0],marker="s",ls="",color="#BD6534",label="External replication")]
    fig.legend(handles=handles,loc="outside lower center",ncol=2,frameon=False)
    for ext in ["png","pdf"]:
        fig.savefig(OUT/f"coordinate_confirmations.{ext}",dpi=220,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    raw,stats=collect_results()
    ref=reference_summary(stats)
    ref.to_csv(OUT/"reference_comparison.tsv",sep="\t",index=False)
    plot(stats,ref)
