"""Publish the frozen analysis once all new networks, rotations and directions are complete."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCH = ROOT.parent / "untrained"
EXPOSURE = ROOT.parent / "protein_extension"
ROT = EXPOSURE / "review_followup" / "rotation"
FINAL = ROOT / "scale_results"
SIZES = ("8m", "35m", "650m")
BLOCK = {"8m": 3, "35m": 6, "650m": 16}
K = 199


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def doc(path):
    return json.loads(read(path))


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def tsv(path, rows):
    if not rows:
        raise ValueError(f"Empty table {path}")
    keys = list(rows[0])
    with path.open("x", encoding="utf8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=keys, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def put(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


def big_null(index):
    if index == 1:
        return ROOT / "sealed" / "timing_650m_init001"
    return ROOT / "sealed" / "main_650m" / f"init_{index:03d}"


def big_artifacts(index):
    base=big_null(index)
    if index==1:
        return base
    receipt=doc(base/"COMPLETE.json")
    folder=Path(receipt["result_folder"]).resolve()
    if (not folder.is_relative_to(base.resolve()) or receipt["index"]!=index or
            receipt["seed"]!=20261118000+index or receipt["status"]!="COMPLETE"):
        raise ValueError(f"Invalid completion pointer for new network {index}")
    return folder


def direction_folder(size, index):
    if size == "650m" and index:
        return big_artifacts(index) / "block16" / "direction"
    return ROOT / "directions" / size / (f"init_{index:03d}" if index else "trained")


def missing_receipts():
    if doc(ROOT/"MAIN_GO.json")["K"] != K or doc(ROOT/"BUDGET.json")["selected_K_new"] != K:
        raise ValueError("Authorized K differs")
    required = [big_null(i)/"COMPLETE.json" for i in range(1,K+1)]
    required += [ROOT/"trained_650m"/"COMPLETE.json", ROOT/"directions"/"COMPLETE.json"]
    required += [direction_folder(size,i)/"COMPLETE.json" for size in ("8m","35m") for i in range(K+1)]
    required.append(direction_folder("650m",0)/"COMPLETE.json")
    return [p for p in required if not p.is_file()]


def pair_order():
    observations = []
    with (EXPOSURE/"structure"/"residue_features.tsv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            if row["split"]=="T" and row["main_clean"]=="True" and row["interface"]=="False":
                observations.append((row["group"],row["accession"],int(row["pos"])))
    observations.sort()
    if len(observations)!=173859 or len(set(observations))!=len(observations):
        raise ValueError("T residue roster changed")
    index={(acc,pos):i for i,(_,acc,pos) in enumerate(observations)}
    pairs=[r for r in table(EXPOSURE/"structure"/"matched_pairs.tsv") if r["split"]=="T"]
    if len(pairs)!=967 or len({r["group"] for r in pairs})!=967:
        raise ValueError("Pair roster changed")
    if not all(float(r["rsa1"])<=.10 and float(r["rsa2"])>=.40 for r in pairs):
        raise ValueError("Pair orientation differs from archive")
    buried=np.array([index[(r["accession"],int(r["pos1"]))] for r in pairs])
    exposed=np.array([index[(r["accession"],int(r["pos2"]))] for r in pairs])
    return pairs,buried,exposed


def act_counts(exposed,buried):
    if len(exposed)!=967 or len(buried)!=967 or not np.isfinite(exposed).all() or not np.isfinite(buried).all():
        raise ValueError("Invalid activation pair")
    return {"exposed_more_active":int(np.sum(exposed>buried)),"buried_more_active":int(np.sum(exposed<buried)),
            "activation_ties":int(np.sum(exposed==buried))}


def old_native_act(model,axis,pairs):
    base=EXPOSURE/"rsa"/"activations"/model/f"T-{axis}"
    buried,exposed=[],[]
    for row in pairs:
        z=np.load(base/(row["accession"]+".npy"),mmap_mode="r",allow_pickle=False)
        buried.append(float(z[int(row["pos1"])-1]))
        exposed.append(float(z[int(row["pos2"])-1]))
    return act_counts(np.array(exposed),np.array(buried))


def readme_link(sender, goal):
    return os.path.relpath(goal,sender).replace("\\","/")


def stats(values):
    if not len(values):
        return (None,None,None)
    arr=np.asarray(values,dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite distribution")
    return tuple(map(float,(arr.min(),np.median(arr),arr.max())))


def rank(value, nulls):
    valid=np.array([x for x in nulls if x is not None],dtype=np.float64)
    denominator=len(valid)+(value is not None)
    if value is None:
        return None,None,denominator
    return int(1+sum(valid>value)),int(1+sum(valid>=value)),denominator


def holm(rows):
    ordered=sorted(rows,key=lambda r:r["p_one_sided"])
    prior=0.0
    for k,row in enumerate(ordered):
        prior=max(prior,min(1.,(len(ordered)-k)*row["p_one_sided"]))
        row["p_holm"]=prior


def network_columns(result, counts):
    fields={"axis":int(result["selection"]["axis_zero_based"]),
            "development_cv_MAE":float(result["selection"]["cv_augmented_mae"])}
    for field in ("sequence_mae","augmented_mae","mae_gain","sequence_brier","augmented_brier","brier_gain"):
        fields[field]=float(result[field])
    for field in ("pair_correct","pair_ties","pair_incorrect"):
        fields[field]=int(result[field])
    for field in ("exposed_more_active","buried_more_active","activation_ties"):
        fields[field]=int(counts[field])
    if sum(fields[k] for k in ("pair_correct","pair_ties","pair_incorrect"))!=967:
        raise ValueError("Incorrect number of pair predictions")
    if sum(fields[k] for k in ("exposed_more_active","buried_more_active","activation_ties"))!=967:
        raise ValueError("Incorrect number of raw activation comparisons")
    for metric in ("mae","brier"):
        error=fields[f"sequence_{metric}"]-fields[f"augmented_{metric}"]-fields[f"{metric}_gain"]
        if abs(error)>1e-12:
            raise ValueError("Gain disagrees with absolute losses")
    return fields


def direction_columns(size,index,axis):
    base=direction_folder(size,index)
    path=base/"DIRECTION_SEALED.json"
    if not path.is_file() and (base/"COMPLETE.json").is_file():
        receipt=doc(base/"COMPLETE.json")
        possible=[p for p in receipt.get("sealed_files_sha256",{}) if Path(p).name=="DIRECTION_SEALED.json"]
        if not possible:
            possible=[str(p) for p in base.glob("attempt*/DIRECTION_SEALED.json")]
        if len(possible)!=1:
            raise ValueError(f"Cannot identify a single finished direction file in {base}")
        pointer=Path(possible[0])
        path=(base/pointer).resolve() if not pointer.is_absolute() else pointer.resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("Completed direction points outside its own folder")
    data=doc(path)
    if (data["selected_axis_zero_based"]!=axis or data["block_one_based"]!=BLOCK[size] or
            data["technical"]["masked_sites"]!=390 or data["technical"]["confirmation_fields_accessed"]):
        raise ValueError(f"Direction and development axis identity disagree: {path}")
    cols={}
    for metric in ("P1","P2"):
        effect=data[metric]
        rho=effect["rho"]
        if rho is None and not effect.get("reason"):
            raise ValueError("Missing reason for undefined rank correlation")
        if rho is not None and not (-1.000000000001 <= float(rho) <= 1.000000000001):
            raise ValueError("Invalid rank correlation")
        cols[metric]=rho
        cols[metric+"_undefined_reason"]=effect.get("reason") if rho is None else None
    cols["direction_source"]=readme_link(FINAL,path)
    return cols


def gather():
    pairs,buried,exposed=pair_order()
    trained_rot=doc(ROOT/"trained_650m"/"TRAINED_RESULTS.json")["blocks"]
    old_replay=doc(ARCH/"TRAINED_REPLAY.json")["models"]
    rows=[]
    for size in SIZES:
        for i in range(K+1):
            sens=None
            if size!="650m":
                if i:
                    folder=ARCH/size/f"init_{i:03d}"
                    name=folder/"RESULT.json"
                    r=doc(name)
                    z=np.load(folder/"T_selected.npy",mmap_mode="r",allow_pickle=False)
                    counts=act_counts(z[exposed],z[buried])
                else:
                    name=ARCH/"TRAINED_REPLAY.json"
                    r=old_replay[size]["results"]
                    counts=old_native_act(size,r["selection"]["axis_zero_based"],pairs)
            elif i:
                folder=big_artifacts(i)
                name=folder/"block16"/"RESULT.json"
                r=doc(name)
                counts=r["T_buried_exposed_activation_counts"]
                second=doc(folder/"block17"/"RESULT.json")
                if second["index"]!=i or second["block_one_based"]!=17:
                    raise ValueError("Mismatch in paired sensitivity network")
                sens=network_columns(second,second["T_buried_exposed_activation_counts"])
            else:
                name=ROOT/"trained_650m"/"TRAINED_RESULTS.json"
                r=trained_rot["16"]["native_result"]
                counts=r["T_buried_exposed_activation_counts"]
                second=trained_rot["17"]["native_result"]
                sens=network_columns(second,second["T_buried_exposed_activation_counts"])
            if i and (r["index"]!=i or r["seed"]!=20261118000+i):
                raise ValueError(f"State index/seed mismatch: {size} {i}")
            cols=network_columns(r,counts)
            row=dict(model=size,network_type="untrained" if i else "trained",index=i,
                     seed=20261118000+i if i else None,primary_block=BLOCK[size])
            row.update(cols)
            row.update(direction_columns(size,i,cols["axis"]))
            for k in cols:
                row["b17_"+k]=None if sens is None else sens[k]
            row["endpoint_source"]=readme_link(FINAL,name)
            rows.append(row)
    if len(rows)!=600 or len({(r["model"],r["index"]) for r in rows})!=600:
        raise ValueError("Expected exactly one row per physical network")
    return rows,trained_rot


def infer(networks,new_rotation):
    old=doc(ARCH/"AGGREGATE.json")["inference"]
    inf=[]
    dirs=[]
    per_size=[]
    for size in SIZES:
        by=[r for r in networks if r["model"]==size]
        trained=by[0]
        null=by[1:]
        if len(null)!=K or trained["network_type"]!="trained":
            raise ValueError("Incomplete network group")
        original=doc(EXPOSURE/"rsa"/"results"/size/"results.json") if size!="650m" else None
        correct_null=[r["pair_correct"] for r in null]
        lo,med,hi=stats(correct_null)
        per_size.append(dict(model=size,block=BLOCK[size],selected_axis=trained["axis"],correct=trained["pair_correct"],
                     incorrect=trained["pair_incorrect"],tie=trained["pair_ties"],T_pairs=967,
                     baseline_correct=537,untrained_minimum=int(lo),untrained_median=med,untrained_maximum=int(hi),
                     more_active_exposed=trained["exposed_more_active"],more_active_buried=trained["buried_more_active"],
                     equal_activation=trained["activation_ties"]))
        for b in ([16,17] if size=="650m" else [BLOCK[size]]):
            prefix="b17_" if b==17 else ""
            for met,ep in (("mae","primary"),("brier","secondary")):
                name=met+"_gain"
                gain=trained[prefix+name]
                vals=np.asarray([r[prefix+name] for r in null])
                lo,med,hi=stats(vals)
                ge=int(np.count_nonzero(vals>=gain))
                ci=(original["paired_95ci"] if met=="mae" else original["buried_exposed_pairs"]["paired_95ci"]) if original else new_rotation[str(b)]["basis_results"][0][ep]["CI95"]
                row=dict(model=size,block=b,endpoint=name,unit="1099 groups" if met=="mae" else "967 residue pairs",
                         K=K,trained_sequence_loss=trained[prefix+"sequence_"+met],
                         trained_augmented_loss=trained[prefix+"augmented_"+met],trained_gain=gain,
                         CI95_low=float(ci[0]),CI95_high=float(ci[1]),
                         untrained_minimum=lo,untrained_median=med,untrained_maximum=hi,
                         null_equal_or_higher=ge,p_one_sided=(ge+1)/(K+1),p_holm=None,
                         family="650M_primary_two_endpoints" if size=="650m" and b==16
                         else "descriptive_adjacent_block" if b==17
                         else "archived_35M_two_endpoints" if size=="35m" else "archived_8M_descriptive")
                if size!="650m":
                    ref=next(v for v in old if v["model"]==size and v["endpoint"]==name)
                    if (abs(row["trained_gain"]-ref["trained_gain"])>1e-12 or ref["p_raw"]!=row["p_one_sided"]
                            or ref["untrained_ge"]!=ge): raise ValueError("Archived inference changed")
                    row["p_holm"]=ref["p_holm"]
                inf.append(row)
        for metric in ("P1","P2"):
            vals=[r[metric] for r in null]
            finite=[v for v in vals if v is not None]
            lo,med,hi=stats(finite)
            start,end,denominator=rank(trained[metric],vals)
            dirs.append(dict(model=size,block=BLOCK[size],metric=metric,trained=trained[metric],
                trained_undefined_reason=trained[metric+"_undefined_reason"],
                rank_best=start,rank_worst=end,rank_denominator=denominator,
                nulls_finite=len(finite),nulls_undefined=K-len(finite),nulls_K=K,
                null_minimum=lo,null_median=med,null_maximum=hi))
    holm([r for r in inf if r["family"]=="650M_primary_two_endpoints"])
    return inf,dirs,per_size


def rotations(new):
    results=doc(ROT/"RESULTS.json")["models"]
    basis=[]
    comparisons=[]
    for size,b,source in [("8m",3,results["8m"]),("35m",6,results["35m"]),("650m",16,new["16"]),("650m",17,new["17"])]:
        projected=source["basis_results"]
        if len(projected)!=10 or projected[0]["basis"]!="native":
            raise ValueError("Expected native axis followed by nine rotations")
        for row in projected:
            for endpoint in ("primary","secondary"):
                item=row[endpoint]
                basis.append(dict(model=size,block=b,basis=row["basis"],axis=row["axis_zero_based"],
                    endpoint="mae_gain" if endpoint=="primary" else "brier_gain",groups=item["groups"],
                    gain=item["gain"],CI95_low=item["CI95"][0],CI95_high=item["CI95"][1]))
        for endpoint,c in source["comparison"].items():
            for target in ("native","mean_nine_rotated","native_minus_mean_rotated"):
                obj=c[target]
                comparisons.append(dict(model=size,block=b,endpoint="mae_gain" if endpoint=="primary" else "brier_gain",
                    contrast=target,gain=obj["gain"],CI95_low=obj["CI95"][0],CI95_high=obj["CI95"][1],
                    native_descriptive_rank_of_ten=c["native_descriptive_rank_of_ten"]))
    return basis,comparisons


def render_figure(out,rows,per_size):
    os.environ["MPLCONFIGDIR"]=str(out.parent/"plot_cache")
    if importlib.util.find_spec("matplotlib") is None:
        plotter=Path(r"external/Anaconda3/envs/neuroaudit/python.exe")
        payload={"rows":[{k:r[k] for k in ("model","index","pair_correct")} for r in rows],"models":per_size}
        subprocess.run([str(plotter),"-B",str(Path(__file__).resolve()),"--render-figure",str(out)],
                       input=json.dumps(payload),text=True,check=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,ax=plt.subplots(figsize=(7,3.65),layout="constrained")
    colors={"init":"#76848b","trained":"#bc5228","baseline":"#35505d"}
    for level,setting in enumerate(per_size):
        y=2-level
        values=[r["pair_correct"] for r in rows if r["model"]==setting["model"] and r["index"]>0]
        v,c=np.unique(values,return_counts=True)
        for value,total in zip(v,c):
            # Equal scores are stacked vertically, with identical bin widths at each size.
            offsets=(np.arange(total)-(total-1)/2)*min(.027,.52/max(total,1))
            ax.scatter(np.full(total,value),y+offsets,s=13,edgecolors="white",linewidths=.25,
                       color=colors["init"],alpha=.78,zorder=2)
        ax.scatter(setting["correct"],y,marker="D",s=66,color=colors["trained"],edgecolors="white",linewidths=.9,zorder=4)
        ax.annotate(str(setting["correct"]),(setting["correct"],y),xytext=(6,9),textcoords="offset points",
                    color="#803311",fontsize=10,weight="semibold")
    ax.axvline(537,color=colors["baseline"],linewidth=1,linestyle="--",zorder=0)
    ax.set_yticks([2,1,0],["ESM-2 8M","ESM-2 35M","ESM-2 650M"])
    ax.set_xlabel("Matched residue pairs ranked correctly (of 967)")
    ax.set_title("Solvent exposure from local sequence and one neuron",loc="left",fontsize=13,weight="semibold",pad=12)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="y",color="#ebebeb",linewidth=.65,zorder=0)
    ax.set_ylim(-.5,2.5)
    ax.margins(x=.09)
    handles=[Line2D([],[],marker="o",linestyle="none",markersize=5,color=colors["init"],label="199 untrained networks per size"),
             Line2D([],[],marker="D",linestyle="none",color=colors["trained"],label="Trained network"),
             Line2D([],[],linestyle="--",color=colors["baseline"],label="Sequence alone (537)")]
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(-.01,-.19),ncol=3,frameon=False,
              handlelength=1.4,columnspacing=1.1,fontsize=8)
    fig.savefig(out/"EXPOSURE_PAIRS.png",dpi=240)
    fig.savefig(out/"EXPOSURE_PAIRS.pdf")
    plt.close(fig)


def score_existing():
    if missing_receipts():
        raise SystemExit("Waiting for the complete frozen cohort; no scientific outputs were opened.")
    if FINAL.exists():
        if (FINAL/"SUMMARY.json").is_file(): return
        raise FileExistsError("Incomplete publication directory")
    rows,new=gather()
    inf,dirs,models=infer(rows,new)
    basis,comparisons=rotations(new)
    curr=[r for r in inf if r["family"]=="650M_primary_two_endpoints"]
    ref=[r for r in inf if r["model"]=="35m"]
    both=all(r["p_holm"]<=.05 for r in curr)
    lower=any(r["trained_gain"] < next(s["trained_gain"] for s in ref if s["endpoint"]==r["endpoint"]) for r in curr)
    decision="A" if both and not lower else "B" if both or any(r["p_holm"]<=.05 for r in curr) else "C"
    with tempfile.TemporaryDirectory(dir=ROOT,prefix="published_scale_") as tmp:
        stage=Path(tmp)/"report"
        stage.mkdir()
        for name,records in [("NETWORK_RESULTS",rows),("INFERENCE",inf),("MODELS",models),("DIRECTION_RANKS",dirs),
                ("ROTATED_BASES",basis),("ROTATION_COMPARISONS",comparisons)]:
            tsv(stage/(name+".tsv"),records)
        render_figure(stage,rows,models)
        put(stage/"SUMMARY.json",dict(finished_utc=datetime.now(timezone.utc).isoformat(),new_scale="650m",K=K,
            network_rows=len(rows),scales=SIZES,main_block=16,sensitivity_block=17,conclusion_case=decision,
            primary_both_Holm_pass=both,at_least_one_gain_smaller_than_35M=lower,
            inference=inf,models=models,directions=dirs,rotation_comparisons=comparisons,
            protocol_sha256=doc(ROOT/"FREEZE.json")["protocol_sha256"],pilot_counted_once=True))
        stage.rename(FINAL)
    print(f"Published {len(rows)} networks, {len(inf)} endpoint records and {len(dirs)} direction records.")


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--render-figure":
        payload=json.load(sys.stdin)
        render_figure(Path(sys.argv[2]),payload["rows"],payload["models"])
    else:
        score_existing()
