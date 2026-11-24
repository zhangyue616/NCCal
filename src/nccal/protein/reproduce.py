"""Recompute protein manuscript quantities from frozen intermediate results."""
from __future__ import annotations
import argparse,json,sys,csv,hashlib,os
from pathlib import Path
import numpy as np
from .statistics import rows,bh,group_equal_r2,group_means,shared_intervals

ROOT=Path(__file__).resolve().parents[3]

def verify(experiment=None):
    base=ROOT/"results/protein";checks=[]
    def check(name,value,expected,tolerance=1e-10,source=""):
        checks.append(dict(name=name,value=float(value),expected=float(expected),tolerance=tolerance,passed=abs(float(value)-float(expected))<=tolerance,source=source))
    def active(number):return experiment is None or experiment.startswith(number)
    if active("08"):
        path=base/"protein_pair_replay/confirmation/T_PREDICTIONS.npz"
        with np.load(path) as data:
            check("masked_alanine_confirmation_groups",len(data["group_ids"]),1304,source=path.relative_to(ROOT).as_posix())
            for name in ["X","S","L"]:
                check("helix_strand_"+name+"_Brier_gain",np.mean(data[name+"_loss"]-data[name+"j_loss"]),np.mean(data["group_contributions"],axis=0)[["X","S","L"].index(name)],source=path.relative_to(ROOT).as_posix())
            z=data["j1246"].reshape(-1,2);case_ids=data["case_ids"].reshape(-1,2)
            strand_minus_helix=np.array([b-a if ":STRAND:" in second else a-b for (a,b),(first,second) in zip(z,case_ids)])
            check("neuron1246_strand_above_helix",np.sum(strand_minus_helix>0),870,source=path.relative_to(ROOT).as_posix())
            gain=data["pair_contributions"][:,0]
            check("composition_improvement_when_strand_higher",np.sum((strand_minus_helix>0)&(gain>0)),716,source=path.relative_to(ROOT).as_posix())
            check("composition_worse_when_helix_higher",np.sum((strand_minus_helix<0)&(gain<0)),383,source=path.relative_to(ROOT).as_posix())
    if active("09"):
        for model,correct in [("35m",632),("8m",532)]:
            p=base/f"protein_exposure/exposure/results/{model}/buried_exposed_group_errors.tsv";r=rows(p)
            check(model+"_exposure_pairs",len(r),967,source=p.relative_to(ROOT).as_posix())
            wins=sum((float(x["augmented_probability"])>.5)==bool(int(x["later_residue_exposed"])) for x in r)
            check(model+"_augmented_correct",wins,correct,source=p.relative_to(ROOT).as_posix())
        p=base/"protein_exposure/exposure/results/35m/buried_exposed_group_errors.tsv";r=rows(p)
        check("sequence_only_correct",sum((float(x["sequence_probability"])>.5)==bool(int(x["later_residue_exposed"])) for x in r),537,source=p.relative_to(ROOT).as_posix())
        p=base/"protein_exposure/exposure/results/35m/group_errors.tsv";r=rows(p)
        check("814_continuous_MAE_gain",np.mean([float(x["sequence_mae"])-float(x["augmented_mae"]) for x in r]),.001142,5e-7,p.relative_to(ROOT).as_posix())
    if active("10"):
        p=base/"protein_reconstruction/neuron_814/results/reconstruction_groups_A_T.tsv";r=rows(p)
        check("814_sequence_R2",group_equal_r2(r,"seq_mse"),.18006953789702118,source=p.relative_to(ROOT).as_posix())
        check("814_contact_R2",group_equal_r2(r,"full_mse"),.217,5e-4,p.relative_to(ROOT).as_posix())
    if active("11"):
        p=base/"protein_scale/scale/scale_results/INFERENCE.tsv";r=rows(p)
        for x in r:
            name=x["model"]+"_b"+x["block"]+"_"+x["endpoint"]
            check(name+"_loss_difference",float(x["trained_sequence_loss"])-float(x["trained_augmented_loss"]),float(x["trained_gain"]),source=p.relative_to(ROOT).as_posix())
            check(name+"_inclusive_reference",(1+float(x["null_equal_or_higher"]))/(1+float(x["K"])),float(x["p_one_sided"]),source=p.relative_to(ROOT).as_posix())
    if active("12"):
        p=base/"native_protein_candidates/native_channels/analysis/locked_pair_scores.tsv";r=rows(p)
        summary=rows(base/"native_protein_candidates/native_channels/analysis/inference_all.tsv")
        for model in ["E1","AMPLIFY"]:
            for task in ["helix_start_vs_internal","helix_vs_strand"]:
                for split in ["T","E"]:
                    selected=[x for x in r if x["model"]==model and x["task"]==task and x["split"]==split]
                    if not selected:continue
                    values={}
                    for x in selected:
                        delta=float(x["delta"]);credit=1 if delta>0 else 0 if delta<0 else .5
                        values.setdefault(x["group"],[]).append(credit)
                    expected=next((x for x in summary if x["model"]==model and x["task"]==task and x["split"]==split and x["scope"]=="all" and x["metric"]=="single_channel"),None)
                    if expected:check(model+"_"+task+"_"+split+"_group_accuracy",np.mean([np.mean(v) for v in values.values()]),float(expected["mean"]),source=p.relative_to(ROOT).as_posix())
    if active("13"):
        p=base/"external_validation/validation/blind_benchmark/simulation_rounds.tsv";r=rows(p)
        null=[x for x in r if x["condition"]=="null_pair_flips"]
        for key,expected in [("D_unadjusted",1000),("D_max_calibrated",4),("D_then_E_locked",30)]:check("random_labels_"+key,sum(int(x[key]) for x in null),expected,source=p.relative_to(ROOT).as_posix())
    if active("14"):
        p=base/"published_label_audit/label_audit/results/per_unit_results.tsv";r=rows(p)
        for frame,pair_count,strong_count in [("T",88,61),("NEW",77,52)]:
            public=[x for x in r if x["source"]=="Hallee" and x["frame"]==frame and x["formal_beta_gamma_eligible"]=="True"]
            check(frame+"_eligible_public_labels",len(public),98,source=p.relative_to(ROOT).as_posix())
            check(frame+"_same_amino_acid_labels",sum(x["same_AA_flag"]=="True" for x in public),pair_count,source=p.relative_to(ROOT).as_posix())
            check(frame+"_sequence_increment_labels",sum(x["strong_increment_flag"]=="True" for x in public),strong_count,source=p.relative_to(ROOT).as_posix())
            for family in ["public","own"]:
                selected=[x for x in r if x["frame"]==frame and (x["source"]=="Hallee")== (family=="public")]
                tests=[(x,key)for x in selected for key in ["pair","strong"]if x[key+"_p"]!=""]
                adjusted=bh([float(x[key+"_p"])for x,key in tests])
                for (x,key),q in zip(tests,adjusted):
                    check(frame+"_"+family+"_"+x["unit_id"]+"_"+key+"_BH",q,float(x[key+"_q"]),source=p.relative_to(ROOT).as_posix())
        by={(x["unit_id"],x["concept"],x["frame"]):x for x in r}
        identifiers=[json.loads(line)for line in(ROOT/"data/protein/sequence_identifiers.jsonl").read_text("utf8").splitlines()if line.strip()]
        universes={f:sorted({x["group"]for x in identifiers if (x["split"]=="T"if f=="T"else x["split"]not in["D","T"])})for f in["T","NEW"]}
        native=json.loads((base/"published_label_audit/native_statistics.json").read_text("utf8"))
        for item in native:
            for frame in ["T","NEW"]:
                groupfile=base/"published_label_audit/per_group"/item["job"]/(frame+"_group_effects.tsv")
                if not groupfile.exists():continue
                values=rows(groupfile);target=by[(item["unit_id"],item["concept"],frame)]
                for column,estimate in [("paired_credit","pair_accuracy"),("linear_Brier_gain","linear_mean"),("strong_Brier_gain","strong_mean")]:
                    if target[estimate]:check(item["job"]+"_"+frame+"_"+column,np.mean([float(x[column]) for x in values]),float(target[estimate]),source=groupfile.relative_to(ROOT).as_posix())
                matrix=np.array([[float(x["paired_credit"]),float(x["linear_Brier_gain"]),float(x["strong_Brier_gain"])]for x in values]);ci=shared_intervals(matrix,[x["group"]for x in values],universes[frame])
                for j,key in enumerate(["pair","linear","strong"]):
                    for side,ix in[("lo",0),("hi",1)]:
                        if target[key+"_CI_"+side]:check(item["job"]+"_"+frame+"_"+key+"_CI_"+side,ci[ix,j],float(target[key+"_CI_"+side]),source=groupfile.relative_to(ROOT).as_posix())
    return dict(status="PASS" if checks and all(x["passed"] for x in checks) else "FAIL",checks=checks,check_count=len(checks),failed_count=sum(not x["passed"] for x in checks),scope="Stored-result arithmetic and row-level aggregates; model forwards are a separate mode.")

def main(experiment=None):
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["verify","forward-subset"],default="verify");parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",type=Path);parser.add_argument("--sequences",type=Path);parser.add_argument("--dependency-dir",type=Path,action="append",default=[]);parser.add_argument("--device",choices=["cpu","cuda"],default="cpu");args=parser.parse_args()
    if args.mode=="verify":result=verify(experiment)
    else:
        if experiment is None:parser.error("Use an experiment entry point for forward-subset.")
        if args.weights is None or args.sequences is None:parser.error("forward-subset requires explicit --weights and --sequences.")
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
        from .models import ProteinModel,check_fixture
        cases=json.loads((ROOT/"data/protein/forward_subset_identifiers.json").read_text("utf8"));fixture=next(c for c in cases if c["experiment"]==experiment)
        sequence=None
        for line in args.sequences.read_text("utf-8-sig").splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get("accession",row.get("id"))==fixture["accession"]:sequence=row["sequence"];break
        if sequence is None:raise ValueError("Required accession is absent from the explicitly supplied sequence file.")
        if hashlib.sha256(sequence.encode()).hexdigest()!=fixture["sequence_sha256"]:raise ValueError("Sequence identity differs from the frozen forward subset.")
        fixture["sequence"]=sequence;model=ProteinModel(fixture["model"],args.weights,args.dependency_dir,args.device)
        try:result=check_fixture(model,fixture)
        finally:model.close()
        result["tested_scope"]="One original input and the explicitly listed model/readout operations; no claim of a full experiment rerun."
        result["device"]=args.device
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),"utf8");print(json.dumps({k:v for k,v in result.items() if k!="checks"}));return 0 if result["status"]=="PASS" else 1

if __name__=="__main__":raise SystemExit(main())
