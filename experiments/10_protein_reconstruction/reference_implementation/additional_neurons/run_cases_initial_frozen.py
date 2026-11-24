"""Prospectively fixed native cases; old scientific inputs are read only.

Reuses the established 814 implementation with explicit configuration changes,
and writes all new fits, predictions and forward checks within this directory.
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import types
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OLD = PROJECT / "protein_extension"
REF = PROJECT / "neuron_814"
LARGE = PROJECT / "scale" / "trained_650m"
CASES = {"8m_801": {"model":"8m", "axis":801, "block":3, "width":1280,
                     "config":(6,320,1280,"gelu"), "weights":str(PROJECT/"assets"/"esm2_8m")},
         "650m_4566": {"model":"650m", "axis":4566, "block":16, "width":5120,
                     "config":(33,1280,5120,"gelu"),
                     "weights":"external/caches/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"}}

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf8")
def plan_check():
    lock=json.loads((HERE/"PROTOCOL_FREEZE.json").read_text())
    assert lock["protocol_sha256"]==sha(HERE/"PROTOCOL.md"), "Protocol drift"
    assert lock["runner_sha256"]==sha(__file__), "Runner drift"
    for name,value in lock["reused_code_sha256"].items():
        assert sha(REF/name)==value, "Reused implementation drift"
    return lock["protocol_sha256"]

def load_module(name,path,replacements=()):
    source=path.read_text(encoding="utf8")
    for old,new in replacements:
        assert old in source, old
        source=source.replace(old,new)
    module=types.ModuleType(name)
    module.__file__=str(path)
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module

def setup(case):
    cfg=CASES[case]
    out=HERE/case
    out.mkdir(exist_ok=True)
    r=load_module("reference_reconstruction",REF/"reconstruction.py",[
        ("BOOT_SEED = 20261121", "BOOT_SEED = 20261122"),
        ("DRAWS = 2000", "DRAWS = 10000"),
        ("confidence = 1 - .05/3", "confidence = 1 - .05/6"),
        ("ci_simultaneous_98_3333", "ci_simultaneous_99_1667")])
    r.HERE=HERE; r.OUT=out; r.check_plan=plan_check

    def dataset(stage,split):
        rows,cohort=r.read_rows(split)
        assert (len(rows),len({x[2] for x in rows}))=={"D":(20974,325),"T":(173859,1099)}[split]
        keys=[(x[0],x[1],x[2]) for x in rows]
        with np.load(REF/"results"/f"geometry_{split}.npz",allow_pickle=False) as f:
            ix=r.alignment(r.site_keys(f["acc"],f["pos"],f["group"]),keys)
            counts=f["counts"][ix]
        assert counts.shape==(len(rows),80)
        assert np.array_equal(counts[:,40:].sum(1),[x[6] for x in rows])
        if stage=="B":
            keep=r.indices_for_b(rows,split)
            rows=[rows[i] for i in keep];keys=[keys[i] for i in keep];counts=counts[keep]
            with np.load(out/f"masked_native_{split}.npz",allow_pickle=False) as f:
                ix=r.alignment(r.site_keys(f["acc"],f["pos"],f["group"]),keys)
                y=f["z"][ix].astype(np.float64)
        elif cfg["model"]=="8m":
            y=r.activations(rows,"8m",split,axis=801).astype(np.float64)
        else:
            from exposure import development_signature
            signature=development_signature(rows)
            if split=="D":
                meta=json.loads((LARGE/"D_ROWS.json").read_text())
                assert np.array_equal(np.array(meta["rows"],dtype=str)[:,:3],np.array(rows,dtype=str)[:,:3])
                assert signature==meta["development_signature"]
                y=np.load(LARGE/"D_BLOCK16.npy",mmap_mode="r")[:,4566].astype(np.float64)
            else:
                meta=json.loads((LARGE/"T_EXTRACTION.json").read_text())
                assert signature==meta["row_signature"]
                y=np.load(LARGE/"T_BLOCK16_PROJECTIONS.npy",mmap_mode="r")[:,0].astype(np.float64)
            assert len(y)==len(rows)
        x=r.design_matrix(rows,cohort)
        if stage=="B":
            offset=16*len(r.ALPHABET)
            columns=np.r_[np.arange(offset),np.arange(offset+len(r.ALPHABET),x.shape[1])]
            x=x[:,columns]
        geo=sparse.csr_matrix(np.log1p(counts.astype(np.float64)))
        fields={"group":np.array([x[2] for x in rows]),"acc":np.array([x[0] for x in rows]),
                "pos":np.array([x[1] for x in rows],dtype=int)}
        assert y.shape==(len(rows),) and np.isfinite(y).all()
        return fields,x,geo,y
    r.dataset=dataset
    replacements=[("UNIT = 814",f"UNIT = {cfg['axis']}"),
        ("path = OLD / \"rsa\" / \"assets\" / \"esm2_35m\"",f"path = Path({cfg['weights']!r})"),
        ("(12,480,1920,\"gelu\")",repr(cfg["config"])),
        ("self.model.esm.encoder.layer[5]",f"self.model.esm.encoder.layer[{cfg['block']-1}]"),
        ("output.shape[-1] != 1920",f"output.shape[-1] != {cfg['width']}"),
        ("\"block_one_based\": 6",f"\"block_one_based\": {cfg['block']}")]
    n=load_module("reference_native",REF/"native_forward.py",replacements)
    n.HERE=HERE;n.OLD=OLD;n.OUT=out;n.TEMP=out/"tmp";n.TEMP.mkdir(exist_ok=True)
    n.plan_check=plan_check
    return r,n

def secondary(r):
    fields,x,geo,y=r.dataset("A","T")
    groups,index,count=np.unique(fields["group"],return_inverse=True,return_counts=True)
    with np.load(r.OUT/"fit_A.npz") as f:
        perm=np.arange(len(y))
        for group in groups:
            ix=np.where(fields["group"]==group)[0]
            assert np.all(np.diff(fields["pos"][ix])>0)
            perm[ix]=np.roll(ix,len(ix)//2)
        shifted=sparse.hstack((x[:,:-1],geo[perm],x[:,-1]),format="csr") @ f["coef_full"]
    with np.load(r.OUT/"site_predictions_A_T.npz") as f:
        native={k:f[k].copy() for k in ("z","seq","full")}
    yz=r.group_average(y,index,count);yz2=r.group_average(y*y,index,count)
    variance=yz2.mean()-yz.mean()**2
    draws=r.draws_for(len(groups))
    bvar=yz2[draws].mean(1)-yz[draws].mean(1)**2
    gm=r.group_average((y-shifted)**2,index,count)
    gf=r.group_average((y-native["full"])**2,index,count)
    cols={"group":groups,"shifted_structure_mse":gm,"full_mse":gf}
    r.save_tsv(r.OUT/"shifted_geometry_groups_T.tsv",cols)
    r.save_npz(r.OUT/"shifted_geometry_sites_T.npz",{**fields,"donor_index":perm,"shifted":shifted})
    pairix=r.indices_for_b(r.read_rows("T")[0],"T").reshape(-1,2)
    assert np.all(fields["group"][pairix[:,0]]==fields["group"][pairix[:,1]])
    dz=y[pairix[:,1]]-y[pairix[:,0]]
    paircols={"group":fields["group"][pairix[:,0]],"actual_delta_z_later_minus_earlier":dz}
    result={"groups":len(groups),"sites":len(y),"singleton_groups":int(np.sum(count==1)),
        "shifted_R2":r.estimate(1-gm.mean()/variance,1-gm[draws].mean(1)/bvar),
        "correct_minus_shifted_R2":r.estimate((gm-gf).mean()/variance,(gm-gf)[draws].mean(1)/bvar),
        "activation_order_pairs":{"denominator":967,"actual_ties":int(np.sum(dz==0))}}
    for key in ("seq","full"):
        dp=native[key][pairix[:,1]]-native[key][pairix[:,0]]
        paircols[key+"_delta_z"]=dp
        result["activation_order_pairs"][key]={"correct":int(np.sum(dp*dz>0)),
            "incorrect":int(np.sum(dp*dz<0)),"prediction_ties":int(np.sum(dp==0)),
            "accuracy_fixed_denominator":float(np.sum(dp*dz>0)/967)}
    r.save_tsv(r.OUT/"activation_order_pairs_T.tsv",paircols)
    r.save_json(r.OUT/"score_secondary.json",result)

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("action",choices=("freeze","fitA","scoreA","nativeD","fitB","nativeT","scoreB","writeback","scoreW","all"))
    parser.add_argument("case",nargs="?",choices=tuple(CASES))
    args=parser.parse_args()
    if args.action=="freeze":
        assert not (HERE/"PROTOCOL_FREEZE.json").exists(), "Do not replace freeze"
        write(HERE/"PROTOCOL_FREEZE.json",{"frozen_utc":utc(),"protocol_sha256":sha(HERE/"PROTOCOL.md"),
            "runner_sha256":sha(__file__),"cases":CASES,
            "reused_code_sha256":{name:sha(REF/name) for name in ("reconstruction.py","native_forward.py")},
            "existing_confirmation_reanalysis":True,"primary_family_size":6,"bootstrap_draws":10000})
        print("PROTOCOL FROZEN",flush=True);return
    plan_check()
    cases=tuple(CASES) if args.case is None else (args.case,)
    for case in cases:
        r,n=setup(case)
        actions=("fitA","nativeD","fitB","scoreA","nativeT","scoreB","writeback","scoreW") if args.action=="all" else (args.action,)
        for action in actions:
            print(utc(),case,action,flush=True)
            if action=="fitA":r.fit("A")
            elif action=="scoreA":r.score("A");secondary(r)
            elif action=="nativeD":n.native("D",8,False)
            elif action=="fitB":r.fit("B")
            elif action=="nativeT":n.native("T",8,True)
            elif action=="scoreB":r.score("B")
            elif action=="writeback":n.writeback(8,True)
            elif action=="scoreW":r.writeback_score()
        write(HERE/"STATUS.json",{"updated_utc":utc(),"case":case,"last_completed_action":actions[-1]})

if __name__=="__main__":
    with threadpool_limits(limits=3):main()
