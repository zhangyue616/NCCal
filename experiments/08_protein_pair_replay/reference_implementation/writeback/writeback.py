"""Offline, fixed-coordinate model-output diagnostic; all writes confined to HERE."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
OFFSETS = [*range(-15, 0), *range(1, 16)]
GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
UNIT, BATCH = 1246, 8
METHODS = ("D_mean", "local", "residual_exchange")

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, obj):
    p = HERE / name
    temp = p.with_name(p.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(p)

def utc():
    return datetime.now(timezone.utc).isoformat()

def local_features(cases):
    x = np.zeros((len(cases), 30 * 21), dtype=np.float64)
    for i, c in enumerate(cases):
        seq, q = c["sequence"], c["q"]
        assert seq[q] == "A" and not set(seq) - set(AA)
        for j, off in enumerate(OFFSETS):
            k = q + off
            cat = AA.index(seq[k]) if 0 <= k < len(seq) else 20
            x[i, j * 21 + cat] = 1
    return x

def ridge(x, y, lam):
    xm, ym = x.mean(0), float(y.mean())
    xc, yc = x - xm, y - ym
    # Exactly mean-case squared loss + lambda * squared coefficient norm.
    a = xc.T @ xc + len(y) * lam * np.eye(x.shape[1])
    coef = cho_solve(cho_factor(a, lower=True, check_finite=False), xc.T @ yc, check_finite=False)
    return coef, ym - xm @ coef

def group_values(v, groups):
    return np.stack([np.mean(v[groups == g], axis=0) for g in sorted(set(groups))])

def prepare():
    if (HERE / "FIT_FREEZE.json").exists():
        raise RuntimeError("D fit already frozen; use --resume to avoid refitting")
    manifest_path = ROOT / "results/revision/protein_D/D_INPUT_MANIFEST.json"
    d = read(manifest_path)
    cases, pairs = d["cases"], d["pairs"]
    groups, pairfold = np.asarray(d["groups"]), np.asarray(d["fold_ids"])
    assert len(cases) == 864 and len(pairs) == 432 and len(set(groups)) == 421
    assert all(p["a_index"] == 2*i and p["b_index"] == 2*i+1 for i,p in enumerate(pairs))
    assert all(len(set(pairfold[groups == g])) == 1 for g in set(groups))
    lookup, logitslookup = {}, {}
    native_sources = []
    for rec in d["native_sources"]:
        p = Path(rec["path"])
        n = np.load(p, allow_pickle=False)
        assert n["z"].shape == (len(n["case_ids"]), 1280)
        for i, cid in enumerate(n["case_ids"].astype(str)):
            assert cid not in lookup
            lookup[cid] = n["z"][i].astype(np.float64)
            if "logits20" in n:
                logitslookup[cid] = n["logits20"][i].astype(np.float64)
        native_sources.append({"path": str(p), "sha256": sha(p)})
    assert set(lookup) == {c["case_id"] for c in cases}
    z = np.stack([lookup[c["case_id"]] for c in cases])
    y, x = z[:, UNIT], local_features(cases)
    casefold = np.repeat(pairfold, 2)
    candidate, all_oof = [], []
    for lam in GRID:
        oof = np.empty(len(y))
        for f in range(4):
            train, test = casefold != f, casefold == f
            coef, intercept = ridge(x[train], y[train], lam)
            oof[test] = x[test] @ coef + intercept
        mse_pair = ((oof-y)**2).reshape(-1,2).mean(1)
        candidate.append({"lambda": lam, "OOF_group_MSE": float(group_values(mse_pair, groups).mean()),
                          "OOF_case_MSE": float(np.mean((oof-y)**2))})
        all_oof.append(oof)
    best = min(range(len(GRID)), key=lambda i:(candidate[i]["OOF_group_MSE"], -GRID[i]))
    coef, intercept = ridge(x, y, GRID[best])
    np.savez_compressed(HERE / "D_FIT.npz", coef=coef, intercept=intercept, D_mean=y.mean(),
                        D_oof=all_oof[best], D_z=y, D_full_z=z)
    identity = read(ROOT / "results/pilot3a/INITIAL_LOCK.json")["model_identity"]
    actual = {name: sha(ROOT / "assets/esm2_8m" / name) for name in identity["files_sha256"]}
    assert actual == identity["files_sha256"], "model identity drift"
    result = {"status":"D_FROZEN_BEFORE_NEW_MODEL_OUTPUT", "created_utc":utc(),
              "D_pairs":432, "D_cases":864, "D_groups":421, "selected_lambda":GRID[best],
              "candidates":candidate, "D_mean":float(y.mean()), "D_SD":float(y.std()),
              "D_OOF_R2":float(1-np.mean((all_oof[best]-y)**2)/np.var(y)),
              "protocol_sha256":sha(HERE / "PROTOCOL.md"), "script_sha256":sha(__file__),
              "D_manifest_sha256":sha(manifest_path), "D_fit_sha256":sha(HERE / "D_FIT.npz"),
              "native_sources":native_sources, "model_identity":identity,
              "T_inputs": {"path":"data/revision/protein_fresh/SEALED_FRESH_CASES.json",
                           "sha256":sha(ROOT / "data/revision/protein_fresh/SEALED_FRESH_CASES.json")}}
    write("FIT_FREEZE.json",result)
    return d, logitslookup

class Runner:
    def __init__(self):
        import torch, transformers
        from transformers import AutoTokenizer, EsmForMaskedLM
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.12)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/esm2_8m", local_files_only=True)
        self.model = EsmForMaskedLM.from_pretrained(ROOT / "assets/esm2_8m", local_files_only=True,
                                                   attn_implementation="eager").cuda().float().eval()
        self.model.requires_grad_(False)
        c = self.model.config
        assert (c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.hidden_act) == (320,1280,6,"gelu")
        self.block = self.model.esm.encoder.layer[2]
        self.aa_ids = self.tokenizer.convert_tokens_to_ids(list(AA))
        assert len(set(self.aa_ids)) == 20
        self.count, self.case_forwards = 0, 0
        self.runtime = {"python":sys.version, "executable":sys.executable, "torch":torch.__version__,
                        "transformers":transformers.__version__, "device":torch.cuda.get_device_name(),
                        "dtype":"float32", "attention":"eager", "threads":4,
                        "hook":"esm.encoder.layer[2].intermediate forward output, post-GELU, q+1, 1246"}

    def run(self, cases, replacement=None):
        torch = self.torch
        tokens = self.tokenizer([c["sequence"] for c in cases], padding=True, return_tensors="pt", truncation=False)
        q = torch.tensor([c["q"]+1 for c in cases], device="cuda", dtype=torch.long)
        rows = torch.arange(len(cases), device="cuda")
        for i,c in enumerate(cases):
            expected = self.tokenizer.convert_tokens_to_ids(list(c["sequence"]))
            assert tokens["input_ids"][i,1:len(expected)+1].tolist() == expected
            assert int(tokens["attention_mask"][i].sum()) == len(expected)+2
        tokens = {k:v.cuda() for k,v in tokens.items()}
        tokens["input_ids"][rows,q] = self.tokenizer.mask_token_id
        assert ((tokens["input_ids"] == self.tokenizer.mask_token_id).sum(1) == 1).all()
        captured = {}
        def intermediate_hook(module, inputs, output):
            assert output.shape[2] == 1280
            captured["z"] = output[rows,q].detach().clone()
            if replacement is not None:
                patched = output.clone()
                patched[rows,q,UNIT] = torch.as_tensor(replacement,device="cuda",dtype=output.dtype)
                assert (patched != output).sum() <= len(cases)
                return patched
        def dense_hook(module, inputs, output):
            captured["fc2"] = output[rows,q].detach().clone()
        h = [self.block.intermediate.register_forward_hook(intermediate_hook),
             self.block.output.dense.register_forward_hook(dense_hook)]
        try:
            with torch.inference_mode():
                out = self.model(**tokens).logits[rows,q][:,self.aa_ids].detach().cpu().numpy()
                z = captured["z"].cpu().numpy()
                fc2 = captured["fc2"].cpu().numpy()
        finally:
            for handle in h:
                handle.remove()
        self.count += 1
        self.case_forwards += len(cases)
        assert np.isfinite(out).all() and np.isfinite(z).all() and np.isfinite(fc2).all()
        return out, z, fc2

def jsd(a,b):
    p,q = softmax(a.astype(np.float64),axis=1),softmax(b.astype(np.float64),axis=1)
    m = (p+q)/2
    return .5*np.sum(p*np.log(p/m)+q*np.log(q/m),axis=1)

def preflight(runner, d, logitslookup):
    from scipy.special import softmax
    fit = np.load(HERE / "D_FIT.npz", allow_pickle=False)
    t = read(ROOT / "data/revision/protein_fresh/SEALED_FRESH_CASES.json")
    n = np.load(ROOT / "results/revision/protein_T/T_NATIVE.npz", allow_pickle=False)
    assert n["case_ids"].astype(str).tolist() == [c["case_id"] for c in t["cases"]]
    # One shortest+one longest existing pair, no model-output-based selection.
    dp = [0, len(d["pairs"])-1]
    idx = np.asarray([2*p+j for p in dp for j in (0,1)])
    dc = [d["cases"][i] for i in idx]
    start = time.perf_counter()
    dl,dz,df = runner.run(dc)
    dr,drz,drf = runner.run(dc,dz[:,UNIT])
    assert np.array_equal(dl,dr) and np.array_equal(df,drf) and np.array_equal(dz,drz), "original restoration not exact"
    patched = dz[:,UNIT]+np.float32(.25)
    pl,pz,pf = runner.run(dc,patched)
    w = runner.block.output.dense.weight[:,UNIT].detach().cpu().numpy()
    response_error = np.max(np.abs(pf-df-(patched-dz[:,UNIT])[:,None]*w))
    assert response_error < 1e-5
    d_act_error = float(np.max(np.abs(dz-fit["D_full_z"][idx])))
    d_logit_error = float(np.max(np.abs(dl-np.stack([logitslookup[c["case_id"]] for c in dc]))))
    chosen = [min(range(len(t["pairs"])),key=lambda i:len(t["cases"][2*i]["sequence"])),
              max(range(len(t["pairs"])),key=lambda i:len(t["cases"][2*i]["sequence"]))]
    ti = np.asarray([2*p+j for p in chosen for j in (0,1)])
    tl,tz,tf = runner.run([t["cases"][i] for i in ti])
    tr,trz,trf = runner.run([t["cases"][i] for i in ti],tz[:,UNIT])
    assert np.array_equal(tl,tr) and np.array_equal(tf,trf)
    t_act_error = float(np.max(np.abs(tz-n["z"][ti])))
    assert max(d_act_error,d_logit_error,t_act_error) < 1e-3, "cache consistency failed"
    elapsed = time.perf_counter()-start
    result = {"status":"PASS", "created_utc":utc(), "runtime":runner.runtime,
              "D_full_z_cache_max_abs_error":d_act_error,"D_20AA_logit_cache_max_abs_error":d_logit_error,
              "T_full_z_cache_max_abs_error":t_act_error,"original_value_logit_max_abs_error":0.0,
              "original_value_JSD_nats":0.0,"dense_column_response_max_abs_error":float(response_error),
              "nonzero_test_JSD_nats_mean":float(jsd(dl,pl).mean()),
              "small_test_case_forwards":runner.case_forwards,"wall_seconds":elapsed,
              "peak_allocated_GB":runner.torch.cuda.max_memory_allocated()/1e9,
              "estimated_full_T_seconds_same_batch_work":elapsed/(runner.case_forwards)*2608*4,
              "T_cache_sha256":sha(ROOT / "results/revision/protein_T/T_NATIVE.npz")}
    write("PREFLIGHT.json",result)
    print(json.dumps(result),flush=True)
    return t, n["z"][:,UNIT].astype(np.float64)

def basic(v):
    return {"mean":float(np.mean(v)),"median":float(np.median(v)),"p05":float(np.quantile(v,.05)),
            "p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99)),"max":float(np.max(v))}

def r2(y,p,reference):
    den = float(np.mean((y-reference)**2))
    return float(1-np.mean((y-p)**2)/den) if den else None

def summarize(t, arr, elapsed, runner):
    fit = np.load(HERE / "D_FIT.npz", allow_pickle=False)
    z,pred = arr["native_z"],arr["prediction"]
    groups = np.asarray([p["group"] for p in t["pairs"]])
    div = arr["JSD"]
    pairdiv = div.reshape(-1,2,3).mean(1)
    gdiv = group_values(pairdiv,groups)
    rng=np.random.default_rng(20261118)
    draws = np.asarray([gdiv[rng.integers(0,len(gdiv),len(gdiv))].mean(0) for _ in range(2000)])
    output={}
    for k,m in enumerate(METHODS):
        output[m]={"case_JSD_nats":basic(div[:,k]),"equal_group_mean_JSD_nats":float(gdiv[:,k].mean()),
                   "cluster_bootstrap_95_CI_mean_JSD_nats":np.quantile(draws[:,k],[.025,.975]).tolist(),
                   "z_MSE":float(np.mean((arr["replacement"][:,k]-z)**2))}
    contrasts={}
    for a,b,name in ((1,0,"local_minus_D_mean"),(2,1,"residual_exchange_minus_local")):
        contrasts[name]={"mean_group_difference_JSD_nats":float((gdiv[:,a]-gdiv[:,b]).mean()),
                         "paired_cluster_bootstrap_95_CI":np.quantile(draws[:,a]-draws[:,b],[.025,.975]).tolist(),
                         "fraction_pairs_below_zero":float(np.mean(pairdiv[:,a] < pairdiv[:,b]))}
    dz=z[0::2]-z[1::2]
    dp=pred[0::2]-pred[1::2]
    D_delta=fit["D_z"][0::2]-fit["D_z"][1::2]
    result={"status":"COMPLETE_FIXED_COORDINATE_DIAGNOSTIC", "created_utc":utc(),
            "cohort":"existing old T; not a newly collected confirmatory test set", "pairs":1304,"cases":2608,
            "known_groups":len(gdiv),"unit":UNIT,"block_0based":2,
            "primary":"JSD over independently re-normalized 20 standard amino-acid logits; natural logs; nats",
            "JSD_range_nats":[0,float(np.log(2))],"absolute_output_divergence":output,"paired_contrasts":contrasts,
            "reconstruction":{"T_z_variance":float(np.var(z)),"T_z_MSE":float(np.mean((pred-z)**2)),
                              "T_R2_T_mean_reference":r2(z,pred,z.mean()),
                              "T_R2_D_mean_reference":r2(z,pred,float(fit["D_mean"])),
                              "paired_delta_MSE":float(np.mean((dp-dz)**2)),
                              "paired_delta_R2_T_mean_reference":r2(dz,dp,dz.mean()),
                              "paired_delta_R2_D_mean_reference":r2(dz,dp,D_delta.mean())},
            "residual_exchange_invariants":{
                "pair_residual_sum_max_abs_error":float(np.max(np.abs(arr["residual"].reshape(-1,2).sum(1)-(arr["replacement"][:,2]-pred).reshape(-1,2).sum(1)))),
                "pair_residual_squared_amplitude_max_abs_error":float(np.max(np.abs((arr["residual"]**2).reshape(-1,2).sum(1)-((arr["replacement"][:,2]-pred)**2).reshape(-1,2).sum(1))))},
            "full_T_cached_z1246_max_abs_error":float(np.max(np.abs(z-arr["cached_z"]))),
            "wall_seconds_full_T":elapsed,"total_batch_calls":runner.count,"total_case_forwards_including_preflight":runner.case_forwards,
            "peak_GPU_allocated_GB":runner.torch.cuda.max_memory_allocated()/1e9,
            "fit_freeze_sha256":sha(HERE / "FIT_FREEZE.json"),"protocol_sha256":sha(HERE / "PROTOCOL.md"),
            "bootstrap":{"seed":20261118,"resamples":2000,"unit":"known group; paired methods"},
            "scope_limits":["single historical selected coordinate; no unit search or matched-coordinate output reference",
                            "local additive reconstruction is not an unrestricted sequence model",
                            "oracle residual exchange tests placement conditional on the local predictor; not a predictive model",
                            "known-group uncertainty does not remove unknown homology or pretraining overlap",
                            "20-AA model-output fidelity is not biological function, mechanism or fitness"],
            "sequences_exported":0,"sequence_candidates_generated":0,"network_requests":0}
    assert max(result["residual_exchange_invariants"].values()) < 1e-10
    write("SUMMARY.json",result)
    return result

def execute(resume=False):
    startall=time.perf_counter()
    if resume:
        freeze=read(HERE / "FIT_FREEZE.json")
        assert freeze["protocol_sha256"] == sha(HERE / "PROTOCOL.md")
        assert freeze["D_fit_sha256"] == sha(HERE / "D_FIT.npz")
        d=read(ROOT / "results/revision/protein_D/D_INPUT_MANIFEST.json")
        logitslookup={}
        for src in d["native_sources"]:
            with np.load(src["path"],allow_pickle=False) as n:
                if "logits20" in n:
                    for cid,l in zip(n["case_ids"].astype(str),n["logits20"]): logitslookup[cid]=l
    else:
        d,logitslookup=prepare()
    fit=np.load(HERE / "D_FIT.npz",allow_pickle=False)
    runner=Runner()
    t,cached=preflight(runner,d,logitslookup)
    assert len(t["cases"]) == 2608 and len(t["pairs"]) == 1304
    assert all(p["a_index"] == 2*i and p["b_index"] == 2*i+1 for i,p in enumerate(t["pairs"]))
    pred=local_features(t["cases"]) @ fit["coef"]+fit["intercept"]
    arr={"prediction":pred,"cached_z":cached,"native_z":np.full(2608,np.nan),
         "JSD":np.full((2608,3),np.nan),"replacement":np.full((2608,3),np.nan),"residual":np.full(2608,np.nan)}
    completed=0
    if resume and (HERE / "NUMERIC_CHECKPOINT.npz").exists():
        old=np.load(HERE / "NUMERIC_CHECKPOINT.npz",allow_pickle=False)
        completed=int(old["completed"])
        assert np.array_equal(old["prediction"],pred)
        arr={k:old[k].copy() for k in arr}
    started=time.perf_counter()
    for i in range(completed,2608,BATCH):
        end=min(i+BATCH,2608)
        cases=t["cases"][i:end]
        natural,z,fc2=runner.run(cases)
        native=z[:,UNIT].astype(np.float64)
        assert np.max(np.abs(native-cached[i:end])) < 1e-3
        local=pred[i:end]
        residual=native-local
        swapped=local+residual.reshape(-1,2)[:,::-1].reshape(-1)
        replacements=np.stack([np.full(end-i,float(fit["D_mean"])),local,swapped],axis=1)
        div=np.empty((end-i,3))
        for k in range(3):
            changed,_,_=runner.run(cases,replacements[:,k])
            div[:,k]=jsd(natural,changed)
        arr["native_z"][i:end]=native
        arr["replacement"][i:end]=replacements
        arr["residual"][i:end]=residual
        arr["JSD"][i:end]=div
        if end % (BATCH*20) == 0 or end == 2608:
            temp=HERE / "NUMERIC_CHECKPOINT.tmp.npz"
            np.savez_compressed(temp,completed=end,**arr)
            temp.replace(HERE / "NUMERIC_CHECKPOINT.npz")
            progress={"status":"RUNNING" if end<2608 else "T_FORWARDS_COMPLETE", "completed_cases":end,
                      "remaining_case_forwards":(2608-end)*4,"wall_seconds":time.perf_counter()-started,
                      "GPU_peak_allocated_GB":runner.torch.cuda.max_memory_allocated()/1e9,"updated_utc":utc()}
            write("PROGRESS.json",progress)
            print(json.dumps(progress),flush=True)
    summary=summarize(t,arr,time.perf_counter()-started,runner)
    del runner.model
    runner.torch.cuda.empty_cache()
    write("GPU_RELEASED.json",{"status":"RELEASED","created_utc":utc(),"total_wall_seconds":time.perf_counter()-startall,
                               "allocated_GB_after_release":runner.torch.cuda.memory_allocated()/1e9})
    print(json.dumps({"status":summary["status"],"summary":str(HERE/"SUMMARY.json"),"GPU":"RELEASED"}),flush=True)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    execute(args.resume)
