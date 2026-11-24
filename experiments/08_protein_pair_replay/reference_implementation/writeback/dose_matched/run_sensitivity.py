"""Post hoc native-anchored dose-matched diagnostic. No fitting, network or sequence export."""
import os
os.environ.update(OMP_NUM_THREADS="3",MKL_NUM_THREADS="3",OPENBLAS_NUM_THREADS="3",
                  HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import sys
sys.dont_write_bytecode=True
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import softmax

H=Path(__file__).resolve().parent
P=H.parent
R=P.parents[1]
AA="ACDEFGHIKLMNPQRSTVWY"
UNIT=1246

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write(name,value):
    p=H/name
    t=p.with_name(p.name+".tmp")
    t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    t.replace(p)
def jsd(a,b):
    p,q=softmax(a.astype(np.float64),axis=1),softmax(b.astype(np.float64),axis=1)
    m=(p+q)/2
    return .5*np.sum(p*np.log(p/m)+q*np.log(q/m),axis=1)

class Runner:
    def __init__(self,device):
        import torch,transformers
        from transformers import AutoTokenizer,EsmForMaskedLM
        torch.set_num_threads(3)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if device=="cuda": torch.cuda.set_per_process_memory_fraction(.12)
        self.torch,self.device=torch,device
        self.tok=AutoTokenizer.from_pretrained(R/"assets/esm2_8m",local_files_only=True)
        self.model=EsmForMaskedLM.from_pretrained(R/"assets/esm2_8m",local_files_only=True,
                                                attn_implementation="eager").to(device).float().eval()
        self.model.requires_grad_(False)
        self.block=self.model.esm.encoder.layer[2]
        self.aa=self.tok.convert_tokens_to_ids(list(AA))
        c=self.model.config
        assert (c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.hidden_act)==(320,1280,6,"gelu")
        self.runtime={"python":sys.version,"executable":sys.executable,"torch":torch.__version__,
                      "transformers":transformers.__version__,"device":device,"threads":3,"dtype":"float32"}
        if device=="cuda": self.runtime["GPU"]=torch.cuda.get_device_name()
        self.calls,self.forwards=0,0

    def run(self,cases,replacement=None):
        t=self.torch
        tokens=self.tok([c["sequence"] for c in cases],padding=True,return_tensors="pt",truncation=False)
        for i,c in enumerate(cases):
            ids=self.tok.convert_tokens_to_ids(list(c["sequence"]))
            assert tokens["input_ids"][i,1:len(ids)+1].tolist()==ids
            assert int(tokens["attention_mask"][i].sum())==len(ids)+2
        tokens={k:v.to(self.device) for k,v in tokens.items()}
        rows=t.arange(len(cases),device=self.device)
        q=t.tensor([c["q"]+1 for c in cases],device=self.device)
        tokens["input_ids"][rows,q]=self.tok.mask_token_id
        assert ((tokens["input_ids"]==self.tok.mask_token_id).sum(1)==1).all()
        captured={}
        def hook(module,inputs,output):
            assert output.shape[2]==1280
            captured["z"]=output[rows,q,UNIT].detach().clone()
            if replacement is not None:
                patched=output.clone()
                patched[rows,q,UNIT]=t.tensor(replacement,device=self.device,dtype=output.dtype)
                assert (patched!=output).sum()<=len(cases)
                return patched
        h=self.block.intermediate.register_forward_hook(hook)
        try:
            with t.inference_mode():
                logits=self.model(**tokens).logits[rows,q][:,self.aa].cpu().numpy()
                z=captured["z"].cpu().numpy().astype(np.float64)
        finally: h.remove()
        self.calls+=1
        self.forwards+=len(cases)
        return logits,z

def basic(v):
    return {"mean":float(v.mean()),"median":float(np.median(v)),"p95":float(np.quantile(v,.95)),
            "p99":float(np.quantile(v,.99)),"max":float(v.max())}

def execute(device):
    if (H/"STARTED.json").exists(): raise RuntimeError("Prior attempt exists; review recovery point before restart")
    parent_files=["PROTOCOL.md","writeback.py","FIT_FREEZE.json","D_FIT.npz","SUMMARY.json","PREFLIGHT.json","NUMERIC_CHECKPOINT.npz","ACCEPTANCE.json"]
    bindings={name:sha(P/name) for name in parent_files}
    old=read(P/"SUMMARY.json")
    freeze=read(P/"FIT_FREEZE.json")
    assert bindings["PROTOCOL.md"]==old["protocol_sha256"]
    assert bindings["D_FIT.npz"]==freeze["D_fit_sha256"]
    data=read(R/freeze["T_inputs"]["path"])
    assert sha(R/freeze["T_inputs"]["path"])==freeze["T_inputs"]["sha256"]
    cases,pairs=data["cases"],data["pairs"]
    assert len(cases)==2608 and len(pairs)==1304
    assert all(p["a_index"]==2*i and p["b_index"]==2*i+1 for i,p in enumerate(pairs))
    prior=np.load(P/"NUMERIC_CHECKPOINT.npz",allow_pickle=False)
    assert int(prior["completed"])==2608
    prediction=prior["prediction"].copy()
    cached_z=prior["native_z"].copy()
    write("STARTED.json",{"status":"POSTHOC_SENSITIVITY_FIXED_BEFORE_NEW_OUTPUT", "created_utc":utc(),
                          "disclosure":"specified after reading original three divergences and residual-exchange dose",
                          "parent_bindings":bindings,"script_sha256":sha(__file__),
                          "protocol_sha256":sha(H/"PROTOCOL_POSTHOC.md"),
                          "device":device,"new_fits":0,"case_forwards_planned":7824})
    runner=Runner(device)
    arrays={"JSD":np.full((2608,2),np.nan),"native_z":np.full(2608,np.nan),
            "local_request":prediction,"control_request":np.full(2608,np.nan)}
    batch=8 if device=="cuda" else 4
    start=time.perf_counter()
    original_error=0.
    for i in range(0,2608,batch):
        end=min(2608,i+batch)
        part=cases[i:end]
        natural,z=runner.run(part)
        local=prediction[i:end]
        delta=local-z
        control=z+delta.reshape(-1,2)[:,::-1].reshape(-1)
        l,_=runner.run(part,local)
        c,_=runner.run(part,control)
        if i==0:
            restored,rz=runner.run(part,z)
            assert np.array_equal(restored,natural) and np.array_equal(rz,z)
        arrays["JSD"][i:end]=np.stack([jsd(natural,l),jsd(natural,c)],axis=1)
        arrays["native_z"][i:end]=z
        arrays["control_request"][i:end]=control
        if end%(batch*40)==0 or end==2608:
            temp=H/"NUMERIC_CHECKPOINT.tmp.npz"
            np.savez_compressed(temp,completed=end,**arrays)
            temp.replace(H/"NUMERIC_CHECKPOINT.npz")
            progress={"completed_cases":end,"remaining_case_forwards":(2608-end)*3,
                      "wall_seconds":time.perf_counter()-start,"device":device,"updated_utc":utc()}
            write("PROGRESS.json",progress)
            print(json.dumps(progress),flush=True)
    elapsed=time.perf_counter()-start
    # All following work is CPU-only numeric acceptance and summary.
    runtime=runner.runtime
    forward_count=runner.forwards
    peak=runner.torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0
    del runner.block,runner.model
    if device=="cuda": runner.torch.cuda.empty_cache()
    write("MODEL_PHASE_COMPLETE.json",{"status":"MODEL_REFERENCES_RELEASED","created_utc":utc(),
                                        "wall_seconds":elapsed,"case_forwards":forward_count,
                                        "peak_GPU_allocated_GB":peak,"runtime":runtime})
    z=arrays["native_z"]
    requested_local=arrays["local_request"]-z
    requested_control=arrays["control_request"]-z
    actual_local=arrays["local_request"].astype(np.float32).astype(np.float64)-z
    actual_control=arrays["control_request"].astype(np.float32).astype(np.float64)-z
    dose_req=np.stack([(requested_local**2).reshape(-1,2).sum(1),(requested_control**2).reshape(-1,2).sum(1)],axis=1)
    dose_actual=np.stack([(actual_local**2).reshape(-1,2).sum(1),(actual_control**2).reshape(-1,2).sum(1)],axis=1)
    groups=np.asarray([p["group"] for p in pairs])
    unique=sorted(set(groups))
    group_dose_req=np.stack([dose_req[groups==g].sum(0) for g in unique])
    group_dose_actual=np.stack([dose_actual[groups==g].sum(0) for g in unique])
    tol=1e-12
    assert np.max(np.abs(dose_req[:,0]-dose_req[:,1]))<tol
    assert np.max(np.abs(group_dose_req[:,0]-group_dose_req[:,1]))<tol
    assert np.max(np.abs(dose_actual[:,0]-dose_actual[:,1]))<1e-6
    pairdiv=arrays["JSD"].reshape(-1,2,2).mean(1)
    gdiv=np.stack([pairdiv[groups==g].mean(0) for g in unique])
    rng=np.random.default_rng(20261118)
    draws=np.stack([gdiv[rng.integers(0,len(gdiv),len(gdiv))].mean(0) for _ in range(2000)])
    summary={"status":"COMPLETE_POSTHOC_NATIVE_ANCHORED_DOSE_MATCHED_SENSITIVITY", "created_utc":utc(),
             "posthoc_disclosure":"specified after reading original three JSD results and their intervention doses; parent results unchanged",
             "unit":UNIT,"block_0based":2,"pairs":1304,"cases":2608,"known_groups":len(unique),
             "primary_readout":"20-AA independently re-normalized JSD; natural logarithms; nats",
             "control_formula":"(z_a + f_b - z_b, z_b + f_a - z_a)",
             "absolute_output_divergence":{
                 name:{"case_JSD_nats":basic(arrays["JSD"][:,i]),"equal_group_mean_JSD_nats":float(gdiv[:,i].mean()),
                       "cluster_bootstrap_95_CI":np.quantile(draws[:,i],[.025,.975]).tolist()}
                 for i,name in enumerate(("local_same_runtime","native_anchored_dose_matched"))},
             "control_minus_local":{"paired_group_mean_JSD_nats":float((gdiv[:,1]-gdiv[:,0]).mean()),
                                    "paired_group_bootstrap_95_CI":np.quantile(draws[:,1]-draws[:,0],[.025,.975]).tolist(),
                                    "fraction_pairs_control_lower":float(np.mean(pairdiv[:,1]<pairdiv[:,0]))},
             "dose_acceptance":{"status":"PASS_ALL_PAIRS_AND_GROUPS_WITH_FLOAT32_ROUNDOFF_DISCLOSED",
                                "pair_count":len(pairs),"group_count":len(unique),
                                "requested_pairs_pass_1e-12":int(np.sum(np.abs(dose_req[:,0]-dose_req[:,1])<tol)),
                                "requested_groups_pass_1e-12":int(np.sum(np.abs(group_dose_req[:,0]-group_dose_req[:,1])<tol)),
                                "requested_pair_L2sq_max_abs_error":float(np.max(np.abs(dose_req[:,0]-dose_req[:,1]))),
                                "requested_group_L2sq_max_abs_error":float(np.max(np.abs(group_dose_req[:,0]-group_dose_req[:,1]))),
                                "actual_float32_pair_L2sq_max_abs_error":float(np.max(np.abs(dose_actual[:,0]-dose_actual[:,1]))),
                                "actual_float32_group_L2sq_max_abs_error":float(np.max(np.abs(group_dose_actual[:,0]-group_dose_actual[:,1]))),
                                "actual_mean_scalar_MSE_local":float(np.mean(actual_local**2)),
                                "actual_mean_scalar_MSE_control":float(np.mean(actual_control**2))},
             "original_value_JSD_nats":original_error,
             "same_runtime_local_vs_parent_JSD_max_abs_error":float(np.max(np.abs(arrays["JSD"][:,0]-prior["JSD"][:,1]))),
             "same_runtime_z_vs_parent_max_abs_error":float(np.max(np.abs(z-cached_z))),
             "bootstrap":{"seed":20261118,"resamples":2000,"unit":"known group, methods paired"},
             "runtime":runtime,"wall_seconds_model_phase":elapsed,"case_forwards_including_original_check":forward_count,
             "peak_GPU_allocated_GB":peak,"new_fits":0,"new_coordinates":0,
             "parent_files_unchanged":all(sha(P/name)==bindings[name] for name in parent_files),
             "limits":["post hoc diagnostic on existing T, not prospective confirmation",
                       "equal pair L2 dose does not imply equal output effect under nonlinear downstream map",
                       "model fidelity measurement only; no biological causation claim"],
             "sequence_generation":0,"sequences_exported":0,"network_requests":0}
    assert summary["parent_files_unchanged"]
    assert summary["same_runtime_z_vs_parent_max_abs_error"]<1e-3
    assert summary["same_runtime_local_vs_parent_JSD_max_abs_error"]<1e-7
    assert np.isfinite(arrays["JSD"]).all() and arrays["JSD"].min()>-1e-14 and arrays["JSD"].max()<=np.log(2)
    np.savez_compressed(H/"DOSE_ACCEPTANCE.npz",pair_requested_L2sq=dose_req,pair_actual_float32_L2sq=dose_actual,
                        group_requested_L2sq=group_dose_req,group_actual_float32_L2sq=group_dose_actual,
                        group_requested_equal=np.abs(group_dose_req[:,0]-group_dose_req[:,1])<tol)
    write("SUMMARY.json",summary)
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    execute(ap.parse_args().device)
