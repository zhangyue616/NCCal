"""Fit with development labels only. Apply locked coefficients on external pairs in a separate stage."""
import os
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="4"
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from time import monotonic
import numpy as np
from analysis import Sites,PairModel,GridRegression,match_channel,slope,rotation,ROOT
from cohorts import ARCHIVE, write_tsv
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/"models"))
FITS=ROOT/"fits"
PRED=ROOT/"predictions"

def write_npz(path,**values):
    temp=path.with_suffix(path.suffix+".partial")
    with temp.open("wb") as handle: np.savez(handle,**values)
    temp.replace(path)

def saved(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}

def guard():
    spec=json.loads((ROOT/"FROZEN.json").read_text(encoding="utf-8"))
    for file,h in spec["files"].items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=h:
            raise ValueError("Protocol or development manifest drift: "+file)
    for p in (FITS,PRED): p.mkdir(exist_ok=True)

def get_acts(data,architecture,initialization,seed=None,development=False,contact_only=False):
    width=1920 if architecture=="esm2_35m" else 512
    values=np.zeros((len(data.keys),width),dtype=np.float32)
    if contact_only:
        idx=set(data.pair_idx["contact"].flatten().tolist())
        byacc={a:[(i,p) for i,p in points if i in idx] for a,points in data.byacc.items()}
        byacc={a:points for a,points in byacc.items() if points}
    else: byacc=data.byacc
    if development and architecture=="esm2_35m" and initialization=="trained":
        cache=ARCHIVE.parent/"rsa"/"activations"/"35m"/"D-all"
        for a,points in byacc.items():
            i,pos=np.array(points).T
            arr=np.load(cache/(a+".npy"),mmap_mode="r",allow_pickle=False)
            if arr.shape!=(len(data.seq[a]["sequence"]),width): raise ValueError("Archived ESM activation shape")
            values[i]=arr[pos-1]
    else:
        from native_extractors import NativeProteinExtractor
        model=NativeProteinExtractor(architecture,initialization,seed=seed,device="cuda")
        accs=sorted(byacc,key=lambda a:(len(data.seq[a]["sequence"]),a))
        for n in range(0,len(accs),8):
            batch=accs[n:n+8]
            acts=model.extract([data.seq[a]["sequence"] for a in batch],
                               positions=[[p for _,p in byacc[a]] for a in batch])
            for a,arr in zip(batch,acts):
                values[[i for i,_ in byacc[a]]] = arr
        del model
    if not np.isfinite(values).all(): raise ValueError("Nonfinite model signals")
    return values

def identity(stage,idx):
    return stage if idx is None else f"{stage}_{idx:03d}"

def pack_source(data,values,grid,exposure):
    raw=values[:,814:815]
    res,coef=grid.fit(raw)
    res=res[:,0]
    z=data.pair_difference(res,"exposure")
    source_coef=slope(z,exposure.y-exposure.oof)
    write_npz(FITS/"source.npz",raw_axis=814,grid_coef=coef[:,0],grid_oof=res[data.grid_idx],brier_slope=source_coef)

def fit_net(data,arch,stage,idx,values,grid,contact):
    target=FITS/(arch+"_"+identity(stage,idx)+".npz")
    if target.exists(): return
    is_rot=stage=="rotation"
    projection=rotation(values.shape[1],idx) if is_rot else None
    delta=data.pair_difference(values,"contact").astype(np.float64)
    if is_rot: delta=delta@projection
    best,k,losses=contact.select_axis(delta)
    contents={"contact_axis":best,"contact_slope":k,"contact_oof_brier":losses[best],
              "contact_oof_local":np.mean((contact.y-contact.oof)**2),"stage":stage,"index":(-1 if idx is None else idx)}
    if is_rot: contents["contact_projection"]=projection[:,best].copy()
    if stage=="trained": contents["contact_allaxis_oof_brier"]=losses
    if arch=="carp_38m":
        teacher=saved(FITS/"source.npz")
        # Ridge residuals transform linearly under an orthogonal change of coordinates.
        # Residualize the trained activations once and reuse the transform for its nine rotations.
        if is_rot and grid.rotated_source is not None: res,allcoef=grid.rotated_source
        else:
            res,allcoef=grid.fit(values)
            if stage in ("trained","rotation"): grid.rotated_source=(res,allcoef)
        rgrid=res[data.grid_idx]
        if is_rot: rgrid=rgrid@projection
        ix,corr,scale=match_channel(teacher["grid_oof"],rgrid,grid.w)
        coef=allcoef@projection[:,ix] if is_rot else allcoef[:,ix]
        contents.update(exposure_axis=ix,teacher_corr=corr,scale=scale,grid_coef=coef)
        if is_rot: contents["exposure_projection"]=projection[:,ix].copy()
    write_npz(target,**contents)

def fit(arch,seed_range,stages):
    import torch
    torch.set_num_threads(4)
    data=Sites("D")
    c=PairModel(data,"contact")
    g=GridRegression.from_data(data) if arch=="carp_38m" or not (FITS/"source.npz").exists() else None
    if g is not None: g.rotated_source=None
    native=None
    if not (FITS/"source.npz").exists():
        native=get_acts(data,"esm2_35m","trained",development=True)
        pack_source(data,native,g,PairModel(data,"exposure"))
        if arch!="esm2_35m": native=None
    for stage in stages:
        for idx in ([None] if stage=="trained" else (range(9) if stage=="rotation" else seed_range)):
            target=FITS/(arch+"_"+identity(stage,idx)+".npz")
            if target.exists(): continue
            started=monotonic()
            if stage=="random":
                acts=get_acts(data,arch,"random",202611220+idx,development=True,contact_only=arch=="esm2_35m")
            else:
                if native is None: native=get_acts(data,arch,"trained",development=True)
                acts=native
            fit_net(data,arch,stage,idx,acts,g,c)
            print(json.dumps({"phase":"fit","architecture":arch,"ref":identity(stage,idx),"seconds":round(monotonic()-started,2)},ensure_ascii=True),flush=True)

def fixed_values(acts,fit,task):
    projection=fit.get(task+"_projection")
    return acts@projection if projection is not None else acts[:,int(fit[task+"_axis"])].astype(np.float64)

def save_predictions(data,arch,name,values,local,fit_spec):
    records=[]
    trained=fit_spec["stage"].item()=="trained"
    for task,bases in local.items():
        if task=="exposure" and arch!="carp_38m": continue
        z=fixed_values(values,fit_spec,task)
        if task=="exposure":
            z=(z-data.x@fit_spec["grid_coef"])*float(fit_spec["scale"])
            beta=float(saved(FITS/"source.npz")["brier_slope"])
        else: beta=float(fit_spec["contact_slope"])
        p=np.clip(bases+beta*data.pair_difference(z,task),0,1)
        for r,b,q in zip(data.pairs[task],bases,p):
            y=int(r["target"])
            records.append(dict(**r,task=task,local_probability=repr(float(b)),augmented_probability=repr(float(q)),
                                gain=repr(float((y-b)**2-(y-q)**2))))
    write_tsv(PRED/(arch+"_"+name+".tsv"),("task","group","accession","early","late","target","local_probability","augmented_probability","gain"),records)
    if trained and arch=="esm2_35m":
        teacher=saved(FITS/"source.npz")
        z=values[:,814]-data.x@teacher["grid_coef"]
        baseline=local["exposure"]
        p=np.clip(baseline+float(teacher["brier_slope"])*data.pair_difference(z,"exposure"),0,1)
        out=[]
        for r,b,q in zip(data.pairs["exposure"],baseline,p):
            y=int(r["target"])
            out.append(dict(**r,task="exposure",local_probability=repr(float(b)),augmented_probability=repr(float(q)),
                            gain=repr(float((y-b)**2-(y-q)**2))))
        write_tsv(PRED/"source_814.tsv",("task","group","accession","early","late","target","local_probability","augmented_probability","gain"),out)

def score(arch,seed_range,stages):
    import torch
    torch.set_num_threads(4)
    receipt=json.loads((ROOT/"DEVELOPMENT_LOCK.json").read_text(encoding="utf-8"))
    locked=receipt["fits"]
    for filename,expected in receipt["code"].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError("Analysis code changed after the development lock: "+filename)
    if hashlib.sha256((FITS/"source.npz").read_bytes()).hexdigest()!=locked["source.npz"]:
        raise ValueError("The source exposure decoder changed after the development lock")
    dev=Sites("D");data=Sites("E")
    local={t:PairModel(dev,t).base_for(data) for t in ("contact","exposure")}
    native=None
    for stage in stages:
        for idx in ([None] if stage=="trained" else (range(9) if stage=="rotation" else seed_range)):
            nm=identity(stage,idx)
            target=FITS/(arch+"_"+nm+".npz")
            if hashlib.sha256(target.read_bytes()).hexdigest()!=locked[target.name]:
                raise ValueError("A fit changed after the development lock")
            if (PRED/(arch+"_"+nm+".tsv")).exists(): continue
            started=monotonic()
            if stage=="random": acts=get_acts(data,arch,"random",202611220+idx,contact_only=arch=="esm2_35m")
            else:
                if native is None: native=get_acts(data,arch,"trained")
                acts=native
            save_predictions(data,arch,nm,acts,local,saved(target))
            print(json.dumps({"phase":"external","architecture":arch,"ref":nm,"seconds":round(monotonic()-started,2)}),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("mode",choices=("fit","external"))
    a.add_argument("architecture",choices=("esm2_35m","carp_38m"))
    a.add_argument("--stage",nargs="+",choices=("trained","random","rotation"),default=("trained","rotation","random"))
    a.add_argument("--start",type=int,default=0);a.add_argument("--stop",type=int,default=199)
    args=a.parse_args()
    if not (0<=args.start<=args.stop<=199): raise ValueError("Reference seed range")
    guard()
    (fit if args.mode=="fit" else score)(args.architecture,range(args.start,args.stop),args.stage)
