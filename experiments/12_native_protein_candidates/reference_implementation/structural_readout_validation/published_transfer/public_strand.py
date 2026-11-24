"""Transfer the original published ESM C-300M beta-strand readout without reselection."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY/"scan"))
import runtime

PREFIX = "ESMC634_"
BLOCK = 29
CHANNEL = 634


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype={"pair_id":str,"group":str,"record_id":str})


def data(row_file, chain_file):
    records = [json.loads(s) for s in chain_file.read_text(encoding="utf-8-sig").splitlines() if s]
    return read_table(row_file), {r["record_id"]:r for r in records}


@torch.inference_mode()
def pole_by_sequence(rows, chains, cache_name):
    stored = {}
    cache_file = HERE/(PREFIX+cache_name+".npz")
    paths = [HERE/(PREFIX+"fresh_helix_readout.npz"), cache_file]
    for path in dict.fromkeys(paths):
        if path.exists():
            with np.load(path) as archive:
                stored.update({key:archive[key] for key in archive.files})
    requested = {chains[r]["sequence_sha256"]:chains[r]["sequence"] for r in rows.record_id.unique()}
    assert all(hashlib.sha256(v.encode()).hexdigest() == k for k,v in requested.items())
    needed = [(k,v) for k,v in requested.items() if k not in stored]
    if needed:
        model, modules, forward, prefix, limit, _ = runtime.load("ESMC")
        assert len(modules) == 30 and modules[BLOCK].weight.shape[1] == 2560 and prefix == 1
        held = {}
        def record_input(module,inputs):
            a = inputs[0][0,:,CHANNEL]
            held["z"] = a.detach().float()
        def record_gate(module,inputs,output):
            gate, up = inputs[0].chunk(2,dim=-1)
            held["direct"] = torch.nn.functional.silu(gate[0,:,CHANNEL].float()) * up[0,:,CHANNEL].float()
        h1 = modules[BLOCK].register_forward_pre_hook(record_input)
        h2 = model.transformer.blocks[BLOCK].ffn[2].register_forward_hook(record_gate)
        try:
            for key, seq in needed:
                assert len(seq)+2 <= limit
                held.clear()
                forward(seq)
                z = held["z"]
                assert len(z)==len(seq)+2 and torch.isfinite(z).all()
                torch.testing.assert_close(z,held["direct"],rtol=2e-5,atol=2e-6)
                stored[key]=z[1:-1].clamp_min(0).cpu().numpy().copy()
        finally:
            h2.remove()
            h1.remove()
    result = {k:stored[k] for k in requested}
    assert all(len(v)==len(requested[k]) and np.isfinite(v).all() and v.min()>=0 for k,v in result.items())
    if not cache_file.exists():
        np.savez_compressed(cache_file,**result)
    return result


def differences(rows,chains,cache):
    signed = []
    for r in rows.itertuples():
        chain=chains[r.record_id]
        seq=chain["sequence"]
        p,n=int(r.index_positive),int(r.index_negative)
        assert chain["group"] == r.group and seq[p] == r.aa == seq[n]
        positions=cache[chain["sequence_sha256"]]
        signed.append(float(positions[p])-float(positions[n]))
    return np.array(signed)


def correctness(delta):
    return (delta>0).astype(float) + .5*(delta==0)


def bootstrap(group_values):
    v=np.array(group_values,dtype=float)
    rng=np.random.default_rng(41624)
    draws=v[rng.integers(len(v),size=(9999,len(v)))].mean(axis=1)
    return dict(mean=float(np.mean(v)),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)))


def summarize(part):
    rates=part.groupby("group",sort=True)["correct"].mean()
    return dict(groups=len(rates),pairs=len(part),wins=int((part.correct==1).sum()),
                losses=int((part.correct==0).sum()),ties=int((part.correct==.5).sum()),**bootstrap(rates))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["fresh-helix","topology","paired"])
    stage=ap.parse_args().stage
    author=read_json(HERE/"esmc634_author_selection.json")
    study_lock=read_json(HERE/"ESMC_STRAND_FREEZE.json")
    assert sha(HERE/"ESMC_STRAND_PROTOCOL.md")==study_lock["protocol_sha256"].lower()
    assert (author["winner"]["layer"],author["winner"]["best_feature"],author["winner"]["arm"])==(BLOCK,CHANNEL,"mlp_positive")
    if stage=="fresh-helix":
        root=STUDY/"fresh"; truth=read_json(root/"COHORT_FREEZE.json")
        assert all(sha(root/f)==truth["files"][f] for f in ("pairs.tsv","chains.jsonl"))
        rows,chains=data(root/"pairs.tsv",root/"chains.jsonl")
        rows=rows.loc[rows.task=="helix_vs_strand"].copy()
        assert set(rows.pair_id)==set(read_table(HERE/"F_pairs.tsv").pair_id)
        pole=pole_by_sequence(rows,chains,"fresh_helix_readout")
        # Existing H/E rows put helix first. The published positive pole prefers strand.
        rows["strand_minus_helix"]=-differences(rows,chains,pole)
        rows["correct"]=correctness(rows.strand_minus_helix)
        rows[["pair_id","record_id","group","aa","strand_minus_helix","correct"]].to_csv(
            HERE/(PREFIX+"F_helix_pairs.tsv"),sep="\t",index=False)
        result=dict(readout="Hallee_ESMC_634",direction="strand above helix",
                    block0=BLOCK,unit0=CHANNEL,author_table_sha256=author["source_sha256"],
                    complete_utc=now(),**summarize(rows))
        (HERE/(PREFIX+"F_helix.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(result,flush=True)
        return
    if stage=="paired":
        pub=read_table(HERE/(PREFIX+"topology_pairs.tsv"))
        native=STUDY/"scan"/"ESMC"/"trained"
        other=pd.concat([read_table(native/"signed_pairs.tsv"),read_table(native/"F_signed_pairs.tsv")])
        keys=["split","group","pair_id"]
        joined=pub.merge(other[keys+["correct"]],on=keys,validate="one_to_one",suffixes=("_fixed","_scanned"))
        assert len(joined)==len(pub)==len(other)
        joined["gap"]=joined.correct_scanned-joined.correct_fixed
        out=[]
        for split,p in joined.groupby("split",sort=True):
            gd=p.groupby("group",sort=True)["gap"].mean()
            out.append(dict(split=split,groups=len(gd),pairs=len(p),**bootstrap(gd)))
        pd.DataFrame(out).to_csv(HERE/(PREFIX+"topology_native_minus_published.tsv"),sep="\t",index=False)
        return
    merged=STUDY/"scan"/"merged_cohorts"
    source=read_json(merged/"READY.json")
    assert sha(merged/"pairs.tsv")==source["pairs_sha256"]
    assert sha(merged/"chains.jsonl")==source["chains_sha256"]
    rows,chains=data(merged/"pairs.tsv",merged/"chains.jsonl")
    dev=rows.loc[rows.split=="D"].copy()
    pole=pole_by_sequence(dev,chains,"D_readout")
    d=differences(dev,chains,pole)
    dev["correct"]=correctness(d)
    # Comparisons exactly at 0.5 choose positive orientation.
    s=sum(Fraction(int((2*p.correct).sum()),2*len(p)) for _,p in dev.groupby("group"))
    sign=1 if 2*s>=dev.group.nunique() else -1
    lock_path=HERE/(PREFIX+"topology_sign_lock.json")
    if lock_path.exists():
        lock=read_json(lock_path)
        assert lock["sign"]==sign and lock["development_source"]==source["pairs_sha256"]
    else:
        with lock_path.open("x",encoding="utf-8") as f:
            json.dump(dict(sign=sign,block0=BLOCK,channel0=CHANNEL,created_utc=now(),
                          development_source=source["pairs_sha256"],development_groups=int(dev.group.nunique()),
                          development_pairs=len(dev),author_table_sha256=author["source_sha256"]),f,indent=2)
    dev["delta"]=sign*d
    frames=[dev]
    # Only after saving the D-only sign may the remaining topology pairs be evaluated.
    held=rows.loc[rows.split!="D"].copy()
    raw=pole_by_sequence(held,chains,"TEH_readout")
    held["delta"]=sign*differences(held,chains,raw)
    frames.append(held)
    sheet=STUDY/"sheet"
    fr,fseq=data(sheet/"pairs_F.tsv",sheet/"chains_F.jsonl")
    fp=pole_by_sequence(fr,fseq,"F_topology_readout")
    fr["delta"]=sign*differences(fr,fseq,fp)
    frames.append(fr)
    allrows=pd.concat(frames,ignore_index=True)
    allrows["correct"]=correctness(allrows.delta)
    allrows[["split","pair_id","record_id","group","aa","delta","correct"]].to_csv(
        HERE/(PREFIX+"topology_pairs.tsv"),sep="\t",index=False)
    out=[dict(split=split,sign=sign,**summarize(p)) for split,p in allrows.groupby("split",sort=True)]
    pd.DataFrame(out).to_csv(HERE/(PREFIX+"topology_summary.tsv"),sep="\t",index=False)
    print(out,flush=True)


if __name__=="__main__":
    main()
