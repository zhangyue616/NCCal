"""Extract natural-sequence FFN post-GELU values at one prespecified relative depth."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from transformers import EsmForMaskedLM

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = PROJECT / "data"
FRESH = DATA / "revision" / "protein_fresh"

def lines(path):
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]

def roster():
    old = json.loads((FRESH / "OLD_SELECTED_FINAL_GROUPS.json").read_text(encoding="utf-8-sig"))
    groups = {(r["origin"], r["id"]): r["current_known_union_group"] for r in old["selected_pair_group_mapping"]}
    result = {}
    for cohort in ("pilot3a", "pilot4a", "pilot5a"):
        for row in lines(DATA / cohort / "proteins.jsonl"):
            result[row["id"]] = ("D", groups[cohort, row["id"]], row["sequence"])
    if len(result) != 432 or len({r[1] for r in result.values()}) != 421:
        raise ValueError("Development cohort differs from the accepted cohort")
    for row in lines(FRESH / "proteins.jsonl"):
        if row["id"] in result:
            raise ValueError("Accession overlap")
        result[row["id"]] = ("T", row["known_group_id"], row["sequence"])
    if len(result) != 1736 or len({r[1] for r in result.values() if r[0] == "T"}) != 1304:
        raise ValueError("Confirmation cohort differs from the accepted cohort")
    return result

def main():
    a = argparse.ArgumentParser(__doc__)
    a.add_argument("model", choices=("8m","35m"))
    a.add_argument("split", choices=("D","T"))
    a.add_argument("--axis", type=int)
    a.add_argument("--device", default="cpu", choices=("cpu","cuda"))
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--ids", type=Path, help="Optional text file listing the accessions with completed structural profiles")
    opt = a.parse_args()
    if opt.split == "T" and opt.axis is None:
        a.error("Select the axis using development labels before extracting confirmation values")
    torch.set_num_threads(opt.threads)
    data = roster()
    ids = {s.strip() for s in opt.ids.read_text().splitlines() if s.strip()} if opt.ids else None
    selected = [(acc,seq) for acc,(split,_,seq) in data.items() if split==opt.split and (ids is None or acc in ids)]
    if ids and ids - {acc for acc,seq in selected}:
        raise ValueError("Requested accession is outside this split")
    selected.sort(key=lambda row:(len(row[1]),row[0]))
    weights = PROJECT/"assets"/"esm2_8m" if opt.model=="8m" else HERE/"assets"/"esm2_35m"
    model = EsmForMaskedLM.from_pretrained(str(weights), local_files_only=True).to(opt.device).eval()
    if model.config.num_hidden_layers != {"8m":6,"35m":12}[opt.model]:
        raise ValueError("Unexpected checkpoint depth")
    block = model.config.num_hidden_layers // 2 - 1
    width = model.config.intermediate_size
    layer = model.esm.encoder.layer[block].intermediate
    vocab = {token:i for i,token in enumerate((weights/"vocab.txt").read_text().splitlines())}
    save = HERE/"activations"/opt.model/(opt.split+"-"+("all" if opt.axis is None else str(opt.axis)))
    save.mkdir(parents=True,exist_ok=True)
    todo = []
    for acc,s in selected:
        f = save/(acc+".npy")
        if f.exists():
            expected = (len(s),width) if opt.axis is None else (len(s),)
            check = np.load(f,mmap_mode="r",allow_pickle=False)
            if check.shape != expected or check.dtype != np.float32: raise ValueError("Bad cache for "+acc)
        else:
            todo.append((acc,s))
    class AtRequestedBlock(Exception):
        pass
    capture = []
    def get_ffn(_module,_args,out):
        capture.append(out.detach() if opt.axis is None else out[:,:,opt.axis].detach())
        raise AtRequestedBlock()
    handle = layer.register_forward_hook(get_ffn)
    t0 = monotonic()
    try:
        with torch.inference_mode():
            for start in range(0,len(todo),opt.batch):
                batch = todo[start:start+opt.batch]
                toks = [[vocab["<cls>"],*[vocab[c] for c in seq],vocab["<eos>"]] for acc,seq in batch]
                seq_ids = torch.full((len(toks),max(map(len,toks))),vocab["<pad>"],dtype=torch.long)
                for i,tokenrow in enumerate(toks): seq_ids[i,:len(tokenrow)] = torch.tensor(tokenrow)
                seq_ids = seq_ids.to(opt.device)
                capture.clear()
                try: model.esm(input_ids=seq_ids,attention_mask=seq_ids.ne(vocab["<pad>"]).long())
                except AtRequestedBlock: pass
                if len(capture)!=1: raise RuntimeError("Post-GELU hook did not run exactly once")
                values = capture[0].cpu().numpy()
                for i,(acc,sequence) in enumerate(batch):
                    array = values[i,1:len(sequence)+1].copy()
                    if not np.isfinite(array).all(): raise ValueError("Non-finite activation in "+acc)
                    tmp = save/(acc+"."+str(os.getpid())+".partial")
                    with tmp.open("wb") as f: np.save(f,array,allow_pickle=False)
                    tmp.replace(save/(acc+".npy"))
                done=min(start+len(batch),len(todo))
                if done==len(todo) or start%(opt.batch*10)==0:
                    print(f"{opt.model} {opt.split}: {done}/{len(todo)} newly extracted, {monotonic()-t0:.1f}s",flush=True)
    finally:
        handle.remove()
    print(json.dumps({"model":opt.model,"split":opt.split,"block_one_based":block+1,"width":width,
                      "axis":opt.axis,"selection":len(selected),"computed":len(todo),"seconds":monotonic()-t0}),flush=True)

if __name__=="__main__":
    main()
