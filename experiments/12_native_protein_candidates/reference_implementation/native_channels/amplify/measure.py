"""Read all native AMPLIFY channels. The local shim implements two xFormers operations."""
import argparse
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel  # Load optional-dependency inventory first.

ROOT = Path(__file__).resolve().parent
COHORT = ROOT.parent / "cohort"
OFFICIAL = ROOT / "official"
TASKS = ("helix_vs_strand", "helix_start_vs_internal")
L, WIDTH = 24, 1712

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)

def sdpa(query, key, value, attn_bias=None, p=0):
    args = (query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return F.scaled_dot_product_attention(*args, attn_mask=attn_bias, dropout_p=p).transpose(1, 2).contiguous()

module = types.ModuleType("xformers")
module.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None, is_package=True)
ops = types.ModuleType("xformers.ops")
ops.SwiGLU = SwiGLU
ops.memory_efficient_attention = sdpa
module.ops = ops
sys.modules["xformers"], sys.modules["xformers.ops"] = module, ops
from official.amplify import AMPLIFY, AMPLIFYConfig

def load_model(device, seed=None):
    torch.manual_seed(88000 if seed is None else seed)
    cfg = json.loads((OFFICIAL / "config.json").read_text())
    net = AMPLIFY(AMPLIFYConfig(**cfg)).eval()
    if seed is None:
        net.load_state_dict(load_file(str(OFFICIAL / "model.safetensors")), strict=True)
    else:
        # Use the reference release's explicit Linear/Embedding initializer.
        net.apply(net._init_weights)
    assert len(net.transformer_encoder) == L
    assert all(b.ffn.w3.weight.shape[1] == WIDTH for b in net.transformer_encoder)
    return net.to(device)

def token_ids(tok, seq):
    ids = tok.encode(seq).ids
    assert len(ids) == len(seq) + 2 and ids[0] == 3 and ids[-1] == 4
    assert len(ids) <= 2048
    return ids

def wins(diff):
    return (diff > 0).astype(np.float32) + .5 * (diff == 0)

def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def select(scores):
    # Integer scores make ties exact before applying the canonical ordering.
    for task, (value, denominator) in zip(TASKS, scores):
        pos = np.stack([value, denominator - value], axis=-1)
        layer, unit, side = np.unravel_index(np.argmax(pos), pos.shape)
        yield {"task": task, "block_1": int(layer + 1), "channel_0": int(unit),
               "sign": 1 if side == 0 else -1, "development_score": float(pos[layer, unit, side] / denominator)}

def group_scores(data, rows):
    out = []
    for task in TASKS:
        sub = rows[rows.task == task]
        counts = sub.groupby("group").size().tolist()
        common = math.lcm(*counts)
        grouped = np.zeros((L, WIDTH), dtype=np.int64)
        for _, members in sub.groupby("group", sort=True):
            chunk = data[members.index.to_numpy()]
            points = 2 * (chunk > 0).sum(axis=0) + (chunk == 0).sum(axis=0)
            grouped += (common // len(members)) * points
        out.append((grouped, 2 * common * len(counts)))
    return out

@torch.inference_mode()
def collect(net, tok, sequences, rows, out, keep, picks=None):
    device = next(net.parameters()).device
    if not len(rows):
        return
    if keep or picks is None:
        buf = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(rows), L, WIDTH))
    else:
        buf = np.full(len(rows), np.nan, dtype=np.float64)
    keys = {t: v for t, v in ((r["task"], r) for r in (picks or []))}
    loc = rows.record_id.map(sequences)
    assert loc.notna().all()
    for sequence, records in rows.groupby(loc, sort=False):
        post = torch.as_tensor(records.index_positive.to_numpy() + 1, device=device)
        neg = torch.as_tensor(records.index_negative.to_numpy() + 1, device=device)
        read = {}
        hook_list = []
        for layer, block in enumerate(net.transformer_encoder):
            def visit(_m, args, ll=layer):
                activation = args[0][0]
                read[ll] = activation[post] - activation[neg]
            if keep or picks is None or any(k["block_1"] == layer + 1 for k in picks):
                hook_list.append(block.ffn.w3.register_forward_pre_hook(visit))
        ids = torch.tensor([token_ids(tok, sequence)], device=device)
        try:
            net(ids)
        finally:
            for hook in hook_list:
                hook.remove()
        if keep or picks is None:
            buf[records.index.to_numpy()] = torch.stack([read[i] for i in range(L)], dim=1).float().cpu().numpy()
        else:
            for task, subset in records.groupby("task"):
                k = keys[task]
                locs = records.index.get_indexer(subset.index)
                col = read[k["block_1"] - 1][:, k["channel_0"]].float().cpu().numpy()
                buf[subset.index] = k["sign"] * col[locs]
    if isinstance(buf, np.memmap):
        buf.flush()
    elif not np.isfinite(buf).all():
        raise ValueError("Missing locked-channel scores")
    return buf

def run(network_id, rows, chains, dev):
    trained = network_id == "trained"
    folder = ROOT / "results" / network_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        return
    net = load_model(dev, None if trained else int(network_id))
    tok = Tokenizer.from_file(str(OFFICIAL / "tokenizer.json"))
    values = []
    for split in ("D", "T", "E"):
        part = rows[rows.canonical_split == split].reset_index(drop=True)
        if not len(part):
            continue
        part.to_csv(folder / f"{split}_pairs.tsv", sep="\t", index=False)
        path = folder / f"{split}_delta.npy"
        if split == "D":
            data = collect(net, tok, chains, part, path, trained)
            selected = list(select(group_scores(data, part)))
            write_json(folder / "locked.json", selected)
        else:
            selected = json.loads((folder / "locked.json").read_text())
            data = collect(net, tok, chains, part, path, trained, selected)
        for k in selected:
            sub = part[part.task == k["task"]]
            if data.ndim == 3:
                col = k["sign"] * data[sub.index, k["block_1"] - 1, k["channel_0"]]
            else:
                col = data[sub.index]
            for i, d in zip(sub.index, col):
                values.append((part.at[i, "pair_id"], split, k["task"], part.at[i, "group"], float(d)))
        del data
        if not trained and split == "D":
            path.unlink()
    pd.DataFrame(values, columns=["pair_id", "split", "task", "group", "delta"]).to_csv(
        folder / "signed_pairs.tsv", sep="\t", index=False)
    write_json(folder / "done.json", {"network": network_id, "n_pairs": len(values)})
    print(network_id, "completed", flush=True)

@torch.inference_mode()
def check_backend():
    tok = Tokenizer.from_file(str(OFFICIAL / "tokenizer.json"))
    source = ROOT.parent.parent / "nccal_evidence_extension" / "cohort" / "chains.jsonl"
    row = next(r for r in map(json.loads, source.read_text(encoding="utf-8").splitlines()) if r["split"] in ["A","D"])
    seq = row["sequence"]
    ids = torch.tensor([token_ids(tok, seq)])
    net = load_model("cpu")
    block = net.transformer_encoder[0]
    x = net.encoder(ids)
    y = block.ffn_norm(x)
    w1, w2 = block.ffn.w12.weight.chunk(2, dim=0)
    explicit = F.silu(y @ w1.T) * (y @ w2.T)
    direct_ffn = float((explicit @ block.ffn.w3.weight.T - block.ffn(y)).abs().max())
    hidden = []
    h = net.transformer_encoder[12].ffn.w3.register_forward_pre_hook(lambda m, a: hidden.append(a[0].detach().cpu()))
    cpu = net(ids).logits.detach().cpu()
    output = net.cuda()(ids.cuda()).logits.detach().cpu()
    h.remove()
    dlog = cpu - output
    dunit = hidden[0] - hidden[1]
    vals = {"source_chain": row["chain_id"], "sequence_length": len(seq), "ffn_explicit_max": direct_ffn,
            "cpu_cuda_logits_max": float(dlog.abs().max()), "cpu_cuda_logits_rms": float(dlog.square().mean().sqrt()),
            "cpu_cuda_mlp13_rms": float(dunit.square().mean().sqrt()), "L": L, "width": WIDTH}
    assert direct_ffn < 1e-4 and vals["cpu_cuda_logits_rms"] < 1e-3 and vals["cpu_cuda_mlp13_rms"] < 1e-3
    write_json(ROOT / "backend_check.json", vals)
    print(vals, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", nargs="*", default=["trained"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if args.check:
        check_backend()
        return
    rows = pd.read_csv(COHORT / "pairs.tsv", sep="\t", dtype={"group":str, "record_id":str})
    rows["canonical_split"] = rows["split"].replace({"A":"D","B":"D"})
    assert set(rows.canonical_split) <= {"D", "T", "E"}
    assert rows.pair_id.is_unique
    assert rows.groupby("group").canonical_split.nunique().max() == 1
    assert set(rows.task) == set(TASKS)
    chains = {r["record_id"]:r["sequence"] for r in map(json.loads, (COHORT / "chains.jsonl").read_text(encoding="utf-8-sig").splitlines())}
    for seed in args.seeds:
        run(seed, rows, chains, "cuda")

if __name__ == "__main__":
    main()
