"""Fixed published coordinate 341; native sequences only; no unit search."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
AA = dict(zip("MRHKDESTNQCUGPAVIFYWLO", range(1, 23)))
AA.update({x: 23 for x in "XZBJ"})
UNIT = 341


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_weights():
    p = HERE / "weights_1900"
    weights = {}
    for key in ["wx", "wh", "wmx", "wmh", "b", "gx", "gh", "gmx", "gmh"]:
        weights[key] = np.load(p / f"rnn_mlstm_mlstm_{key}_0.npy")
    weights["embed"] = np.load(p / "embed_matrix_0.npy")
    for key, gain in [("wx", "gx"), ("wh", "gh"), ("wmx", "gmx"), ("wmh", "gmh")]:
        w = weights[key]
        # TensorFlow l2_normalize dim=0 uses max(sum(x*x),epsilon=1e-12).
        weights[key] = w / np.sqrt(np.maximum((w * w).sum(axis=0), 1e-12)) * weights[gain]
    return weights


@torch.inference_mode()
def forward(seqs, weights):
    w = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    bsize = len(seqs)
    lengths = [len(s) for s in seqs]
    ids = torch.zeros((max(lengths) + 1, bsize), dtype=torch.long)
    ids[0, :] = 24
    for k, s in enumerate(seqs):
        ids[1:len(s) + 1, k] = torch.tensor([AA[a] for a in s])
    c = torch.zeros((bsize, 1900))
    h = torch.zeros_like(c)
    result = np.empty((bsize, max(lengths)), dtype=np.float32)
    # Input-only projections can be precomputed without changing the recurrence.
    x = w["embed"][ids]
    x_m = x @ w["wmx"]
    x_z = x @ w["wx"]
    for t in range(len(ids)):
        m = x_m[t] * (h @ w["wmh"])
        z = x_z[t] + m @ w["wh"] + w["b"]
        i, f, o, u = z.chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * u.tanh()
        h = o.sigmoid() * c.tanh()
        if t > 0:
            result[:, t - 1] = h[:, UNIT].numpy()
    return [result[k, :n].copy() for k, n in enumerate(lengths)]


def numpy_reference(seq, weights):
    c = np.zeros(1900, np.float32)
    h = c.copy()
    result = []
    for token in [24] + [AA[a] for a in seq]:
        x = weights["embed"][token]
        m = (x @ weights["wmx"]) * (h @ weights["wmh"])
        z = x @ weights["wx"] + m @ weights["wh"] + weights["b"]
        i, f, o, u = np.split(z, 4)
        sigmoid = lambda v: 1 / (1 + np.exp(-v))
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(u)
        h = sigmoid(o) * np.tanh(c)
        result.append(float(h[UNIT]))
    return np.array(result[1:])


def aggregate(pairs):
    groups = pairs.groupby(["split", "task", "group"], sort=True).agg(
        n_pairs=("delta", "size"), n_chains=("record_id", "nunique"),
        mean_delta=("delta", "mean"), median_delta=("delta", "median"),
        fraction_positive=("correct_positive_direction", "mean"),
        n_ties=("tie", "sum"),
    ).reset_index()
    rows = []
    rng = np.random.default_rng(20261122)
    for (split, task), ps in pairs.groupby(["split", "task"], sort=True):
        gs = groups[(groups.split == split) & (groups.task == task)]
        accuracy = gs.fraction_positive.to_numpy()
        sample = rng.choice(len(gs), (10000, len(gs)), replace=True)
        ci = np.quantile(accuracy[sample].mean(1), [.025, .975])
        ds = gs.mean_delta.to_numpy()
        nonzero = ds[ds != 0]
        rows.append(dict(split=split, task=task,
            inference="published_H_gt_E" if task == "helix_vs_strand" else "descriptive_start_minus_interior_not_original_claim",
            n_groups=len(gs), n_chains=ps.record_id.nunique(), n_pairs=len(ps),
            n_positive=int((ps.delta > 0).sum()), n_negative=int((ps.delta < 0).sum()),
            n_ties=int((ps.delta == 0).sum()),
            pooled_fraction_positive=float(ps.correct_positive_direction.mean()),
            macro_fraction_positive=float(accuracy.mean()),
            macro_fraction_positive_group_bootstrap_95ci=ci.tolist(),
            pooled_mean_delta=float(ps.delta.mean()), macro_mean_delta=float(ds.mean()),
            n_groups_mean_delta_positive=int((ds > 0).sum()),
            group_mean_sign_one_sided_binomial_p=float(binomtest(int((nonzero > 0).sum()),len(nonzero),.5,alternative="greater").pvalue) if len(nonzero) else None,
        ))
    return groups, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=HERE.parent / "cohort")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    records = [json.loads(s) for s in (args.cohort / "chains.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    pairs = pd.read_csv(args.cohort / "pairs.tsv", sep="\t")
    assert set(pairs.task).issubset({"helix_vs_strand", "helix_start_vs_internal"})
    records = sorted(records, key=lambda r: (len(r["sequence"]), r["record_id"]))
    record_map = {r["record_id"]: r for r in records}
    assert len(record_map) == len(records)
    # Data validation independent of activations, before model forward.
    for p in pairs.itertuples():
        r = record_map[p.record_id]
        assert r["group"] == p.group and r["split"] == p.split
        assert r["sequence"][p.index_positive] == p.aa == r["sequence"][p.index_negative]
    weights = load_weights()
    # Independent formulas, true natural sequence prefix, fixed unit; no unit search.
    prefix = records[0]["sequence"][:12]
    np_ref = numpy_reference(prefix, weights)
    torch_ref = forward([prefix], weights)[0]
    error = float(np.max(np.abs(np_ref - torch_ref)))
    assert error < 2e-5, error
    batch = forward([records[0]["sequence"], records[1]["sequence"]], weights)
    single = forward([records[0]["sequence"]], weights)[0]
    batch_error = float(np.max(np.abs(batch[0] - single)))
    assert batch_error < 2e-5, batch_error
    acts = {}
    for j in range(0, len(records), args.batch_size):
        rs = records[j:j + args.batch_size]
        values = forward([r["sequence"] for r in rs], weights)
        for r, v in zip(rs, values):
            acts[r["record_id"]] = v
        print(f"forward {min(j + args.batch_size,len(records))}/{len(records)} elapsed={time.time()-started:.1f}s", flush=True)
    np.savez_compressed(HERE / "hidden341_per_chain.npz", **acts)
    pairs["positive_h341"] = [float(acts[p.record_id][p.index_positive]) for p in pairs.itertuples()]
    pairs["negative_h341"] = [float(acts[p.record_id][p.index_negative]) for p in pairs.itertuples()]
    pairs["delta"] = pairs.positive_h341 - pairs.negative_h341
    pairs["tie"] = pairs.delta == 0
    pairs["correct_positive_direction"] = np.where(pairs.delta > 0, 1.0, np.where(pairs.delta < 0, 0.0, .5))
    pairs.to_csv(HERE / "pair_results.tsv", sep="\t", index=False)
    groups, summary = aggregate(pairs)
    groups.to_csv(HERE / "group_results.tsv", sep="\t", index=False)
    summary_obj = dict(unit_zero_based=341, unit_one_based=342, observation="hidden_h_after_current_AA",torch=torch.__version__,device="CPU",threads=args.threads,
        n_chains=len(records), n_residues=sum(len(r["sequence"]) for r in records),
        validation=dict(numpy_vs_torch_max_abs_error=error, single_vs_batch_max_abs_error=batch_error, mapping_pairs_same_AA=True, original_TF_executed=False),
        elapsed_seconds=time.time()-started, summaries=summary)
    (HERE / "RESULTS.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    sources = [HERE / "LOCKED_PROTOCOL.md", Path(__file__), args.cohort / "chains.jsonl", args.cohort / "pairs.tsv", HERE / "unirep.py", HERE / "data_utils.py", HERE / "source_repo/UniRep-analysis-master/figures/figure2/FINAL_Fig2f_alpha-beta_neuron.ipynb"] + sorted((HERE / "weights_1900").glob("*.npy"))
    manifest = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources]
    (HERE / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
