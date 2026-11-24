"""Literature-table-selected original ESM2 MLP coordinate; no new-cohort selection."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from run_unirep341 import aggregate, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNIT = 786
BLOCK = 4


def main():
    started = time.time()
    torch.set_num_threads(4)
    cohort = HERE.parent / "cohort"
    records = [json.loads(s) for s in (cohort / "chains.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    pairs = pd.read_csv(cohort / "pairs.tsv", sep="\t")
    records = sorted(records, key=lambda r: (len(r["sequence"]), r["record_id"]))
    record_map = {r["record_id"]: r for r in records}
    for p in pairs.itertuples():
        r = record_map[p.record_id]
        assert r["group"] == p.group and r["split"] == p.split
        assert r["sequence"][p.index_positive] == p.aa == r["sequence"][p.index_negative]
    assert max(len(r["sequence"]) for r in records) <= 1022
    model_path = ROOT / "assets/esm2_8m"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True).eval().float()
    captured = {}
    validation = {"direct_coordinate_check": False}
    def hook(module, inputs, output):
        if not validation["direct_coordinate_check"]:
            direct = inputs[0] @ module.weight[UNIT] + module.bias[UNIT]
            error = float((output[:, :, UNIT] - direct).abs().max())
            assert error < 2e-5, error
            validation.update(direct_coordinate_check=True, direct_coordinate_max_abs_error=error)
        captured["pre"] = output[:, :, UNIT].detach().clone()
    handle = model.esm.encoder.layer[BLOCK].intermediate.dense.register_forward_hook(hook)
    raw, poles = {}, {}
    with torch.inference_mode():
        for j in range(0, len(records), 8):
            batch = records[j:j + 8]
            encoded = tokenizer([r["sequence"] for r in batch], return_tensors="pt", padding=True)
            # No masked tokens, crops, substitutions, or interventions.
            assert not torch.any(encoded["input_ids"] == tokenizer.mask_token_id)
            model(**encoded)
            values = captured["pre"].numpy()
            for k, r in enumerate(batch):
                n = len(r["sequence"])
                assert int(encoded["attention_mask"][k].sum()) == n + 2
                raw[r["record_id"]] = values[k, 1:n + 1].copy()
                poles[r["record_id"]] = np.maximum(raw[r["record_id"]], 0)
            print(f"Hallee786 forward {min(j+8,len(records))}/{len(records)} elapsed={time.time()-started:.1f}s", flush=True)
    handle.remove()
    np.savez_compressed(HERE / "hallee786_raw_pre_per_chain.npz", **raw)
    np.savez_compressed(HERE / "hallee786_positive_per_chain.npz", **poles)
    pairs["positive_pole786"] = [float(poles[p.record_id][p.index_positive]) for p in pairs.itertuples()]
    pairs["negative_pole786"] = [float(poles[p.record_id][p.index_negative]) for p in pairs.itertuples()]
    pairs["delta"] = pairs.positive_pole786 - pairs.negative_pole786
    pairs["tie"] = pairs.delta == 0
    pairs["correct_positive_direction"] = np.where(pairs.delta > 0, 1., np.where(pairs.delta < 0, 0., .5))
    pairs.to_csv(HERE / "hallee786_pair_results.tsv", sep="\t", index=False)
    groups, summary = aggregate(pairs)
    for row in summary:
        if row["task"] == "helix_vs_strand":
            row["inference"] = "published_positive_Helix_direction_transferred_to_new_H_vs_E_contrast"
    groups.to_csv(HERE / "hallee786_group_results.tsv", sep="\t", index=False)
    result = dict(model="esm2_8m", model_revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        layer_one_based=5, block_zero_based=4, unit_zero_based=786, unit_one_based=787,
        observation="pre-GELU FFN positive pole max(z_786,0)", origin="author-public-archive row, not a main-text named unit",
        original_author_F1=0.08819554796412499,
        original_task="Helix annotation vs remainder, not same-AA H-vs-E",
        current_comparison="same-AA same-chain H-vs-E, new-contrast transfer of original positive-Helix direction",
        original_sample_overlap="UNKNOWN", device="CPU",precision="float32", torch=torch.__version__,
        n_chains=len(records), n_residues=sum(len(r["sequence"]) for r in records),
        validation=validation, summaries=summary, elapsed_seconds=time.time()-started)
    (HERE / "HALLEE786_RESULTS.json").write_text(json.dumps(result, indent=2),encoding="utf8")
    files=[Path(__file__), HERE / "HALLEE_HELIX_LOCK.json", HERE / "hallee_retest.py", HERE / "hallee_e1_per_concept.parquet",model_path / "model.safetensors",model_path / "config.json",model_path / "vocab.txt",cohort / "chains.jsonl",cohort / "pairs.tsv"]
    (HERE / "HALLEE786_INPUT_MANIFEST.json").write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
