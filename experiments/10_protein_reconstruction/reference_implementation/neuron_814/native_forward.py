"""Offline ESM-2 block-six / coordinate-814 masked-context forward passes.

Only this extension is written. Native extraction, D technical checks, and T
writeback are separately resumable; this module never fits or scores outcomes.
"""
from __future__ import annotations

import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                  MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "protein_extension"
OUT = HERE / "results"
TEMP = HERE / "native_forward_tmp"
AA = "ACDEFGHIKLMNPQRSTVWY"
UNIT = 814
sys.path.insert(0, str(OLD / "rsa"))
from model_inputs import read_rows


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def atomic_npz(path, fields):
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    tmp.replace(path)


def plan_check():
    actual = sha(HERE / "PROTOCOL.md")
    if actual.lower() != json_read(HERE / "PROTOCOL_FREEZE.json")["protocol_sha256"].lower():
        raise ValueError("Frozen protocol hash mismatch")
    return actual


def fit_check(authorized):
    if not authorized:
        raise ValueError("T forward requires explicit --fit-authorized after the coordinator's sealed-fit notice")
    freeze = json_read(OUT / "fit_B_FREEZE.json")
    if freeze["protocol_sha256"] != plan_check() or freeze["fit_sha256"] != sha(OUT / "fit_B.npz"):
        raise ValueError("B fit is not the sealed fit for this protocol")
    return sha(OUT / "fit_B_FREEZE.json")


def cohort_cases(split):
    rows, cohort = read_rows(split)
    expected = {"D": (20974, 325), "T": (173859, 1099)}[split]
    if (len(rows), len({r[2] for r in rows})) != expected:
        raise ValueError("Archived clean cohort composition changed")
    if split == "D":
        selected = []
        by_group = {}
        for row in rows:
            by_group.setdefault(row[2], []).append(row)
        for group in sorted(by_group):
            group_rows = sorted(by_group[group], key=lambda r: (r[0], r[1]))
            if len({r[0] for r in group_rows}) != 1:
                raise ValueError("Multiple representative proteins in a D group")
            ix = np.linspace(0, len(group_rows)-1, min(len(group_rows), 32), dtype=int)
            selected.extend(group_rows[i] for i in ix)
    else:
        paired_keys = set()
        pair_groups = set()
        with (OLD / "structure" / "matched_pairs.tsv").open(encoding="utf-8-sig", newline="") as handle:
            for pair in csv.DictReader(handle, delimiter="\t"):
                if pair["split"] != "T":
                    continue
                if pair["group"] in pair_groups:
                    raise ValueError("Multiple pairs per T group")
                pair_groups.add(pair["group"])
                for field in ("pos1", "pos2"):
                    key = (pair["accession"], int(pair[field]), pair["group"])
                    if key in paired_keys:
                        raise ValueError("Repeated paired T site")
                    paired_keys.add(key)
        selected = [r for r in rows if (r[0], r[1], r[2]) in paired_keys]
        if len(selected) != 1934 or len(pair_groups) != 967 or len(paired_keys) != len(selected):
            raise ValueError("Fixed 967-pair T selection failed")
    selected.sort(key=lambda r: (r[2], r[0], r[1]))
    cases = []
    for row in selected:
        acc, pos, group = row[:3]
        seq = cohort[acc][2]
        if not 1 <= pos < len(seq) or seq[pos-1] != row[3]:
            raise ValueError(f"Missing immediate C-terminal readout or source mismatch: {acc}:{pos}")
        cases.append({"acc": acc, "pos": pos, "group": group, "sequence": seq})
    return cases


def fields_for(cases):
    return {"acc": np.array([c["acc"] for c in cases]),
            "pos": np.array([c["pos"] for c in cases], dtype=np.int64),
            "group": np.array([c["group"] for c in cases])}


def assert_keys(data, fields):
    if any(not np.array_equal(data[k], v) for k, v in fields.items()):
        raise ValueError("Case keys or ordering disagree")


def batches_for(cases, maximum):
    # Scheduling uses only sequence length and original stable case index.
    order = sorted(range(len(cases)), key=lambda i: (len(cases[i]["sequence"]), i))
    cursor = 0
    while cursor < len(order):
        length = len(cases[order[cursor]]["sequence"]) + 2
        # Eager attention has quadratic temporary storage; long sequences use
        # smaller batches. Changing this schedule does not change any inputs.
        batch = min(maximum, max(1, int(12_000_000 / (length * length))))
        ix = np.asarray(order[cursor:cursor+batch], dtype=np.int64)
        cursor += len(ix)
        yield ix


class Runner:
    def __init__(self):
        import torch
        import transformers
        from transformers import EsmForMaskedLM
        self.torch = torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        path = OLD / "rsa" / "assets" / "esm2_35m"
        self.vocab = {token:i for i,token in enumerate((path / "vocab.txt").read_text().splitlines())}
        self.aa_ids = [self.vocab[aa] for aa in AA]
        self.model = EsmForMaskedLM.from_pretrained(str(path), local_files_only=True,
                        attn_implementation="eager", dtype=torch.float32).cuda().float().eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        if (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act) != (12,480,1920,"gelu"):
            raise ValueError("Unexpected ESM-2 35M configuration")
        if any(p.dtype != torch.float32 for p in self.model.parameters()) or self.model.training:
            raise ValueError("Model must be float32 and eval")
        self.block = self.model.esm.encoder.layer[5]
        self.forwards = 0
        self.case_forwards = 0
        self.started = time.perf_counter()
        self.runtime = {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "GPU": torch.cuda.get_device_name(), "dtype": "float32", "attention": "eager",
            "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32, "eval": not self.model.training,
            "block_one_based": 6, "coordinate_zero_based": UNIT,
            "source_token_index": "one-based source pos", "mask_token_index": "source pos + 1",
            "canonical_AA_order": AA, "network_requests": 0}

    def run(self, cases, replacement=None, inspect_dense=False):
        torch = self.torch
        device = "cuda"
        tokens = [[self.vocab["<cls>"], *[self.vocab[c] for c in case["sequence"]], self.vocab["<eos>"]]
                  for case in cases]
        input_ids = torch.full((len(cases), max(map(len,tokens))), self.vocab["<pad>"], dtype=torch.long)
        for i, row in enumerate(tokens):
            input_ids[i,:len(row)] = torch.tensor(row)
        source = torch.tensor([c["pos"] for c in cases], dtype=torch.long, device=device)
        target = source + 1
        rr = torch.arange(len(cases), device=device)
        input_ids = input_ids.to(device)
        original_source = input_ids[rr,source].clone()
        input_ids[rr,target] = self.vocab["<mask>"]
        if not bool(((input_ids == self.vocab["<mask>"]).sum(1) == 1).all()):
            raise ValueError("Each input must contain exactly one mask")
        if not torch.equal(input_ids[rr,source], original_source):
            raise ValueError("Source residue was altered")
        captured = {}
        def hook(_module, _args, output):
            if output.shape[-1] != 1920 or output.dtype != torch.float32:
                raise ValueError("Unexpected post-GELU tensor")
            captured["z"] = output[rr,source,UNIT].detach().clone()
            if replacement is not None:
                patch = torch.as_tensor(replacement, device=device, dtype=torch.float32)
                if patch.shape != source.shape or not bool(torch.isfinite(patch).all()):
                    raise ValueError("Invalid replacement shape or nonfinite patch")
                changed = output.clone()
                changed[rr,source,UNIT] = patch
                captured["actual_replacement"] = changed[rr,source,UNIT].detach().clone()
                return changed
        def dense_hook(_module, _args, output):
            captured["dense"] = output[rr,source].detach().clone()
        handles = [self.block.intermediate.register_forward_hook(hook)]
        if inspect_dense:
            handles.append(self.block.output.dense.register_forward_hook(dense_hook))
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=input_ids.ne(self.vocab["<pad>"]).long()).logits
                result = {"logits": logits[rr,target][:,self.aa_ids].detach().cpu().numpy(),
                          **{k:v.cpu().numpy() for k,v in captured.items()}}
        finally:
            for h in handles:
                h.remove()
        self.forwards += 1
        self.case_forwards += len(cases)
        if any(not np.isfinite(v).all() or v.dtype != np.float32 for v in result.values()):
            raise ValueError("Nonfinite or non-float32 forward result")
        return result

    def release(self):
        self.runtime.update(batch_calls=self.forwards, case_forwards=self.case_forwards,
            wall_seconds=time.perf_counter()-self.started,
            peak_GPU_allocated_bytes=int(self.torch.cuda.max_memory_allocated()))
        del self.block
        del self.model
        self.torch.cuda.empty_cache()
        self.runtime["GPU_allocated_bytes_after_release"] = int(self.torch.cuda.memory_allocated())
        return self.runtime


def technical_checks(runner, cases):
    shortest = min(range(len(cases)), key=lambda i: (len(cases[i]["sequence"]), i))
    longest = max(range(len(cases)), key=lambda i: (len(cases[i]["sequence"]), -i))
    records = []
    for i in sorted({shortest,longest}):
        sample = [cases[i]]
        native = runner.run(sample, inspect_dense=True)
        restored = runner.run(sample, native["z"], inspect_dense=True)
        if any(not np.array_equal(native[k], restored[k]) for k in ("z","dense","logits")):
            raise ValueError("Native-value restoration is not bitwise exact on D")
        proposed = native["z"] + np.float32(0.25)
        changed = runner.run(sample, proposed, inspect_dense=True)
        dose = changed["actual_replacement"] - native["z"]
        column = runner.block.output.dense.weight[:,UNIT].detach().cpu().numpy()
        expected = dose[:,None] * column[None,:]
        error = float(np.max(np.abs(changed["dense"] - native["dense"] - expected)))
        actual_dense_change = float(np.max(np.abs(changed["dense"]-native["dense"])))
        if error > 2e-5 or actual_dense_change == 0 or not np.all(dose != 0):
            raise ValueError(f"D fixed-patch dense-column response failed: {error}")
        records.append({**{k:cases[i][k] for k in ("acc","pos","group")},
            "sequence_length":len(cases[i]["sequence"]), "fixed_patch_requested":0.25,
            "fixed_patch_actual_float32":dose.tolist(), "restoration_bitwise_exact":True,
            "dense_column_response_max_abs_error":error,
            "dense_fixed_patch_max_abs_change":actual_dense_change})
    return {"status":"PASS", "scope":"D shortest and longest selected inputs", "cases":records}


def checkpoint_load(path, fields, arrays, identity):
    if not path.exists():
        return np.zeros(len(fields["pos"]), dtype=bool)
    with np.load(path, allow_pickle=False) as saved:
        assert_keys(saved,fields)
        if str(saved["identity"]) != identity:
            raise ValueError("Checkpoint belongs to another protocol/input phase")
        for key in arrays:
            if saved[key].shape != arrays[key].shape:
                raise ValueError("Checkpoint array shape mismatch")
            arrays[key][:] = saved[key]
        return saved["done"].copy()


def update_receipt(name, value):
    path = OUT / "FORWARD_CHECK.json"
    receipt = json_read(path) if path.exists() else {"protocol_sha256": plan_check(), "canonical_AA_order": AA}
    if receipt["protocol_sha256"] != plan_check():
        raise ValueError("Forward receipt belongs to another protocol")
    receipt[name] = value
    receipt["updated_utc"] = utc()
    atomic_json(path,receipt)


def progress(phase, done, runner):
    state = {"phase":phase, "completed":int(done.sum()), "total":len(done), "updated_utc":utc(),
        "phase_seconds":time.perf_counter()-runner.started, "batch_calls":runner.forwards,
        "case_forwards":runner.case_forwards,
        "peak_GPU_allocated_bytes":int(runner.torch.cuda.max_memory_allocated())}
    atomic_json(TEMP / (phase+"_progress.json"),state)
    print(json.dumps(state),flush=True)


def native(split, maximum, authorized):
    protocol = plan_check()
    fit = fit_check(authorized) if split == "T" else None
    cases = cohort_cases(split)
    fields = fields_for(cases)
    dest = OUT / f"masked_native_{split}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as saved:
            assert_keys(saved, fields)
            if saved["z"].shape != (len(cases),) or saved["logits"].shape != (len(cases),20):
                raise ValueError("Completed native output has invalid shape")
            if any(saved[k].dtype != np.float32 or not np.isfinite(saved[k]).all() for k in ("z","logits")):
                raise ValueError("Completed native output invalid")
        print(f"Completed native {split} already exists and validates",flush=True)
        return
    identity = protocol+"|native|"+split+"|"+str(fit)
    arrays = {"z": np.full(len(cases),np.nan,dtype=np.float32),
              "logits": np.full((len(cases),20),np.nan,dtype=np.float32)}
    path = TEMP / f"native_{split}_checkpoint.npz"
    done = checkpoint_load(path,fields,arrays,identity)
    runner = Runner()
    technical = technical_checks(runner,cases) if split == "D" else None
    if technical is not None:
        update_receipt("D_technical",technical)
    last_save = time.perf_counter()
    try:
        for batch_number, ix in enumerate(batches_for(cases,maximum)):
            if done[ix].all():
                continue
            # Do not reshape a partially finished batch: compute identical full
            # batch geometry and overwrite only the same selected native cases.
            result = runner.run([cases[i] for i in ix])
            for key in arrays:
                arrays[key][ix] = result[key]
            done[ix] = True
            if time.perf_counter()-last_save > 45 or done.all():
                atomic_npz(path,{**fields,**arrays,"done":done,"identity":np.asarray(identity)})
                progress("native_"+split,done,runner)
                last_save = time.perf_counter()
        if not done.all() or any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Native extraction incomplete")
        atomic_npz(dest,{**fields,**arrays})
    finally:
        runtime = runner.release()
    update_receipt("native_"+split,{"status":"COMPLETE", "sites":len(cases),
        "groups":len(set(fields["group"])), "output_sha256":sha(dest),
        "fit_freeze_sha256":fit, "runtime":runtime})
    print(f"NATIVE {split} COMPLETE: {dest}",flush=True)


def writeback(maximum, authorized):
    protocol = plan_check()
    fit = fit_check(authorized)
    cases = cohort_cases("T")
    fields = fields_for(cases)
    replacement_path = OUT / "B_replacements_T.npz"
    with np.load(replacement_path,allow_pickle=False) as saved:
        assert_keys(saved,fields)
        replacements = {k:saved[k].copy() for k in ("z","mean","seq","full","swapped")}
    if any(v.dtype != np.float32 or v.shape != (len(cases),) or not np.isfinite(v).all() for v in replacements.values()):
        raise ValueError("Replacements must be finite float32 arrays aligned to T")
    with np.load(OUT / "masked_native_T.npz",allow_pickle=False) as saved:
        assert_keys(saved,fields)
        native_z, native_logits = saved["z"].copy(), saved["logits"].copy()
    if not np.array_equal(native_z,replacements["z"]):
        raise ValueError("Replacement native z is from a different context")
    identity = protocol+"|writeback|"+fit+"|"+sha(replacement_path)
    arrays = {"z":np.full(len(cases),np.nan,dtype=np.float32)}
    for arm in ("native","restored","mean","seq","full","swapped"):
        arrays["logits_"+arm] = np.full((len(cases),20),np.nan,dtype=np.float32)
    for arm in ("mean","seq","full","swapped"):
        arrays["actual_"+arm] = np.full(len(cases),np.nan,dtype=np.float32)
    path = TEMP / "writeback_T_checkpoint.npz"
    done = checkpoint_load(path,fields,arrays,identity)
    runner = Runner()
    last_save = time.perf_counter()
    try:
        for ix in batches_for(cases,maximum):
            if done[ix].all():
                continue
            batch = [cases[i] for i in ix]
            native = runner.run(batch)
            if not np.array_equal(native["z"],native_z[ix]) or np.max(np.abs(native["logits"]-native_logits[ix])) > 1e-4:
                raise ValueError("Native replay disagrees with matching masked-context cache")
            arrays["z"][ix] = native["z"]
            arrays["logits_native"][ix] = native["logits"]
            restored = runner.run(batch,native["z"])
            if not np.array_equal(restored["logits"],native["logits"]) or not np.array_equal(restored["actual_replacement"],native["z"]):
                raise ValueError("T native restoration is not bitwise exact")
            arrays["logits_restored"][ix] = restored["logits"]
            for arm in ("mean","seq","full","swapped"):
                output = runner.run(batch,replacements[arm][ix])
                if not np.array_equal(output["z"],native["z"]) or not np.array_equal(output["actual_replacement"],replacements[arm][ix]):
                    raise ValueError("Patch altered the upstream native activation or dose")
                arrays["logits_"+arm][ix] = output["logits"]
                arrays["actual_"+arm][ix] = output["actual_replacement"]
            done[ix] = True
            if time.perf_counter()-last_save > 45 or done.all():
                atomic_npz(path,{**fields,**arrays,"done":done,"identity":np.asarray(identity)})
                progress("writeback_T",done,runner)
                last_save = time.perf_counter()
        if not done.all() or any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Writeback extraction incomplete")
        atomic_npz(OUT / "writeback_T.npz",{**fields,**arrays})
    finally:
        runtime = runner.release()
    # This is a mechanical float32 intervention-dose accounting check only.
    full_dose = (arrays["actual_full"].astype(np.float64)-native_z).reshape(-1,2)
    swapped_dose = (arrays["actual_swapped"].astype(np.float64)-native_z).reshape(-1,2)
    full_sum = (full_dose**2).sum(1)
    swap_sum = (swapped_dose**2).sum(1)
    relative = float(np.max(np.abs(full_sum-swap_sum)/np.maximum(1.0,full_sum)))
    if relative > 2e-6:
        raise ValueError("Equal pairwise float32 dose accounting failed")
    update_receipt("writeback_T",{"status":"COMPLETE","sites":len(cases),"groups":967,
        "native_replay_max_logit_error":float(np.max(np.abs(arrays["logits_native"]-native_logits))),
        "native_restoration_bitwise_exact_all_sites":True,
        "pair_squared_dose_max_abs_error":float(np.max(np.abs(full_sum-swap_sum))),
        "pair_squared_dose_max_scaled_error":relative,
        "actual_float32_patches_recorded":True,
        "output_sha256":sha(OUT / "writeback_T.npz"),"replacement_sha256":sha(replacement_path),
        "fit_freeze_sha256":fit,"runtime":runtime})
    print(f"WRITEBACK T COMPLETE: {OUT / 'writeback_T.npz'}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("native","writeback"))
    parser.add_argument("split",nargs="?",choices=("D","T"))
    parser.add_argument("--batch",type=int,default=12)
    parser.add_argument("--fit-authorized",action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Batch must be positive")
    OUT.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    if args.phase == "native":
        if args.split is None:
            parser.error("Native phase needs D or T")
        native(args.split,args.batch,args.fit_authorized)
    else:
        if args.split not in (None,"T"):
            parser.error("Writeback is defined only on T")
        writeback(args.batch,args.fit_authorized)
