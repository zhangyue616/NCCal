"""Single post hoc sequence-displacement swapped arm; no outcome scoring."""
from __future__ import annotations
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                  MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
import sys
sys.dont_write_bytecode = True
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
EXPECTED_ADDENDUM = "49a282a2f9687482621aa6735159fa1d42cf6fd0df288501f543b15348ead3e5"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def main():
    started_utc = utc()
    lock = json.loads((HERE / "ADDENDUM_FREEZE.json").read_text(encoding="utf-8-sig"))
    actual_addendum = sha(HERE / "ADDENDUM.md")
    if actual_addendum != EXPECTED_ADDENDUM or lock["addendum_sha256"] != EXPECTED_ADDENDUM:
        raise ValueError("Post hoc addendum changed")
    input_hashes = {name:sha(PARENT / name) for name in lock["original_inputs"]}
    if input_hashes != lock["original_inputs"]:
        raise ValueError("Original archived input changed after addendum freeze")
    sys.path.insert(0,str(PARENT))
    import native_forward as original
    original.plan_check()
    original.fit_check(True)
    cases = original.cohort_cases("T")
    fields = original.fields_for(cases)
    if len(cases) != 1934:
        raise ValueError("Expected 1,934 frozen contexts")
    archived = read_npz(PARENT / "results" / "writeback_T.npz")
    native = read_npz(PARENT / "results" / "masked_native_T.npz")
    replacements = read_npz(PARENT / "results" / "B_replacements_T.npz")
    for data in (archived,native,replacements):
        original.assert_keys(data,fields)
    z32 = native["z"]
    seq32 = archived["actual_seq"]
    if z32.dtype != np.float32 or seq32.dtype != np.float32:
        raise ValueError("Original native z and actual sequence patches must be float32")
    if not np.array_equal(z32,archived["z"]) or not np.array_equal(z32,replacements["z"]):
        raise ValueError("Native activations have inconsistent lineage")
    if not np.array_equal(seq32,replacements["seq"]) or not np.array_equal(native["logits"],archived["logits_native"]):
        raise ValueError("Original actual sequence patches or native logits disagree")
    groups, inverse, counts = np.unique(fields["group"],return_inverse=True,return_counts=True)
    if len(groups) != 967 or not np.all(counts == 2):
        raise ValueError("Expected one frozen pair per group")
    pair_index = np.argsort(inverse,kind="stable").reshape(-1,2)
    mate = np.empty(len(cases),dtype=np.int64)
    mate[pair_index[:,0]], mate[pair_index[:,1]] = pair_index[:,1], pair_index[:,0]
    if not np.array_equal(mate[mate],np.arange(len(cases))) or np.any(mate == np.arange(len(cases))):
        raise ValueError("Invalid fixed within-pair permutation")
    z64, seq64 = z32.astype(np.float64), seq32.astype(np.float64)
    original_delta = seq64 - z64
    requested = (z64 + original_delta[mate]).astype(np.float32)
    original_dose = (original_delta[pair_index]**2).sum(axis=1)
    proposed_delta = requested.astype(np.float64)-z64
    proposed_dose = (proposed_delta[pair_index]**2).sum(axis=1)
    proposed_error = np.abs(proposed_dose-original_dose)
    scaled_error = proposed_error / np.maximum(1.0,original_dose)
    if np.max(scaled_error) > 2e-6:
        raise ValueError("Requested float32 swapped sequence dose violates pair tolerance")
    output_logits = np.full((len(cases),20),np.nan,dtype=np.float32)
    actual = np.full(len(cases),np.nan,dtype=np.float32)
    native_error = 0.0
    z_error = 0.0
    runner = original.Runner()
    pid = os.getpid()
    started = time.perf_counter()
    try:
        for number, ix in enumerate(original.batches_for(cases,12)):
            batch = [cases[i] for i in ix]
            replay = runner.run(batch)
            logit_error = float(np.max(np.abs(replay["logits"]-native["logits"][ix])))
            activation_error = float(np.max(np.abs(replay["z"]-z32[ix])))
            native_error = max(native_error,logit_error)
            z_error = max(z_error,activation_error)
            if not np.array_equal(replay["z"],z32[ix]) or not np.array_equal(replay["logits"],native["logits"][ix]):
                raise ValueError("Native masked context is not a bitwise replay of the archive")
            changed = runner.run(batch,requested[ix])
            if not np.array_equal(changed["z"],z32[ix]):
                raise ValueError("New arm changed upstream native activation")
            if not np.array_equal(changed["actual_replacement"],requested[ix]):
                raise ValueError("Registered patch differs from requested float32 dose")
            actual[ix] = changed["actual_replacement"]
            output_logits[ix] = changed["logits"]
            if number % 50 == 0:
                print(json.dumps({"status":"RUNNING","batches":number+1,
                    "case_forwards":runner.case_forwards,"updated_utc":utc()}),flush=True)
        if not np.isfinite(output_logits).all() or not np.isfinite(actual).all():
            raise ValueError("New arm is incomplete")
    finally:
        runtime = runner.release()
        gc.collect()
        runner.torch.cuda.empty_cache()
        runtime["GPU_allocated_bytes_after_gc"] = int(runner.torch.cuda.memory_allocated())
    actual_delta = actual.astype(np.float64)-z64
    actual_dose = (actual_delta[pair_index]**2).sum(axis=1)
    dose_error = np.abs(actual_dose-original_dose)
    if not np.array_equal(actual,requested) or np.max(dose_error/np.maximum(1.0,original_dose)) > 2e-6:
        raise ValueError("Actual intervention fails final per-pair dose verification")
    path = HERE / "sequence_swap_T.npz"
    original.atomic_npz(path,{**fields,"logits_seq_swapped":output_logits,
                              "actual_seq_swapped":actual})
    receipt = {"status":"COMPLETE_SINGLE_POSTHOC_ARM", "started_utc":started_utc,
        "completed_utc":utc(),"PID":pid,"sites":len(cases),"groups":len(groups),
        "posthoc":True,"addendum_sha256":actual_addendum,
        "addendum_sealed_utc":lock["sealed_utc"],"original_input_hashes":input_hashes,
        "script_sha256":sha(__file__),"output_sha256":sha(path),
        "canonical_AA_order":original.AA,"batch_maximum":12,
        "native_logits_bitwise_exact_all_sites":True,
        "native_logits_max_abs_error":native_error,
        "upstream_z_bitwise_exact_all_sites":True,"upstream_z_max_abs_error":z_error,
        "patch_equals_requested_float32_all_sites":True,
        "within_pair_permutation_involution_checked":True,
        "pair_squared_dose_max_abs_error":float(np.max(dose_error)),
        "pair_squared_dose_max_scaled_error":float(np.max(dose_error/np.maximum(1.0,original_dose))),
        "pair_dose_scale":"max(1, original sequence pair squared dose sum)",
        "pair_squared_dose_original_mean":float(original_dose.mean()),
        "pair_squared_dose_swapped_mean":float(actual_dose.mean()),
        "raw_swapped_delta_vs_permuted_original_max_abs_float32_error":float(np.max(np.abs(actual_delta-original_delta[mate]))),
        "forward_seconds":time.perf_counter()-started,"runtime":runtime,
        "outcome_scoring_performed":False,"other_interventions_computed":False}
    original.atomic_json(HERE / "sequence_swap_receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"sites":len(cases),
        "native_logits_error":native_error,"upstream_z_error":z_error,
        "pair_dose_scaled_error":receipt["pair_squared_dose_max_scaled_error"],
        "output":str(path),"PID":pid}),flush=True)


if __name__ == "__main__":
    main()
