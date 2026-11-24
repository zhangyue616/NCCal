"""Authorized offline trained-650M CPU extraction, D locks and Haar controls.

Only this directory is written. Mathematical helpers are imported from the
frozen archived exposure/rotation implementation. Standard output is technical
progress only; scientific endpoints remain in the local sealed result files.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import sys
sys.dont_write_bytecode = True
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import scipy
import torch
import transformers
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
ROOT = HERE.parent
OLD = ROOT / "protein_extension"
RSA = OLD / "rsa"
ROTATION = OLD / "review_followup/rotation/run_rotation.py"
SNAPSHOT = Path("external/caches/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
sys.path.insert(0, str(RSA))
from model_inputs import read_rows, design_matrix, grouped_folds, weights, group_means, interval
from exposure import PENALTIES, sequence_fit, residual_slopes, development_signature

BLOCKS, WIDTH, BATCH = (16, 17), 5120, 8
STAGE = "STARTING"
START = time.monotonic()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("Output is outside trained_650m")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temp.replace(path)


def savez(path, **fields):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temp.replace(path)


def tsv(path, columns, rows):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)
    temp.replace(path)


def progress(stage, **technical):
    global STAGE
    STAGE = stage
    record = {"stage": stage, "utc": utc(), "pid": os.getpid(), "elapsed_seconds": time.monotonic() - START,
              "cwd": str(OUT), "executable": sys.executable, **technical}
    write(OUT / "STATUS.json", record)
    print(json.dumps(record), flush=True)


class BudgetStop(Exception):
    pass


def guard(estimated_seconds=0):
    go = read(HERE / "MAIN_GO.json")
    deadline = datetime.fromisoformat(go["active_deadline_utc"])
    if go.get("status") != "AUTHORIZED_MAIN" or (deadline - datetime.now(timezone.utc)).total_seconds() <= estimated_seconds:
        raise BudgetStop("MAIN_GO hard deadline or next-stage budget guard")
    if shutil.disk_usage(OUT).free < 4 * 1024**3:
        raise BudgetStop("Less than 4 GiB free; preserve queue/output space")


def hard_stop():
    write(OUT / "HARD_STOP.json", {"utc": utc(), "stage": STAGE, "reason": "MAIN_GO hard deadline", "pid": os.getpid()})
    os._exit(124)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def preflight():
    guard()
    freeze, go = read(HERE / "FREEZE.json"), read(HERE / "MAIN_GO.json")
    if sha(HERE / "PROTOCOL_C.md") != freeze["protocol_sha256"] or sha(HERE / "FREEZE.json") != go["freeze_receipt_sha256"]:
        raise ValueError("Authorization/protocol freeze changed")
    critical = [ROTATION, RSA / "exposure.py", RSA / "model_inputs.py", RSA / "paired_rsa.py", RSA / "extract_axes.py",
                OLD / "structure/residue_features.tsv", OLD / "structure/matched_pairs.tsv"]
    for path in critical:
        expected = next(v for k, v in freeze["files"].items() if Path(k).resolve() == path.resolve())
        if sha(path) != expected:
            raise ValueError("A frozen input/source changed")
    if (torch.__version__, transformers.__version__, np.__version__, scipy.__version__) != ("2.7.1+cpu", "4.44.2", "1.26.4", "1.14.1"):
        raise ValueError("Use the archived CPU environment")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    deadline = datetime.fromisoformat(go["active_deadline_utc"])
    timer = threading.Timer(max(0, (deadline - datetime.now(timezone.utc)).total_seconds()), hard_stop)
    timer.daemon = True
    timer.start()
    receipt = {"protocol_sha256": freeze["protocol_sha256"], "freeze_sha256": sha(HERE / "FREEZE.json"),
               "script_sha256": sha(Path(__file__)), "snapshot": str(SNAPSHOT), "threads": 2,
               "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
               "transformers": transformers.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
               "inference": "CPU float32, eval, eager; float64 Haar and regression", "active_deadline_utc": go["active_deadline_utc"],
               "free_disk_bytes_at_start": shutil.disk_usage(OUT).free,
               "source_sha256": {str(p): sha(p) for p in critical}}
    if (OUT / "RUN_IDENTITY.json").exists():
        old = read(OUT / "RUN_IDENTITY.json")
        for key in ("protocol_sha256", "freeze_sha256", "script_sha256", "snapshot", "threads", "torch", "transformers", "numpy", "scipy"):
            if old[key] != receipt[key]:
                raise ValueError("Resume identity differs")
    else:
        write(OUT / "RUN_IDENTITY.json", receipt)
    return module("trained650_archived_rotation", ROTATION)


class Development:
    def __init__(self, rotation):
        self.rows, self.cohort = read_rows("D")
        self.signature = development_signature(self.rows)
        self.groups = np.asarray([r[2] for r in self.rows])
        self.y = np.asarray([r[5] for r in self.rows])
        if (len(self.rows), len(np.unique(self.groups))) != (20974, 325):
            raise ValueError("D composition changed")
        self.x, self.w = design_matrix(self.rows, self.cohort), weights(self.groups)
        fold = grouped_folds(self.groups)
        self.splits = [(fold != i, fold == i) for i in range(4)]
        self.losses = {}
        for penalty in PENALTIES:
            prediction = np.empty_like(self.y)
            for train, valid in self.splits:
                prediction[valid] = self.x[valid] @ sequence_fit(self.x[train], self.y[train], weights(self.groups[train]), penalty)
            self.losses[penalty] = float(group_means(np.abs(self.y - prediction), self.groups).mean())
        self.penalty = min(PENALTIES, key=self.losses.get)
        self.coef = sequence_fit(self.x, self.y, self.w, self.penalty)
        self.residuals = []
        for train, valid in self.splits:
            coef = sequence_fit(self.x[train], self.y[train], weights(self.groups[train]), self.penalty)
            self.residuals.append((self.y[train] - self.x[train] @ coef, self.y[valid] - self.x[valid] @ coef))
        self.dp = rotation.pairs(self.rows, self.cohort, "D")
        if len(self.dp[3]) != 195:
            raise ValueError("D pair composition changed")
        savez(OUT / "sequence_model.npz", coef=self.coef)
        write(OUT / "D_ROWS.json", {"development_signature": self.signature,
              "row_count": len(self.rows), "shape": [len(self.rows), WIDTH], "groups": 325, "columns": ["acc", "pos", "group", "aa", "secondary", "rsa", "remote_contacts"],
              "row_order": "model_inputs.read_rows: group, accession, one-based position",
              "row_keys": [[r[0], r[1], r[2]] for r in self.rows], "rows": [list(r) for r in self.rows]})


def load_model():
    guard(60)
    weight = SNAPSHOT / "model.safetensors"
    provenance = read(HERE / "WEIGHTS_OFFLINE.json")
    expected = next(f["sha256"] for f in provenance["target_650M"]["snapshots"][0]["files"] if f["name"] == weight.name)
    if sha(weight) != expected:
        raise ValueError("650M source weight identity differs")
    net = transformers.EsmForMaskedLM.from_pretrained(str(SNAPSHOT), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager").cpu().float().eval()
    net.requires_grad_(False)
    if (net.config.num_hidden_layers, net.config.hidden_size, net.config.intermediate_size) != (33, 1280, WIDTH):
        raise ValueError("650M model dimensions changed")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError("Trained path must remain CPU float32")
    tied = net.esm.embeddings.word_embeddings.weight.data_ptr() == net.lm_head.decoder.weight.data_ptr()
    if not tied:
        raise ValueError("Pretrained model embedding/head not tied")
    digest = hashlib.sha256()
    for key, value in net.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    identity = {"loaded_utc": utc(), "snapshot": str(SNAPSHOT), "revision": SNAPSHOT.name,
                "weight_sha256": expected, "state_sha256": digest.hexdigest(), "embedding_lm_head_tied": tied,
                "blocks": 33, "hidden_size": 1280, "ffn_width": WIDTH, "cpu_float32_eval": True,
                "torch": torch.__version__, "transformers": transformers.__version__}
    path = OUT / "MODEL_IDENTITY.json"
    if path.exists() and read(path)["state_sha256"] != identity["state_sha256"]:
        raise ValueError("Resume loaded model state changed")
    if not path.exists():
        write(path, identity)
    vocab = {token: i for i, token in enumerate((SNAPSHOT / "vocab.txt").read_text().splitlines())}
    progress("MODEL_LOADED", cpu_float32=True, weight_identity_matches=True)
    return net, vocab


class AtLastBlock(Exception):
    pass


def extract(net, vocab, rows, cohort, split, vectors=None):
    signature = development_signature(rows)
    receipt_path = OUT / f"{split}_EXTRACTION.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["row_signature"] != signature:
            raise ValueError("Existing extraction rows differ")
        return {b: np.load(OUT / receipt["files"][str(b)], mmap_mode="r", allow_pickle=False) for b in BLOCKS}
    by_acc = {}
    for i, r in enumerate(rows):
        by_acc.setdefault(r[0], []).append((i, r[1] - 1))
    accessions = sorted(by_acc, key=lambda acc: (len(cohort[acc][2]), acc))
    state_path = OUT / f"{split}_EXTRACTION_PROGRESS.json"
    state = read(state_path) if state_path.exists() else {"row_signature": signature, "completed_accessions": [], "seconds": 0.0}
    if state["row_signature"] != signature:
        raise ValueError("Partial extraction rows differ")
    done = set(state["completed_accessions"])
    names = {b: f"D_BLOCK{b}.npy" if split == "D" else f"T_BLOCK{b}_PROJECTIONS.npy" for b in BLOCKS}
    arrays, partial = {}, {}
    shape = (len(rows), WIDTH if vectors is None else 10)
    dtype = np.float32 if vectors is None else np.float64
    for b in BLOCKS:
        final = OUT / names[b]
        partial[b] = final.with_name(final.name + ".partial")
        if final.exists() and len(done) == len(accessions):
            arrays[b] = np.load(final, mmap_mode="r", allow_pickle=False)
        elif partial[b].exists():
            arrays[b] = np.lib.format.open_memmap(partial[b], mode="r+")
        else:
            if done:
                raise ValueError("Resume has completed proteins but a matrix is absent")
            arrays[b] = np.lib.format.open_memmap(partial[b], mode="w+", dtype=dtype, shape=shape)
        if arrays[b].shape != shape or arrays[b].dtype != dtype:
            raise ValueError("Partial matrix shape/dtype differs")
    capture = {}
    def hook(block):
        def inner(_module, _args, output):
            if output.device.type != "cpu" or output.dtype != torch.float32 or output.shape[-1] != WIDTH:
                raise ValueError("Unexpected trained CPU intermediate tensor")
            capture[block] = output.detach().numpy()
            if block == BLOCKS[-1]:
                raise AtLastBlock()
        return inner
    handles = [net.esm.encoder.layer[b - 1].intermediate.register_forward_hook(hook(b)) for b in BLOCKS]
    started = time.monotonic()
    previous_batch = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, len(accessions), BATCH):
                ids = accessions[start:start + BATCH]
                if set(ids) <= done:
                    continue
                guard(max(30, previous_batch * 1.2))
                tick = time.monotonic()
                tokens = [[vocab["<cls>"], *[vocab[c] for c in cohort[a][2]], vocab["<eos>"]] for a in ids]
                inp = torch.full((len(ids), max(map(len, tokens))), vocab["<pad>"], dtype=torch.long)
                for k, row in enumerate(tokens):
                    inp[k, :len(row)] = torch.tensor(row)
                if bool((inp == vocab["<mask>"]).any()):
                    raise ValueError("Natural extraction must have no mask")
                capture.clear()
                try:
                    net.esm(input_ids=inp, attention_mask=inp.ne(vocab["<pad>"]).long())
                except AtLastBlock:
                    pass
                if set(capture) != set(BLOCKS):
                    raise ValueError("Both CPU hooks must run")
                for b in BLOCKS:
                    for k, acc in enumerate(ids):
                        order, positions = np.asarray(by_acc[acc]).T
                        values = capture[b][k, positions + 1]
                        arrays[b][order] = values if vectors is None else values @ vectors[b]
                    arrays[b].flush()
                done.update(ids)
                previous_batch = time.monotonic() - tick
                state.update(completed_accessions=sorted(done), seconds=state.get("seconds", 0.0) + previous_batch)
                write(state_path, state)
                progress(f"{split}_NATURAL_CPU", proteins_complete=len(done), proteins_total=len(accessions),
                         last_batch_seconds=previous_batch, saved_width=shape[1])
    finally:
        for handle in handles:
            handle.remove()
    for b in BLOCKS:
        if not np.isfinite(arrays[b]).all():
            raise ValueError("Nonfinite extracted values")
        if hasattr(arrays[b], "flush"):
            arrays[b].flush()
    arrays.clear()
    gc.collect()
    for b in BLOCKS:
        if partial[b].exists():
            partial[b].replace(OUT / names[b])
    write(receipt_path, {"status": "COMPLETE", "split": split, "row_signature": signature,
          "shape_each_block": list(shape), "dtype": str(np.dtype(dtype)), "files": {str(b): names[b] for b in BLOCKS},
          "proteins": len(accessions), "residues": len(rows), "seconds": state["seconds"], "finished_utc": utc(),
          "CPU_float32_two_threads": True, "shared_forward_blocks": list(BLOCKS), "batch_size": BATCH,
          "full_T_width_saved": False, "before_T_all_D_locks": split == "D" or (OUT / "D_SELECTION_LOCK.json").exists()})
    progress(f"{split}_EXTRACTION_COMPLETE", shape_each_block=list(shape), proteins=len(accessions),
             elapsed_this_invocation_seconds=time.monotonic() - started)
    return {b: np.load(OUT / names[b], mmap_mode="r", allow_pickle=False) for b in BLOCKS}


def fit_basis(ctx, rotation, block, basis, z, vector_source=None):
    folder = OUT / f"b{block}" / ("native" if basis == 0 else f"haar_{basis:02d}")
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / "SELECTION_LOCK.json"
    if lock_path.exists():
        old = read(lock_path)
        if old["development_data_signature"] != ctx.signature or old["basis_index"] != basis:
            raise ValueError("Existing basis lock differs")
        return old
    guard(180)
    tick = time.monotonic()
    losses = np.zeros(WIDTH)
    for (train, valid), (rt, rv) in zip(ctx.splits, ctx.residuals):
        guard(30)
        mu, beta = residual_slopes(z[train], rt, weights(ctx.groups[train]))
        ids = np.flatnonzero(valid)
        for start in range(0, len(ids), 512):
            take = ids[start:start + 512]
            error = rv[start:start + 512, None] - (z[take] - mu) * beta
            losses += ctx.w[take] @ np.abs(error)
    axis = int(np.argmin(losses))
    selected = z[:, axis].astype(np.float64)
    mu, beta = residual_slopes(z[:, [axis]], ctx.y - ctx.x @ ctx.coef, ctx.w)
    early, late, pair_x, pair_y, pair_groups = ctx.dp
    secondary = rotation.pair_fit_d(pair_x, selected[early] - selected[late], pair_y, pair_groups)
    vector = np.eye(1, WIDTH, axis).reshape(-1) if basis == 0 else vector_source[:, axis].copy()
    save(folder / "D_axis_losses.npy", losses)
    save(folder / "D_selected.npy", selected)
    save(folder / "VECTOR.npy", vector)
    savez(folder / "sequence_model.npz", coef=ctx.coef, mu=mu, beta=beta)
    record = {"model": "650m", "block_one_based": block, "basis": "native" if basis == 0 else f"haar_{basis:02d}",
              "basis_index": basis, "axis_zero_based": axis, "axis_candidates": WIDTH,
              "cv_augmented_mae": float(losses[axis]), "cv_baseline_mae": ctx.losses[ctx.penalty],
              "penalty": ctx.penalty, "mean": float(mu[0]), "slope": float(beta[0]),
              "development_data_signature": ctx.signature, "selected_utc": utc(),
              "before_confirmation_extraction": True, "secondary_readout": secondary,
              "files": {name: str(folder / name) for name in ("D_axis_losses.npy", "D_selected.npy", "VECTOR.npy", "sequence_model.npz")},
              "seconds": time.monotonic() - tick}
    write(lock_path, record)
    progress("D_BASIS_LOCKED", block=block, basis_index=basis, locked_basis_count=sum(1 for p in OUT.glob("b*/*/SELECTION_LOCK.json")),
             fit_seconds=record["seconds"])
    return record


def lock_development(ctx, rotation, matrices):
    final = OUT / "D_SELECTION_LOCK.json"
    if final.exists():
        lock = read(final)
        if lock["development_signature"] != ctx.signature or lock["basis_locks"] != 20:
            raise ValueError("Combined D lock differs")
        return lock
    for b in BLOCKS:
        fit_basis(ctx, rotation, b, 0, matrices[b])
    rotations = OUT / "rotations"
    rotations.mkdir(exist_ok=True)
    rotation_records = []
    for r in range(1, 10):
        guard(300)
        path, record_path = rotations / f"haar_{r:02d}.npy", rotations / f"haar_{r:02d}.json"
        tick = time.monotonic()
        if path.exists() and record_path.exists():
            q, record = np.load(path, allow_pickle=False), read(record_path)
            if q.shape != (WIDTH, WIDTH) or q.dtype != np.float64 or record["seed"] != 2026111800 + r:
                raise ValueError("Existing Haar matrix differs")
        else:
            rng = np.random.default_rng(2026111800 + r)
            q, rr = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
            q *= np.where(np.diag(rr) >= 0, 1., -1.)[None, :]
            del rr
            error = float(np.max(np.abs(q.T @ q - np.eye(WIDTH))))
            if error >= 1e-12:
                raise ValueError("Haar matrix not orthogonal")
            save(path, q)
            record = {"r": r, "path": str(path), "seed": 2026111800 + r, "sha256": sha(path), "width": WIDTH,
                      "max_orthogonality_error": error, "dtype": "float64", "convention": "Gaussian QR with diagonal sign correction; Z@Q",
                      "shared_blocks": list(BLOCKS), "generation_seconds": time.monotonic() - tick}
            write(record_path, record)
        rotation_records.append(record)
        progress("HAAR_READY", basis_index=r, shared_blocks=list(BLOCKS), generation_seconds=record["generation_seconds"])
        for b in BLOCKS:
            folder = OUT / f"b{b}" / f"haar_{r:02d}"
            if (folder / "SELECTION_LOCK.json").exists():
                continue
            guard(180)
            zz = matrices[b] @ q
            fit_basis(ctx, rotation, b, r, zz, q)
            del zz
            gc.collect()
        del q
        gc.collect()
    blocks = {}
    for b in BLOCKS:
        records = [read(OUT / f"b{b}" / ("native" if r == 0 else f"haar_{r:02d}") / "SELECTION_LOCK.json") for r in range(10)]
        selected = np.column_stack([np.load(Path(r["files"]["D_selected.npy"]), allow_pickle=False) for r in records])
        vectors = np.column_stack([np.load(Path(r["files"]["VECTOR.npy"]), allow_pickle=False) for r in records])
        path = OUT / f"b{b}/D_FROZEN.npz"
        savez(path, selected=selected, vectors=vectors, mu=np.array([r["mean"] for r in records]),
              beta=np.array([r["slope"] for r in records]), groups=ctx.groups)
        blocks[str(b)] = {"width": WIDTH, "records": records, "D_FROZEN_path": str(path), "frozen_sha256": sha(path)}
    lock = {"locked_utc": utc(), "basis_locks": 20, "blocks": blocks, "development_signature": ctx.signature,
            "D_groups": 325, "D_residues": 20974, "D_pair_groups": 195, "sequence_penalty": ctx.penalty,
            "sequence_penalty_cv_mae": ctx.losses, "all_pair_coefficients_frozen": True,
            "new_T_outcomes_computed": False, "rotations": rotation_records, "model_identity_sha256": sha(OUT / "MODEL_IDENTITY.json")}
    write(final, lock)
    progress("ALL_TWENTY_D_LOCKS_COMPLETE", blocks=list(BLOCKS), basis_locks=20, paired_fits_frozen=True)
    return lock


def score(ctx, rotation, lock, projections):
    rows, _ = read_rows("T")
    groups, y = np.asarray([r[2] for r in rows]), np.asarray([r[5] for r in rows])
    if (len(rows), len(np.unique(groups))) != (173859, 1099):
        raise ValueError("T composition changed")
    base = design_matrix(rows, ctx.cohort) @ ctx.coef
    bgroup = group_means(np.abs(y - base), groups)
    de, dl, dx, dy, dgroups = ctx.dp
    te, tl, tx, ty, tgroups = rotation.pairs(rows, ctx.cohort, "T")
    if len(tgroups) != 967:
        raise ValueError("T pair composition changed")
    kernel = (tx @ dx.T).toarray()
    target = (ty + 1) / 2
    all_blocks, flat = {}, []
    for b in BLOCKS:
        guard(60)
        frozen = np.load(lock["blocks"][str(b)]["D_FROZEN_path"], allow_pickle=False)
        values = projections[b]
        basis_results, primary_gains, pair_gains = [], [], []
        for i, rec in enumerate(lock["blocks"][str(b)]["records"]):
            folder = OUT / f"b{b}" / rec["basis"]
            aug = base + (values[:, i] - rec["mean"]) * rec["slope"]
            a = group_means(np.abs(y - aug), groups)
            gain = bgroup - a
            dz = frozen["selected"][de, i] - frozen["selected"][dl, i]
            tz = values[te, i] - values[tl, i]
            preds = [rotation.pair_predict(p, kernel, dz, tz) for p in rec["secondary_readout"]]
            losses = [(target - p) ** 2 for p in preds]
            pgain = losses[0] - losses[1]
            correct = int(np.sum(((preds[1] > .5) & (target == 1)) | ((preds[1] < .5) & (target == 0))))
            ties = int(np.sum(preds[1] == .5))
            buried, exposed = np.where(ty == 1, te, tl), np.where(ty == 1, tl, te)
            counts = {"exposed_more_active": int(np.sum(values[exposed, i] > values[buried, i])),
                      "buried_more_active": int(np.sum(values[buried, i] > values[exposed, i])),
                      "activation_ties": int(np.sum(values[buried, i] == values[exposed, i])), "pairs": 967}
            primary_path, paired_path = folder / "group_errors.tsv", folder / "buried_exposed_group_errors.tsv"
            tsv(primary_path, ["group", "sequence_mae", "augmented_mae", "gain"], zip(np.unique(groups), bgroup, a, gain))
            tsv(paired_path, ["group", "target", "sequence_probability", "augmented_probability", "sequence_brier", "augmented_brier", "gain"],
                zip(tgroups, target, *preds, *losses, pgain))
            result = {"selection": rec, "T_groups": 1099, "T_residues": 173859, "T_pairs": 967,
                      "sequence_mae": float(bgroup.mean()), "augmented_mae": float(a.mean()), "mae_gain": float(gain.mean()),
                      "sequence_brier": float(losses[0].mean()), "augmented_brier": float(losses[1].mean()), "brier_gain": float(pgain.mean()),
                      "pair_correct": correct, "pair_ties": ties, "pair_incorrect": 967 - correct - ties,
                      "pair_penalties": [p["penalty"] for p in rec["secondary_readout"]], "T_buried_exposed_activation_counts": counts,
                      "primary_CI95": interval(gain), "secondary_CI95": interval(pgain),
                      "primary_group_file": str(primary_path), "paired_group_file": str(paired_path)}
            write(folder / "RESULT.json", result)
            row = {"model": "650m", "block_one_based": b, "basis": rec["basis"], "axis_zero_based": rec["axis_zero_based"],
                   "primary": {"groups": 1099, "sequence_mae": result["sequence_mae"], "augmented_mae": result["augmented_mae"], "gain": result["mae_gain"], "CI95": result["primary_CI95"]},
                   "secondary": {"groups": 967, "sequence_brier": result["sequence_brier"], "augmented_brier": result["augmented_brier"], "gain": result["brier_gain"], "CI95": result["secondary_CI95"]},
                   "result_path": str(folder / "RESULT.json"), "primary_group_file": str(primary_path), "paired_group_file": str(paired_path)}
            basis_results.append(row)
            primary_gains.append(gain)
            pair_gains.append(pgain)
            flat.append([b, rec["basis"], rec["axis_zero_based"], result["sequence_mae"], result["augmented_mae"], result["mae_gain"], result["sequence_brier"], result["augmented_brier"], result["brier_gain"], correct, ties, 967 - correct - ties])
        comparison = {}
        for endpoint, gain_matrix, ids in (("primary", np.asarray(primary_gains), np.unique(groups)), ("secondary", np.asarray(pair_gains), tgroups)):
            native, rotated = gain_matrix[0], gain_matrix[1:].mean(axis=0)
            mean_values = gain_matrix.mean(axis=1)
            low = int(1 + np.sum(mean_values[1:] > mean_values[0]))
            high = int(np.sum(mean_values >= mean_values[0]))
            comparison[endpoint] = {name: {"gain": float(v.mean()), "CI95": interval(v)} for name, v in
                                   (("native", native), ("mean_nine_rotated", rotated), ("native_minus_mean_rotated", native - rotated))}
            comparison[endpoint].update(native_descriptive_rank_of_ten=low, native_rank_occupied_interval=[low, high],
                                        native_tied_basis_count=high - low + 1, groups=len(native), interpretation="Descriptive rank conditional on this trained block and nine Haar bases; not an exact p value")
            path = OUT / f"b{b}/{endpoint}_native_vs_mean_groups.tsv"
            tsv(path, ["group", "native_gain", "mean_nine_rotated_gain", "native_minus_mean_rotated_gain"], zip(ids, native, rotated, native - rotated))
            comparison[endpoint]["paired_group_file"] = str(path)
        all_blocks[str(b)] = {"native_result": read(OUT / f"b{b}/native/RESULT.json"), "native_result_path": str(OUT / f"b{b}/native/RESULT.json"),
                              "basis_results": basis_results, "comparison": comparison}
    report = {"status": "COMPLETE", "completed_utc": utc(), "blocks": all_blocks,
              "D_lock_sha256": sha(OUT / "D_SELECTION_LOCK.json"), "trained_model_identity": read(OUT / "MODEL_IDENTITY.json"),
              "bootstrap": "Paired groups, 2000 draws, seed 20261118. Mean nine rotated gains within group before bootstrapping.",
              "main_block": 16, "descriptive_block": 17, "no_untrained_test_computed_here": True}
    write(OUT / "TRAINED_RESULTS.json", report)
    tsv(OUT / "BASIS_RESULTS.tsv", ["block", "basis", "axis", "sequence_mae", "augmented_mae", "mae_gain", "sequence_brier", "augmented_brier", "brier_gain", "pair_correct", "pair_ties", "pair_incorrect"], flat)
    write(OUT / "COMPLETE.json", {"status": "COMPLETE", "finished_utc": utc(), "blocks": list(BLOCKS), "basis_results": 20,
          "all_D_locked_before_T": True, "CPU_float32_two_threads": True, "T_saved_width_per_block": 10,
          "report": str(OUT / "TRAINED_RESULTS.json"), "D_lock": str(OUT / "D_SELECTION_LOCK.json"),
          "elapsed_this_invocation_seconds": time.monotonic() - START, "remaining_disk_bytes": shutil.disk_usage(OUT).free})
    progress("COMPLETE", blocks=list(BLOCKS), bases_per_block=10, scientific_results_sealed=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(2):
        rotation = preflight()
        if (OUT / "COMPLETE.json").exists():
            progress("ALREADY_COMPLETE", scientific_results_sealed=True)
            return
        progress("CONTEXT_PREPARATION")
        ctx = Development(rotation)
        net, vocab = load_model()
        matrices = extract(net, vocab, ctx.rows, ctx.cohort, "D")
        lock = lock_development(ctx, rotation, matrices)
        matrices.clear()
        gc.collect()
        vectors = {b: np.load(lock["blocks"][str(b)]["D_FROZEN_path"], allow_pickle=False)["vectors"] for b in BLOCKS}
        rows, cohort = read_rows("T")
        projections = extract(net, vocab, rows, cohort, "T", vectors)
        del net
        gc.collect()
        progress("SCORING_SEALED")
        score(ctx, rotation, lock, projections)


if __name__ == "__main__":
    try:
        main()
    except BudgetStop as error:
        progress("STOP_BUDGET_OR_SPACE", reason=str(error), resumable=True)
    except Exception as error:
        write(OUT / "ERROR.json", {"utc": utc(), "stage": STAGE, "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "resumable": True})
        progress("STOP_TECHNICAL_ERROR", error_type=type(error).__name__, resumable=True)
        raise
