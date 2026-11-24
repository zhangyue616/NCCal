"""Resumable D-only directions for old 8M/35M states and trained 650M.

Only the central forward queue may invoke `run --gpu-lease ...`.  No model is
created by import, inventory, or synthetic-selfcheck.  New direction values are
never parsed or printed by this runner; the frozen readout writes sealed files.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import sys
sys.dont_write_bytecode = True
import argparse
import gc
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCALE = HERE.parent
ROOT = SCALE.parent
ARCHIVE = ROOT / "untrained"
RSA = ROOT / "exposure"
REFERENCE = ARCHIVE / "run_experiment.py"
SNAPSHOT = Path("external/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
AA = "ACDEFGHIKLMNPQRSTVWY"
STATE_TOLERANCE = 1e-12


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("Direction runner output is outside its owned directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def jobs():
    result = []
    for model in ("8m", "35m"):
        for label in ("trained", *(f"init_{i:03d}" for i in range(1, 200))):
            folder = HERE / model / label
            result.append({"id": f"{model}/{label}", "model": model, "network": label,
                           "result": str(folder / "DIRECTION_SEALED.json"),
                           "arrays": str(folder / "direction_vectors_SEALED.npz"),
                           "receipt": str(folder / "COMPLETE.json"), "status": "PENDING"})
    folder = HERE / "650m/trained"
    result.append({"id": "650m/trained", "model": "650m", "network": "trained",
                   "result": str(folder / "DIRECTION_SEALED.json"),
                   "arrays": str(folder / "direction_vectors_SEALED.npz"),
                   "receipt": str(folder / "COMPLETE.json"), "status": "PENDING"})
    return result


def index_update(completed=None):
    path = HERE / "INDEX.json"
    obj = read(path) if path.exists() else {"scope": "8m/35m trained+199 old states each; 650m trained b16 only",
                                           "expected_networks": 401, "networks": jobs()}
    if completed is not None:
        found = [x for x in obj["networks"] if x["id"] == completed]
        if len(found) != 1:
            raise ValueError("Unknown direction job in index")
        found[0]["status"] = "COMPLETE"
    obj["completed_networks"] = sum(x["status"] == "COMPLETE" for x in obj["networks"])
    obj["status"] = "COMPLETE" if obj["completed_networks"] == 401 else "IN_PROGRESS"
    obj["updated_utc"] = utc()
    write(path, obj)
    if obj["status"] == "COMPLETE":
        write(HERE / "COMPLETE.json", {"status": "COMPLETE_401_DIRECTION_NETWORKS", "completed_utc": utc(),
                                      "index": str(path), "index_sha256": sha(path), "networks": 401})
    return obj


def gates():
    go, freeze = read(SCALE / "MAIN_GO.json"), read(SCALE / "FREEZE.json")
    if go.get("status") != "AUTHORIZED_MAIN" or go.get("author_message") != 'Scientific narrative is documented in PROTOCOL.md.':
        raise ValueError("Explicit MAIN_GO authorization is absent")
    if go["frozen_protocol_sha256"] != sha(SCALE / "PROTOCOL_C.md"):
        raise ValueError("Protocol changed after MAIN_GO")
    if go["freeze_receipt_sha256"] != sha(SCALE / "FREEZE.json"):
        raise ValueError("MAIN_GO freeze binding changed")
    required = [SCALE / "PROTOCOL_C.md", SCALE / "direction_readout.py", SCALE / "RSA_PROPENSITY.json",
                REFERENCE, RSA / "model_inputs.py", RSA / "exposure.py", RSA / "paired_rsa.py"]
    frozen = {str(Path(p).resolve()).lower(): h for p, h in freeze["files"].items()}
    for path in required:
        if frozen.get(str(path.resolve()).lower()) != sha(path):
            raise ValueError("A required frozen source/input changed")
    deadline = datetime.fromisoformat(go["active_deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("The global MAIN_GO deadline has elapsed")
    return go, deadline


def check_deadline(deadline, estimated_seconds):
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 1.2 * estimated_seconds:
        raise TimeoutError("The next complete network does not fit the global 12-hour deadline")


class DevelopmentContext:
    """Only D attributes; reuse the archived fixed sequence fit and folds."""
    def __init__(self, ref):
        self.d, roster = ref.read_rows("D")
        self.cohort = {a: roster[a] for a in {r[0] for r in self.d}}
        if any(x[0] != "D" for x in self.cohort.values()):
            raise ValueError("The context contains nondevelopment sequences")
        self.y = np.asarray([r[5] for r in self.d], dtype=np.float64)
        self.g = np.asarray([r[2] for r in self.d])
        if (len(self.d), len(np.unique(self.g))) != (20974, 325):
            raise ValueError("The fixed development population changed")
        self.w = ref.weights(self.g)
        self.x = ref.design_matrix(self.d, self.cohort)
        source = ARCHIVE / "8m/init_001"
        old = read(source / "SELECTION_LOCK.json")
        self.penalty, self.basecv = old["penalty"], old["cv_baseline_mae"]
        with np.load(source / "sequence_model.npz", allow_pickle=False) as data:
            self.coef = data["coef"].copy()
        if ref.development_signature(self.d) != old["development_data_signature"]:
            raise ValueError("D row/order signature differs from the original lock")
        folds = ref.grouped_folds(self.g)
        self.splits = [(folds != k, folds == k) for k in range(4)]
        self.residuals = []
        for train, val in self.splits:
            fit = ref.sequence_fit(self.x[train], self.y[train], ref.weights(self.g[train]), self.penalty)
            self.residuals.append((self.y[train] - self.x[train] @ fit, self.y[val] - self.x[val] @ fit))
        self.dp = ref.Context.pairs(self, self.d, "D")
        if len(self.dp[0]) != 195:
            raise ValueError("The fixed 195 development pairs changed")


def state_hash(net):
    digest = hashlib.sha256()
    for key, value in net.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def old_folder(model, label):
    return ARCHIVE / (f"trained_replay/{model}" if label == "trained" else f"{model}/{label}")


def load_small(ref, model, label, folder):
    torch = ref.torch
    path = ROOT / "assets/esm2_8m" if model == "8m" else RSA / "assets/esm2_35m"
    config = ref.EsmConfig.from_pretrained(str(path), local_files_only=True)
    config._attn_implementation = "eager"
    if label == "trained":
        net = ref.EsmForMaskedLM.from_pretrained(str(path), local_files_only=True,
                    dtype=torch.float32, attn_implementation="eager").float().eval()
        expected = None
    else:
        index = int(label.split("_")[1])
        expected = read(old_folder(model, label) / "MODEL_IDENTITY.json")
        if expected["seed"] != 20261118000 + index:
            raise ValueError("An archived initialization seed changed")
        if (ref.torch.__version__, ref.transformers.__version__) != (expected["torch"], expected["transformers"]):
            raise ValueError("The original initialization runtime is not available")
        if sha(path / "config.json") != expected["config_sha256"]:
            raise ValueError("The archived initialization configuration changed")
        torch.manual_seed(expected["seed"])
        with torch.device("cpu"):
            net = ref.EsmForMaskedLM(config).float().eval()
    identity = state_hash(net)
    if expected is not None and identity != expected["state_sha256"]:
        raise ValueError("Recreated network differs from its original exact state hash")
    tied = net.esm.embeddings.word_embeddings.weight.data_ptr() == net.lm_head.decoder.weight.data_ptr()
    if not tied or any(p.device.type != "cpu" for p in net.parameters()):
        raise ValueError("The original CPU construction/tied-head invariant changed")
    receipt = {"state_sha256": identity, "matches_original_state": None if expected is None else identity == expected["state_sha256"],
               "state_reference": None if expected is None else str(old_folder(model, label) / "MODEL_IDENTITY.json"),
               "config_sha256": sha(path / "config.json"), "CPU_construction": True,
               "torch": ref.torch.__version__, "transformers": ref.transformers.__version__,
               "attention": "eager", "dtype": "float32", "TF32": False,
               "embedding_lm_head_tied": True, "completed_utc": utc()}
    write(folder / "MODEL_IDENTITY.json", receipt)
    net.requires_grad_(False)
    vocab = {token: i for i, token in enumerate((path / "vocab.txt").read_text().splitlines())}
    return net.to("cuda"), vocab, receipt


def verify_development(ref, ctx, z, model, label, folder):
    source = old_folder(model, label)
    expected = read(source / "SELECTION_LOCK.json")
    with np.load(source / "sequence_model.npz", allow_pickle=False) as data:
        if not np.array_equal(data["coef"], ctx.coef):
            raise ValueError("The archived sequence-reference coefficients changed")
    replay = folder / "development_replay"
    replay.mkdir(parents=True, exist_ok=True)
    actual, selected = ref.Context.select(ctx, z, replay, model)
    archived_losses = np.load(source / "D_axis_losses.npy", allow_pickle=False)
    reproduced_losses = np.load(replay / "D_axis_losses.npy", allow_pickle=False)
    old_selected = np.load(source / "D_selected.npy", allow_pickle=False)
    loss_error = float(np.max(np.abs(archived_losses - reproduced_losses)))
    activation_error = float(np.max(np.abs(old_selected.astype(float) - selected.astype(float))))
    slope_error = abs(actual["slope"] - expected["slope"])
    mean_error = abs(actual["mean"] - expected["mean"])
    if actual["axis_zero_based"] != expected["axis_zero_based"] or actual["penalty"] != expected["penalty"]:
        raise ValueError("The original development selection did not replay")
    if (actual["development_data_signature"] != expected["development_data_signature"] or
            max(loss_error, slope_error, mean_error, activation_error) > STATE_TOLERANCE or
            np.sign(actual["slope"]) != np.sign(expected["slope"])):
        raise ValueError("Original D losses/activation polarity failed exact-tolerance replay")
    return int(expected["axis_zero_based"]), {"status": "PASS", "D_only": True,
             "source_lock": str(source / "SELECTION_LOCK.json"), "source_lock_sha256": sha(source / "SELECTION_LOCK.json"),
             "all_axis_losses_max_abs_error": loss_error, "selected_activation_max_abs_error": activation_error,
             "selected_slope_abs_error": slope_error, "selected_mean_abs_error": mean_error,
             "selected_axis_matches": True, "polarity_matches": True, "absolute_tolerance": STATE_TOLERANCE}


def load_650(ref, ctx, folder):
    source = SCALE / "trained_650m"
    required = [source / "D_BLOCK16.npy", source / "D_ROWS.json", source / "b16/native/SELECTION_LOCK.json",
                source / "b16/native/D_selected.npy", source / "MODEL_IDENTITY.json"]
    if not all(p.exists() for p in required):
        raise FileNotFoundError("Trained 650M D matrix/row lock/state identity are not all ready")
    rows = read(source / "D_ROWS.json")
    if rows["development_signature"] != ref.development_signature(ctx.d):
        raise ValueError("650M development row signature differs")
    selection = read(source / "b16/native/SELECTION_LOCK.json")
    if selection["development_data_signature"] != rows["development_signature"]:
        raise ValueError("650M D selection lock has a different population")
    z = np.load(source / "D_BLOCK16.npy", mmap_mode="r", allow_pickle=False)
    if z.shape != (20974, 5120) or z.dtype != np.float32:
        raise ValueError("650M trained D activation shape/dtype changed")
    axis = int(selection["axis_zero_based"])
    selected = np.load(source / "b16/native/D_selected.npy", allow_pickle=False)
    if not np.array_equal(selected, z[:, axis]):
        raise ValueError("650M selected activation column differs from its D lock")
    torch = ref.torch
    net = ref.EsmForMaskedLM.from_pretrained(str(SNAPSHOT), local_files_only=True,
                    dtype=torch.float32, attn_implementation="eager").float().eval()
    identity = state_hash(net)
    expected = read(source / "MODEL_IDENTITY.json")
    if identity != expected["state_sha256"]:
        raise ValueError("650M trained network differs from the natural-analysis state")
    model_identity = {"state_sha256": identity, "matches_original_state": True,
                      "state_reference": str(source / "MODEL_IDENTITY.json"), "snapshot": str(SNAPSHOT),
                      "CPU_construction": True, "attention": "eager", "dtype": "float32", "TF32": False}
    write(folder / "MODEL_IDENTITY.json", model_identity)
    net.requires_grad_(False)
    vocab = {token: i for i, token in enumerate((SNAPSHOT / "vocab.txt").read_text().splitlines())}
    return net.to("cuda"), vocab, z, axis, model_identity, {
        "status": "PASS", "D_only": True, "source_lock": str(source / "b16/native/SELECTION_LOCK.json"),
        "source_lock_sha256": sha(source / "b16/native/SELECTION_LOCK.json"),
        "development_signature_matches": True, "selected_column_exact": True}


def completed(model, label):
    path = HERE / model / label / "COMPLETE.json"
    if not path.exists():
        return None
    receipt = read(path)  # Technical receipt only, never the sealed direction result.
    if receipt["status"] != "COMPLETE" or receipt["direction_source_sha256"] != sha(SCALE / "direction_readout.py"):
        raise ValueError("Completed direction receipt has a different frozen implementation")
    for filename, digest in receipt["sealed_files_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("A completed sealed direction artifact changed")
    index_update(f"{model}/{label}")
    return receipt


def run_one(ref, direction, ctx, propensity, model, label, deadline, lease, estimate):
    done = completed(model, label)
    if done is not None:
        return {"status": "ALREADY_COMPLETE", "model": model, "network": label,
                "total_seconds": done["timing"]["total_seconds"]}
    check_deadline(deadline, estimate)
    folder = HERE / model / label
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "DIRECTION_SEALED.json").exists() or (folder / "direction_vectors_SEALED.npz").exists():
        raise RuntimeError("Unreceipted sealed output exists; preserve it for targeted technical recovery")
    error_file = folder / "ERROR.json"
    attempts = read(error_file).get("attempts", 0) if error_file.exists() else 0
    if attempts >= 3:
        raise RuntimeError("This same network has reached the three-attempt safety limit")
    start, started = time.perf_counter(), utc()
    write(folder / "RUN_STARTED.json", {"started_utc": started, "PID": os.getpid(), "gpu_lease": lease,
          "deadline_utc": deadline.isoformat(), "runner_sha256": sha(__file__),
          "direction_source_sha256": sha(SCALE / "direction_readout.py"), "attempt": attempts + 1})
    net = None
    try:
        initialize = time.perf_counter()
        if model == "650m":
            net, vocab, z, axis, identity, replay = load_650(ref, ctx, folder)
            init_seconds = time.perf_counter() - initialize
            extraction_seconds = 0.0
            selection_seconds = 0.0
            block = 16
        else:
            net, vocab, identity = load_small(ref, model, label, folder)
            init_seconds = time.perf_counter() - initialize
            extract_start = time.perf_counter()
            z = (ref.activations(ctx.d, model, "D", all_axes=True) if label == "trained"
                 else ref.extract(net, vocab, ctx.d, ctx.cohort))
            extraction_seconds = time.perf_counter() - extract_start
            selection_start = time.perf_counter()
            axis, replay = verify_development(ref, ctx, z, model, label, folder)
            selection_seconds = time.perf_counter() - selection_start
            block = net.config.num_hidden_layers // 2
        write(folder / "DEVELOPMENT_REPLAY_TECHNICAL.json", replay)
        result = direction.run_direction(net, vocab, ctx, z, axis, block, propensity, folder,
                                         site_batch=8, axis_batch=16)
        # The returned dictionary contains no P1/P2 or selected-axis values.
        write(folder / "DIRECTION_RETURN_TECHNICAL.json", result)
        timing = {"initialization_state_hash_transfer_seconds": init_seconds,
                  "D_natural_activation_seconds": extraction_seconds,
                  "D_lock_replay_seconds": selection_seconds,
                  "direction": result["timing"], "total_seconds": time.perf_counter() - start}
        sealed_hashes = {p: sha(p) for p in result["sealed_files"]}
        receipt = {"status": "COMPLETE", "model": model, "network": label, "started_utc": started,
                   "completed_utc": utc(), "timing": timing, "technical": result["technical"],
                   "runner_sha256": sha(__file__), "direction_source_sha256": sha(SCALE / "direction_readout.py"),
                   "model_state_sha256": identity["state_sha256"], "development_replay": replay,
                   "sealed_files_sha256": sealed_hashes, "scientific_values_disclosed": False}
        write(folder / "COMPLETE.json", receipt)
        index_update(f"{model}/{label}")
        return {"status": "COMPLETE", "model": model, "network": label,
                "total_seconds": timing["total_seconds"], "receipt": str(folder / "COMPLETE.json")}
    except Exception as exc:
        write(error_file, {"status": "STOP_ERROR", "attempts": attempts + 1, "utc": utc(),
                           "type": type(exc).__name__, "reason": str(exc), "scientific_values_disclosed": False})
        raise
    finally:
        if net is not None:
            del net
        gc.collect()
        ref.torch.cuda.empty_cache()
        ref.torch.cuda.synchronize()


def inventory():
    plan = jobs()
    missing = []
    for entry in plan:
        if entry["model"] == "650m":
            continue
        source = old_folder(entry["model"], entry["network"])
        for name in ("SELECTION_LOCK.json", "D_axis_losses.npy", "D_selected.npy", "sequence_model.npz"):
            if not (source / name).exists():
                missing.append(str(source / name))
        if entry["network"] != "trained" and not (source / "MODEL_IDENTITY.json").exists():
            missing.append(str(source / "MODEL_IDENTITY.json"))
    return {"status": "READY" if not missing else "MISSING_INPUTS", "planned_networks": len(plan),
            "small_archived_networks": 400, "missing_files": missing,
            "index": str(HERE / "INDEX.json"), "complete": str(HERE / "COMPLETE.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "synthetic-selfcheck", "run"))
    parser.add_argument("--model", choices=("8m", "35m", "650m"))
    parser.add_argument("--trained", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=199)
    parser.add_argument("--gpu-lease", help="Explicit identifier supplied by the sole forward queue")
    parser.add_argument("--deadline-utc", help="Optional scheduler deadline; cannot extend MAIN_GO")
    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps(inventory(), indent=2)); return
    direction = module("scale_direction_frozen", SCALE / "direction_readout.py")
    if args.command == "synthetic-selfcheck":
        assert len(jobs()) == 401 and len({x["id"] for x in jobs()}) == 401
        print(json.dumps(direction.synthetic_selfcheck(), indent=2)); return
    if not args.model or not args.gpu_lease:
        parser.error("The forward queue must supply --model and --gpu-lease")
    if args.model == "650m" and not args.trained:
        parser.error("650M null directions belong exclusively to the forward queue")
    if not 1 <= args.start <= args.stop <= 199:
        parser.error("Initialization indices must be within 1..199")
    go, deadline = gates()
    if args.deadline_utc:
        requested_deadline = datetime.fromisoformat(args.deadline_utc)
        if requested_deadline.tzinfo is None or requested_deadline > deadline:
            raise ValueError("The scheduler deadline must include timezone and cannot extend MAIN_GO")
        deadline = requested_deadline
    check_deadline(deadline, 30)
    ref = module("scale_direction_reference", REFERENCE)
    propensity = np.asarray(read(SCALE / "RSA_PROPENSITY.json")["values"], dtype=np.float64)
    ctx = DevelopmentContext(ref)
    index_update()
    labels = ["trained"] if args.trained else [f"init_{i:03d}" for i in range(args.start, args.stop + 1)]
    estimate = 60.0 if args.model == "650m" else 20.0
    for label in labels:
        result = run_one(ref, direction, ctx, propensity, args.model, label, deadline, args.gpu_lease, estimate)
        if result["status"] == "COMPLETE":
            estimate = max(estimate, float(result["total_seconds"]))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
