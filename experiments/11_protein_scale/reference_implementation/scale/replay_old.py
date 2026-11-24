"""Replay 8M/35M from their archived activations using the selector/scorer used in the pilot."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "untrained"
STRUCTURE = HERE.parent / "protein_extension"
sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD))
import run_experiment as old


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    seal = json.loads((HERE / "FREEZE.json").read_text(encoding="utf8"))
    if digest(HERE/"PROTOCOL_C.md") != seal["protocol_sha256"]:
        raise ValueError("Protocol differs from freeze")
    for path,expected in seal["files"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    c = old.Context()
    stored = old.read(OLD/"TRAINED_REPLAY.json")
    reports = {}
    for model,expected_index in (("8m",801),("35m",814)):
        folder = HERE/"replay"/model
        folder.mkdir(parents=True,exist_ok=True)
        frozen = old.read(STRUCTURE/"rsa"/"results"/model/"results.json")
        reference = frozen["selection"]
        d = old.activations(c.d,model,"D",all_axes=True)
        selected,winning_d = c.select(d,folder,model)
        del d
        for key in ("axis_zero_based","axis_candidates","penalty","development_data_signature"):
            if selected[key] != reference[key]:
                raise ValueError(f"{model}: selection mismatch for {key}")
        if selected["axis_zero_based"] != expected_index:
            raise ValueError("Unexpected archived winner")
        t = old.activations(c.t,model,"T",axis=selected["axis_zero_based"])
        result = c.score(selected,winning_d,t,folder)
        old_pairs = frozen["buried_exposed_pairs"]
        fields = {"development_CV_reference_MAE":(selected["cv_baseline_mae"],reference["cv_baseline_mae"]),
                  "development_CV_selected_MAE":(selected["cv_augmented_mae"],reference["cv_augmented_mae"]),
                  "T_sequence_MAE":(result["sequence_mae"],frozen["sequence_mae"]),
                  "T_augmented_MAE":(result["augmented_mae"],frozen["augmented_mae"]),
                  "T_MAE_gain":(result["mae_gain"],frozen["group_mae_difference"]),
                  "T_sequence_Brier":(result["sequence_brier"],old_pairs["sequence_brier"]),
                  "T_augmented_Brier":(result["augmented_brier"],old_pairs["augmented_brier"]),
                  "T_Brier_gain":(result["brier_gain"],old_pairs["improvement"])}
        check = {key:{"replayed":a,"archived":b,"abs_error":float(abs(a-b))} for key,(a,b) in fields.items()}
        max_error = max(item["abs_error"] for item in check.values())
        old_selection_result = stored["models"][model]["results"]
        if (max_error>1e-12 or result["pair_penalties"] != old_pairs["penalties"] or
                result["pair_correct"] != old_selection_result["pair_correct"] or result["pair_ties"]!=old_selection_result["pair_ties"]):
            raise ValueError(f"Replay mismatch: {model}, {check}")
        mu,beta = old.residual_slopes((-winning_d)[:,None],c.y-c.x@c.coef,c.w)
        scalar = (winning_d.astype(float)-selected["mean"])*selected["slope"]
        sign_error=float(np.max(np.abs(scalar-(-winning_d.astype(float)-mu[0])*beta[0])))
        early,late,*_ = c.dp
        dz=winning_d[early].astype(float)-winning_d[late]
        if sign_error>1e-12 or not np.array_equal(np.outer(dz,dz),np.outer(-dz,-dz)):
            raise ValueError("Polarity invariance failed")
        reports[model] = {"axis_zero_based":expected_index,"penalty":selected["penalty"],
                          "pair_penalties":result["pair_penalties"],"pair_correct":result["pair_correct"],
                          "pair_ties":result["pair_ties"],"T_groups":result["T_groups"],"T_sites":result["T_residues"],
                          "T_pairs":result["T_pairs"],"D_signature":selected["development_data_signature"],
                          "sign_flip_max_abs_error":sign_error,"checks":check,"max_abs_error":max_error}
        print(f"{model}: axis {expected_index}, max abs error {max_error:.3g}",flush=True)
    old.write(HERE/"TRAINED_REPLAY.json",{"status":"PASS","completed_utc":datetime.now(timezone.utc).isoformat(),
                    "protocol_sha256":seal["protocol_sha256"],"impl":"original Context.select and Context.score",
                    "used_archived_activations":True,"new_scales_run":False,"models":reports})


if __name__ == "__main__":
    main()
