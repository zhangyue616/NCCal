"""Development-only direct-path read/write direction for the scale extension.

Importing this file performs no data/model access.  The coordinator must freeze
the protocol and authorize the timing/main run before calling run_direction().
The caller owns the model, model provenance, and execution authority.  This
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

AA = "ACDEFGHIKLMNPQRSTVWY"
SLOPE_VARIANCE_FLOOR = 1e-12  # The archived residual_slopes rule.
PROJECTION_ATOL = 2e-5


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def _npz(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    temporary.replace(path)


def _moments(z, residual, weights):
    """Use the archived float64 weighted-moment order, without refitting."""
    mean = np.zeros(z.shape[1], dtype=np.float64)
    second = np.zeros_like(mean)
    cross = np.zeros_like(mean)
    minimum = np.full_like(mean, np.inf)
    maximum = np.full_like(mean, -np.inf)
    for start in range(0, len(weights), 512):
        block = np.asarray(z[start:start + 512], dtype=np.float64)
        w = weights[start:start + 512]
        mean += w @ block
        second += w @ (block * block)
        cross += (w * residual[start:start + 512]) @ block
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
    variance = np.maximum(0.0, second - mean * mean)
    # A truly constant native axis has exactly zero population SD; do not let
    # cancellation of its raw moments manufacture a tiny displacement.
    variance[minimum == maximum] = 0.0
    covariance = cross - mean * np.dot(weights, residual)
    beta = np.divide(covariance, variance, out=np.zeros_like(variance),
                     where=variance > SLOPE_VARIANCE_FLOOR)
    sigma = np.sqrt(variance)  # Population SD, using the same group-equal weights.
    return mean, variance, beta, sigma, beta * sigma


def _rank_correlation(x, y, reason=None):
    """Average-rank Spearman; retain finite zero entries and expose degeneracy."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[good], y[good]
    out = {"n_requested": len(x), "n_finite": int(good.sum()),
           "ties": "average ranks; finite zero values retained"}
    for name, values in (("x", xv), ("y", yv)):
        _, counts = np.unique(values, return_counts=True)
        out[name + "_distinct_values"] = int(len(counts))
        out[name + "_tied_observations"] = int(counts[counts > 1].sum())
    if reason is None and len(xv) < 2:
        reason = "fewer_than_two_finite_pairs"
    if reason is None and len(np.unique(xv)) < 2:
        reason = "constant_first_rank_vector"
    if reason is None and len(np.unique(yv)) < 2:
        reason = "constant_second_rank_vector"
    if reason is not None:
        return {**out, "rho": None, "status": "UNDEFINED", "reason": reason}
    rx, ry = rankdata(xv, method="average"), rankdata(yv, method="average")
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float(np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return {**out, "rho": rho, "status": "DEFINED"}


def _log_odds(logits):
    """Strict one-versus-the-other-19 canonical-AA log odds, in float64.

    g_a = l_a - logsumexp(l_b for b != a).  This uses only the canonical
    20 logits and is not log probability, a probability ratio to the entire
    vocabulary, or a linearized/first-order head response.
    """
    import torch
    values = logits.to(dtype=torch.float64)
    if values.shape[-1] != 20:
        raise ValueError("Exactly 20 canonical amino-acid logits are required")
    diagonal = torch.eye(20, dtype=torch.bool, device=values.device)
    others = values.unsqueeze(-2).masked_fill(diagonal, -torch.inf)
    return values - torch.logsumexp(others, dim=-1)


def _development_inputs(ctx, z, propensity):
    # Do not access ctx.t, ctx.xt, ctx.yt, ctx.gt, ctx.tp or any test output.
    rows = ctx.d
    if len(rows) != 20974:
        raise ValueError("The fixed D residue count changed")
    groups = np.asarray([r[2] for r in rows])
    levels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(levels) != 325 or not np.array_equal(groups, ctx.g):
        raise ValueError("The fixed D groups changed")
    weights = 1.0 / (len(levels) * counts[inverse])
    if not np.array_equal(weights, np.asarray(ctx.w, dtype=np.float64)):
        raise ValueError("Context weights differ from the D group-equal weights")
    y = np.asarray(ctx.y, dtype=np.float64)
    if not np.array_equal(y, np.asarray([r[5] for r in rows], dtype=np.float64)):
        raise ValueError("Development RSA labels differ from the row roster")
    propensity = np.asarray(propensity, dtype=np.float64)
    if propensity.shape != (20,) or not np.isfinite(propensity).all():
        raise ValueError("RSA propensity must contain 20 finite frozen values")
    aa = np.asarray([r[3] for r in rows])
    aa_counts = np.asarray([(aa == a).sum() for a in AA])
    if np.any(aa_counts == 0) or aa_counts.sum() != len(rows):
        raise ValueError("D must contain every canonical amino-acid class")
    actual_propensity = np.asarray([y[aa == a].mean() for a in AA])
    if np.max(np.abs(actual_propensity - propensity)) > 1e-15:
        raise ValueError("Frozen propensity differs from actual-site arithmetic D RSA means")
    residual = y - np.asarray(ctx.x @ ctx.coef).reshape(-1)
    if residual.shape != (len(rows),) or not np.isfinite(residual).all():
        raise ValueError("Invalid all-D sequence-reference residuals")
    early, late = (np.asarray(ctx.dp[i], dtype=np.int64) for i in (0, 1))
    if early.shape != (195,) or late.shape != (195,):
        raise ValueError("The fixed 195 development pairs changed")
    selected = np.column_stack((early, late)).reshape(-1)
    if len(np.unique(selected)) != 390 or selected.min() < 0 or selected.max() >= len(rows):
        raise ValueError("Development pair sites are repeated or out of range")
    if np.any(groups[early] != groups[late]) or len(np.unique(groups[early])) != 195:
        raise ValueError("Expected one two-site pair per D group")
    for i, j in zip(early, late):
        if rows[i][0] != rows[j][0] or not rows[i][1] < rows[j][1]:
            raise ValueError("D pair membership or canonical position order differs")
    if z.shape[0] != len(rows) or not np.isfinite(z).all():
        raise ValueError("Development activation array is invalid")
    cases = []
    for index in selected:
        accession, pos, group, amino_acid = rows[index][:4]
        split, stored_group, sequence = ctx.cohort[accession]
        if split != "D" or group != stored_group or not 1 <= pos <= len(sequence):
            raise ValueError("A direction input is not in the fixed D cohort")
        if sequence[pos - 1] != amino_acid or any(a not in AA for a in sequence):
            raise ValueError("Direction sequence or source-residue identity differs")
        cases.append({"acc": accession, "pos": int(pos), "group": group, "sequence": sequence})
    return residual, weights, propensity, aa_counts, cases


def run_direction(net, vocab, ctx, z_D, axis, block_one_based, rsa_propensity, folder,
                  *, site_batch=8, axis_batch=16):
    """Run only after the coordinator's frozen-protocol GO.

    Return technical/timing metadata only.  P1/P2, selected-axis information,
    and direction vectors are written to the named SEALED files for later
    authorized analysis.  The caller retains and releases the supplied net.
    """
    import torch
    started_utc, start = _utc(), time.perf_counter()
    folder = Path(folder)
    result_path, array_path = folder / "DIRECTION_SEALED.json", folder / "direction_vectors_SEALED.npz"
    if result_path.exists() or array_path.exists():
        raise ValueError("Direction output already exists; do not silently rerun")
    if site_batch < 1 or axis_batch < 1:
        raise ValueError("Positive technical batch sizes are required")
    if net.training or any(p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError("Direction model must be float32 and eval")
    device = next(net.parameters()).device
    if device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("TF32 must be disabled by the caller")
    if getattr(net.config, "_attn_implementation", None) != "eager":
        raise ValueError("The fixed eager-attention implementation is required")
    if not 1 <= int(block_one_based) <= len(net.esm.encoder.layer):
        raise ValueError("The specified one-based FFN block is out of range")
    layer = net.esm.encoder.layer[int(block_one_based) - 1]
    final_norm = net.esm.encoder.emb_layer_norm_after
    if not isinstance(final_norm, torch.nn.LayerNorm) or not hasattr(net, "lm_head"):
        raise ValueError("Unexpected ESM final encoder norm or original LM head")
    weight = layer.output.dense.weight
    width, hidden = int(weight.shape[1]), int(weight.shape[0])
    z = np.asarray(z_D)
    if z.ndim != 2 or z.shape[1] != width or z.dtype != np.float32 or not 0 <= int(axis) < width:
        raise ValueError("D activations/selected axis do not match the selected FFN module")
    aa_ids = [int(vocab[a]) for a in AA]
    if len(set(aa_ids)) != 20:
        raise ValueError("Canonical amino-acid vocabulary indices are not distinct")
    residual, weights, propensity, aa_counts, cases = _development_inputs(ctx, z, rsa_propensity)
    moment_start = time.perf_counter()
    mu, variance, beta, sigma, t = _moments(z, residual, weights)
    moments_seconds = time.perf_counter() - moment_start
    # Capture BEFORE encoder.emb_layer_norm_after.  Model outputs/hidden_states
    # already passed through that layer and must never be normalized again.
    x_cpu = np.empty((390, hidden), dtype=np.float32)
    original_logits = np.empty((390, 20), dtype=np.float32)
    order = sorted(range(len(cases)), key=lambda i: (len(cases[i]["sequence"]), i))
    full_calls = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), site_batch):
            indices = np.asarray(order[offset:offset + site_batch], dtype=np.int64)
            selected = [cases[i] for i in indices]
            tokens = [[vocab["<cls>"], *[vocab[a] for a in c["sequence"]], vocab["<eos>"]]
                      for c in selected]
            inputs = torch.full((len(indices), max(map(len, tokens))), int(vocab["<pad>"]),
                                dtype=torch.long, device=device)
            for i, values in enumerate(tokens):
                inputs[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            rr = torch.arange(len(indices), device=device)
            positions = torch.tensor([c["pos"] for c in selected], dtype=torch.long, device=device)
            inputs[rr, positions] = int(vocab["<mask>"])  # Mask the source itself, not source+1.
            if not bool(((inputs == int(vocab["<mask>"])).sum(1) == 1).all()):
                raise ValueError("Each direction context must have exactly one source-site mask")
            captured = []

            def before_norm(_module, args):
                captured.append(args[0][rr, positions, :].detach().clone())

            handle = final_norm.register_forward_pre_hook(before_norm)
            try:
                output = net(input_ids=inputs, attention_mask=inputs.ne(int(vocab["<pad>"])).long()).logits
            finally:
                handle.remove()
            if len(captured) != 1 or captured[0].shape != (len(indices), hidden):
                raise ValueError("The pre-final-normalization state was not captured exactly once")
            x_cpu[indices] = captured[0].cpu().numpy()
            original_logits[indices] = output[rr, positions][:, aa_ids].cpu().numpy()
            full_calls += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        head_start = time.perf_counter()
        x = torch.as_tensor(x_cpu, dtype=torch.float32, device=device)
        # Define the displacement from the archived float32 x/weight values
        # and float64 D population SD; round the resulting x only once.
        x64 = x.to(dtype=torch.float64)
        sigma_gpu = torch.as_tensor(sigma, dtype=torch.float64, device=device)
        delta = np.empty((width, 20), dtype=np.float64)
        baseline = {}
        projection_error = 0.0
        zero_shift_error = 0.0
        for first in range(0, width, axis_batch):
            last = min(width, first + axis_batch)
            size = last - first
            # A matching tail batch shape makes a zero shift an exact native
            # projection.  Cache this baseline once per technical batch size.
            if size not in baseline:
                native_x = x.unsqueeze(0).expand(size, -1, -1).contiguous()
                native_logits = net.lm_head(final_norm(native_x))[..., aa_ids]
                err = float(torch.max(torch.abs(native_logits - torch.as_tensor(
                    original_logits, device=device).unsqueeze(0))).item())
                projection_error = max(projection_error, err)
                if err > PROJECTION_ATOL:
                    raise ValueError(f"Original-value LN/head projection failed: {err}")
                baseline[size] = _log_odds(native_logits)
            shifts = weight[:, first:last].T.to(dtype=torch.float64) * sigma_gpu[first:last, None]
            shifted = (x64.unsqueeze(0) + shifts[:, None, :]).to(dtype=torch.float32)
            changed_logits = net.lm_head(final_norm(shifted))[..., aa_ids]
            change = _log_odds(changed_logits) - baseline[size]
            change = change - change.mean(dim=-1, keepdim=True)
            # The order is pair-adjacent: first average two sites, then 195 groups.
            averaged = change.reshape(size, 195, 2, 20).mean(dim=2).mean(dim=1)
            delta[first:last] = averaged.cpu().numpy()
            zeros = np.flatnonzero(sigma[first:last] == 0)
            if len(zeros):
                zero_shift_error = max(zero_shift_error, float(torch.max(torch.abs(change[zeros])).item()))
                if zero_shift_error > 1e-12:
                    raise ValueError("A zero-SD axis produced a nonzero direct-path change")
                delta[first + zeros] = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        head_seconds = time.perf_counter() - head_start
    if not np.isfinite(delta).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite direct-path direction values")
    statistics_start = time.perf_counter()
    rcenter = propensity - propensity.mean()
    rden = float(np.dot(rcenter, rcenter))
    if rden == 0:
        u = np.full(width, np.nan)
        p1 = _rank_correlation(np.zeros(20), propensity, "constant_RSA_propensity")
        p2 = _rank_correlation(t, u, "constant_RSA_propensity")
    else:
        u = (delta - delta.mean(axis=1, keepdims=True)) @ rcenter / rden
        p1 = _rank_correlation(np.sign(t[int(axis)]) * delta[int(axis)], propensity,
                               "selected_axis_reading_score_is_zero" if t[int(axis)] == 0 else None)
        p2 = _rank_correlation(t, u)
    technical = {"D_residues": 20974, "D_groups": 325, "masked_sites": 390,
                 "masked_groups": 195, "full_forward_calls": full_calls,
                 "all_FFN_units_processed": width, "hidden_size": hidden,
                 "source_mask": "the selected site itself",
                 "capture": "forward_pre_hook of esm.encoder.emb_layer_norm_after",
                 "tail": "original final encoder LayerNorm, then original complete LM head",
                 "projection_max_abs_logit_error": projection_error,
                 "projection_absolute_tolerance": PROJECTION_ATOL,
                 "zero_SD_max_abs_logodds_change": zero_shift_error,
                 "site_batch_maximum": site_batch, "axis_batch_maximum": axis_batch,
                 "no_checkpoint_load_or_network": True, "confirmation_fields_accessed": False,
                 "displacement_numeric": "float64(x_float32 + W_float32 * SD_float64), then one cast to float32",
                 "dtype": "native float32 tail; float64 log-odds and weighted statistics"}
    timing = {"D_moments_seconds": moments_seconds, "masked_390_full_forward_seconds": forward_seconds,
              "all_units_head_readout_seconds": head_seconds,
              "direction_summary_statistics_seconds": time.perf_counter() - statistics_start,
              "total_seconds_before_sealed_write": time.perf_counter() - start}
    # All scientific values stay in the caller's sealed result folder.
    folder.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    _npz(array_path, {"alphabet": np.asarray(AA), "activation_mean": mu, "activation_variance": variance,
                      "activation_SD": sigma, "D_residual_slope": beta, "t": t, "u": u,
                      "delta_centered_logodds": delta, "rsa_propensity": propensity,
                      "rsa_propensity_site_counts": aa_counts, "pre_final_norm_x": x_cpu,
                      "native_20_logits": original_logits,
                      "acc": np.asarray([c["acc"] for c in cases]),
                      "pos": np.asarray([c["pos"] for c in cases]),
                      "group": np.asarray([c["group"] for c in cases])})
    science = {"status": "COMPLETE_DEVELOPMENT_ONLY_DESCRIPTIVE", "started_utc": started_utc,
               "completed_utc": _utc(), "block_one_based": int(block_one_based),
               "selected_axis_zero_based": int(axis), "P1": p1, "P2": p2,
               "log_odds": "l_a-logsumexp(other 19 canonical logits); intervention-minus-native, centered over 20",
               "t": "all-D group-equal residual slope times identically weighted population activation SD",
               "r": "actual-site arithmetic mean RSA per canonical AA in all eligible D residues",
               "u": "OLS slope with intercept of delta(a) on r(a), 20 equally weighted AA classes",
               "D_signature": hashlib.sha256("".join(repr(r) + "\n" for r in ctx.d).encode()).hexdigest(),
               "zero_SD_axes": int((sigma == 0).sum()),
               "slope_floor_axes": int((variance <= SLOPE_VARIANCE_FLOOR).sum()),
               "slope_variance_floor": SLOPE_VARIANCE_FLOOR,
               "technical": technical, "timing": timing}
    timing["sealed_array_write_seconds"] = time.perf_counter() - write_start
    timing["total_seconds"] = time.perf_counter() - start
    _json(result_path, science)
    return {"status": "COMPLETE", "timing": timing, "technical": technical,
            "sealed_files": [str(result_path), str(array_path)]}


def synthetic_selfcheck():
    """CPU-only numerical identities on invented arrays; no ESM/data access."""
    import torch
    gen = np.random.default_rng(731)
    logits = gen.normal(size=(3, 20)) * 2
    got = _log_odds(torch.tensor(logits, dtype=torch.float64)).numpy()
    expected = np.empty_like(logits)
    for row in range(len(logits)):
        for a in range(20):
            other = np.delete(logits[row], a)
            m = other.max()
            expected[row, a] = logits[row, a] - m - np.log(np.exp(other - m).sum())
    odds_error = float(np.max(np.abs(got - expected)))
    centered = got - got.mean(axis=1, keepdims=True)
    offset = _log_odds(torch.tensor(logits + 17.0, dtype=torch.float64)).numpy()
    odds_offset_error = float(np.max(np.abs(got - offset)))
    z = gen.normal(size=(32, 4)).astype(np.float32)
    z[:, 0] = 0.75
    residual, weights = gen.normal(size=32), np.full(32, 1 / 32)
    mu, var, beta, sd, t = _moments(z, residual, weights)
    manual_sd = np.sqrt(np.sum(weights[:, None] * (z.astype(float) - mu) ** 2, axis=0))
    sd_error = float(np.max(abs(sd - manual_sd)))
    assert var[0] == beta[0] == sd[0] == t[0] == 0
    assert _rank_correlation([0, 0], [1, 2])["rho"] is None
    assert abs(_rank_correlation([0, 0, 2], [1, 1, 3])["rho"] - 1) < 1e-14
    # Pure synthetic pre-LN projection: exact original tail and a zero shift.
    torch.manual_seed(731)
    norm = torch.nn.LayerNorm(8).double()
    head = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(),
                               torch.nn.LayerNorm(8), torch.nn.Linear(8, 20)).double()
    x = torch.tensor(gen.normal(size=(6, 8)), dtype=torch.float64)
    with torch.inference_mode():
        direct = head(norm(x))
        repeated = head(norm(x[None].expand(2, -1, -1).contiguous()))
        zero = _log_odds(head(norm(x + torch.zeros_like(x)))) - _log_odds(direct)
    projection_error = float(torch.max(abs(repeated - direct)).item())
    zero_error = float(torch.max(abs(zero)).item())
    assert odds_error < 1e-13 and odds_offset_error < 1e-13 and sd_error < 1e-13
    assert np.max(abs(centered.sum(1))) < 1e-13 and projection_error < 1e-13 and zero_error == 0
    return {"status": "PURE_SYNTHETIC_PASS", "one_vs_rest_max_error": odds_error,
            "logit_offset_invariance_max_error": odds_offset_error,
            "weighted_population_SD_max_error": sd_error,
            "synthetic_tail_projection_max_error": projection_error, "zero_shift_error": zero_error}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-selfcheck", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_selfcheck:
        parser.error("Only --synthetic-selfcheck is available as a standalone command")
    print(json.dumps(synthetic_selfcheck(), indent=2, allow_nan=False))
