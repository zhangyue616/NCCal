# Masked alanine helix and strand readouts

The readout is the fixed ESM-2 8M block-3 coordinate 1246. Each target alanine is independently masked in its natural full-length sequence. Development selection uses the original development fit/select components; the retrospective network and rotation controls repeat this same selection opportunity. Do not replace this estimand by reselection on all 432 paired development observations.

Composition, ordered sequence-window and interaction kernels use development-only trace normalization and regularization. The confirmation frame contains 1304 homology groups. The source case identifiers encode HELIX or STRAND, and the order of these labels changes across pairs: never assume the second stored row is the strand. Pair contributions store the three reference-minus-augmented Brier differences. Whole groups are resampled with selections and predictors held fixed.

The original single-centre calibration and later paired-kernel diagnostics are distinct. Single-centre ranks are retrospective. One rotated reading exceeds the native single-site gain. The later equal-displacement writeback control is more faithful than the computed sequence replacement; original oracle-residual exchange and equal-displacement exchange must remain separate. All output divergences concern frozen model output fidelity in nats, not physical structure.

The original source byte digests, file sizes, preserved freeze timestamps and the corresponding English protocol are listed in frozen_records.json. Date stamps identify freezes; they do not denote separate reproduction outputs.
