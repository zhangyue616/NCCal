# Masked alanine helix and strand readouts

Run from the repository root:

```sh
python experiments/08_protein_pair_replay/reproduce.py --mode verify --output artifacts/08_protein_pair_replay.json
```

The verifier recomputes row-level means, decision counts or calibration arithmetic from stored results. It does not load model weights. PROTOCOL.md records scientific estimands; DEVIATIONS.md retains execution departures and negative outcomes. Original protocol byte digests and available freeze times are preserved in frozen_records.json.

The reference_implementation directory contains the authored analysis modules, separated from external packages and model weights. Full original environment layouts are not implied by these files; see the repository reproduction coverage record for forward-subset checks.

## Forward subset

```sh
python experiments/08_protein_pair_replay/reproduce.py --mode forward-subset --weights MODEL_DIRECTORY --sequences INPUT_SEQUENCES.jsonl --device cpu --output artifacts/08_protein_pair_replay_forward.json
```

Use Transformers 4.56.2, a compatible PyTorch build and the pinned local weights. `--dependency-dir PACKAGE_DIRECTORY` can be repeated for an explicitly installed model SDK. E1 additionally needs its pinned upstream source and runtime dependencies. No package or weight download is implicit. `data/protein/acquire.py` is an explicit optional retrieval command; it was not run during the offline build. Input sequence checksums must match the frozen identifiers.

Covered by the actual CPU forward subset: Masked alanine, block-3 post-GELU coordinate 1246; native activation compared with the original cache.

Uncovered by this subset: Development coordinate search, full retrospective 199-network calibration, kernel fitting and all writeback arms were not rerun by this subset.

Full GPU execution of the complete reference implementations has not been validated from a clean checkout. Their archived scientific outputs and row-level statistics remain available; a forward-subset PASS does not certify those complete workflows.
