# Reconstruction and native writeback of protein units

Run from the repository root:

```sh
python experiments/10_protein_reconstruction/reproduce.py --mode verify --output artifacts/10_protein_reconstruction.json
```

The verifier recomputes row-level means, decision counts or calibration arithmetic from stored results. It does not load model weights. PROTOCOL.md records scientific estimands; DEVIATIONS.md retains execution departures and negative outcomes. Original protocol byte digests and available freeze times are preserved in frozen_records.json.

The reference_implementation directory contains the authored analysis modules, separated from external packages and model weights. Full original environment layouts are not implied by these files; see the repository reproduction coverage record for forward-subset checks.

## Forward subset

```sh
python experiments/10_protein_reconstruction/reproduce.py --mode forward-subset --weights MODEL_DIRECTORY --sequences INPUT_SEQUENCES.jsonl --device cpu --output artifacts/10_protein_reconstruction_forward.json
```

Use Transformers 4.56.2, a compatible PyTorch build and the pinned local weights. `--dependency-dir PACKAGE_DIRECTORY` can be repeated for an explicitly installed model SDK. E1 additionally needs its pinned upstream source and runtime dependencies. No package or weight download is implicit. `data/protein/acquire.py` is an explicit optional retrieval command; it was not run during the offline build. Input sequence checksums must match the frozen identifiers.

Covered by the actual CPU forward subset: Neighbour-masked 35M neuron 814, original twenty-amino-acid logits, and actual replacement by the frozen structure predictor.

Uncovered by this subset: Full development ridge cross-validation and all 1934 writeback inputs were not rerun.

Full GPU execution of the complete reference implementations has not been validated from a clean checkout. Their archived scientific outputs and row-level statistics remain available; a forward-subset PASS does not certify those complete workflows.
