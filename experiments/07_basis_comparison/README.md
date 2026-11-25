# Reproduction interface

Run the CPU calculation from this checkout:

```sh
python experiments/07_basis_comparison/reproduce.py --mode verify
```

Run the bounded real-checkpoint subset using a separately obtained public snapshot:

```sh
python experiments/07_basis_comparison/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda
```

The snapshot contains `config.json` and `pytorch_model.bin` from the exact public revision in `data/molecular/model_sources.json`. Derived integer model inputs and verified atom anchors are included. This command needs neither the original study directory nor a model download. Source structures and weights are not redistributed. Optional public CID retrieval is in `data/molecular/download_inputs.py`; it has not been run during this offline build.

CPU coverage: all 17 rows of counts, medians and retention categories from per-direction measurements. Real smoke: one archived molecule, rotated/PCA/TopK coefficients compared with archived readouts and a real writeback in each basis. Full basis fitting and all intervention conditions are not re-executed.

Both commands check their actual covered outputs against frozen artifacts or explicit engineering invariants. A passing smoke does not validate the full experiment, its statistical population, fitting or all model architectures. Full end-to-end extraction and fitting have not yet been migrated to a complete portable command. Historical source code and data availability are recorded in the assembly report; this is an implementation gap, not evidence that the material is absent.

The rewritten-fit sensitivity can be recounted with the Python standard library:

```sh
python -B src/nccal/rewrite_sensitivity.py --out ../rewrite-sensitivity.json
```

This command uses 7,888 archived direction-level records across the native,
PCA, nine rotated and three full dictionary bases. Main retention requires
`canonical_usable AND retained`; the sensitivity additionally requires finite
rewritten R2 at least 0.5. The archived retained flag already records the frozen
write-back comparisons. Original NaN values remain in the data and in the full
physical denominator. The frozen variance-selected 464-feature subsets are
reported separately. No new fit or confidence interval is calculated.
