# Reproduction interface

Run the CPU calculation from this checkout:

```sh
python experiments/04_sparse_dictionary/reproduce.py --mode verify
```

Run the bounded real-checkpoint subset using a separately obtained public snapshot:

```sh
python experiments/04_sparse_dictionary/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda
```

The snapshot contains `config.json` and `pytorch_model.bin` from the exact public revision in `data/molecular/model_sources.json`. Derived integer model inputs and verified atom anchors are included. This command needs neither the original study directory nor a model download. Source structures and weights are not redistributed. Optional public CID retrieval is in `data/molecular/download_inputs.py`; it has not been run during this offline build.

CPU coverage: every recipe T1/T2 fold, rank p and conditional concept-bootstrap interval from per-concept scores. It does not retrain the 504 dictionaries, recompute every training-quality record, or repeat all sensitivity fitting.

Both commands check their actual covered outputs against frozen artifacts or explicit engineering invariants. A passing smoke does not validate the full experiment, its statistical population, fitting or all model architectures. Full end-to-end extraction and fitting have not yet been migrated to a complete portable command. Historical source code and data availability are recorded in the assembly report; this is an implementation gap, not evidence that the material is absent.
