# Reproduction interface

Run the CPU calculation from this checkout:

```sh
python experiments/06_molecular_edits/reproduce.py --mode verify
```

Run the bounded real-checkpoint subset using a separately obtained public snapshot:

```sh
python experiments/06_molecular_edits/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda
```

The snapshot contains `config.json` and `pytorch_model.bin` from the exact public revision in `data/molecular/model_sources.json`. Derived integer model inputs and verified atom anchors are included. This command needs neither the original study directory nor a model download. Source structures and weights are not redistributed. Optional public CID retrieval is in `data/molecular/download_inputs.py`; it has not been run during this offline build.

CPU coverage: all 4,640 per-unit R-squared values from raw paired activations, preserving original SHA ordering and moment arithmetic. The full contrast/bootstrap/restoration tables are retained but their every interval and statistic is not yet independently recomputed by this wrapper. Real smoke: one controlled hydroxyl edit, matched source/product activations and their change. The separate prepare_pairs.py command rebuilds complete chemical pair inputs from external public structures, the fixed 2,000-CID manifest and two public tokenizer directories. Class/site replay matches all 2,000 parents; a 16-parent full-preparation subset matches all 20 archived records by canonical JSON hash. See LEVEL2_INPUTS.md for commands and source identity guards. The complete neural measurement and restoration queue is not yet ported.

Both commands check their actual covered outputs against frozen artifacts or explicit engineering invariants. A passing smoke does not validate the full experiment, its statistical population, fitting or all model architectures. Full end-to-end extraction and fitting have not yet been migrated to a complete portable command. Historical source code and data availability are recorded in the assembly report; this is an implementation gap, not evidence that the material is absent.
