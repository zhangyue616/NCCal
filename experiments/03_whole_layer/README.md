# Reproduction interface

Run the CPU calculation from this checkout:

```sh
python experiments/03_whole_layer/reproduce.py --mode verify
```

Run the bounded real-checkpoint subset using a separately obtained public snapshot:

```sh
python experiments/03_whole_layer/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda
```

The snapshot contains `config.json` and `pytorch_model.bin` from the exact public revision in `data/molecular/model_sources.json`. Derived integer model inputs and verified atom anchors are included. This command needs neither the original study directory nor a model download. Source structures and weights are not redistributed. Optional public CID retrieval is in `data/molecular/download_inputs.py`; it has not been run during this offline build.

CPU coverage: all stored curve deviations and benchmark metrics from raw predictions or grouped score/error sufficient statistics. Whole-layer weighted variance tables, task-head refitting and every decomposition uncertainty table are retained but not all independently recomputed by this wrapper. The smoke tests a real whole-layer writeback; it does not validate the complete named-task experiment.

Both commands check their actual covered outputs against frozen artifacts or explicit engineering invariants. A passing smoke does not validate the full experiment, its statistical population, fitting or all model architectures. Full end-to-end extraction and fitting have not yet been migrated to a complete portable command. Historical source code and data availability are recorded in the assembly report; this is an implementation gap, not evidence that the material is absent.
