# Protein replay and molecular analyses

The READMEs in `scripts/`, `results/`, `data/`, `docs/`, `reproduction/` and `source_data/` explain the short codes used in those folders.

This guide covers the protein extension, the ChemBERTa development/confirmation analyses, the scaffold-unseen and standardized-rotation checks, the paired-dose analysis, and chemical activation replacement. The development-only sequence-rule analysis is recorded in `reproduction/protein/HISTORICAL_SEQUENCE_RULES.json`.

`docs/` holds English translations (`*.en.md`) of the analysis protocols written during the study; the Chinese originals are not part of this release and govern if the two differ.

## Executable protein replay

From the repository root, with Python 3.11 and `requirements-protein.txt`:

```sh
python reproduction/protein/unpack_inputs.py --output-root replay-inputs
python reproduction/protein/replay_protein.py --project-root . --input-root replay-inputs --output-dir replay-output --chemistry-python /path/to/python
```

Use new, empty input and output directories. `--chemistry-python` only reads installed-package metadata from the given interpreter, so the current interpreter can be used. The replay needs no downloads and no ESM-2 checkpoint.

The archive holds all 116 inputs: the raw records needed to rebuild the exclusion graph, both 28-page UniProt snapshots, the 1,304 pairs (2,608 cases), saved confirmation-set activations, the six kernel programs fitted on the development set, and the confirmation-set predictions. The replay rebuilds the graph and case selection and reruns the kernel readout with the four original source modules, then writes `REPLAY_RECEIPT.json`. It does not run the model, refit, or recompute intervals or p values.

UniProt inputs derive from UniProt release 2026_03 (2 September 2026), attributed to the UniProt Consortium and provided under [CC BY 4.0](https://rest.uniprot.org/help/license). The repository includes selected UniProt records and data derived from them (sequence and annotation projections, component joins, case selections, and readouts); these are changes to the source material. The analysis uses existing PDB-covered secondary-structure annotations; no new structures or DSSP assignments were generated.

## Molecular code and saved results

| Analysis | Code | Saved results |
|---|---|---|
| Development/confirmation corpus and fixed-coordinate confirmation | `scripts/fixed_coordinate_analyses/chem_data.py`, `chem_run.py`, `chem_T_data.py`, `chem_T_run.py`, `chem_statistics.py` | `data/fixed_coordinate_analyses/chemistry/`, `results/fixed_coordinate_analyses/chemistry_D/`, `chemistry_T/` |
| Finite-input and full-clamp controls | `chem_input_reference.py`, `chem_D_secondary.py`, `chem_T_secondary.py`, `chem_T_intervention.py` | Corresponding `secondary/` result directories |
| Scaffold-unseen subset, lineage, and anchor mapping | `scripts/sensitivity_analyses/a_lineage.py`, `a_h4_mapping.py`, `a_scaffold_subset.py` | `results/sensitivity_analyses/lineage/`, `mapping/`, `scaffold_subset/` |
| Standardized rotation | `scripts/sensitivity_analyses/b_standardized_rotation.py` | `results/sensitivity_analyses/standardized_rotation/` |
| Paired dose | `scripts/sensitivity_analyses/c_paired_dose.py`, `c_geometry_recovery.py` | `results/sensitivity_analyses/paired_dose/` |
| Activation reconstruction and replacement | `scripts/reconstruction_and_replacement/data_fit.py`, `runner.py`, `summarize.py` | `results/reconstruction_and_replacement/fit/`, `summary/`, saved forward arrays |

`scripts/fixed_coordinate_analyses/chem_T_oracle_memory_resume.py` completed the descriptive oracle re-selection on the confirmation set after a memory-related interruption, and `c_geometry_recovery.py` recomputed the baseline and 18 dose conditions for one molecule using its original four-molecule batch; the reported results come from these completed runs.

To extract the stored chemical replacement output arrays into a new directory:

```sh
python reproduction/unpack_chemical_outputs.py --output-root replacement-output-copy
```

This only extracts the stored arrays: baseline outputs (10,000 × 199), condition outputs (2 × 5 × 10,000 × 199), realized doses, and permutation metadata.

## Inputs and acquisition

The protein replay uses only the included inputs. A molecular rerun needs the following.

1. `reproduction/OMITTED_INPUTS.json` lists the 129 files that are not included (about 3.60 GB of molecular activation caches, plus intermediate reference predictions, per-rotation arrays, and bootstrap arrays), with their paths, sizes, original hashes, timestamps, and the extraction stage that produced them. Included are the development/confirmation atom labels, ordered SMILES, mapping anchors, concept dictionaries, fitted models, selections, result summaries, and the bootstrap replicates behind the reported intervals (`T_FIXED_REPLICATES.npz`, `scaffold_subset/SLICE_FIXED_REPLICATES.npz`, `secondary/input/PANEL_BRIER_BOOTSTRAP.npy`, `secondary/mechanism/`).
2. Download model weights with `python reproduction/fetch_models.py --model all`. The script fetches pinned revisions (ESM-2: `facebook/esm2_t6_8M_UR50D` at `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; ChemBERTa: `DeepChem/ChemBERTa-77M-MTR` at `66b895cab8adebea0cb59a8effa66b2020f204ca`), including the eight ChemBERTa model, configuration, and tokenizer files, and checks their sizes and hashes.
3. Rebuilding the corpus from PubChem requires the September 2026 `CID-SMILES.gz` snapshot (1,486,837,335 bytes; SHA-256 in `reproduction/OMITTED_INPUTS.json`). PubChem replaces this file over time, and `chem_source.py` rejects a file whose size, date, or hash differs; the ordered development/confirmation inputs used in the study are included.
4. Beyond the omitted activation caches and model weights, a full molecular rerun requires additional study materials outside this release, including the original protocols and test modules read by some preparation and readiness steps.
5. `a_lineage.py` and `a_h4_mapping.py` read `NATIVE_UNIT_LEGACY_ROOT` (default: the repository root) for `data/derived/cross_model_comparison/corpus/H4_used.tsv`, `scripts/cross_model_comparison/models.py`, `scripts/cross_model_comparison/check_chemberta_alignment.py`, `models/download_log.csv`, and the saved alignment check. The primary-cohort mapping comparison loads no weights and expects tokenizers at `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mlm/` and `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mtr/` (default `external/models/...`). Fetch the six tokenizer files and `config.json` for each model listed in `acquisition/EXTERNAL_ASSETS.json`, for example `python scripts/cross_model_comparison/companion.py fetch-asset --id models/chemberta_77m_mlm/tokenizer.json --asset-root external`. Three small corpus and concept source files are in `scripts/fixed_coordinate_analyses/frozen_source/`. Some molecular runners monitor resources through Windows APIs; full runs on Linux or macOS have not been tested.

The primary-cohort tokenizer-alignment check is saved in `cross_model_comparison/results_posthoc/alignment_check.json`, with the model download log in `models/download_log.csv`; the atom-mapping checks are in `results/fixed_coordinate_analyses/mapper/ENGINEERING_CHECKS.json`.

## Validation

`reproduction/protein/VALIDATION.json` records a complete replay of the protein analysis from the released files: all 45 checks passed, with a maximum score difference of 4.66 × 10⁻¹⁵, and no network access, model calls, refitting, or recomputed statistics.
