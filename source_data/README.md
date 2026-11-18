# Figure values and labels

This folder contains saved numerical values, labels, example tiles' metadata and matrices used by the released figures.

| File group | Contents | Manuscript or SI section |
| --- | --- | --- |
| `calibration_and_transfer_values.json` | Cross-model calibrated folds and fixed-coordinate/scaffold transfer | Figures 2b–c, 3a–c; SI S-B, S-D, S-I |
| `label_prediction_and_protein_values.json` | Molecular label-prediction reductions and paired protein values | Figures 1f, 4a, 6a–b; SI S-D, S-G |
| `reconstruction_and_replacement_values.json` | Two displayed molecules' observed and predicted atom activations | Figure 4e; SI S-J |
| `data_S1_best_units.csv`, `figure_S_B1_plot_data.json` | Unit-level initial/subset-cohort and readout/subset comparisons | SI S-B |
| `figure_S_C*_plot_data.json`, `figure_S_C1_tokens.csv`, `figure_S_C5_gates.json`, `figure_S_C5_matrices/` | Tokenizer, curated examples, best-lift matrices and exploratory coordinate shares | SI S-C; Figure 2a |
| `figure_S_E1_plot_data.json`, `figure_S_I1_plot_data.json`, `figure_S_J1_plot_data.json` | Full-clamp, lineage, atom-anchor and paired-dose values | SI S-E, S-I; Figure 5b |

## Names used here

| Name in files, columns or saved table fields | Meaning |
| --- | --- |
| H2 / H3 / H4 | Initial cross-model cohort / subset-evaluation cohort / primary cohort. |
| v3 / v5 / v6 | Initial-cohort comparison / eight-checkpoint comparison / separate MoLFormer evaluation. |
| J1 / J2 | Coordinate readability / basis preference; the figure suffix in `figure_S_J1_plot_data.json` separately identifies Figure S-J1. |
| twin | Matched randomly initialized network, including the corresponding column-name prefixes. |
| gate | Prespecified decision rule, in the saved comparison file name. |
| D / T | Development set / confirmation set. |
| M1 / M2 | Binary-concept-mean / local-graph activation predictions in replacement values; separate from the pilot readout-location check. |
| 293 / 454 | Original aliphatic hydroxyl / carbonyl post-GELU coordinates in ChemBERTa-77M-MTR. |

See `../FIGURES.md` for the complete panel-to-source map. Layout records and references to working files were removed from the cleaned source-data files; their numerical values are unchanged.
