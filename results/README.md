# Saved analysis results

This folder contains saved numerical results for the fixed-coordinate molecular analyses, protein extension, post hoc sensitivity analyses and two-coordinate reconstruction and replacement.

| Folder or file group | Contents | Manuscript or SI section |
| --- | --- | --- |
| `fixed_coordinate_analyses/chemistry_D/`, `chemistry_T/` | Development selections, confirmation transfer, fitted input references and output interventions | SI S-D–S-E, S-H |
| `fixed_coordinate_analyses/protein_D/`, `protein_T/` | Fitted protein programs, predictions and group-level inference | SI S-G–S-H |
| `fixed_coordinate_analyses/mapper/` | Molecular atom-mapping checks | SI S-D |
| `sensitivity_analyses/` | Lineage, atom-anchor eligibility, scaffold-unseen transfer, standardized rotations and paired doses | SI S-I; population eligibility in S-L |
| `reconstruction_and_replacement/` | Fitted activation models, stored forward outputs, reconstruction and output-preservation summaries | SI S-J |

## Names used here

| Name in files or saved fields | Meaning |
| --- | --- |
| D / T | Development set / confirmation set. |
| H4 | Primary cohort (10,000 molecules), drawn after excluding the original pool. |
| D_mean | Development-set all-atom mean, in the replacement `condition` column. |
| M1 / M2 | Binary-concept mean / hierarchical radius-0/radius-1 local-graph activation prediction, in the replacement `condition` column. These are separate from the pilot readout-location check. |
| M1_delta_perm / M2_delta_perm | Within-molecule residual-permutation conditions for the binary-concept and local-graph predictions. |
| 293 / 454 | Positive post-GELU coordinates for aliphatic hydroxyl / carbonyl in the original ChemBERTa-77M-MTR two-coordinate analysis. |
| 24 / 25 / 26 | Carbonyl / carbonyl excluding COOH / aliphatic hydroxyl, in the `concept` column. Concepts 24 and 25 selected the same coordinate 454. |
| gX / gS / gL | Paired Brier-loss reduction from adding coordinate 1246 to the composition / sequence-window / interaction reference, respectively. |

The recorded execution repairs are described in `../docs/DEVIATIONS.md`; the dose recovery changed one row's recorded response and dose values while preserving the scientific parameters and samples.
