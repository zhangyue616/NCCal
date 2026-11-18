# Analysis scripts

This folder contains code for the cross-model comparison, fixed-coordinate molecular analyses, protein extension and post hoc sensitivity analyses.

| Folder or file group | Contents | Manuscript or SI section |
| --- | --- | --- |
| `cross_model_comparison/` | Cross-model calibration, corpus construction, decision rules and post hoc comparisons | Dual-reference calibration; SI S-A–S-C, S-E–S-F, S-H |
| `fixed_coordinate_analyses/chem_*.py` | Fixed-coordinate transfer, fitted input references and frozen-model interventions | SI S-D–S-E, S-H |
| `fixed_coordinate_analyses/protein_*.py` | Protein source grouping, fitted references and confirmation evaluation | Protein fitted-reference extension; SI S-G–S-H |
| `sensitivity_analyses/` | Lineage, atom anchors, scaffold-unseen, standardized-rotation and paired-dose analyses | SI S-I; population eligibility in S-L |
| `reconstruction_and_replacement/` | Two-coordinate activation reconstruction and pretrained-head replacement | SI S-J |
| `pilot3a.py`, `prepare_pilot5a_data.py` | Protein input and development materials used by the saved-data replay | SI S-G–S-H |

## Names used here

| Name in files | Meaning |
| --- | --- |
| D / T | Development set / confirmation set. |
| v5 / v6 | Eight-checkpoint comparison / separate MoLFormer evaluation. |
| gate | Prespecified decision rule. |
| audit | Cross-model comparison, in script names. |
| m1 / m4 | Pilot readout-location check (Table S-E3) / tensor-shuffling comparison (Table S-F1), in script names. |

## Working and release folder names

| Folder name used during the study | Release folder name |
| --- | --- |
| lineB | cross_model_comparison |
| revision_20261017 | fixed_coordinate_analyses |
| strengthening_20261025 | sensitivity_analyses |
| chemical_replacement_20261027 | reconstruction_and_replacement |

The four modules loaded by the protein replay are unchanged: `fixed_coordinate_analyses/protein_metadata_gate.py`, `fixed_coordinate_analyses/protein_fresh_data.py`, `fixed_coordinate_analyses/protein_krr.py` and `prepare_pilot5a_data.py`.
