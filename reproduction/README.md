# Saved-data replay and input acquisition

This folder contains the saved-data protein replay, archived molecular replacement outputs, input inventories and acquisition helpers.

| Folder or file group | Contents | Manuscript or SI section |
| --- | --- | --- |
| `protein/inputs.tar.gz`, `protein/unpack_inputs.py` | Protein source records, case selections, activations and fitted readouts | SI S-G–S-H |
| `protein/replay_protein.py`, `protein/VALIDATION.json` | Saved-data protein replay and its recorded validation | SI S-G–S-H |
| `protein/HISTORICAL_SEQUENCE_RULES.json` | Development-only sequence-rule proposals; none enters the reported reference models | SI Use of AI in executing the study |
| `chemical_replacement_outputs.tar.gz`, `unpack_chemical_outputs.py` | Stored original and replaced molecular outputs, doses and permutation metadata | SI S-J |
| `OMITTED_INPUTS.json`, `fetch_models.py` | Omitted-input inventory and pinned model acquisition helper | SI S-A, S-D–S-G, S-J |

## Names used here

| Name in archived files or saved fields | Meaning |
| --- | --- |
| D / T | Development set / confirmation set. |
| D_mean / M1 / M2 | Development-mean / binary-concept-mean / local-graph activation predictions in replacement conditions, separate from the pilot readout-location check. |
| gX / gS / gL | Paired Brier-loss reduction from adding coordinate 1246 to the composition / sequence-window / interaction reference. |

The two archives retain the folder and path names used during the study; the four working-to-release folder-name pairs are listed in `../scripts/README.md`. The protein replay uses the four unchanged source modules listed there.
