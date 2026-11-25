# Figure 1: from a name to its uses

From the repository root, run `python -B figures/render.py 1 --output ../nccal-overview`.
The entry dispatches to `native/main_1.py`, using the shared `native/style.py`
canvas. It reads archived numerical results and molecular geometry from this
repository. Rendering performs no model inference or fitting.

The overview connects four tests of a neuron name to molecular uses and matched
protein readouts. Panels b–e are calibration, transfer, reconstruction and output
preservation. Panel f follows the already selected hydroxyl example; panel g
introduces the two protein questions. The source map below distinguishes the
illustrative structures from the observations that determine the plotted values.

| Panel | Drawing and source |
|---|---|
| a | The ketone molecular layout is in `data/artwork/molecules/fig1_mol2d.json`; atom-to-token anchors are in `selected_atom_alignment.json` in the same directory. The encoder, grid and arrows are schematic; the grid is not an activation heat map. |
| b | The trained/untrained and native/rotated comparisons explain calibration. Histogram shapes and their marker placement are schematic. The 199-network and nine-rotation design is the frozen calibration design. `results/artwork/main/calibration.json` supplies the corresponding fold quantities when numerical annotations are requested. |
| c | The chosen atom's measured response is read from `results/artwork/main/atom_activations.npz`, using the exact slice, coordinate 454 and atom order recorded in `data/artwork/molecules/ketone2_heat.json`. The scaffold-unseen 7,205-molecule label describes the confirmation subset. The transfer statistic is from `results/artwork/main/calibration.json`. |
| d | Observed, binary-concept and bonding-environment responses come from the same atom array, with the saved M1/M2 predictions. The figure checks agreement with the exact stored molecule and atom-label mask. Reconstruction summaries are from `results/artwork/main/activation_output.json`. |
| e | The frozen-network replacement diagram and task-head blocks are schematic. Bar lengths use the saved whole-molecule output-deviation estimates for coordinate 454 in `results/artwork/main/activation_output.json`. |
| f, left | `results/artwork/examples/Fig5d_CID66277342_three_SMILES.json` supplies one fixed molecule, three actual SMILES encodings, the marked carbon, its measured post-GELU readings and frozen rule values. These are the same selected example and measurements used in Figure 5. |
| f, middle | `results/molecular/edits/raw_pairs.npz` supplies the original, controlled-product and predicted changes at the same marked carbon for CID 66277342, edit class 1. `data/artwork/examples/hydroxyl_pair_inputs.jsonl.gz` supplies its actual parent/product graphs. The code checks that the parent's reading equals the canonical reading in the notation example. |
| f, right | `results/artwork/use/5c_and_s_s1_basis_summary.tsv` supplies retained directions for 464 native neurons, nine 464-direction rotated bases and all 928 features in each of three dictionaries. Native retention is 26; rotated and dictionary raw counts are 0–4 and 53–55. Dictionary counts are multiplied by 464/928 for the displayed directions-per-464 scale. Bars use the median, with minimum–maximum spans where they differ. The selected 464-feature dictionary subsets remain separate source records and are not used in this panel. |
| g, structure | The cached `data/artwork/generated/prot_gray_two_sites.png` and `protein_masked_sites_manifest.json` illustrate two alanines in PDB 2CZL, chain A. Their source backbone is `data/artwork/protein_masked_geometry.json`. |
| g, dots | `results/artwork/examples/Fig6d_Q9I398_E1_1900.json` supplies the fixed E1 neuron 1900 readings for the helix and strand arginines in UniProt Q9I398. The two displayed values are approximately 0.576677 and −0.001257. These values belong to Q9I398, separately from the 2CZL alanine illustration. |
| g, bars | `results/artwork/main/exposure_example.json` supplies the ESM-2 35M neuron 814 exposure comparison. The sequence-only and neuron-augmented Brier losses are approximately 0.2433 and 0.2152 across 967 held-out groups. The values describe the exposure-pair analysis; they are not readings at the two illustrated 2CZL sites. |

The full numerical inputs remain in their existing source locations.
`figure_1_sources.json` records the local paths, sizes and SHA-256 values of the
17 drawing inputs, including the source geometry for the cached protein component.
All were already part of the released repository. The original records, endpoint
definitions, interval families and example-selection scope remain unchanged.
`FIGURES_5_6.md` gives the molecular and E1 selection details.

`published/Figure_1.png`, `Figure_1.svg` and `Figure_1.pdf` retain the
author-selected exports. They are reference outputs and are never drawing
inputs. The canvas is 504 × 404 points; the 600-dpi PNG is 4200 × 3366 pixels.
The optional display environment in `requirements.txt` reproduced that PNG byte
for byte. SVG visible elements agree after normalizing export metadata and
generated element identifiers. PDF bytes agree after normalizing export dates
and document identifiers; the decoded page stream and extracted text also agree.
The original vector files are retained byte for byte. Matplotlib export dates
and generated SVG identifiers can change between renders, and a different font
or display environment can change typography and pixels.
