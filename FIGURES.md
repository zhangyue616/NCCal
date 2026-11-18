# Figures and source data

`figures/` contains the 18 PNG figures exactly as they appear in the manuscript and Supplementary Information: Figures 1–6, S-B1, S-C1–S-C6, S-E1, S-I1, S-J1, S-M1 and S-N1. Figure captions are in those documents.

[figure_manifest.json](figure_manifest.json) lists each PNG and its SHA-256, panel-level value sources and available renderers. The current artwork is supplied as PNG.

## Main figures by panel

Figure 1 combines schematic workflow elements with stored atom activations and the output/Brier bars repeated in Figures 5a and 6b. The source list identifies those values.

| Panel | Stored value sources | Content |
|---|---|---|
| Figure 1a | See description | Schematic inputs, frozen encoders and atom-by-coordinate activation matrix. |
| Figure 1b | See description | Schematic calibration histograms and pins; the 199 initializations and nine rotations describe the comparison design. |
| Figure 1c | `results/reconstruction_and_replacement/fit/T_activations.npz`, `data/fixed_coordinate_analyses/chemistry/sealed/T.tsv` | Stored observed carbonyl-coordinate activations for confirmation molecule CID 678545; surrounding workflow elements are schematic. |
| Figure 1d | `results/reconstruction_and_replacement/fit/T_activations.npz`, `data/fixed_coordinate_analyses/chemistry/sealed/T.tsv` | Stored observed activations, binary-concept means and local-graph predictions for the same molecule. |
| Figure 1e | `results/reconstruction_and_replacement/summary/OUTPUT_METRICS.csv` | Stored carbonyl pretrained-head output MAE bars, repeated in Figure 5a. Replacement pathways and task-head boxes are schematic. |
| Figure 1f | `source_data/label_prediction_and_protein_values.json` | Stored paired Brier losses, repeated in Figure 6b; the protein/input workflow is schematic. |
| Figure 2a | `source_data/figure_S_C6_plot_data.json` | Exploratory matched-coordinate shares and individual initialization/rotation references for the three graph checkpoints; the pilot block of the source file. |
| Figure 2b | `source_data/calibration_and_transfer_values.json` | Primary-cohort checkpoint folds, concept-bootstrap intervals and calibrated support in the saved comparison records. |
| Figure 2c | `source_data/calibration_and_transfer_values.json` | The two calibrated folds across all twelve MoLFormer layers; the saved depth records. |
| Figure 3a | `source_data/calibration_and_transfer_values.json`, `results/fixed_coordinate_analyses/chemistry_T/T_FIXED_SUMMARY.json` | Per-concept basis-preference transfer and the saved aggregate mean/interval on the confirmation set. |
| Figure 3b | `source_data/calibration_and_transfer_values.json`, `results/sensitivity_analyses/scaffold_subset/SLICE_FIXED_SUMMARY.json` | Per-concept scaffold-unseen versus full-confirmation transfer, with the saved scaffold-subset summary. |
| Figure 3c | `source_data/calibration_and_transfer_values.json`, `results/fixed_coordinate_analyses/chemistry_T/T_FIXED_SUMMARY.json` | Per-concept readability transfer versus basis-preference transfer, with saved aggregate means. |
| Figure 4a | `source_data/label_prediction_and_protein_values.json` | Twelve per-concept Brier-loss reductions and their saved equal-weight confirmation-set mean. |
| Figure 4b | `results/reconstruction_and_replacement/summary/ACTIVATION_METRICS.csv` | Pooled-atom reconstruction R² and saved marginal 95% intervals for binary-concept and local-graph predictions of coordinates 293 and 454. |
| Figure 4c | See description | Artwork only; extension analysis files are not part of this release. Coordinate identities and per-coordinate R²: Tables S-M1, S-M2a–S-M2b; equal-coordinate means and contrasts: Table S-M3. |
| Figure 4d | `results/reconstruction_and_replacement/fit/T_activations.npz` | Saved observed and predicted activations and concept labels for all 246,722 confirmation atoms. |
| Figure 4e | `source_data/reconstruction_and_replacement_values.json`, `results/reconstruction_and_replacement/fit/T_activations.npz` | Stored atom-level observed and predicted activations for the two displayed molecules. |
| Figure 5a | `results/reconstruction_and_replacement/summary/OUTPUT_METRICS.csv`, `reproduction/chemical_replacement_outputs.tar.gz`, `results/reconstruction_and_replacement/forward/RUN_BINDING.json` | Saved output-MAE estimates and intervals; the archive contains original and replaced outputs for the per-molecule scatter. Coordinate, condition and output axes are recorded in the run binding. |
| Figure 5b | `source_data/figure_S_J1_plot_data.json` | Full-clamp dose means and H intervals, together with post hoc paired-dose markers, ranges and Hpair(1) intervals. The two comparisons use their respective saved uncertainty estimates. |
| Figure 5c | See description | Artwork only; BACE/ESOL joint-replacement analysis files are not part of this release. Absolute deviations and contrasts: Tables S-N1–S-N4. |
| Figure 5d | See description | Artwork only; single-coordinate task-head analysis files are not part of this release. Identities: Table S-M1; related reference-minus-local contrasts: Tables S-N5–S-N8. The full absolute-deviation heatmap matrix is not supplied. |
| Figure 6a | `source_data/label_prediction_and_protein_values.json` | Saved protein, chain, masked-site and coordinate identities define the paired design. The current protein ribbon is supplied as artwork. |
| Figure 6b | `source_data/label_prediction_and_protein_values.json` | Six stored paired Brier losses for the three fitted references, each without and with the coordinate. |
| Figure 6c | `results/fixed_coordinate_analyses/protein_T/T_SUMMARY.json` | Saved reductions and the reference-shift contrast, with 95% and 98.75% intervals and Holm-adjusted p values. |
| Figure 6d | `results/fixed_coordinate_analyses/protein_T/T_PREDICTIONS.npz`, `results/fixed_coordinate_analyses/protein_T/T_SUMMARY.json` | Stored per-group reductions supply the 1,304 scatter pairs; aggregate means are in the saved summary. |
| Figure 6e | `results/fixed_coordinate_analyses/protein_T/T_PREDICTIONS.npz`, `results/fixed_coordinate_analyses/protein_T/T_SUMMARY.json` | Stored masked-center activations, case pairing and per-group composition-reference reductions. The 870/434 count labels are not separate summary fields; this release supplies the underlying arrays. |

## Supplementary figures

| Figure | Stored value sources | Rendering materials and content |
|---|---|---|
| Figure S-B1 | `source_data/figure_S_B1_plot_data.json` | Saved readout and concept-subset comparison folds. |
| Figure S-C1 | `source_data/figure_S_C1_plot_data.json`, `source_data/figure_S_C1_tokens.csv` | Tokenizer examples and all 593 vocabulary-row norms. |
| Figure S-C2 | `source_data/figure_S_C2_plot_data.json` | Six ChemBERTa-77M-MLM examples; saved values and image tiles. The renderer draws these figures from the saved values; label wording and layout differ slightly from the published artwork. Renderer: `figure_source/supplementary/plot_S_C2_to_C5.py`. Tiles: `figure_source/assets/example_tiles/`. |
| Figure S-C3 | `source_data/figure_S_C3_plot_data.json` | Six ChemBERTa-77M-MTR examples; saved values and image tiles. The renderer draws these figures from the saved values; label wording and layout differ slightly from the published artwork. Renderer: `figure_source/supplementary/plot_S_C2_to_C5.py`. Tiles: `figure_source/assets/example_tiles/`. |
| Figure S-C4 | `source_data/figure_S_C4_plot_data.json` | Six CheMeleon examples; saved values and image tiles. The renderer draws these figures from the saved values; label wording and layout differ slightly from the published artwork. Renderer: `figure_source/supplementary/plot_S_C2_to_C5.py`. Tiles: `figure_source/assets/example_tiles/`. |
| Figure S-C5 | `source_data/figure_S_C5_plot_data.json`, `source_data/figure_S_C5_gates.json`, `source_data/figure_S_C5_matrices/v4_chemberta_L0.npz`, `source_data/figure_S_C5_matrices/v4_chemberta_L1.npz`, `source_data/figure_S_C5_matrices/v4_chemberta_L2.npz`, `source_data/figure_S_C5_matrices/v4_chemeleon_main.npz`, `source_data/figure_S_C5_matrices/v4_chemeleon_mp3.npz` | Saved best-lift matrices and comparison summaries. The renderer draws these figures from the saved values; label wording and layout differ slightly from the published artwork. Renderer: `figure_source/supplementary/plot_S_C2_to_C5.py`. |
| Figure S-C6 | `source_data/figure_S_C6_plot_data.json` | Exploratory matched-coordinate shares, individual reference values and heterogeneous atom-response examples. |
| Figure S-E1 | `source_data/figure_S_E1_plot_data.json` | Full-clamp contrasts, associations and dose values. |
| Figure S-I1 | `source_data/figure_S_I1_plot_data.json` | Population lineage, scaffold intersections and atom-anchor verification counts. |
| Figure S-J1 | `source_data/figure_S_J1_plot_data.json` | Full-clamp and paired-dose trajectories, contrasts, intervals and doses; panel D is schematic. |
| Figure S-M1 | See description | Artwork only; extension analysis files are not included. Matched-control identities and per-coordinate reconstruction: Tables S-M1, S-M2a–S-M2b; control means are reported in the figure caption. |
| Figure S-N1 | See description | Artwork only; extension analysis files are not included. Coordinate identities: Table S-M1; single-coordinate contrasts: Tables S-N5–S-N8; joint matched-control contrasts: Table S-N9. |

`source_data/data_S1_best_units.csv` provides the unit-level table for the initial cross-model and subset-evaluation cohorts. Layout records and references to working files were removed from the source-data files; their numerical values are unchanged.

## Render Figures S-C2–S-C5 from saved results

`figure_source/supplementary/plot_S_C2_to_C5.py` uses NumPy ≥ 1.17 and Pillow to assemble the example panels for Figures S-C2–S-C4 from saved image tiles and draw Figure S-C5 from five saved matrices. The renderer draws these figures from the saved values; label wording and layout differ slightly from the published artwork. Its S-C5 branch reproduces the 16 saved effect estimates to within 1.1 × 10⁻¹⁶.
