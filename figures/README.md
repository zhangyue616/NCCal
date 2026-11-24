# Figure rendering

All six main and 24 supplementary figures have executable renderers using repository-local numerical intermediates and derived molecular/protein geometry. They do not open supplied finished manuscript figures, load model weights, run inference, fit predictors or resample uncertainty.

Run from the repository root:

```bash
python -B figures/render.py --all --output figures/rendered
```

Individual identifiers also work, for example `python -B figures/render.py 5 6 S-C2 S-M1`. Shared renderers produce their companion figures together. Every figure exports PNG, SVG and PDF. This layer requires NumPy, Matplotlib, Pillow and RDKit; model frameworks and CairoSVG are unnecessary. Arial is preferred when installed and is not redistributed.

`inventory.tsv` and `inventory.json` give the inputs, drawing code and command for each of the 30 figures. The renderers use stored study results. All six main figures share the canvas and drawing styles in `native/style.py`.

| Figures | Drawing code | Inputs |
|---|---|---|
| Main 1 | `native/main_1.py` | Calibration, reconstruction and output tables; the fixed Figure 5 molecule, basis summaries and Figure 6 protein example; selected molecular and protein geometry |
| Main 2–4 | `native/main_2.py` through `main_4.py` | Main result tables, atom arrays and selected geometry |
| Main 5 | `native/main_5.py` | Edit, rewrite and basis tables, archived pair arrays, minimal molecular graphs and the real three-notation export |
| Main 6 | `native/main_6.py` | Public-label, fixed-channel and exposure tables, the real Q9I398 export and derived backbone geometry |
| S-Q1/R1/T1 | `plot_use_results.py` | Edit, rewrite and public-label tables |
| S-U1 | `plot_figure_s_u1.py` | Supplied English single-cell source values: 16 basis counts, 32 rates and 63 donor-order points |
| S-G5 | `native/protein_panels.py`, composition script | Paired result tables and derived backbone geometry |
| S-B1/I1 | `native/supplement_audits.py` | Readout, lineage and anchor-audit JSON |
| S-C1 | `native/supplement_tokenizer.py` | All 593 archived embedding rows and summary |
| S-C2/C3/C4/C5 | `native/supplement_examples.py` | Fixed units, atom responses, 108 molecule geometries, calibration matrices/gates |
| S-C6 | `native/supplement_pilot.py` | Pilot values and saved example atom responses |
| S-E1 | `native/supplement_mechanism.py` | Mechanism and full-clamp dose summaries |
| S-G1/G3 | `native/supplement_masked_components.py`, `compose_masked_supplement.py` | 1,304-group values, paired-site activations and derived backbone |
| S-G2 | `native/supplement_exposure_endpoints.py` | Network, inference and rotation tables |
| S-G4 | `native/supplement_protein_validation.py` | Null selection, known basis, repeated structures and decision policies |
| S-J1 | `native/supplement_paired_dose.py` | Frozen paired-dose and full-clamp results |
| S-J2 | `native/supplement_notation.py` | Twelve frozen estimates and intervals |
| S-M1/N1 | `native/supplement_matched_controls.py` | Saved reconstruction and fidelity values |
| S-O1/P1 | `native/supplement_layer_data.py`, `supplement_layer_dictionary.py` | Whole-layer and dictionary tables |

Numerical inputs are in `results/artwork/`; derived geometry is in `data/artwork/`. S-C2–C4 regenerate their atom-colour layers from saved response vectors and molecule geometry. Their original raster tiles are not inputs. S-G1/G3/G5 and main 6 compose newly drawn numerical panels. SVG axes/text remain vector objects; protein/RDKit depictions are embedded raster components. Protein composite PDFs use full-resolution raster composition.

The protein renderers use derived 2CZL/2XMX backbone geometry (270 residues each). Raw PDB/SIFTS snapshots, papers, manuscripts and external icon libraries are absent. The optional structure download helper prints public IDs/URLs by default and requires `--download` to acquire files; rendering never invokes it. See `data/artwork/README.md` for asset provenance.

The rewrite advantage is a difference in RMS output deviation across 199 normalized outputs, displayed in 0.001 units. Edit panels show atom-level points and read the frozen molecule-equal R2. All four E1 helix-versus-strand rows use 140 new pairs. Confidence intervals, multiplicity flags and descriptive-reference scopes are read from frozen results. S-C5 checks all 16 displayed effects against the gates; S-M1/N1 check 54 reconstruction points, 216 single contrasts, 12 joint-control intervals and all axis ranges against saved values.

The complete earlier batch was tested with Python 3.11.17, NumPy 2.4.6, Matplotlib 3.11.2, Pillow 12.3.0 and RDKit 2026.03.6. Main 2–4 and S-G2/G4/O1/P1 matched the corresponding Word PNGs in RGB pixels. S-Q1/R1/T1 matched in the original display environment (Python 3.7.0, NumPy 1.15.1, Matplotlib 2.2.3). The other supplementary renderers use retained numerical inputs; fonts, depiction rendering, panel lettering or composition can differ.

The main Figure 1, 5 and 6 PNGs reproduced byte for byte through `render.py` in the modern display environment. The SVGs match after normalizing export metadata and generated element identifiers. The PDF page content also matches. The author-selected files in `published/` serve as reference outputs. `FIGURE_1.md` maps the overview panels to their stored data and distinguishes the schematic drawings. `FIGURE_1.md` maps the overview panels to their stored data and distinguishes the schematic drawings. `FIGURES_5_6.md` documents the source data, example selection and frequency-axis ranges.
