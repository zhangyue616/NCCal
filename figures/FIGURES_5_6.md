# Figures 5 and 6

From the repository root, run `python -B figures/render.py 5 6`.
Outputs are written to `figures/rendered/`; `--output PATH` selects another directory.
The entry dispatches to `native/main_5.py` and `native/main_6.py`.
Both use the same `native/style.py` canvas as main Figures 1–4. The scripts
read repository-local inputs and also export their panel labels and annotations
as `Figure_5_strings.txt` and `Figure_6_strings.txt`.

The optional display environment is listed in `figures/requirements.txt`.
Exact reproduction was demonstrated with Python 3.11.17 and the listed package
versions. The selected artwork uses the Arial font family through Matplotlib's
normal font discovery. No machine-specific font location or font file is included.
Another font or renderer version can change typography and pixels. The optional
display requirements do not replace the repository's numerical CPU environment.

`figures/published/Figure_5.{png,svg,pdf}` and `Figure_6.{png,svg,pdf}` are the
author-selected exports. They are distributed outputs and are never drawing
inputs. The PNGs are 600 dpi: 4200 x 3416 and 4200 x 3316 pixels respectively.
Their canvases are 504 x 410 and 504 x 398 points. Fresh rendering reproduced
both PNGs byte for byte. SVG visible elements matched after disregarding
metadata and generated identifiers. PDF page streams, visible text, and
144-dpi rasterization matched; export timestamps differed.

Numerical figure tables remain in `results/artwork/use/`; the Figure 5a stored
atom responses are in `results/molecular/edits/raw_pairs.npz`. The two numerical
example exports and their English field definitions are in
`results/artwork/examples/`. Minimal molecular graphs and structural selection
inputs are in `data/artwork/examples/`. The structural component is generated
from `data/artwork/protein_exposure_geometry.json` by
`native/protein_exposure.py` and `native/protein_primitives.py`. Its small cached
PNG and site anchors are in `data/artwork/generated/`. The exposure anchors
are in `protein_exposure_pair_manifest.json`. `validate_exposure_example.py` checks the frozen
selection and geometry, and regenerates that component if its cache is absent.

Figure 5a selection first chooses, within each of the 60 eligible hydroxyl-to-H
pairs, the affected shared atom having the largest absolute rule-predicted
change. It computes the absolute prediction error at that atom and the median
error over the 60 pairs. It then selects the error nearest that median among
the 14 molecules having at most 16 heavy atoms. This selects CID 66277342,
canonical carbon index 9. The selected atom is not assumed to be an edited
carbon in every pair. Figure 5d uses that molecule's canonical encoding and
the two smallest frozen rewrite IDs. Figure 6a selection is structural and
does not read activations. Figure 6d is a post hoc illustrative selection;
its length bound and complete candidate ranking are documented with the JSON.

The top histogram in Figure 5e uses a frequency limit of 18: its largest
stacked bin is 16 and all 54 neurons are counted. The right histogram in
Figure 6b uses a frequency limit of 38: its largest stacked bin is 35 and all
98 labels are counted. Runtime assertions check that the plotted scatter
points lie inside the axis limits, histogram stacks fit inside the
frequency limits, and every included observation is counted.

`plot_use_results.py` draws supplementary Figures S-Q1, S-R1 and S-T1.
For S-G5, `native/protein_panels.py` generates the components assembled by
`plot_protein_supplement.py`.

Figure 5f reads all 928 features from each dictionary and scales every count by
464/928 before the square-root plotting transform. Native neurons, rotations
and principal axes have 464 directions. Right-side labels retain raw counts
and their denominators: dictionaries 53–55 of 928, neurons 26 of 464, principal
axes 8 of 464 and rotations 0–4 of 464.

Figure 6c adds a descriptive paired-accuracy rank row for each protein frame.
A label beats all nine rotated bases when `rotation_K` is 9 and
`rotation_pair_accuracy_upper_rank` is at most 0.1 (within floating-point
tolerance). The counts are 18 of 98 confirmation labels and 13 of 98 labels
on new proteins. This row is separate from the same-amino-acid discrimination
and sequence-predictor-increment tests; it does not show the rank of the
sequence-predictor increment.
