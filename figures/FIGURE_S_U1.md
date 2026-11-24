# Supplementary Figure S-U1

`plot_figure_s_u1.py` renders the supplied English artwork from `results/artwork/ctp_case/figure_data.tsv`. The drawing and source values were integrated without new model inference or a new scientific analysis. The original figure-data numeric strings were retained.

- Panel a contains 16 same-name-pair counts, bound to the same-name count column in the corresponding basis rows of `results/ctp_case/Table_S-U2.tsv`.
- Panel b contains 32 recovery rates. Each is the matching JDM or Wells value in `Table_S-U1.tsv` (both members beat the rival and rank in the top five), multiplied by 100. Each point is the median of per-donor conditional pair fractions for that basis's own evaluable pairs.
- Panel c contains the supplied 63 donor-order plotting points, grouped as UCSF 20, JDM 20 and GSE313642 23. They show native minus the median of eight rotations on the locked protein exam. Horizontal marks show cohort donor means. These points are not reconstructed from Table 5's cohort summaries. UCSF and JDM are old cohorts scored after locking; GSE313642 was previously unused.

```sh
python -B figures/render.py S-U1 --output ../nccal-ctp-figure
```

The author-selected PDF, SVG and PNG files are preserved in `figures/published`. Figure 1 and its sources remain unchanged. Provider-specific terms for cTP-case derived inputs are in `data/ctp_case/RIGHTS.md` and `data_inventory.tsv`.
