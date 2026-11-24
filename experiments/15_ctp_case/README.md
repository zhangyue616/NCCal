# cTP-net supplementary case

This case asks which level a phenotype name locates. The five source tables are in `results/ctp_case/Table_S-U1.tsv` through `Table_S-U5.tsv`; they use the scientific English translations and retain the source row order, definitions and numerical precision. These are supplementary results, not a separate study or an independently registered replication of every cohort.

Run from the repository root with Python 3.11 or newer and the cTP-case environment:

```sh
python -m pip install -r environment/ctp-case.txt
python -B experiments/15_ctp_case/reproduce.py --mode verify --output ../ctp-case-verification.json
```

The command rebuilds the five tables from the minimal derived inputs in `data/ctp_case/intermediates`, writes the tables next to the receipt, and compares every cell with the archived outputs. It does not read the archived five tables until the comparison step. It performs no network request, matrix processing, model fitting, label nomination, new statistical test or public-input model rerun. Confidence intervals, adjusted p-values and other summary-only fields are retrieved from frozen records, not re-estimated. The main repository verifier invokes this entry point as experiment 15.

| Table | Calculated from archived observations | Retrieved frozen fields and limits |
|---|---|---|
| 1: name recovery | Donor-by-pair margins, ranks and gains reconstruct per-donor rates, their medians, fixed-8,128 opportunity gains, and evaluable-pair ranges. MS target ranks are donor-row medians. Fixed historical rates are calculated from archived counts and their own valid-donor denominators. | Naming and locked-pair inventories, historical dual-success counts, fixed-descriptor target choices, positive-control correlations and the original Holm-adjusted comparisons are frozen summaries. The maximum Holm p = 0.0302 belongs to the 24-comparison fixed-8,128 main endpoint family, not a recovery-rate test. |
| 2: opportunities across bases | For each of 16 bases, possible pairs are calculated as C(variable units, 2), followed by same-name fractions and the release reference. | Same-name counts, variable-unit counts, global top-five variance and median within-coordinate top-five variance are per-basis archived summaries. |
| 3: physical sorting | Library coverage counts are summed within each of two pools. | AUROCs and class sizes come from archived pool-by-contrast records. Four healthy donors and five iPD donors were pooled before sequencing; these are two scored pools, not nine independent donor AUROCs. Per-cell scores are absent. |
| 4: third antibody | CD226 and PD-1 examples use donor purity rows to calculate purity medians, paired gains and panel-versus-fixed-panel differences. Inventory operations count 704 original selector rows, 17 selected rows, distinct unit pairs and the 0/17 flag total. | KLRG1 effects, intervals and Holm p-values are archived comparison summaries. The 17 units are outcome-selected selector rows, not donors. No new significance test or independent confirmation is added. |
| 5: frozen protein exam | Donor-comparison rows reconstruct paired mean effects, donor medians and common-exam ranges. Donor-by-basis records reconstruct absolute medians and question ranges. | Nine coordinate-specific winner totals are archived exam-summary counts, not re-counted individual questions. Confidence intervals, Holm p-values and family sizes are frozen tests. UCSF and JDM are old cohorts scored after locking; GSE313642 was previously unused. |

`limited_reference/scvi.json` is a four-row source check for the single scVI limitation statement. It is not a sixth scientific result table. The model chose its own coordinates under its own 128-coordinate search budget; no cross-model biological equivalence of coordinate numbers is asserted.

## File identities and privacy

`data/ctp_case/source_archive.json` lists the current input paths and the digests of the distributed excerpts. `data/ctp_case/original_record_map.tsv` separately records original source identifiers, original-file digests and excerpt digests. Rows and columns not needed by the original assembly logic were removed. Donor keys were replaced consistently within their cohort; no identity crosswalk, cell identifier, matrix or clinical record is included. Selector inventories retain only the cardinality, uniqueness or frozen flags needed by Table 4. The Table 4 freeze excerpt omits private `source_paths` and refers to original input identities by neutral record IDs; its selection facts and numeric thresholds are unchanged.

In the map, `semantic_record` is a content-based reference name and `archive_record` identifies an actual distributed file. A blank archive path means the original record is not included. `UNKNOWN` means no verifiable original-file digest was supplied; directory identifiers have no file digest. Original digests carried by the accepted source archive or locks are distinguished from digests calculated on the distributed excerpts. Inventory references beginning `record:` select a map row, with any following line or JSON selector retained as source-location metadata.

The two files in `locks/LOCK1.json` and `locks/LOCK2.json` are original bytes with their original `.sha256` records. They contain relative archival paths and public source identifiers. The map preserves their original file identifiers under the exam archive and records which selected excerpts are distributed. Verification checks lock byte identities and distributed excerpt identities. It does not check all thousands of files referenced inside the locks: most of those upstream training, nomination and acquisition artifacts are outside this minimal archive. Lock integrity is not a claim of complete upstream reproduction or of independent cohort provenance.

The source licence inventory and the object-specific review checked on 2026-11-24 are in `data/ctp_case/data_inventory.tsv`. Source data and donor-derived records have provider-specific terms. The first-party code licence does not license third-party matrices or resolve missing submitter terms.

## Figure

`figures/plot_figure_s_u1.py` draws Figure S-U1 from `results/artwork/ctp_case/figure_data.tsv`. The archived drawing has 16 basis counts, 32 recovery rates and 63 donor-order points. Panels a and b are bound to Tables 2 and 1 respectively. Panel c uses only the supplied 63 plotting points: Table 5's cohort summaries cannot recover those individual points.

```sh
python -B figures/plot_figure_s_u1.py --data results/artwork/ctp_case/figure_data.tsv --output ../ctp-case-figure
```
