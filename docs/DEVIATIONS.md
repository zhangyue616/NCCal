# Recorded execution deviations

## Memory interruptions and execution adapter

During the chemical execution, two memory-related interruptions occurred. The fixed-coordinate bootstrap resumed from completed results with the same stored molecule weights, and the oracle calculation completed once an adapter released already-validated input objects before the unchanged oracle function. This adapter was introduced as an execution repair after confirmation outputs were observed, as reported in Supplementary Section S-G.

The adapter is `scripts/fixed_coordinate_analyses/chem_T_oracle_memory_resume.py`. Its preparation, execution and authorization are recorded in `results/fixed_coordinate_analyses/chemistry_T/oracle_memory_engineering/ADAPTER_READY.json`, `EXECUTION.json` and `ROOT_REVIEW_AND_GO.json` in that directory. The execution record states that winners, candidate count, data types and bootstrap replicates were unchanged and that fixed winners were not replaced. The oracle result is descriptive, with point estimates only.

## Numerical-threshold exceedance

`results/sensitivity_analyses/paired_dose/engineering_recovery/AMENDMENT.json` records an extra all-output baseline comparison that stopped at an unselected output: the maximum absolute difference was 1.1444091796875e-05 against a 1e-5 threshold. The frozen target output's maximum difference was 4.172325134277344e-07. The record attributes the difference to float32 rounding under different deterministic batch composition and right-padding.

The repair retained the all-output diagnostic and the unchanged 1e-5 tolerance for each frozen target output, reused completed results and the saved baseline, and changed none of the pairing, target dose, intervention coefficients, hook, statistics or samples. The recorded exceedance was outside the frozen target output. Supplementary Section S-I.4 identifies this recorded exceedance.

## Dose-verification recovery

For one molecule (confirmation-set index 8381), dose verification initially gave an absolute error of 1.5744881362023122e-06 against the original per-condition tolerance of 1.500690181674742e-06. Restoring the original four-molecule batch reduced the error to 2.0747876361415507e-08. The baseline and all 18 conditions for that molecule were restored using the same geometry; the original baseline and condition arrays were preserved, and the final result arrays integrate explicit one-row replacements. Response and dose values for that row therefore changed; the pairing, target doses, intervention coefficients, target output, means, norms, hook, bootstrap and all samples did not change.

The recovery script is `scripts/sensitivity_analyses/c_geometry_recovery.py`. The six files in `results/sensitivity_analyses/paired_dose/geometry_recovery/` are `AMENDMENT.json`, `DIAGNOSTIC.json`, `DIAGNOSTIC_ARRAYS.npz`, `RESTORATION.json`, `RESTORED_ARRAYS.npz` and `APPLIED_TRACE.json`. They retain the diagnostic, restored arrays and application trace. Supplementary Section S-I.4 identifies this dose-verification recovery.
