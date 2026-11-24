# Numerical evidence

`run.py` checks displayed numerical positions against named calculations and frozen evidence selectors. `paper_numbers.tsv` records each position's source, operation and display scale; `claim_sources.json` assigns selectors to their modules.

The public-label distribution in `effects_claim_bindings.py` includes every trained, formally eligible public label in frame T, regardless of whether either test passed. It computes medians across labels and the 25th/75th percentiles with NumPy's linear quantile method. Each label's paired accuracy already averages its pair credits equally across homology groups. Annotation-versus-unannotated residue AUROC and same-amino-acid paired accuracy remain separate estimands.

