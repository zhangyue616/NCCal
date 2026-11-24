# Native, rotated, principal and dictionary directions

The same development rules, molecules, six representations, ten within-molecule displacement controls and molecule bootstrap are applied to 464 native directions, nine 464-dimensional Haar rotations, 464 principal components and three 928-feature TopK dictionaries: 7,888 directions. Rotations use seeds 30000 through 30008. PCA is fitted to all development heavy atoms. The dictionary seeds are 20261118 + 1000000007*rep, rep 0,1,2, with k=16 and validation-only reconstruction selection. Evaluation identities are excluded from dictionary fitting and validation.

Each dictionary also reports the 464 features with largest unlabelled validation variance contribution, selected before evaluation. Dead or nonvariable features remain in physical denominators; their R-squared is missing. Orthogonal transforms have identical whole-layer variance-weighted explained fractions, although counts crossing a per-direction threshold differ. The maximum archived numerical discrepancy is 7.1e-9.

There is no new inferential hypothesis family. Original/rotated basis ranks and recipe comparisons are descriptive. The archived guide tag was category III because the two dictionary-versus-native directions were mixed, not a prespecified inferential category. Native rewriting retention, rotated failures, full-width and variance-selected dictionaries and all three seeds are retained. The intervention dictionaries use block-two post-GELU activations, distinct from the final-block pre-GELU calibration dictionaries.

