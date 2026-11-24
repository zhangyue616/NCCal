# Solvent exposure beyond sequence

This analysis measures continuous relative solvent accessibility in natural unmasked proteins. Development and confirmation comprise 325 and 1099 homology groups, with 20974 and 173859 eligible residues. Structural annotations use experimentally resolved, exact-mapped assembly residues. Missing or uncertain mappings are excluded rather than treated as negative observations. The 194 reserved groups do not enter the original confirmation denominator.

The sequence reference uses a 31-residue one-hot window, relative position, squared position, chain length and intercept. Four-fold group-balanced development cross-validation selects regularization. Every candidate axis receives the same scalar residual-correction opportunity, and coordinate ties select the smaller index. Exposure readouts are ESM-2 8M block-3 neuron 801 and 35M block-6 neuron 814.

The prespecified primary endpoint is 8M continuous MAE gain, whose interval spans zero. The secondary matched-pair endpoint uses one same-chain, same-amino-acid, same-secondary-structure buried/exposed pair per group, with RSA at most 0.10 versus at least 0.40 and nonoverlapping local windows. There are 195 development and 967 confirmation pairs. Antisymmetric sequence-difference kernels produce paired Brier losses. Continuous losses average residues within groups then groups equally. Long-range-contact strata are descriptive and do not identify a physical pathway.

The original source byte digests, file sizes, preserved freeze timestamps and the corresponding English protocol are listed in frozen_records.json. Date stamps identify freezes; they do not denote separate reproduction outputs.
