# Numerical example exports and selection

These UTF-8 JSON files contain measured and frozen-rule values, not rendering
manifests. Plotting reads them directly. Documentary source pointers identify
released subsets or public scientific IDs; SHA-256 values associated with a
source entry refer to its archived original full artifact as stated in
`sha256_scope`. No original archived array is claimed to be a byte-identical
copy of its released JSON subset. All numerical, boolean and null fields are
unchanged from the actual exports. No model forward or fit was run to make
these figures.

## Figure 5d: CID 66277342

`Fig5d_CID66277342_three_SMILES.json` supplies `Nc1nc(NCCOCCO)n[nH]1`
and the two numerically smallest frozen rewrite IDs, 1 and 2. It belongs to
the qualified frozen cohort (zero-based cohort rank 1026; evaluation index 26),
with 13 heavy atoms. This molecule was already present; it is not an added
sample. The full cohort/evaluation sizes are 3000/2000. `cohort` records the
sampling and gate status. `measurement` defines ChemBERTa-MTR natural block 2
(zero-based block 1), post-GELU units 293 and 454. `encodings` holds the strings,
model tokens, atom-to-character/token mapping, measured values and two frozen
rule values per atom. `prediction` gives the frozen lookup/backoff definition.
The atom-matched rule is invariant across these encodings. Twelve atom labels
use the radius-1 lookup; one uses the radius-0 backoff.

Canonical atom indices are zero-based; one-based counterparts are explicit.
Character spans are zero-based half-open intervals. Token indices include
special tokens, which are explicitly listed. Missing or ambiguous positional
anchors remain null; they must never be interpreted as zero. The mapping
records expose missing source characters, cross-atom tokens and identity loss,
including the `[nH]` atom. The figure's character boxes shade element anchors;
they do not assert that every bracket or H character survives tokenization.

`highlighted_site_from_figure_5a_edit` binds the edit independently of activity:
canonical carbon index 9 is adjacent to hydroxyl oxygen index 10 (one-based
numbers 10 and 11). Their character and token correspondence is given for
all three encodings. Figure 5a's separate median-error selection is described
in `figures/FIGURES_5_6.md`; its 60 selected atoms are not all assumed to be
edited carbons.

## Figure 6d: Q9I398, E1 channel 1900

`Fig6d_Q9I398_E1_1900.json` retains the actual unmasked full-chain input (268
residues) and signed native E1-150M readout: natural block 20, zero-based block
19, zero-based channel 1900, post-SwiGLU before the output projection, fixed
sign +1. This is an amplitude, not a probability. Two model prefix tokens are
excluded from residue positions. `readout_definition` states the tensor site,
index conventions and sign. `selected.residues` contains every input residue,
frozen labels and measurements. `selected.structure` and per-residue mappings
give PDB 6TD9 chain A annotation evidence. Its 250-residue construct maps to
UniProt 19-268; it is not the 268-residue model input. Missing coordinate
mapping is null and does not remove the full-chain readout.

`frozen_pair_audit` preserves all 140 NEW pairs: 122 strictly correct, 18
incorrect and 0 ties. The illustrative selection first computes the median
signed helix-minus-strand difference over all 122 correct pairs, before a
length bound: 0.6957096392288804. `all_correct_pairs_used_for_median` retains
those 122 values. With a full-input bound of <=250 residues there are **zero
correct candidates**: the only pair in that length range is the incorrectly
ordered Q9D6Y7 (length 233), recorded in `strict_original_250`.

The author therefore used <=275 full-input residues. Twelve pairs meet that
bound; ten are correct. `ranked_eligible_candidates` lists all ten, sorted by
absolute distance to the above median, with original frozen pair-row order
as the tie-break. Q9I398 is rank 1: arginines at helix position 191 and strand
position 49 have readouts 0.5766767263412476 and -0.0012573824496939778,
difference 0.5779341087909415 and distance 0.11777553043793887 from the median.
This author-requested post hoc illustration does not change the aggregate
frozen result or constitute an additional confirmatory test.

| Rank | UniProt | Full input length | Signed helix minus strand | Distance from median |
| --- | --- | ---: | ---: | ---: |
| 1 | Q9I398 | 268 | 0.577934108791 | 0.117775530438 |
| 2 | P0C0T5 | 274 | 0.871000394225 | 0.175290754996 |
| 3 | Q6PAF4 | 270 | 0.924325637519 | 0.22861599829 |
| 4 | Q7M135 | 269 | 0.448295196053 | 0.247414443176 |
| 5 | P46670 | 268 | 0.402412697673 | 0.293296941556 |
| 6 | P13598 | 275 | 0.325615115464 | 0.370094523765 |
| 7 | Q5JIZ8 | 261 | 0.182758249342 | 0.512951389886 |
| 8 | P55969 | 260 | 1.2459545508 | 0.550244911574 |
| 9 | P13021 | 272 | 0.135588571429 | 0.5601210678 |
| 10 | L0E4F8 | 264 | 1.3793862462 | 0.683676606975 |
