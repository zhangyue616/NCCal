# Protein input identities

Only identifiers, checksums and derived scientific values are distributed. Raw bulk structure, mapping and protein database snapshots and all pretrained model weights are excluded. `model_sources.json` fixes available model identities; `public_label_sources.json` fixes external author tables without redistributing originals.

`forward_subset_identifiers.json` gives small original-input model checks. Its residue positions are one-based in the natural sequence. The masked-alanine archive and E1 pair tables store zero-based indices; those were explicitly converted. Actual CPU subset results are recorded outside the repository in the build verification record.

For the published ESM C helix example, obtain the provider's ESM C 600M MLP concept table from the pinned dataset in `public_label_sources.json`. Its SHA-256 is `7f192247ed759ca9798d62e533f15ff4a2128000e129de2701627fe3e90412ba`. This distribution contains the selected unit identity and the measurements on the study proteins. The original six-row author-table extract is identified by SHA-256 `16f302883c297d728ede0f8cfeead03f9d34deee7c9804695c131ea3390be862` and remains with the upstream table.

Optional retrieval commands make public network requests only when explicitly invoked:

```sh
python data/protein/acquire.py --kind forward-sequences --output inputs/protein
python data/protein/acquire.py --kind public-label-tables --output inputs/public_labels
```

Retrieval was not run during the offline build. A changed public sequence or table is rejected by its frozen checksum. Current public endpoints do not guarantee availability of every historical byte; the author may need a separately licensed release attachment for a changed input.

External model implementations and author demonstration code are not part of this authored-code release. This includes UniRep, AMPLIFY, Protify unit readers and FastPLMs checkpoint-conversion/modeling modules. Legacy mapping audits accept explicitly supplied source files or directories. Their complete historical layouts and remote checkpoint conversion were not rerun by the seven portable CPU forward subsets.
