# Rebuilding the controlled-edit inputs

The parent molecules are identified by the `evaluation` rows of `data/molecular/input_ids.tsv`. Source acquisition goes into an external cache. The deterministic, pre-activation chemical-selection stage is portable:

```sh
python experiments/06_molecular_edits/select_sites.py --input EXTERNAL_CANONICAL_STRUCTURES.tsv --output EXTERNAL_SITE_SELECTION.json
```

The input columns are `cid` and `canonical_smiles`. Use the frozen standardization and RDKit 2026.3.6 canonical atom order before comparing atom indices. The selector enumerates the fixed hydroxyl, ketone and acid motifs, applies isotope/chiral/bond-stereo restrictions, and includes aromatic fluorination only at graph distance at least four from a target site. It chooses the class by the minimum SHA-256 of `20261123/edit-class/CID/class`, and the site by the minimum SHA-256 of `20261123/edit-site/CID/comma-separated-indices`. It therefore selects at most one class and one site for each parent without activation information.

The independent local replay matched candidate counts, chosen classes and chosen sites for all 2,000 original parents. Of these, 1,509 have no eligible site. The retained input records give the subsequent six qualification failures, the 485 mapped pairs and the 467 controlled-string primary pairs. The chosen class/site is not itself proof that a product qualifies for the scientific endpoint.

The complete input preparation is also portable:

```sh
python experiments/06_molecular_edits/prepare_pairs.py --input EXTERNAL_CANONICAL_STRUCTURES.tsv --fixed-cohort data/molecular/evaluation_ids.tsv --chemberta-tokenizer-dir PUBLIC_CHEMBERTA_TOKENIZER_DIRECTORY --molformer-tokenizer-dir PUBLIC_MOLFORMER_TOKENIZER_DIRECTORY --output-dir EXTERNAL_OUTPUT_DIRECTORY
```

Run from the repository root. The public tokenizer providers and exact tokenizer hashes are in `data/molecular/pair_tokenizer_identity.json`. This command requires Python 3.11, RDKit 2026.3.6, NumPy and tokenizers; it needs no network, model weights or private study directory. It implements fixed graph editing, product standardization/qualification, parent-order noncanonical serialization, preservation of unrelated stereo, graph lineage, complete atom spans under both tokenizers, radius-one environment keys and the nonlocal-token syntax check. Historical overlap is recalculated from the bundled CID/canonical-identity exclusion manifest. The outputs are `pair_inputs.jsonl.gz`, `pair_eligibility.jsonl.gz` and `summary.json` in the explicitly supplied external output directory.

The input must contain `cid` and `canonical_smiles` columns. If starting with another source representation, `--standardize-source` applies the frozen standardization before identity checking. Current source acquisition is not assumed to reproduce historical normalization: each structure must match its recorded canonical SHA-256, and both tokenizer JSON hashes must match. On drift the command reports the expected and observed hashes and stops. The fixed manifest determines the 2,000-parent ordering. `--limit 16` runs the first 16 parents as an explicitly partial engineering subset. That subset regenerated four accepted pair records and sixteen eligibility records; all twenty canonical JSON record hashes matched the archived originals. No model forward was used. The complete 2,000-parent preparation was not rerun during assembly; the separate class/site replay did cover all 2,000.

The original accepted-pair mappings, rejected-case reasons, source algorithms and measured arrays were found; no irrecoverable input is currently identified. A tokenizer/source hash mismatch is a concrete source-identity failure, rather than permission to replace the cohort or omit a negative case.

The bundled real-model subset uses a single accepted, frozen hydroxyl-to-hydrogen pair in derived integer form and compares its matched activations with the archive. The CPU command recomputes all 4,640 R-squared values from raw paired activations. It does not yet reconstruct every contrast interval or restoration statistic, despite retaining those result tables and raw restoration arrays. Input preparation is executable for the full fixed cohort; full neural measurement for all five edits and the restoration queue have not been migrated. Complete scientific Level 2 reproduction is therefore not claimed.
