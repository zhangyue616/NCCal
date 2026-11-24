# Molecular reproduction coverage

The seven experiment entry points share the numerical definitions in this package. The CPU commands use archived model outputs, per-concept profiles, per-molecule intervention advantages, paired activations, or derived sufficient statistics. They do not require model weights. The separate real-checkpoint command checks a bounded subset; its PASS must not be read as validation of the complete experiment.

For each experiment name in the table below, the portable commands are:

```sh
python experiments/EXPERIMENT/reproduce.py --mode verify
python experiments/EXPERIMENT/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda
```

`MODEL_DIRECTORY` is a separately obtained snapshot of `DeepChem/ChemBERTa-77M-MTR`, revision `66b895cab8adebea0cb59a8effa66b2020f204ca`, containing `config.json` and `pytorch_model.bin`. The command uses the bundled integer-input and verified atom-anchor fixtures, not a private study-directory layout. The fixture CID, model revision and source checkpoint checksums are in `data/molecular/`. Source molecules can be retrieved into an external cache using the explicit CID-download command; this offline assembly did not run that network operation.

| EXPERIMENT | Actual real-checkpoint subset | Complete Level 2 status |
|---|---|---|
| `01_molecular_calibration` | One public-input fixture through the actual pretrained encoder; archived block-two activations/output are checked, and the final pre-GELU calibration site is exercised. | The nine-model, 199-initialization, nine-rotation extraction and all robustness arms have not been migrated to one portable full-run command. CPU calibration covers the eight-checkpoint family and the separate twelve-layer MoLFormer selection. |
| `02_molecular_reconstruction` | One molecule, frozen graph prediction substituted at neuron 293, original-value sham, real 199-output forward. | The scalar-replacement interface works. Complete development refitting, independent panel extraction, matched-dose inference and all uncertainty tables are not yet run from a clean environment. |
| `03_whole_layer` | The same real molecule with all 464 channels replaced by graph predictions or the development mean. | This tests whole-layer writeback into the anonymous pretrained head. It does not exercise the frozen BACE/ESOL heads, their fitting or the complete task populations. Those task curves and metrics are recalculated on CPU from archived outputs and derived score/error statistics. |
| `04_sparse_dictionary` | Real final-block pre-GELU activations passed through the archived primary 928-feature TopK dictionary and decoder; finite reconstruction and at most 16 active features are checked. | No clean-environment SAE retraining, 504-fit reproduction or complete concept-score extraction is claimed. CPU rank/fold/interval calculations cover every saved recipe. |
| `05_molecular_rewrites` | All six representations of one CID. All 464 same-atom activations and the 199-output vectors are compared with the archived readouts. | This is a genuine archived-value inference replay. The 3,000-molecule selection, full 2,000-molecule census, rescue fitting and reference-network extraction have not all been executed by a portable full-run driver. |
| `06_molecular_edits` | One frozen controlled hydroxyl-to-hydrogen pair, with source/product matched activations and their differences checked against the archive. | The complete input-preparation command takes external public structures and the fixed cohort, edits and qualifies products, constructs controlled strings, and maps atoms under both tokenizers. Class/site selection matches all 2,000 original parents; the full preparation chain matches 20 archived records for a 16-parent subset. The complete neural measurement and restoration queue has not been ported. See `experiments/06_molecular_edits/LEVEL2_INPUTS.md`. |
| `07_basis_comparison` | One real molecule; rotation, PCA and TopK coefficients compared with archived readouts; a real writeback along each basis is executed. | No new fitting of PCA/three dictionaries or replay of every direction and intervention condition is claimed. CPU summaries cover all seventeen basis rows. |

The actual installed execution environment was Python 3.11 with NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.11.0+cu128, Transformers 5.20.0, tokenizers 0.23.3, scikit-learn 1.9.1, RDKit 2026.3.6 and threadpoolctl 3.7.0. These are observed versions, not a fresh-install lock. The tiny neural replays used strict float32 with TF32 disabled on an RTX 5090. Their model is small; full-suite GPU/VRAM costs cannot be inferred from this subset. Cold imports dominate these bounded checks, which are expected to take seconds to tens of seconds locally; a separate per-experiment timing benchmark was not performed. The clean-environment test and environment lock remain a repository-level acceptance responsibility.

## Public-input and weight boundaries

The CID manifests identify the molecular cohorts. They do not claim that a current rolling PubChem feed reproduces an old feed byte for byte. Fresh source acquisition must check the frozen identities and use the original standardization and tokenizer revisions. The integer fixture is a small derived input, not a bulk PubChem snapshot. Benchmark source structures and per-input measured targets are excluded; derived model predictions and grouped score/error sufficient statistics support the CPU task metrics.

The real-model smoke uses one trained chemical checkpoint. It does not validate every graph checkpoint, MoLFormer, random initialization or training recipe. The same seed under newer Torch/Transformers did not reproduce the historical initialized ChemBERTa states; exact full reference replay must use the recorded old initialization environment or the bound state artifacts. Model weights are excluded, and their provider's terms apply separately from this code.

Full-run commands are intentionally not advertised where orchestration remains incomplete. The original scripts and omitted intermediate files were found locally; the open issues are migration, complete numerical coverage and a verified clean environment, rather than an asserted loss of the research evidence. Consult each experiment's protocol, source-identity mapping, deviations and actual coverage note before interpreting a PASS.
