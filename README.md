# NCCal

NCCal tests what a name attached to an individual neuron can predict. A molecular example follows a hydroxyl-related neuron from its response on a new molecule through a functional-group edit and replacement of its activation in the frozen model. The protein experiments ask which published and newly identified neurons distinguish matched residues after accounting for their amino acid and surrounding sequence.

This repository accompanies *NCCal: reading, testing and replacing individual neurons in molecular and protein language models*. The software version is **v2.0.0**. The article and Supplementary Information identify summary results by figure, table and section. The numbered worksheets belong to the article's separately distributed Supplementary Data 1.

## Recalculate stored results on a CPU

From the top of the extracted repository, use Python 3.11.15 and the pinned packages. Install from a local wheelhouse using `--no-index --find-links WHEELHOUSE` when working offline.

```sh
python -m venv .venv
# Activate .venv in the current shell.
python -m pip install -r environment/cpu.txt
python -B verify/run.py --experiments --out ../nccal-verification
```

The command writes all ten manuscript and supplementary table grids, a record for each displayed numerical position, and an experimental verification result for each of the 14 directories. Exit status 0 means all recorded positions are accounted for; 1 indicates a disagreement or a failed experimental check; 2 indicates missing coverage. The `source` and `field_or_operation` columns in `verify/paper_numbers.tsv` identify calculations, archived study statistics, protocol settings and notation. For example, the primary sparse-dictionary calibration fold is recalculated from 173 concepts and 199 untrained networks. Protein point estimates are recalculated where the contributing observations are available, and frozen interval endpoints are retrieved by task, readout and comparison. The command runs independently of the article workbook. Further keyed checks on archived intermediates appear in `keyed-trace.tsv`.

To rebuild the tables or check two main figures separately:

```sh
python -B src/nccal/tables.py --out ../nccal-tables
python -B figures/render.py 2 3 --output ../nccal-figures
```

The figures use a separate display environment. Versions, rendering commands and the inputs behind individual panels appear in [the figure guide](figures/README.md). The [complete artwork index](figures/inventory.tsv) covers the six main and 23 supplementary figures. The scientific table inputs are in [results/tables](results/tables).

## Recompute model outputs

The study uses sequence and graph encoders, sparse dictionaries and task heads. The inference commands use separately acquired weights. `python data/fetch.py` lists pinned public inputs. `python data/fetch.py KEY --check FILE` checks an existing file without a network request. See [source identities and reuse terms](data/SOURCES.md) before obtaining restricted inputs. The identifier lists include the PubChem sequence-rewriting splits, PDB and UniProt accessions, and the structure-file checksums.

A molecular example performs a frozen-checkpoint forward pass on the original and rewritten forms of a molecule:

```sh
python -B experiments/05_molecular_rewrites/reproduce.py --mode smoke --model-dir MODEL_DIRECTORY --device cuda --output ../rewrite-example.json
```

Each experiment has its own protocol and executable scope. The [protocol index](experiments/protocol_index.tsv) links the English protocols to original file digests and recorded times. The structural readout validation entries use the release folder name in their record identifiers. Their checksums refer to the original archived files. The [molecular guide](src/nccal/molecular/README.md) and [protein coverage table](data/protein/reproduction_coverage.json) describe exactly which operations were tested. A protein subset can be run with the corresponding checkpoint and public sequences:

```sh
python -B data/protein/acquire.py --kind forward-sequences --output ../nccal-inputs
python -B experiments/09_protein_exposure/reproduce.py --mode forward-subset --weights MODEL_DIRECTORY --sequences ../nccal-inputs/forward_sequences.jsonl --device cpu --output ../exposure-example.json
```

The full nine-encoder calibration, refitting of all downstream heads, training every dictionary and checking all protein reference networks require additional computation and input reconstruction. The [environment guide](environment/README.md) gives hardware and timing information. Some historical full-run scripts refer to the original research layout. The per-experiment coverage records identify these scripts and distinguish them from the tested subset commands.

## Find an analysis

| Main figures or supplementary sections | Experiment directories | Principal output |
|---|---|---|
| Figures 1 and 2, S-A, S-B, S-C, S-L | 01 molecular calibration, 02 molecular reconstruction | Model and cohort definitions, native and reference profiles, differences between native and rotated axes, examples and atom mapping |
| Figures 2 and 3, S-D, S-J, S-M | 01, 02 | Transfer to held-out molecules, label prediction and single-neuron and matched-panel reconstruction |
| Figures 3 and 4, S-E, S-F, S-I, S-N | 02, 03 whole layer | Intervention dose, output deviation, placement controls and BACE/ESOL performance |
| Figure 5, S-O, S-P | 03, 04 sparse dictionary, 07 basis comparison | Layer variance decomposition, frozen-head and refitted-head results, dictionary calibration |
| Figure 5, S-Q, S-R, S-S | 05 molecular rewrites, 06 molecular edits, 07 | Same-atom tracking, functional-group edit responses, retention across bases |
| Figure 6, S-G | 08 protein pair replay, 09 protein exposure, 10 protein reconstruction, 11 protein scale | Within-protein readouts, sequence controls, exposure and replacement |
| Figure 6, S-G, S-T | 12 native protein candidates, 13 external validation, 14 published label audit | External protein cases, new-protein validation, fixed and public readouts |
| S-H, S-K | Protocols and deviations in the corresponding experiments | Execution departures, scope of each comparison and guidance for interpretation |

The numerical and plotting inputs are under `results/molecular`, `results/protein` and `results/artwork`. The `data` directory holds identifiers, small study-derived fixtures and acquisition code. Original third-party label tables, pretrained model weights and bulk database snapshots remain at their sources. Study records retain unfavourable comparisons, including the whole-layer placement controls and the prespecified exposure endpoints.

## Citation and licence

Cite Yue Zhang, *NCCal*, software version v2.0.0. The [citation record](CITATION.cff) identifies the intended project URL [zhangyue616/NCCal](https://github.com/zhangyue616/NCCal). The [MIT licence](LICENSE) applies to first-party code. Original data are subject to their providers' terms. The source and data guide identifies the licence for first-party derived material.
