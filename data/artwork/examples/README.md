# Minimal drawing inputs

`hydroxyl_pair_inputs.jsonl.gz` contains 61 frozen hydroxyl-to-hydrogen graph
records, including the 60 passing the main string-control gate. Only CID,
edit class, source/product SMILES, removed atom IDs, edit site and gate status
are retained. Native Figure 5a responses and predicted changes are read from
`results/molecular/edits/raw_pairs.npz`; this graph file supplies no new response.

`exposure_pair_selection.tsv` contains the 967 frozen confirmation exposure
pairs. Native sequence length and mapped-chain fraction are retained only for
alanine/helical candidates, the only rows eligible for this example. The
selection specification is in `exposure_pair_selection.json`: retain A/H
pairs with mapped fraction at least 0.95, then order by distance of full native
sequence length from 270 and by homology-group ID. There are 52 candidates;
the selected pair is UniProt P05820 / PDB 2XMX chain A, alanines 86 and 248.
Their RSA values are 0.003479374240535174 and 0.5785970761794452. The full
native sequence has 271 residues; the retained backbone has 270, positions
2 through 271. Selection uses no activation or prediction.

`data/artwork/protein_exposure_geometry.json` contains only derived N, CA, C,
O coordinates, secondary-structure classes, site metadata and provenance IDs.
The renderer generates its ribbon and selected-site spheres locally. The
orange sites and central orange sequence cells mark the chosen buried/exposed
alanines; orange is not a magnitude scale for a neuron or a public label.
Dark backbone windows correspond to the seven-residue sequence windows.
No raw PDB/mmCIF, SIFTS file, third-party database, model or font is required.
