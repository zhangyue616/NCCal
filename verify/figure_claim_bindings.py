"""Article-position bindings to the actual figure data and drawing geometry.

Each value is in its article display unit. Design quantities and plot geometry
are identified as such; no search for equal numbers in the article is used.
"""
from pathlib import Path
import ast
import csv
import json
import re
import numpy as np


def bindings(root):
    root = Path(root)
    result = {}

    def load(path):
        return json.loads((root / path).read_text(encoding="utf-8-sig"))

    def rows(path):
        with (root / path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.endswith(".tsv") else ","))

    def put(ids, value, source, operation):
        for identifier in ids.split():
            if identifier in result:
                raise ValueError("Duplicate figure binding " + identifier)
            result[identifier] = {"value": float(value), "source": source, "operation": operation}

    def common(values):
        values = list(values)
        assert values and len(set(values)) == 1, values
        return values[0]

    def tree(path):
        return ast.parse((root / path).read_text(encoding="utf-8-sig"))

    def function(path, name):
        return next(n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name == name)

    def assigned(path, name):
        found = [n.value for n in ast.walk(tree(path)) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(found) == 1, (path, name)
        return ast.literal_eval(found[0])

    def axis_limits(path, axis):
        found = []
        for n in ast.walk(function(path, "build")):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == axis and n.func.attr == "set"):
                found += [ast.literal_eval(k.value) for k in n.keywords if k.arg == "xlim"]
        assert len(found) == 1, (path, axis)
        return found[0]

    def ci_key_level(mapping, prefix="CI"):
        keys = [k for k in mapping if re.match(prefix + r"\d", k)]
        assert keys
        key = keys[0]
        match = re.match(prefix + r"(\d+)(?:_(\d+))?(?:_|$)", key)
        assert match, key
        return float(match[1] + ("." + match[2] if match[2] else ""))

    art = "results/artwork/"
    supp = art + "supplement/"

    # Main Figure 2a: three graph checkpoints, three old initializations and ten
    # pilot rotations are distinct from the 199-network confirmatory family.
    source = art + "main/pilot.json"
    pilot = load(source)["pilot"]
    put("MS-0355 MS-0386", len(pilot), source,
        "Figure 2a: count the three named pilot graph checkpoints, not all nine confirmatory checkpoints.")
    put("MS-0387", common(len(r["initialization_dots_pct"]) for r in pilot.values()), source,
        "Figure 2a design: number of initialization-network dots per graph checkpoint.")
    put("MS-0388", common(len(r["rotation_dots_pct"]) for r in pilot.values()), source,
        "Figure 2a design: number of pilot rotated-axis readings per checkpoint.")
    source = art + "main/calibration.json"
    calibration = load(source)["panel_AB_frozen_source"]["audit"]
    h4 = [m["H4"] for m in calibration["models"]]
    put("MS-0389", common(m["K"] for m in h4), source,
        "Figure 2b: H4.K initialization references; separate from the older pilot and H3 families.")
    put("MS-0390", common(m["R"] for m in h4), source,
        "Figure 2b: H4.R rotated readings used within each trained/initialization model.")

    # Main Figure 3b/c/e: selectors follow physical coordinates and the actual
    # representative-molecule array, rather than another quantity equal to two.
    source = art + "main/activation_output.json"
    activation = load(source)
    put("MS-0424", len({r["source_row"]["coordinate"] for r in activation["activation_rows"]}), source,
        "Figure 3b: distinct focal coordinate identifiers in the saved pooled-atom activation rows.")
    level = common(float(re.search(r"marginal_(\d+)_percentile", r["source_row"]["uncertainty"])[1])
                   for r in activation["activation_rows"])
    put("MS-0427", level, source,
        "Figure 3b interval design: nominal percent encoded by each source row's marginal_95_percentile uncertainty field.")
    source = art + "main/reconstruction_panel.json"
    reconstruction = load(source)
    first = reconstruction["row_order"]["chemberta_mtr"][0]["coordinate"]
    focal = [r for r in reconstruction["coordinates"] if r["model"] == "chemberta_mtr"
             and r["coordinate"] == first and r["predictor"] != "Development mean"]
    put("MS-0429", len({r["predictor"] for r in focal}), source,
        "Figure 3c: count the binary-concept and local-graph predictions joined for each physical neuron; exclude the development-mean reference.")
    put("MS-0430", len({r["coordinate"] for r in focal}), source,
        "Figure 3c: the two joined predictions refer to one and the same physical coordinate.")
    source = "data/artwork/molecules/fig4_mol2d.json"
    molecules = load(source)["molecules"]
    put("MS-0431", len(molecules), source,
        "Figure 3e: count the two actual representative molecular geometries supplied to the panel.")

    # Representative hydroxyl molecule in the opening text. Match the frozen
    # depiction graph to its saved concept-label mask without a chemistry runtime.
    hydroxyl = next(m for m in molecules if m["name"] == "oh")
    atoms = {int(a["index"]): a for a in hydroxyl["atoms"]}
    carbon_ids = {i for i, atom in atoms.items() if atom["element"] == "C"}
    put("MS-0037", len(carbon_ids), source,
        "Opening example: count carbon atoms in the exact hydroxyl representative molecular graph, not total heavy atoms.")
    label_source = art + "main/activation_output.json"
    saved = [r["molecule"] for r in activation["C_saved_atom_values"]["molecules"]
             if int(r["molecule"]["coordinate"]) == 293
             and int(r["molecule"]["concept_id"]) == 26
             and r["molecule"]["smiles"] == hydroxyl["smiles"]]
    assert len(saved) == 1
    saved = saved[0]
    assert int(saved["n_atoms"]) == len(atoms)
    assert set(saved["graph_atom_ids"]) == set(atoms)
    target_ids = set(saved["target_graph_atom_ids"])
    assert target_ids == {i for i, positive in zip(saved["graph_atom_ids"], saved["target_labels"]) if positive}
    put("MS-0038", len(carbon_ids & target_ids), source + "; " + label_source,
        "Opening example: count carbon atoms in the exact saved hydroxyl target-atom mask; assert matching SMILES, coordinate 293, concept 26 and graph-atom identities.")

    source = art + "main/activation_output.json"
    put("MS-0476", common(float(re.search(r"marginal_(\d+)_percentile", r["source_row"]["uncertainty"])[1])
                          for r in activation["output_rows"]), source,
        "Figure 4a interval design: nominal whole-molecule percentile coverage encoded in the exact output-row uncertainty field.")
    objects = activation["historical_contrast_rows"]
    carbonyl = [r for r in objects if str(r.get("concept_id", r.get("exact_id", ""))) in ("24", "25")]
    # Use the explicit chemical-object records, shared with the displayed clamp panels.
    source = supp + "figure_s_e1.json"
    mechanism = load(source)
    carbonyl = [r for r in mechanism["objects"] if int(r["exact_id"]) in (24, 25)]
    put("MS-0479", len(carbonyl), source,
        "Figure 4b: the two carbonyl query definitions share selected physical coordinate 454.")
    put("MS-0480", len({r["D_selected_unit"] for r in carbonyl}), source,
        "Figure 4b: distinct selected physical coordinates among the two carbonyl definitions.")

    # Main Figure 5: canonical denominator, physical basis width and recipe counts.
    source = art + "use/5b_and_s_q1_census.tsv"
    census = rows(source)
    put("MS-0551", sum(r["canonical_usable"] == "True" for r in census), source,
        "Figure 5b: count canonically usable ChemBERTa neurons before classifying their rewrite outcomes.")
    put("SI-2118", len(census), source,
        "Figure S-Q1: count all physical native neurons, including those not canonically usable.")
    source = art + "use/5c_and_s_s1_basis_summary.tsv"
    basis_rows = rows(source)
    native = next(r for r in basis_rows if r["basis"] == "native" and r["subset"] == "full")
    put("MS-0554", native["physical_n"], source,
        "Figure 5c: native physical width used as the common directions-per-width denominator.")
    put("MS-0555", len([r for r in basis_rows if r["basis"].startswith("rot_") and r["subset"] == "full"]), source,
        "Figure 5c: count distinct complete Haar-rotation bases.")
    put("MS-0556", len([r for r in basis_rows if r["basis"].startswith("sae_rep") and r["subset"] == "full"]), source,
        "Figure 5c: count the three fitted dictionary recipes; do not double-count their top-464 subsets.")
    source = "experiments/05_molecular_rewrites/PROTOCOL.md"
    protocol = (root / source).read_text(encoding="utf-8")
    assert "five distinct random SMILES" in protocol
    put("MS-0553", 5, source,
        "Design: five distinct random SMILES per molecule, excluding the canonical representation; literal protocol rule.")

    # Main Figure 6a: actual display-site geometries establish the two alanines.
    source = "data/artwork/protein_exposure_geometry.json"
    geometry = load(source)
    sites = geometry["sites"]
    assert all(s["residue"].startswith("ALA") and s["class"] == "H" for s in sites.values())
    put("MS-0630", len(sites), source,
        "Figure 6a: count the two selected helical alanine display sites in 2XMX chain A.")
    put("MS-0632", sum(k == "buried" for k in sites), source,
        "Figure 6a: number of the selected pair's sites labelled buried; RSA remains in the source record.")
    put("MS-0633", sum(k == "exposed" for k in sites), source,
        "Figure 6a: number of the selected pair's sites labelled exposed; RSA remains in the source record.")
    source = supp + "figure_s_c2.json"
    first_example = load(source)["examples"][0]
    precision_key = next(k for k in first_example if k.startswith("precision_top"))
    put("SI-1977", int(precision_key.removeprefix("precision_top")), source,
        "Figure S-C2: precision@k denominator encoded by the original precision_top100 field, not the number of displayed molecules.")
    source = "figures/native/supplement_examples.py"
    threshold = [n.comparators[0].left.value for n in ast.walk(function(source, "draw_cell_grid"))
                 if isinstance(n, ast.Compare) and len(n.comparators) == 1
                 and isinstance(n.comparators[0], ast.BinOp) and isinstance(n.comparators[0].op, ast.Mult)
                 and isinstance(n.comparators[0].left, ast.Constant)
                 and isinstance(n.comparators[0].right, ast.Name) and n.comparators[0].right.id == "wmax"]
    put("SI-1979 SI-1984 SI-1988", 100 * common(threshold), source,
        "Figure S-C2–C4 depiction design: actual non-target highlight cutoff in draw_cell_grid, multiplied by 100 for percent of each molecule's peak.")
    source = "figures/native/supplement_examples.py"
    node = function(source, "fig_s3")
    alpha = [c.value for n in ast.walk(node) if isinstance(n, ast.Compare)
             and isinstance(n.left, ast.Name) and n.left.id == "p"
             for c in n.comparators if isinstance(c, ast.Constant)]
    put("SI-1996", common(alpha), source,
        "Figure S-C5 design: actual nominal p threshold used to fill a positive-effect diamond; this is not a family-adjusted decision.")
    source = "figures/native/supplement_audits.py"
    references = [n.args[0].value for n in ast.walk(function(source, "fold_axis"))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "axvline" and n.args and isinstance(n.args[0], ast.Constant)]
    put("SI-1966", common(references), source,
        "Figure S-B1 geometry: fold-axis neutral reference line, a ratio of one.")
    source = "figures/native/supplement_examples.py"
    references = [n.args[0].value for n in ast.walk(function(source, "fig_s3"))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "axvline" and n.args and isinstance(n.args[0], ast.Constant)]
    put("SI-1997", common(references), source,
        "Figure S-C5 geometry: neutral fold-one line on each logarithmic axis.")

    # S-G1 and S-G3 use the masked 1246 assay, not the exposure assay.
    source = art + "main/concept_protein.json"
    protein = load(source)
    put("SI-2019 SI-2033", protein["coordinate"]["unit"], source,
        "Figure S-G1c/S-G3: actual D-selected masked helix–strand coordinate identifier 1246.")
    gains = [r for r in protein["endpoints"] if r["endpoint"] in ("gX", "gS", "gL")]
    put("SI-2034", len(gains), source,
        "Figure S-G3 top rows: one paired-Brier gain for composition, sequence-window and interaction references.")
    put("SI-2035", sum(r["Holm_reject"] == "true" for r in gains), source,
        "Figure S-G3 badge: count of those three reference-specific gains passing the stored Holm criterion.")
    put("SI-2036", ci_key_level({k: v for k, v in gains[0].items() if k.startswith("CI95")}), source,
        "Figure S-G3 thick bars: nominal percent encoded by the stored CI95 lower/upper fields.")
    put("SI-2037", ci_key_level({k: v for k, v in gains[0].items() if k.startswith("CI98_75")}), source,
        "Figure S-G3 thin bars: nominal percent encoded by stored CI98_75 lower/upper fields.")
    assert all(r["Holm_reject"] == "true" for r in protein["endpoints"])
    put("SI-2039", len(protein["endpoints"]), source,
        "Figure S-G3 family: count the three reference gains plus the prespecified composition-minus-window contrast; all four Holm flags checked.")

    # S-G2 exposes two endpoints and two distinct model widths.
    source = supp + "exposure_inference.tsv"
    inference = rows(source)
    put("SI-2025", len({r["endpoint"] for r in inference}), source,
        "Figure S-G2: distinct endpoint names mae_gain and brier_gain, not the number of model widths.")
    put("SI-2028", common(int(r["K"]) for r in inference), source,
        "Figure S-G2: untrained networks per model and endpoint from the saved inference denominator K.")
    source = supp + "exposure_rotations.tsv"
    rotations = rows(source)
    put("SI-2029", common(sum(r["model"] == m and r["basis"] != "native" for r in rotations)
                          for m in {r["model"] for r in rotations}), source,
        "Figure S-G2: count non-native basis rows per model; these nine rotations are dependent references.")
    put("SI-2030", next(r["axis_zero_based"] for r in rotations if r["model"] == "35m" and r["basis"] == "native"), source,
        "Figure S-G2: D-selected native ESM-2 35M exposure coordinate identifier; not the masked coordinate.")

    # S-G4a/b: widths, splits and trial denominators from the exact simulation.
    source = supp + "protein_null_summary.json"
    null = load(source)
    put("SI-2044", null["data"]["width"], source,
        "Figure S-G4a design: physical activation axes searched per random-label trial, excluding the separate sign count.")
    put("SI-2045", null["data"]["development_groups"], source,
        "Figure S-G4a: actual development homology-group count.")
    put("SI-2046", null["calibration"]["n"], source,
        "Figure S-G4a: independent null full-search maxima used for maximum-search calibration.")
    put("SI-2047", null["data"]["confirmation_groups"], source,
        "Figure S-G4a: other homology groups used to test the development-locked axis.")
    put("SI-2049", next(r["n_rounds"] for r in null["conditions"] if r["condition"] == "null_pair_flips"), source,
        "Figure S-G4a: number of random-label trials, distinct from independent calibration searches.")
    put("SI-2050", common(r["n_rounds"] for r in null["conditions"] if r["condition"] in ("single_4", "distributed_4")), source,
        "Figure S-G4b: number of trials in each displayed known-generator signal condition.")
    key = next(k for k in null["summary"][0] if k == "retention_mc95_low")
    put("SI-2048", float(re.search(r"mc(\d+)_", key)[1]), source,
        "Figure S-G4a design: nominal Wilson Monte Carlo interval percent encoded in retention_mc95 fields.")
    source = "results/protein/external_validation/validation/blind_benchmark/fixed_pair_background.npz"
    with np.load(root / source, allow_pickle=False) as background:
        put("SI-2043", background["delta"].shape[0] / len(set(background["groups"].tolist())), source,
            "Figure S-G4a design: one fixed natural residue-pair activation difference row per distinct homology group.")
    source = "experiments/13_external_validation/PROTOCOL.md"
    protocol = (root / source).read_text(encoding="utf-8")
    assert "single- and eight-axis generators" in protocol
    put("SI-2051", 1, source, "Figure S-G4b design: the frozen protocol's single-axis generating condition.")
    put("SI-2052", 8, source, "Figure S-G4b design: the frozen protocol's eight-axis distributed generating condition.")
    source = "figures/native/supplement_protein_validation.py"
    headings = [n.args[2] for n in ast.walk(tree(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "zip" and len(n.args) == 3
                and isinstance(n.args[0], ast.Name) and n.args[0].id == "axes"
                and isinstance(n.args[2], ast.List)]
    assert len(headings) == 1
    put("SI-2042", len(ast.literal_eval(headings[0])), source,
        "Figure S-G4 layout: count the four named interpretation checks a–d, excluding the separate decision ledger e.")

    # S-G4c/d/e: known-basis support and held-out groups, paired structures and routing.
    source = "results/protein/external_validation/validation/known_basis/SUMMARY.json"
    known = load(source)
    identity = next(a for a in known["arms"] if a["arm"] == "identity")
    put("SI-2055", identity["selected_single_axis"], source,
        "Figure S-G4c: original ESM-2 35M coordinate in the identity basis; transformed selected axes are not substituted.")
    arms = [a for a in known["arms"] if a["arm"] in ("signed_swap", "rotation45", "hadamard4", "hadamard16")]
    put("SI-2056", min(a["source_support_size"] for a in arms), source,
        "Figure S-G4c: smallest planted support size among the four plotted function-preserving transforms.")
    put("SI-2057", max(a["source_support_size"] for a in arms), source,
        "Figure S-G4c: largest planted support size among the four plotted transforms.")
    put("SI-2058", known["counts"]["T"]["groups"], source,
        "Figure S-G4c: held-out T homology groups used to evaluate activation recovery.")
    source = supp + "protein_repeated_structures.tsv"
    repeat = [r for r in rows(source) if r["quality_atom_match"] == "True"]
    put("SI-2059", common(len({r["original_pdb"].lower(), r["second_pdb"].lower()}) for r in repeat), source,
        "Figure S-G4d: distinct original and second PDB entries for each atom-matched repeated-structure residue pair.")
    source = supp + "protein_decision_policies.tsv"
    policy = next(r for r in rows(source) if r["frame"] == "814 original")
    put("SI-2060", int(policy["frame"].split()[0]), source,
        "Figure S-G4e: original exposure-neuron identifier in the selected decision-policy row.")
    put("SI-2062", policy["selected"], source,
        "Figure S-G4e: actual fixed routing budget, not all original exposure pairs.")
    source = "results/protein/external_validation/validation/decision_use/group_decisions.tsv"
    decision = [r for r in rows(source) if r["frame"] == "814 original"]
    target = {int(r["later_exposed"]) for r in decision}
    assert target == {0, 1}
    midpoint = (min(target) + max(target)) / 2
    selected = [r for r in decision if r["selected"] == "1"]
    other = [r for r in decision if r["selected"] == "0"]
    assert len(selected) == int(policy["selected"])
    assert max(abs(float(r["sequence"]) - midpoint) for r in selected) <= min(abs(float(r["sequence"]) - midpoint) for r in other)
    put("SI-2063", midpoint, source,
        "Figure S-G4e design: binary probability midpoint; verify that all selected routing pairs are closest to this midpoint.")

    # The omitted range is stored as display-axis edges; convert to original units.
    source = "figures/native/supplement_paired_dose.py"
    scale_source = supp + "figure_s_j1.json"
    scale = load(scale_source)["display_specification"]["unit_multiplier_for_A_and_B"]
    put("SI-2078", axis_limits(source, "low")[1] / scale, source + "; " + scale_source,
        "Figure S-J1b geometry: right edge of the lower full-clamp axis divided by its display multiplier, yielding original output units.")
    put("SI-2079", axis_limits(source, "high")[0] / scale, source + "; " + scale_source,
        "Figure S-J1b geometry: left edge of the upper full-clamp axis divided by its display multiplier; this is the omitted range's upper endpoint.")

    source = supp + "matched_control_panels.json"
    matched = load(source)
    put("SI-2098", ci_key_level(matched["reconstruction"][0]), source,
        "Figure S-M1 design: nominal percent encoded in the saved reconstruction CI95 fields, descriptive rather than family-adjusted.")
    put("SI-2102", ci_key_level(matched["joint_controls"][0]), source,
        "Figure S-N1 design: nominal percent encoded in joint-control CI95 fields; unadjusted scaffold-bootstrap intervals.")
    source = supp + "p_primary_four_tests.csv"
    dictionary = rows(source)
    key = next(k for k in dictionary[0] if k == "CI95_concept_bootstrap")
    put("SI-2110", float(re.search(r"CI(\d+)_", key)[1]), source,
        "Figure S-P1 design: nominal confidence percent encoded in the frozen concept-bootstrap interval field.")
    put("SI-2113", common(int(r["n_initializations"]) for r in dictionary), source,
        "Figure S-P1: shared initialization-network count in all four primary dictionary tests.")

    source = art + "use/s_g5a_paired_activations_814.tsv"
    exposure_unit = common(int(re.search(r"activation_(\d+)$", key)[1])
                           for key in rows(source)[0] if re.search(r"activation_(\d+)$", key))
    put("SI-2115", exposure_unit, source,
        "Figure S-G5a: exposure coordinate identifier encoded in both actual buried/exposed activation columns; independent of the masked assay.")
    source = art + "use/5a_and_s_r1_changed_atom_deltas.tsv"
    edit = rows(source)
    focal_units = {int(r["unit"]) for r in edit}
    assert focal_units == {293, 454}
    unit_source = supp + "figure_s_e1.json"
    hydroxyl_unit = int(next(r["D_selected_unit"] for r in mechanism["objects"] if int(r["exact_id"]) == 26))
    carbonyl_unit = int(next(r["D_selected_unit"] for r in mechanism["objects"] if int(r["exact_id"]) == 24))
    assert hydroxyl_unit in focal_units and carbonyl_unit in focal_units
    put("SI-2119 SI-2122", hydroxyl_unit, source + "; " + unit_source,
        "Figure S-Q1/S-R1: the designated hydroxyl coordinate present in the frozen changed-atom rows.")
    put("SI-2120 SI-2123", carbonyl_unit, source + "; " + unit_source,
        "Figure S-Q1/S-R1: the designated carbonyl coordinate present in the frozen changed-atom rows.")
    put("SI-2124", common(len({r["edit_class"] for r in edit if int(r["unit"]) == unit}) for unit in focal_units), source,
        "Figure S-R1: count displayed edit classes per focal neuron, including the remote aryl-fluorine control.")
    return result
