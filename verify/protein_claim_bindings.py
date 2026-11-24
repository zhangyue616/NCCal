"""Explicit paragraph-level protein quantities, independently of display matching."""
from pathlib import Path
import csv,json
import numpy as np

def get_bindings(repo_root=None):
    root=Path(repo_root)if repo_root is not None else Path(__file__).resolve().parents[1]
    base="results/protein/";out={}
    def js(rel):return json.loads((root/rel).read_text("utf8"))
    def rows(rel):
        with(root/rel).open(encoding="utf-8-sig",newline="")as f:return list(csv.DictReader(f,delimiter="\t"if rel.endswith(".tsv")else","))
    def put(number,value,source,operation):
        key=number if isinstance(number,str)else"SI-"+str(number)
        out[key]={"value":float(value),"source":source,"operation":operation}
    def field(number,rel,value,operation):put(number,value,rel,operation)
    def many(mapping,source,operation):
        for number,value in mapping.items():put(number,value,source,operation)

    # SI22 and SI26: the original linear-reference analyses retain their own estimands.
    rel=base+"protein_scale/scale/scale_results/INFERENCE.tsv";scale=rows(rel)
    r=next(x for x in scale if x["model"]=="35m"and x["endpoint"]=="mae_gain")
    r650=next(x for x in scale if x["model"]=="650m"and x["block"]=="16"and x["endpoint"]=="brier_gain")
    many({1367:int(r["K"]),1368:int(r["K"]),1370:int(r650["block"]),1371:len({x["endpoint"]for x in scale if x["model"]=="650m"and x["block"]=="16"})},rel,"Original exposure network-reference count, prespecified 650M block and its two endpoints")
    r8=next(x for x in scale if x["model"]=="8m"and x["endpoint"]=="mae_gain")
    if float(r8["CI95_low"])<0<float(r8["CI95_high"]):put(1363,0,rel,"The prespecified 8M continuous-MAE gain interval contains the null gain zero")
    rel=base+"protein_pair_replay/calibration/NETWORK_REFERENCE.csv";r=next(x for x in rows(rel)if x["network"]=="trained_gpu")
    many({1413:float(r["single_center_native"]),1414:float(r["paired_gS_native"])},rel,"The trained row's single-centre and paired sequence-window gains are different frozen estimands")
    rel=base+"protein_pair_replay/writeback/SUMMARY.json";wb=js(rel)
    many({1412:wb["unit"],1415:100*wb["reconstruction"]["T_R2_T_mean_reference"]},rel,"Fixed unit identity and local-sequence reconstruction R2 relative to the T mean, displayed as percent")
    rel=base+"external_validation/alanine_transfer/SUMMARY.json";transfer=js(rel);tr={x["identity"]:x for x in transfer["results"]if x["frame"]=="direct_sequence_screen"}
    many({1422:tr["A"]["n_pairs"],1423:tr["L"]["n_pairs"],1424:tr["V"]["n_pairs"],1429:100*(1-.05/len([tr["L"],tr["V"]])),1437:len([tr["L"],tr["V"]])},rel,"Direct-sequence-screened transfer counts by residue identity; L/V are the two prespecified primary comparisons")
    rel=base+"native_protein_candidates/cross_architecture/results_sequence_screened/SUMMARY.json";screened=js(rel);full=js(base+"native_protein_candidates/cross_architecture/results/SUMMARY.json");added=[x for x in screened if x!="source_814_exposure"]
    many({1441:len(added),1442:100*(1-.05/len(added)),1443:screened[added[0]]["rotation"]["n"],1444:95,1447:full["source_814_exposure"]["n"],1451:screened["source_814_exposure"]["n"]},rel,"The three added cross-model comparisons and their joint criterion; source-814 populations remain original versus sequence-screened")

    # SI29 and SI30: same structural dataset, separately defined held-out and CV analyses.
    rel=base+"external_validation/disulfide/ANALYSIS.json";a=js(rel);c=a["counts"];disc=a["discrimination"]["fixed361"]
    many({1455:c["chains"],1456:c["homology_groups"],1457:disc["all_observed"]["n"],1458:disc["Cys_only"]["positive"],1459:disc["Cys_only"]["negative"],1460:c["test_pairs"],1461:c["test_pair_groups"]},rel,"counts or fixed361 test-population discrimination denominators")
    put(1463,361,rel,"Identity encoded by the fixed361 readout branch; externally fixed native coordinate")
    put(1464,4181,rel,"Identity encoded by the sae4181 readout branch; externally fixed sparse feature")
    put(1469,disc["Cys_only"]["positive"]/disc["Cys_only"]["n"],rel,"AP of the residue-letter C score: all positives are cysteines, so AP equals positive cysteines / all cysteines")
    put(1470,.5,rel,"Within cysteines the residue-letter C score is constant; all positive-negative comparisons tie and receive one half")
    lockrel=base+"external_validation/disulfide/SELECTION_LOCK.json";lock=js(lockrel)["arms"]["native"]
    many({1471:lock["searched_poles"],1474:lock["unit"]},lockrel,"arms.native searched-pole denominator and development-locked unit")
    rankrel=base+"external_validation/disulfide_controls/RESULT.json";rank=js(rankrel)
    many({1475:rank["n_init"],1476:rank["n_bases"],1477:(1+rank["untrained_at_least_as_high"])/(1+rank["n_init"]),1478:rank["untrained_at_least_as_high"],1479:rank["n_init"],1480:rank["rotation_at_least_as_high"],1481:rank["n_bases"]},rankrel,"Development-selected native disulfide reading; inclusive empirical reference numerator and denominator")
    cvrel=base+"external_validation/disulfide_controls/FIXED_ANNOTATION_CV.json";cv=js(cvrel);sae=cv["out_of_fold"]["local_plus_SAE4181"]
    many({1483:len(cv["groups_per_fold"]),1484:cv["n_groups"],1485:sae["gain"],1486:100*(1-.05/2),1487:sae["simultaneous_97_5_gain"][0],1488:sae["simultaneous_97_5_gain"][1],1489:sae["Brier"],1490:(1-.5)**2,1491:.5},cvrel,"All-group five-fold fixed-annotation CV; SAE4181 effect and two-comparison Bonferroni interval; constant-probability reference is p=0.5")

    # SI32: whole-protein label; coordinate tuple is intentionally not treated as 4204.
    rel=base+"external_validation/protein_level_label/RESULT.json";a=js(rel);sheet=next(x for x in a["primary"]if x["measure"]=="sheet_fraction")
    many({1501:0,1503:a["n"],1504:7.4,1506:100*(1-.05/2),1511:sheet["simultaneous_97_5"][1],1512:a["high_high"]["n"],1513:100*a["high_high"]["prob_larger_than_other"],1514:a["high_high"]["other"],1515:len(a["fold_counts"])},rel,"Whole-protein result: zero-based author coordinate convention; charge74 denotes pH7.4; high_high pairwise comparison and held-out folds")

    # SI33: fixed generating conditions, not a numeric match to unrelated trials.
    rel=base+"external_validation/validation/blind_benchmark/DATA_SUMMARY.json";d=js(rel)
    many({1524:d["protein_homology_groups"],1525:d["development_groups"],1526:d["confirmation_groups"],1527:d["width"],1528:d["coordinate_sign_candidates"]/d["width"]},rel,"Fixed same-amino-acid benchmark populations, width and sign-search opportunity")
    trialsrel=base+"external_validation/validation/blind_benchmark/simulation_rounds.tsv";trials=rows(trialsrel)
    null=[x for x in trials if x["condition"]=="null_pair_flips"];single=[x for x in trials if x["condition"]=="single_4"];distributed=[x for x in trials if x["condition"]=="distributed_4"]
    many({1529:len(null),1531:sum(int(x["D_max_calibrated"])for x in null),1532:sum(int(x["D_then_E_locked"])for x in null),1533:sum(int(x["D_then_E_locked"])and int(x["recovered_generator_member_and_sign"])for x in single),1534:len(single),1537:len(distributed),1538:sum(int(x["D_then_E_locked"])for x in distributed),1539:len(distributed),1540:sum(int(x["recovered_generator_member_and_sign"])for x in distributed),1541:len(distributed)},trialsrel,"Filter the prespecified null, strong single-axis or strong eight-axis generator, then count the explicitly named selection/confirmation events")
    rel=base+"external_validation/validation/blind_benchmark/RESULT.json";a=js(rel);condition=next(x for x in a["conditions"]if x["condition"]=="distributed_4");real=a["real_structural_test"]
    many({1530:a["calibration"]["n"],1535:len(d["truth_original_zero_based_indices"]["distributed"]),1536:condition["oracle_confirmation_retained"],1542:real["D_win"]*real["D_groups"],1543:real["D_groups"],1544:real["E_win"]*real["E_groups"],1545:real["E_groups"]},rel,"Independent maximum-search calibration; known eight-axis oracle; real long-range label's fixed D/E success fractions")

    # SI34: reconstruction is separate from function-preserving numerical equivalence.
    rel=base+"external_validation/validation/known_basis/SUMMARY.json";a=js(rel);arm=next(x for x in a["arms"]if x["arm"]=="hadamard16")
    many({1551:next(x for x in a["arms"]if x["arm"]=="identity")["selected_single_axis"],1552:next(x for x in a["arms"]if x["arm"]=="identity")["source_support_size"],1553:arm["source_support_size"],1555:a["counts"]["T"]["groups"],1556:a["counts"]["T"]["rows"],1557:arm["source_support_size"],1562:round(arm["OMP_support_recall"]*arm["source_support_size"]),1563:arm["source_support_size"],1565:arm["tracked_R2"]},rel,"Identity and prespecified hadamard16 arms; T population; recovered generating support and exact tracked combination")
    rel=base+"external_validation/validation/known_basis/OUTPUT_SUMMARY.json";a=js(rel)
    many({1569:a["site_records"],1570:a["chains"]},rel,"Declared full-forward output sites and chains, distinct from algebraic linear-equivalence rows")
    rel=base+"external_validation/validation/known_basis/output_summary.tsv";eq=rows(rel)
    exact=[x for x in eq if x["comparison"]in["compensated_noop","tracked_nonzero_writeback","single_decoder_inverse_equivalence","OMP_decoder_inverse_equivalence"]]
    if exact:put(1566,max(float(x["max_abs_logit_error"])for x in exact)*1e5,rel,"Maximum full-forward logit error over compensated operations, displayed in units of 10^-5")

    # SI35: repeat annotations with frozen predictions.
    rel=base+"external_validation/validation/repeat_structure/summary.json";a=js(rel);primary=a["primary_binary_second_structure"];secondary=a["secondary_continuous_order_second_structure"]
    many({1572:a["input_groups"],1573:a["native_exact_framed_groups"],1574:a["selected_second_entries"],1575:a["quality_atom_pairs"],1576:a["ordering_direction_preserved_groups"],1577:primary["groups"],1578:.10,1579:.40,1580:len(a["intermediate_cases"]),1581:primary["groups"],1582:primary["improvement"],1583:95,1584:primary["improvement_95ci"][0],1585:primary["improvement_95ci"][1],1586:secondary["groups"],1587:secondary["improvement"],1588:secondary["improvement_95ci"][0],1589:secondary["improvement_95ci"][1]},rel,"Fixed-prediction repeated-structure funnel and primary/secondary populations; binary thresholds RSA<=0.10 and RSA>=0.40")
    rel=base+"external_validation/validation/decision_use/policies.tsv";policies=rows(rel);orig=next(x for x in policies if x["frame"]=="814 original");external=next(x for x in policies if x["frame"]=="814 external strict")
    many({1591:3,1592:.5,1594:int(orig["n"]),1596:int(external["n"]),1597:2,1598:100*(1-.05/2),1599:95,1604:2},rel,"Prespecified floor(n/3) allocation by distance from probability 0.5; exactly two primary frames and their Bonferroni family")

    # SI37: exact task, scope, model and frame filters are explicit.
    rel=base+"native_protein_candidates/native_channels/analysis/inference_all.tsv";allrows=rows(rel)
    def e1(task,split,scope="all",metric="single_channel"):
        return next(x for x in allrows if x["model"]=="E1"and x["task"]==task and x["split"]==split and x["scope"]==scope and x["metric"]==metric)
    startT=e1("helix_start_vs_internal","T");startE=e1("helix_start_vs_internal","E");sameT=e1("helix_start_vs_internal","T","same_helix");sameE=e1("helix_start_vs_internal","E","same_helix");helixT=e1("helix_vs_strand","T");helixE=e1("helix_vs_strand","E")
    many({1623:int(helixT["groups"]),1624:int(helixE["groups"]),1625:int(startT["groups"]),1626:int(startE["groups"]),1630:100*float(startT["mean"]),1631:95,1634:int(startT["pairs"]),1635:100*float(startE["mean"]),1638:int(startE["pairs"]),1640:int(sameT["groups"]),1641:int(sameT["pairs"]),1643:int(sameE["groups"]),1644:int(sameE["pairs"]),1648:len({(x["model"],x["task"])for x in allrows if x["split"]=="T"and x["scope"]=="all"and x["metric"]=="single_channel"})},rel,"E1 helix-start or helix-strand analysis, fixed T/E frames, all versus same_helix scopes; percentage values are 100 times group means")
    rel="data/protein/native_chain_identifiers.jsonl";chains=[json.loads(s)for s in(root/rel).read_text("utf8").splitlines()if s.strip()]
    many({1620:len(chains),1621:len({x["sequence_sha256"]for x in chains}),1622:len({x["group"]for x in chains})},rel,"Distinct observed chains, natural-sequence identities and homology groups in the frozen native-channel panel")
    rel=base+"native_protein_candidates/native_channels/analysis/reference_comparison.tsv";ref=rows(rel)
    for number,kind in [(1646,"orthogonal basis"),(1647,"untrained initialization")]:
        r=next(x for x in ref if x["model"]=="E1"and x["task"]=="helix_start_vs_internal"and x["split"]=="T"and x["reference"]==kind)
        put(number,int(r["n"]),rel,"E1 helix-start T descriptive reference count: "+kind)
    rel=base+"native_protein_candidates/native_channels/terminal_specificity/split_summary.tsv";terminal=rows(rel)
    for number,task,split in[(1673,"N_vs_C","T"),(1674,"N_vs_C","E"),(1676,"C_vs_interior4","T")]:
        matched=[x for x in terminal if x["task"]==task and x["split"]==split]
        if matched:put(number,100*float(matched[0]["group_equal_accuracy"]),rel,"Post hoc terminal specificity, task="+task+", split="+split+", group-equal percentage")

    # SI38-40: independent fresh frame, localization and topology must not be merged.
    rel=base+"native_protein_candidates/structural_readout_validation/fresh/RESULTS.json";a=js(rel)
    put(1679,a["n_total_native_sequences"],rel,"Fresh frame's total natural-sequence identities")
    rel="data/protein/fresh_structure_identifiers.jsonl";identities=[json.loads(s)for s in(root/rel).read_text("utf8").splitlines()if s.strip()]
    put(1680,len({x["group"]for x in identities}),rel,"Distinct frozen fresh-frame homology groups, counted independently of chain count")
    rel="data/protein/fresh_prior_exclusion_identifiers.jsonl";prior=[json.loads(s)for s in(root/rel).read_text("utf8").splitlines()if s.strip()]
    put(1681,len({x["sequence_sha256"]for x in prior}),rel,"Distinct previously used natural-sequence identities screened before fresh recruitment")
    rel=base+"native_protein_candidates/structural_readout_validation/fixed_axes/F_summary.tsv";summary=rows(rel)
    r=next(x for x in summary if x["metric"]=="rate"and x["question"]=="interior_vs_start"and x["unit"]=="unit2192")
    t=next(x for x in summary if x["metric"]=="rate"and x["question"]=="helix_vs_strand"and x["unit"]=="unit1900")
    many({1687:int(r["groups"]),1691:int(r["pairs"]),1698:int(t["pairs"]),1703:int(r["groups"])},rel,"F frame restricted to the same 13 groups supporting both fixed E1 questions")
    rel=base+"published_label_audit/label_audit/results/per_unit_results.tsv";audit=rows(rel);r=next(x for x in audit if x["source"]!="Hallee"and x["model"]=="e1_150m"and x["unit0"]=="1900"and x["concept"]=="OWN_HELIX_VS_BETA"and x["frame"]=="NEW")
    put(1704,int(r["groups"]),rel,"Own E1-1900 NEW helix-versus-strand paired groups, after previously-used exclusions")
    rel=base+"native_protein_candidates/structural_readout_validation/localization/fresh_summary.tsv";local=rows(rel);r=next(x for x in local if x["measure"]=="sequence_correct")
    many({1708:int(r["windows"]),1709:int(r["groups"]),1710:100*float(r["mean"])},rel,"F five-position localization, sequence-only exact-start correctness and group/window counts")
    rel=base+"native_protein_candidates/structural_readout_validation/published_transfer/F_tie_coverage.tsv";tie=rows(rel);r=next(x for x in tie if x["unit"]=="Hallee_ESM2_786"and x["scope"]=="this readout distinguishes")
    many({1716:int(r["total_pairs"]),1720:int(r["covered_pairs"]),1721:int(r["wins"])},rel,"Published ESM-2 unit786 on every F helix-strand pair; covered means non-tied, wins counted among covered pairs")
    rel=base+"native_protein_candidates/structural_readout_validation/scan/E1/trained/selection_lock.json";r=js(rel)["selection"];put(1726,r["channel_0"],rel,"Development-locked E1 sheet-edge channel, zero-based coordinate")
    rel=base+"native_protein_candidates/structural_readout_validation/scan/ESMC/trained/selection_lock.json";r=js(rel)["selection"];put(1728,r["channel_0"],rel,"Development-locked ESM C sheet-edge channel, zero-based coordinate")
    rel=base+"native_protein_candidates/structural_readout_validation/strand_reference/fresh_summary.tsv";strand=rows(rel)
    for number,model in[(1730,"E1"),(1731,"ESMC")]:
        r=next(x for x in strand if x["split"]=="F"and x["model"]==model and x["metric"]=="brier_gain")
        put(number,float(r["mean"]),rel,"F whole-strand-reference Brier gain for the development-selected "+model+" channel; preserve the negative sign")
    rel=base+"native_protein_candidates/structural_readout_validation/published_transfer/ESMC634_topology_summary.tsv";r=next(x for x in rows(rel)if x["split"]=="F")
    many({1735:100*float(r["mean"]),1736:int(r["groups"]),1737:int(r["pairs"])},rel,"Fixed published ESM C-300M strand pole634, D-oriented, on F edge-versus-central pairs")
    rel=base+"native_protein_candidates/structural_readout_validation/published_transfer/ESMC634_increment_summary.tsv";records=rows(rel)
    for number,split in[(1738,"T"),(1739,"E"),(1740,"H"),(1741,"F")]:
        r=next(x for x in records if x["reference"]=="whole_strand"and x["split"]==split)
        put(number,float(r["brier_gain"]),rel,"Exploratory D-only fit, whole-strand sequence features, split="+split+"; signed Brier gain")

    # Remaining repeated locations and explicitly encoded protocol definitions.
    # These definitions are tied to source tokens; they are not inferred by
    # finding an unrelated stored quantity with the same displayed number.
    def definition(numbers,value,source,needle,operation):
        if needle not in(root/source).read_text("utf8"):
            raise ValueError("Protocol/source token changed: "+source+" :: "+needle)
        for number in numbers:put(number,value,source,operation)
    rel=base+"protein_scale/scale/scale_results/INFERENCE.tsv"
    put(1372,next(int(x["block"])for x in rows(rel)if x["model"]=="650m"and x["block"]=="17"),rel,"Archived block17 sensitivity analysis, separate from prespecified block16")
    rel=base+"protein_reconstruction/additional_neurons/SUMMARY.json";additional=js(rel)
    primary_jsd=[v["writeback"]["contrasts"][k]["ci_simultaneous_99_1667"]for v in additional.values()for k in["seq_minus_full_JSD","swapped_minus_full_JSD"]]
    family=len(additional)+len(primary_jsd)
    many({1405:len(primary_jsd)//len(additional),1406:family,1407:100*(1-.05/family),1408:sum(lo<0<hi for lo,hi in primary_jsd)},rel,"Two fixed additional neurons: one unmasked full-minus-sequence reconstruction contrast and two JSD contrasts each; count simultaneous intervals containing zero")
    if all(lo<0<hi for lo,hi in primary_jsd):put(1409,0,rel,"Every primary writeback interval contains the null contrast zero")
    rel="experiments/10_protein_reconstruction/PROTOCOL.md"
    definition([1402],1,rel,"immediate C-terminal neighbour","The target is the immediate next residue: source index plus one")
    definition([1403],2,rel,"swaps the pair's prediction-minus-observation displacements","Exactly two source displacements are exchanged within a pair")
    definition([1416],20,rel,"twenty-amino-acid distribution","Writeback JSD uses the normalized distribution over the twenty canonical amino acids")
    rel=base+"external_validation/alanine_transfer/SUMMARY.json";a=js(rel)
    # The identity of the transferred unit is part of the archived protocol.
    rel=base+"external_validation/alanine_transfer/MODEL_IDENTITY.json"
    put(1421,js(rel)["unit"],rel,"The externally transferred helix-strand readout retains the model-identity record's native unit")
    rel=base+"native_protein_candidates/cross_architecture/results/SUMMARY.json"
    put(1445,814,rel,"source_814_exposure is the source readout identity, distinct from the three newly added cross-model comparisons")
    definition([1448],95,"experiments/12_native_protein_candidates/reference_implementation/native_channels/analysis/inference.py","np.quantile(means,.025)","Central 95-percent group-bootstrap interval uses quantiles0.025 and0.975")
    rel="experiments/13_external_validation/reference_implementation/disulfide/run_assay.py"
    definition([1482],5,rel,"range(-5,6)","Local cysteine predictor uses flanks at offsets minus five through plus five, excluding centre")
    rel="experiments/12_native_protein_candidates/reference_implementation/native_channels/cohort/build_structure_panel.py"
    definition([1616],1,rel,"p-s<2","First biological helix position is offset zero plus one")
    definition([1617],2,rel,"p-s<2","Helix-start role consists of offsets zero and one, biological positions1 through2")
    definition([1618],3,rel,"p-s>=3 and e-p>=3","Interior requires three residues from both annotated helix ends")
    definition([1619],10,rel,"complete and length>=10","Complete annotated helices must have length at least ten")
    rel=base+"native_protein_candidates/native_channels/e1/trained/selection_lock.json";selection=js(rel)["selections"]
    start=next(x for x in selection if x["task"]=="helix_start_vs_internal");helix=next(x for x in selection if x["task"]=="helix_vs_strand")
    many({1615:sum(x["task"]=="helix_start_vs_internal"for x in selection),1628:start["layer1"],1629:start["unit0"],1649:start["unit0"],1672:start["unit0"],1688:start["unit0"],1689:helix["unit0"],1695:helix["unit0"],1696:start["unit0"],1706:start["unit0"],1711:start["unit0"]},rel,"Development-locked E1 selection, keyed by structural question; layer1 is one-based and unit0 is zero-based")
    rel="experiments/12_native_protein_candidates/reference_implementation/native_channels/analysis/inference.py"
    definition([1650],4,rel,"[-4,-3,-2,-1,1,2,3,4]","Sequence predictor uses four flanking residues in each direction, excluding the fixed identical centre")
    definition([1652,1713],95,rel,"np.quantile(means,.025)","Central percentile interval is100 times(0.975-0.025)")
    definition([1669],1,rel,"0.5 * (x == 0)","Tie-credit numerator is one")
    definition([1670,1722],.5,rel,"0.5 * (x == 0)","An exactly tied oriented difference receives one half credit")
    rel=base+"native_protein_candidates/native_channels/analysis/coordinate_confirmations.tsv";r=next(x for x in rows(rel)if x["kind"]=="published_fixed"and x["model"]=="ESM-2 8M"and x["task"]=="helix_vs_strand"and x["split"]=="T")
    many({1663:int(r["channel_0"]),1718:int(r["channel_0"])},rel,"Externally fixed ESM-2 8M published helix-strand native coordinate, independently of its accuracy")
    rel="data/protein/fresh_structure_query.json";date=js(rel)["query"]["nodes"][0]["parameters"]["value"];startdate=date["from"].split("-");enddate=date["to"].split("-")
    many({1682:int(startdate[2]),1683:int(startdate[0]),1684:int(enddate[2]),1685:int(enddate[0])},rel,"Fixed RCSB first-release-date query bounds; do not equate date eligibility with known pretraining exclusion")
    rel=base+"native_protein_candidates/structural_readout_validation/fixed_axes/F_summary.tsv";adv=[x for x in rows(rel)if x["metric"]=="preassigned_advantage"]
    if len({float(x["holm"])for x in adv})!=1:raise ValueError("The two archived Holm probabilities are no longer identical")
    put(1701,float(adv[0]["holm"]),rel,"Both fixed question-specific group-paired sign-flip tests, adjusted jointly by Holm")
    definition([1692],95,"experiments/12_native_protein_candidates/reference_implementation/native_channels/analysis/inference.py","np.quantile(means,.025)","Central 95-percent group-bootstrap interval")
    rel="experiments/12_native_protein_candidates/reference_implementation/structural_readout_validation/strand_reference/run_reference.py"
    definition([1729],6,rel,"['log1p_length','charged_fraction','polar_fraction','hydropathy','hydropathy_alternation','polar_alternation']","Six specified whole-strand sequence features, prior to adding the neuron")
    rel=base+"native_protein_candidates/structural_readout_validation/published_transfer/esmc634_author_selection.json";a=js(rel)
    put(1733,int(a["winner"]["best_feature"]),rel,"Published ESM C-300M positive strand pole identity selected from the pinned author table")
    rel=base+"published_label_audit/label_audit/results/per_unit_results.tsv";audit=rows(rel)
    exposure=next(x for x in audit if x["source"]!="Hallee"and x["model"]=="esm2_35m"and x["unit0"]=="814"and x["concept"]=="OWN_EXPOSURE"and x["frame"]=="NEW")
    put("MS-0664",int(exposure["groups"]),rel,"Own native814 exposure, NEW paired groups, beyond the stronger sequence predictor")
    definition(["MS-0600"],95,"experiments/14_published_label_audit/PROTOCOL.md","Two thousand shared group bootstrap samples","Central95-percent group-bootstrap interval for the own E1 helix-strand increment")
    definition(["MS-0584"],2,"experiments/14_published_label_audit/PROTOCOL.md","jointly to same-amino-acid and strong-increment tests","The two formal tests summarized are paired discrimination and strong sequence increment")
    put("MS-0305",361,base+"external_validation/disulfide/ANALYSIS.json","Fixed361 branch denotes the externally published pre-GELU disulfide coordinate")
    rel="figures/plot_use_results.py"
    definition(["MS-0573"],.7,rel,"a.axvline(.7,","Descriptive vertical AUROC reference line at0.70; not the BH significance threshold")
    definition([1388],10,"src/nccal/other_metrics.py","np.sort(delta)[-10:]","The concentration diagnostic sums the ten largest signed group-level JSD differences")
    rel=base+"external_validation/alanine_transfer/PAIR_SELECTION.json";pairs=js(rel)["pairs"]
    per_group={}
    for pair in pairs:
        key=(pair["group"],pair["identity"]);per_group[key]=per_group.get(key,0)+1
    if set(per_group.values())!={1}:raise ValueError("Transfer pairs are no longer one per group and identity")
    put(1425,max(per_group.values()),rel,"Count selected pairs independently for every homology-group and residue-identity combination")
    rel=base+"external_validation/alanine_transfer/SUMMARY.json";a=js(rel)
    intervals=[x["comparisons"]["L_minus_Lj"]["CI97_5_simultaneous_percentile"]for x in a["results"]if x["frame"]=="direct_sequence_screen"and x["identity"]in["L","V"]]
    if len(intervals)!=2 or not all(lo<0<hi for lo,hi in intervals):raise ValueError("L/V primary interval statement changed")
    put(1438,0,rel,"The two prespecified L/V interaction-reference gain intervals both contain the null gain zero")
    rel="experiments/13_external_validation/reference_implementation/disulfide/run_assay.py"
    definition([1472],2,rel,"layer[1].intermediate.dense","Disulfide pole search uses zero-based block1, the second block")
    definition([1473],361,rel,"w[[361,1280+361]]=-np.inf","Both signed poles of target361 are excluded from every search arm")
    definition([1492,1493],2,rel,"idx=np.array([p.positive_index0,p.negative_index0])","Each output-fidelity intervention targets exactly the two cysteines of its fixed positive/negative pair")
    rel=base+"external_validation/validation/blind_benchmark/DATA_SUMMARY.json";a=js(rel)
    put(1523,a["chosen_chains"]/a["protein_homology_groups"],rel,"One chosen natural structural pair per original homology group; chosen47 chains /47 groups")
    rel="experiments/12_native_protein_candidates/reference_implementation/native_channels/terminal_specificity/run_terminal.py"
    definition([1671],2,rel,"range(start, start + 2), range(end - 1, end + 1)","Fixed N-versus-C definition compares the first two and last two residues of the same annotated helix")
    definition([1675],4,rel,"range(start + 4, end - 3)","C-versus-interior requires interior sites at least four residues from each helix end")
    rel="experiments/12_native_protein_candidates/reference_implementation/structural_readout_validation/localization/localize.py"
    definition([1705],5,rel,"for a in range(5)","Exact-start localization chooses among five supplied biological positions")
    rel="experiments/12_native_protein_candidates/reference_implementation/structural_readout_validation/sheet/build_sheet.py"
    definition([1723],1,rel,"bins[(s['record_id'],s['sheet_id'],s['aa'])]","Edge/central pairs are restricted to one record, one deposited open sheet and one amino-acid identity")
    rel=base+"native_protein_candidates/native_channels/e1/trained/selection_lock.json";a=js(rel)
    put(1942,next(x["unit0"]for x in a["selections"]if x["task"]=="helix_start_vs_internal"),rel,"The sequence-order limitation concerns the frozen E1 helix-start unit2192")
    rel=base+"published_label_audit/label_audit/results/per_unit_results.tsv"
    put(1943,len({x["unit_id"]for x in rows(rel)if x["source"]!="Hallee"}),rel,"Number of distinct own-study readout definitions across both frames, retaining separate E1 question definitions")
    return out

bindings=get_bindings
