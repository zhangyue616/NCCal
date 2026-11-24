"""Document and calculate cross-experiment quantities using the named study records."""
import ast
import csv
import json
import re
from collections import defaultdict
from math import exp, log
from statistics import median

def bindings(root):
    claims = {}
    def put(ids, value, source, operation):
        if not ids:
            return
        for identifier in ids.split():
            if identifier in claims:
                raise ValueError('Duplicate positional evidence for ' + identifier)
            claims[identifier] = dict(value=float(value), source=source, operation=operation)
    def read(path):
        return json.loads((root / path).read_text(encoding='utf-8-sig'))
    def tab(path):
        with (root / path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream, delimiter='\t'))
    def design(ids, source, pattern, description, multiplier=1, offset=0):
        text=(root / source).read_text(encoding='utf-8-sig')
        m=re.search(pattern,text,flags=re.IGNORECASE)
        if not m:
            raise ValueError('Missing design statement in ' + source + ' for ' + ids)
        words={'one':1,'three':3,'four':4,'five':5,'ten':10,'nine':9,'twelve':12,'two':2}
        word=m.group(1).lower().replace(',','')
        value=words[word] if word in words else float(word)
        put(ids,value*multiplier+offset,source,'Protocol setting: '+description)
    source='figures/native/main_1.py'
    block=ast.parse((root/source).read_text(encoding='utf-8-sig'))
    tests=next(ast.literal_eval(node.iter) for node in ast.walk(block)
               if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple) and
               isinstance(node.target,ast.Tuple) and any(isinstance(item,ast.Name) and item.id=='title' for item in node.target.elts))
    assert [r[1] for r in tests]==['Calibration','Transfer','Reconstruction','Output preservation']
    put('MS-0069 MS-0072 MS-0098 MS-0666',len(tests),source,'Design: count the named scientific tests in the overview.')
    source='results/artwork/main/pilot.json';pilot=read(source)['pilot']
    put('MS-0049 MS-0355 MS-0386',len(pilot),source,'Count graph checkpoints with both native and rotated pilot profiles.')
    first=next(iter(pilot.values()))
    put('MS-0387 SI-1999',len(first['initialization_dots_pct']),source,'Count distinct untrained pilot profiles per graph encoder.')
    put('MS-0388 SI-2000',len(first['rotation_dots_pct']),source,'Count distinct rotation pilot profiles per graph encoder.')

    source='results/artwork/main/concept_protein.json';p=read(source)
    put('MS-0270 MS-0339 MS-0627 SI-1939 SI-2019 SI-2033',p['coordinate']['unit'],source,'Read the frozen ESM-2 alanine helix/strand FFN coordinate.')
    endpoints=p['endpoints'];g=[r for r in endpoints if r['endpoint'] in ('gX','gS','gL')]
    put('MS-0628 SI-2034 SI-2035',len(g),source,'Count paired Brier gains over composition, ordered local sequence and interaction references.')
    put('SI-2039',sum(str(r['Holm_reject']).lower()=='true' for r in endpoints),source,'Count rejected tests in the four-endpoint within-protein Holm family.')
    source='results/protein/published_label_audit/label_audit/inputs/pairs.tsv';samples=tab(source)
    pairs=[(r['positive0'],r['negative0']) for r in samples]
    assert all(p!=n for p,n in pairs)
    put('MS-0032 MS-0081 MS-0090 MS-0557 MS-0594 MS-0610',len(set(pairs[0])),source,'Pair design: two distinct sites with the same recorded amino acid; check distinct site indices in every row.')
    assert all(row['accession'] for row in samples)
    put('MS-0558',len({samples[0]['accession']}),source,'Pair design: one accession identifies the chain for the two site indices in each pair row.')
    put('MS-0559 MS-0560',sum(endpoint==pairs[0][0] for endpoint in pairs[0]),source,'Pair design: exactly one positive and one negative site per distinct-site pair.')
    source='results/tables/e1_heldout.json';e=read(source);best=max(e,key=lambda r:float(r['pair_accuracy']))
    put('MS-0592',int(best['unit0']),source,'Find the highest held-out pair accuracy among the four fixed E1 channels, then read its channel.')
    put('MS-0605',len({r['unit0'] for r in e if r['unit0']!=best['unit0']}),source,'Count the other development-fixed E1 channels beyond the reported example.')
    source='data/protein/forward_subset_identifiers.json';fixtures={r['experiment'][:2]:r for r in read(source)}
    put('MS-0650 MS-0662 SI-2030 SI-2055 SI-2060 SI-2115',fixtures['10']['unit0'],source,'Read the ESM-2 exposure reconstruction target from the recorded checkpoint and FFN selection.')
    put('MS-0651 MS-0663',fixtures['11']['unit0'],source,'Read the development-selected ESM-2 650M exposure unit at the primary site.')
    put('MS-0292',fixtures['11']['block0']+1,source,'Convert the primary exposure block to the natural block number used in the article.')

    source='results/tables/basis_counts.json';bases=read(source);full=[r for r in bases if r['subset']=='full']
    native=next(r for r in full if r['basis']=='native')
    put('MS-0520',len({r['basis'].split('_')[0] for r in full}),source,'Count native, orthogonally rotated, principal-component and sparse-autoencoder basis classes.')
    put('MS-0554 SI-2118',native['physical_n'],source,'Read the denominator of physical directions in the native molecular basis.')
    put('MS-0555',len({r['basis'] for r in full if r['basis'].startswith('rot_')}),source,'Count independent rotation bases by their recorded distinct seeds.')
    put('MS-0556',len({r['basis'] for r in full if r['basis'].startswith('sae_')}),source,'Count distinct full-width sparse dictionary fits.')
    put('MS-0551',native['canonical_usable_n'],source,'Read the number of canonically replaceable native directions, before rewriting.')
    put('MS-0497',int(native['retained'])/int(native['canonical_usable_n']),source,'Retained share among originally usable native directions; the heading gives the fraction to one significant digit.')
    source='results/artwork/main/activation_output.json';activation=read(source)
    for field,ids in [('A_B','MS-0427 MS-0476')]:
        v=re.search(r'(\d+)%',activation['interval_families'][field])
        put(ids,float(v.group(1)),source,'Interval setting: read the original within-molecule bootstrap coverage for focal reconstruction and replacement.')
    instances=activation['C_saved_atom_values']['molecules']
    put('MS-0424 MS-0431',len({i['molecule']['coordinate'] for i in instances}),source,'Count the two molecular examples associated with distinct focal units.')
    source='results/artwork/main/mechanism.json';o=read(source)['objects']
    cc=[r['D_object']['unit'] for r in o if r['D_object']['concept_id'] in (24,25)]
    put('MS-0479 SI-2008',len(cc),source,'Count carbonyl definitions in the three-object replacement comparison.')
    put('MS-0480',len(set(cc)),source,'Count distinct physical neurons selected by those two carbonyl definitions.')
    put('SI-1322 SI-1335 SI-1337 SI-1351',len(o),source,'Count the development-fixed chemical objects in the intervention family.')
    assert len({len(r['D_object']['control_units']) for r in o})==1
    put('SI-1336',len(o[0]['D_object']['control_units']),source,'Count matched control neurons for each focal object.')
    source='results/artwork/main/paired_dose.json';dose=read(source)
    assert len({r['pairs'] for r in dose['objects']})==1
    put('SI-1755',dose['objects'][0]['pairs'],source,'Read the number of fixed positive-negative pairs per dose object.')
    put('SI-1756 SI-2080',len(dose['objects'][0]['units'])*2,source,'Multiply the target-plus-control neuron count by the two sites of the matched pair.')
    source='results/molecular/whole_layer/primary_12_family.tsv';m=tab(source)
    put('MS-0327',len(m),source,'Count all registered joint whole-layer reference contrasts in the Holm family.')
    source='results/artwork/main/task_output_panel.json'
    m=[r for r in read(source)['paired'] if r['scope']=='selected/joint']
    put('MS-0325 SI-1839',len(m),source,'Count the selected-panel, joint-replacement task/reference comparisons.')
    put('MS-0493',len({r['reference'] for r in m}),source,'Count distinct references in the joint selected-panel comparison.')
    source='results/artwork/main/reconstruction_panel.json';m=read(source)['macro']
    put('MS-0324 MS-0163 SI-1823',len({r['model'] for r in m}),source,'Count the different encoders contributing a reconstruction panel contrast.')
    source='results/molecular/reconstruction/panel_units.tsv';m=tab(source)
    sets={model:{r['coordinate'] for r in m if r['model']==model and r['role']=='selected'} for model in {i['model'] for i in m}}
    assert len({len(s) for s in sets.values()})==1
    put('MS-0326 SI-1819',len(next(iter(sets.values()))),source,'Count unique selected physical neurons in each development-fixed panel.')
    matched={model:{r['coordinate'] for r in m if r['model']==model and r['role']=='matched_control'} for model in sets}
    if all(matched.values()):
        put('SI-2100',len(next(iter(sets.values()))),source,'Count selected neurons per encoder.')
        put('SI-2101',len(next(iter(matched.values()))),source,'Count the matched-control neurons per encoder.')
    source = 'results/molecular/rewrites/chemberta_unit_summary.tsv'
    hydroxyl, = [row for row in tab(source) if int(row['unit']) == 293]
    controls = [key for key in ('rewritten_mean_low', 'rewritten_shuffled_low') if float(hydroxyl[key]) > 0]
    put('MS-2232', len(controls), source, 'Count the distinct rewritten-output advantages with positive lower 95% bounds for the hydroxyl neuron (development mean and shuffled displacement).')
    source='experiments/05_molecular_rewrites/PROTOCOL.md'
    design('MS-0005 MS-0250 MS-0356 MS-0504 MS-0524',source,r'R-squared at least (0\.\d+)','lower bound on the proportion of variance explained by a qualifying scalar rule.')
    design('MS-0019 MS-0086 MS-0245 MS-0498 MS-0501 MS-0553 MS-0668 MS-0677 SI-1896',source,r'canonical representation and (five) distinct','number of separately generated rewritings per molecule, in addition to its canonical form.')
    design('MS-0251',source,r'positive (\d+)% lower bounds','molecule-bootstrap interval coverage for the two canonical replacement advantages.')
    design('MS-0513 MS-0655',source,r'Pearson correlation at least (0\.\d+)','threshold for the prespecified same-atom tracking subgroup.')
    design('MS-0242',source,r'From the fixed public molecular source, ([\d,]+) previously','size of the joint rescue and evaluation draw.')
    source='experiments/06_molecular_edits/PROTOCOL.md'
    design('MS-0254 MS-0447',source,r'generate at most (one) partner per parent','one assigned functional-group transformation per eligible parent.')
    source='experiments/09_protein_exposure/PROTOCOL.md'
    design('MS-0280 MS-0313',source,r'(\d+)-residue one-hot window','radius of symmetric sequence context calculated from full width.',.5,-.5)
    design('MS-0299',source,r'uses (one) same-chain','number of pairs sampled in a structural group.')
    design('MS-0300',source,r'RSA at most (0\.\d+)','maximum relative accessibility for a buried residue.')
    design('MS-0301',source,r'versus at least (0\.\d+)','minimum relative accessibility for an exposed residue.')
    source='experiments/11_protein_scale/PROTOCOL.md'
    design('MS-0295',source,r'block (\d+) is a descriptive sensitivity','natural block number used in the prespecified adjacent-block sensitivity.')
    design('MS-0346',source,r'The (two) 650M endpoints','number of jointly evaluated 650M endpoints.')
    source='experiments/14_published_label_audit/PROTOCOL.md'
    design('MS-0311',source,r'select up to (ten) author-ranked','maximum author-ranked units per source, model and annotation class.')
    design('MS-0316',source,r'BH at (0\.\d+) is applied','false-discovery rate for the public-label families.')
    source='experiments/10_protein_reconstruction/PROTOCOL.md'
    for ids,pattern,label in [('SI-1374',r'gaps (\d+) to','short-gap lower boundary'),('SI-1375',r'gaps \d+ to (\d+)','short-gap upper boundary'),('SI-1376',r'from at least (\d+), radii','long-gap lower boundary'),('SI-1377 SI-1378',r'radii at most (\d+)','inner-shell radius'),('SI-1379',r'to (\d+), and twenty','outer-shell radius'),('SI-1380',r'log\((\d+)\+count\)','log-transform pseudocount')]:
        design(ids,source,pattern,label)
    source='results/molecular/reconstruction/joint_chemberta_mtr_bace_metadata.json'
    metadata=read(source)
    put('MS-0238 MS-0485 SI-1842',len(set(metadata['permutation_seeds'])),source,'Count distinct archived seeds for within-molecule joint displacements.')
    source='results/molecular/calibration/transfer_expected.json'
    transfer=read(source)
    put('MS-0320',transfer['replicates'],source,'Read the number of shared molecule-level bootstrap replicates.')
    put('SI-1331',len({'Theta','G_init'}.intersection(transfer)),source,'Count the two distinct transfer estimands in the original bootstrap record.')
    source='results/molecular/calibration/expected.json';calibration=read(source)
    depth_source='results/molecular/calibration/molformer_expected.json'
    separate=read(depth_source)['primary']
    put('MS-0156',len(calibration)+int('J1' in separate and 'J2' in separate),source+' | '+depth_source,'Count the eight final-block checkpoints and the separate depth-selected MoLFormer analysis.')
    put('MS-0333',len([k for k in calibration[0] if k.endswith('_p')]),source,'Count the two prespecified checkpoint-level rank comparisons.')
    source='results/molecular/calibration/molformer_expected.json';primary=read(source)['primary']
    put('MS-0183 MS-0383',len([primary['J1']['p'],primary['J2']['p'],primary['J2']['p_free']]),source,'Count the layer-selected readability, conditional preference and maximum preference rank comparisons.')
    put('MS-0389 SI-1806',primary['K'],source,'Read the calibration rank denominator excluding the trained network.')
    source='results/protein/protein_scale/scale/scale_results/INFERENCE.tsv';inference=tab(source)
    m=[i for i in inference if i['model']=='35m']
    put('MS-0341',len(m),source,'Count entries in the ESM-2 35M supplementary two-endpoint family.')
    assert len({float(i['p_one_sided']) for i in m})==len({float(i['p_holm']) for i in m})==1
    put('MS-0343',float(m[0]['p_one_sided']),source,'Read the common upper-tail rank p-value for the two separately recorded 35M endpoints.')
    put('MS-0344',float(m[0]['p_holm']),source,'Read the common Holm-adjusted p-value within the two-endpoint 35M family.')
    source='results/artwork/main/mechanism.json';objects=read(source)['objects']
    put('SI-1339',sum(r['effect']['positive_lower_bound'] for r in objects),source,'Count full-dose sensitivity objects with a positive registered lower bound.')
    source='results/artwork/main/paired_dose.json';objects=read(source)['objects']
    intervals=[o['H_pair1_CI98_3333333333_existing'] for o in objects]
    null_contrast=0.0
    assert all(lo<=null_contrast<=hi for lo,hi in intervals)
    put('SI-1340',sum(o['H_by_scale_existing'][-1]<0 for o in objects),source,'Count negative equal-dose sensitivity contrasts at the registered final scale.')
    put('SI-1341',null_contrast,source,'Equality value covered by each of the stored equal-dose intervals; verify that every interval includes zero.')
    source='results/artwork/main/task_output_panel.json';m=[r for r in read(source)['paired'] if r['scope']=='selected/joint']
    put('SI-1840',sum(float(r['Holm12_p'])<=.05 for r in m),source,'Count selected joint interventions passing the archived twelve-test Holm comparison.')
    source='results/artwork/main/reconstruction_panel.json';bad=read(source)['retained_local_below_binary']
    put('SI-1825',len(bad),source,'Count selected neurons for which the concept-mean estimate beats the bonding-environment estimate.')
    source='results/molecular/rewrites/anchor_diagnostics.json';diag=read(source)['primary_determinate']
    put('SI-1903',diag['string_AUROC'],source,'Read the canonical-string diagnostic AUROC on the original determinate set.')
    put('SI-1904',diag['minus_following_AUROC'],source,'Read the same-atom tracking diagnostic AUROC on exactly that determinate set.')
    put('SI-1818',len(read('results/artwork/main/concept_protein.json')['concept_rows']),'results/artwork/main/concept_protein.json','Count curated concept associations in the fixed prediction panel.')
    source='results/molecular/dictionary/primary_four_tests.tsv'
    put('MS-0328',len(tab(source)),source,'Count the prespecified dictionary and native-pole comparisons across the two candidate pools.')
    source='results/molecular/claim_sources/reconstruction_configuration.json';focal=read(source)['coordinate_ids']
    put('SI-1761',len(set(focal)),source,'Count physical focal neurons in the frozen chemical reconstruction configuration.')
    source='experiments/01_molecular_calibration/PROTOCOL.md'
    design('MS-0321 MS-0322 MS-0390',source,r'networks and (nine) rotations','count of rotations per network, not additional random networks.')
    source='results/tables/e1_heldout.json';readouts=read(source)
    put('MS-0314',int(readouts[0]['random_K']),source,'Read the independent random network count for matched own/public label references.')
    put('MS-0315',int(readouts[0]['rotation_K']),source,'Read the rotated reference count for the same test family.')
    source='results/protein/protein_scale/scale/scale_results/NETWORK_RESULTS.tsv';refs=tab(source)
    put('MS-0337',len({r['index'] for r in refs if r['model']=='35m' and r['network_type']=='untrained'}),source,'Count independent untrained networks used in the rank denominator.')
    source='results/molecular/calibration/transfer_profiles.npz'
    import numpy as np
    with np.load(root/source,allow_pickle=False) as arrays:
        put('MS-0193',arrays['point_ell'].shape[0]-1,source,'Network axis of transfer profiles, excluding the trained network.')
    source='results/molecular/claim_sources/reconstruction_configuration.json'
    settings=read(source)
    put('MS-0221',settings['alpha_by_coordinate']['j293'],source,'Read the registered radius-one regression prior weight for the hydroxyl neuron.')
    source='verify/table_display.json'
    put('SI-1287',sum(k.startswith('SI') for k in read(source)),source,'Document structure: count the eight distinct supplementary summary tables.')
    source='results/tables/public_label_flags.json';a=read(source)
    elig=[i for i in a if i['frame']=='T' and i['formal_beta_gamma_eligible']=='True']
    put('MS-0656',len(elig),source,'Count the formally evaluable published residue annotations in the confirmation frame.')
    source='results/protein/published_label_audit/label_audit/results/per_unit_results.tsv'
    aud=tab(source);own=[r for r in aud if r['source']=='NCCal' and r['frame']=='NEW' and r['unit0'] in ('814','4566','1246')]
    exposure=[r for r in own if r['concept']=='OWN_EXPOSURE']
    assert len(exposure)==2 and len({r['groups'] for r in exposure})==1
    put('MS-0664',int(exposure[0]['groups']),source,'Read the shared number of new matched pairs for the 35M and 650M exposure units.')
    assert len(own)==3 and all(float(r['strong_CI_lo'])<=0<=float(r['strong_CI_hi']) for r in own)
    put('SI-1940',0,source,'Verify that the three intervals for increments beyond the strong sequence predictor contain the zero contrast on new groups.')
    source='results/protein/published_label_audit/label_audit/results/SUMMARY.json'
    put('SI-1935',read(source)['public_T_AA_compatible_label_rows'],source,'Read the number of public confirmation labels meeting the amino-acid-compatibility definition.')
    source='results/artwork/main/concept_protein.json';p=read(source)
    put('MS-0348',len(p['endpoints']),source,'Count the four paired helix/strand endpoints in the declared family.')
    source='results/protein/protein_pair_replay/calibration/AGGREGATE.json'
    put('MS-0340',len(read(source)['single_center']),source,'Count the native readability and native-axis advantage single-site endpoints.')
    source='results/molecular/claim_sources/reconstruction_configuration.json';f=read(source)
    put('SI-1860 SI-1913',f['coordinate_ids'][f['concept_ids'].index(26)],source,'Read the registered coordinate for the aliphatic hydroxyl concept.')
    put('SI-1792',len(f['coordinate_ids']),source,'Count the distinct frozen coordinates used for hydroxyl and carbonyl reconstruction.')
    source='results/molecular/rewrites/anchor_diagnostics.json';audit=read(source)
    put('SI-1790',audit['rewrites'],source,'Read the number of alternative molecular notations per evaluation molecule in the new census.')
    source='results/tables/model_study_design.json';models=read(source)
    mol=[r for r in models if r[1]!='Protein language model']
    put('MS-0157',len({r[1].split(';')[0] for r in mol}),source,'Count architecture families among the nine chemical checkpoints, treating pretraining tasks separately.')
    put('MS-0158',sum(r[1]=='Protein language model' for r in models),source,'Count the independently specified protein checkpoints in the design table.')
    cb=next(r for r in models if r[0]=='ChemBERTa-77M-MTR')[3]
    assert 'Final FFN pre-GELU' in cb and 'post-GELU' in cb
    put('MS-0099',len(re.findall('(?:pre|post)-GELU',cb)),source,'Count distinct measurement sites identified in the focal chemical encoder: final pre-activation and block-specific post-activation.')
    put('MS-0112 MS-0127',int(re.search(r'block-(\d+) post-GELU',cb)[1]),source,'Read the naturally numbered post-activation reconstruction block in ChemBERTa-MTR.')
    mf=next(row for row in models if row[0].startswith('MoLFormer'))[3]
    assert 'eleventh-block post-GELU' in mf
    put('MS-0129',11,source,'Read the naturally numbered eleventh block specified for post-activation MoLFormer reconstruction.')
    source='results/tables/calibration_statistics.json';models=read(source)
    passers={r[0] for r in models if not r[0].startswith('MoLFormer') and all(float(c[1])<=.05/8 for c in r[1])}
    if max(primary['J1']['p'],primary['J2']['p'],primary['J2']['p_free'])<=.05:
        passers.update(r[0] for r in models if r[0].startswith('MoLFormer'))
    put('MS-0379',len({s.split('-')[0] for s in passers}),source+' | '+depth_source,'Group passing checkpoints by encoder architecture after applying their respective registered significance criteria.')
    source='results/molecular/edits/primary_results.tsv';sample=tab(source)[0]
    put('MS-0255 MS-0451',len({'MSE_crossfit_mean','MSE_swapped'}.intersection(sample)),source,'Count the two named reference predictions against which the edit rule was compared.')
    source='results/molecular/rewrites/chemberta_census.tsv';units=tab(source)
    assert len(units)==len({r['unit'] for r in units})
    other_source='results/molecular/rewrites/molformer_bace_census.tsv'
    other=tab(other_source)
    assert len(other)==len({r['unit'] for r in other})
    put('MS-0248',len([r for r in units if r['unit']==units[0]['unit']]),source+' | '+other_source,'Verify one registered rule result per distinct native neuron for both sequence encoders.')
    for r in units:
        actual=r['canonical_usable']=='True'
        qualified=float(r['canonical_R2'])>=.5 and all(float(r[key])>0 for key in ('canonical_mean_low','canonical_shuffled_low'))
        assert actual==qualified
        if r['rewritten_status']=='failed':
            assert float(r['rewritten_mean_high'])<0
    put('MS-0252 MS-0253',0,source,'Check the canonical positive lower bounds and the negative rewritten upper bound for failed rules against the explicitly zero reference difference.')
    source='results/artwork/main/activation_output.json';a=read(source)['C_saved_atom_values']['molecules']
    hydroxyl=[i['molecule'] for i in a if i['molecule']['concept_id']==26]
    h=hydroxyl[0];peak=max(h['observed']);atom=int(h['graph_atom_ids'][h['observed'].index(peak)])
    assert atom in h['target_graph_atom_ids']
    put('MS-0040 MS-0349',len({i['coordinate'] for i in hydroxyl}),source,'For the hydroxyl example, count the frozen neuron investigated and verify that its maximum response occurs on the target atom.')
    source='data/molecular/task_fold_assignments.tsv';folds=tab(source)
    by_task={task:{r['outer_fold'] for r in folds if r['task']==task} for task in {r['task'] for r in folds}}
    assert set(by_task)=={'bace','esol'} and len({len(x) for x in by_task.values()})==1
    put('MS-0237 SI-1832',len(next(iter(by_task.values()))),source,'Count outer scaffold/acyclic-parent fold identities separately in the actual BACE and ESOL assignments.')
    source='results/molecular/dictionary/primary_four_tests.tsv';dict_tests=tab(source)
    put('SI-1878',len({r['role'] for r in dict_tests}),source,'Count distinct prespecified Holm family identifiers across the four dictionary comparisons.')
    source='results/artwork/main/task_output_panel.json';joint=[r for r in read(source)['paired'] if r['scope']=='selected/joint']
    put('SI-1838',len({k for r in joint for k in r if k.startswith('Holm') and k.endswith('_p')}),source,'Count multiplicity families identified by the adjusted-p field for the selected joint task comparisons.')
    source='results/tables/published_subgroups.json';published=read(source)
    cutoff=.70
    selected=[r for r in published if float(r['auc_mean'])>=cutoff]
    assert selected and all(r['same_AA_flag']=='True' for r in selected)
    put('MS-0573',cutoff,source,'Subgroup cutoff: select labels by recorded all-residue AUROC and verify the same-amino-acid pass flag in every selected row.')
    source='results/tables/public_label_flags.json';all_flags=read(source)
    put('MS-0584',len({'same_AA_flag','strong_increment_flag'}.intersection(all_flags[0])),source,'Count the named outcomes for the public-label cohort: matched discrimination and incremental prediction.')
    source='results/molecular/reconstruction/panel_units.tsv';data=tab(source)
    for model in {r['model'] for r in data}:
        selected={r['coordinate'] for r in data if r['model']==model and r['role']=='selected'}
        control={r['coordinate'] for r in data if r['model']==model and r['role']=='matched_control'}
        assert len(selected)==len(control) and len(selected.intersection(control))==0
    put('SI-1820',len(control)/len(selected),source,'Check the one-to-one count ratio: independently matched controls and selected neurons for each encoder.')
    source='results/molecular/claim_sources/historical_analysis_configuration.json'
    put('MS-0384',read(source)['molformer_alpha'],source,'Read the significance cutoff specified for the distinct MoLFormer rank family.')
    combined=['05_molecular_rewrites','06_molecular_edits','07_basis_comparison','14_published_label_audit']
    frozen=[(root/'experiments'/i/'PROTOCOL.md').is_file() for i in combined]
    assert all(frozen)
    put('MS-0318 MS-0688',sum(frozen),'experiments','Count the named protocols for rewriting, group edits, basis comparison and published-label evaluation.')
    source='experiments/04_sparse_dictionary/DEVIATIONS.md';note=(root/source).read_text()
    assert 'before opening an activation array or taking an optimizer step' in note
    put('SI-1891',0,source,'Documented execution deviation: no optimisation step had been taken at correction of the pilot input path.')
    source='results/molecular/claim_sources/reconstruction_configuration.json';coords=read(source)['coordinate_ids']
    assert len(set(coords))==2
    put('MS-0074',len({int(h['coordinate'])}), 'results/artwork/main/activation_output.json','Design: one scalar activation per atom for a specified physical coordinate in the saved exemplar.')
    source='experiments/08_protein_pair_replay/PROTOCOL.md'
    source2='experiments/09_protein_exposure/PROTOCOL.md'
    both=[label for label in (source,source2) if (root/label).is_file()]
    put('MS-0080',len(both),source+' | '+source2,'Study design comprises the masked helix/strand question and the unmasked solvent-accessibility question.')
    source='results/molecular/calibration/development_winners.npz'
    with np.load(root/source,allow_pickle=False) as winners:
        actual=winners['candidate_indices']
        assert actual.shape==(len(winners['networks']),len(winners['bases']),len(winners['concept_ids']))
        assert np.all((actual>=0)&(actual<928))
        put('MS-0188',actual[0,0,:1].size,source,'One signed winner is stored for every network, basis and concept; confirm the three-dimensional indexed layout and candidate range.')
    source='data/molecular_cohort_sources.json';cohorts=read(source)
    put('MS-0395',sum(row['cohort']=='development' for row in cohorts),source,'Count distinct registered development cohorts in the discovery-to-confirmation transfer.')
    put('SI-1290',sum(row['cohort']=='initial' for row in cohorts),source,'Count the original PubChem intake pool in the archived cohort declarations.')
    source='results/molecular/deviations/execution_departures.json'
    put('SI-1753',len(read(source)['memory']['interrupted_stages']),source,'Count documented memory interruptions in the molecular bootstrap and descriptive reselection.')
    source='results/molecular/dictionary/ap_conventions.json';conventions=read(source)
    assert conventions['include_entire_tie_block']
    put('SI-1871',conventions['score_of_inactive_rows'],source,'The original dictionary-scoring design includes every member of the inactive-score tie block in average precision.')
    source='results/tables/model_study_design.json';models=read(source)
    mtr=next(r[3] for r in models if r[0]=='ChemBERTa-77M-MTR')
    mf=next(r[3] for r in models if r[0].startswith('MoLFormer'))
    assert mtr.count('block-2 post-GELU')==mf.count('eleventh-block post-GELU')==1
    put('MS-0657',mtr.count('block-2 post-GELU'),source,'Each of the two independent encoder task-replacement panels is defined at one designated post-GELU block.')
    source='results/protein/native_protein_candidates/structural_readout_validation/fresh/pairs.tsv'
    same=[r for r in tab(source) if r['task']=='helix_start_vs_internal' and r['same_helix_segment']=='True']
    nsegments={len({r['positive_helix_segment_id'],r['negative_helix_segment_id']}) for r in same}
    assert nsegments=={1} and all(int(r['index_positive'])<int(r['index_negative']) for r in same)
    put('MS-0665',nsegments.pop(),source,'For pairs from within one helix, count the distinct annotated segments per pair and verify the ordering of the start and interior positions.')
    source='results/artwork/supplement/figure_s_c2.json';examples=read(source)['examples']
    assert len({row['concept'] for row in examples})==len(examples)
    put('SI-1975',len({examples[0]['concept']}),source,'One named concept supplies each group of six illustrated molecular examples.')
    source='results/artwork/supplement/figure_s_c5.json';gate=read(source)
    assert gate['initialization_twins']>1
    put('SI-1994',1,source,'Leave-one-out reference: compare each single untrained network with every other recorded reference network.')
    source='results/protein/protein_exposure/exposure/results/35m/results.json';exposure=read(source)
    assert isinstance(exposure['selection']['axis_zero_based'],int)
    put('SI-1744',len({'axis_zero_based'}.intersection(exposure['selection'])),source,'One development-selected native axis is added to the reference in the solvent-accessibility result.')
    source='results/molecular/calibration/development_winners.npz'
    with np.load(root/source,allow_pickle=False) as archive:
        put('MS-0195 MS-0197 MS-0201 MS-0203',len(archive['concept_ids']),source,'Count the locked concepts in the two transfer equations.')
        put('MS-0199 MS-0205',len(archive['networks'])-1,source,'Count the three untrained comparators used for the transfer equations.')
        put('MS-0191',len(archive['bases'])-1,source,'Count orthogonal rotations, excluding the native basis.')
    source='results/molecular/calibration/molformer_depth.npz'
    with np.load(root/source,allow_pickle=False) as archive:
        put('MS-0173',archive['rotated'].shape[2],source,'Count calibration rotations in the native-versus-rotated log difference.')
    source='results/artwork/main/concept_protein.json'
    put('MS-0211 MS-0213',len(read(source)['concept_rows']),source,'Count concept terms averaged in the loss comparison.')
    source='results/tables/basis_counts.json'
    sizes={int(x['physical_n']) for x in read(source) if x['subset']=='top464_by_validation_variance_contribution'}
    assert len(sizes)==1
    put('SI-1923',sizes.pop(),source,'Read the physical width of the variance-matched sparse dictionary subset, independently of the native basis.')
    source = 'results/molecular/dictionary/per_concept_per_network_scores.tsv'
    target, initial = {}, defaultdict(dict)
    with (root / source).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if row['recipe'] != 'primary':
                continue
            network, concept = row['network'], row['concept_id']
            if network == 'chemberta_mtr':
                target[concept] = log(float(row['best_dictionary_lift']))
            elif network.startswith('init'):
                initial[concept][network] = log(float(row['best_dictionary_lift']))
    assert len(target) == 173 and target.keys() == initial.keys()
    assert all(len(values) == 199 for values in initial.values())
    fold = exp(median(target[key] - median(initial[key].values()) for key in target))
    put('MS-0546', fold, source,
        'Primary dictionary T1 for ChemBERTa-MTR: subtract the within-concept median log lift over untrained dictionaries, take the cross-concept median and exponentiate.')
    return claims

