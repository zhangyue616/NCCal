"""Named molecular evidence bindings for article positions.

Values are in the units displayed at the article position (percent where the
text displays percent). This module does not read the article or look for equal
numbers in a directory. Every position has an explicit scientific selector.
Operations beginning 'Design' identify protocol or plotting parameters, rather
than estimated effects. Source fragments retain their original SHA identities.
"""
from pathlib import Path
import ast,csv,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def compute(root=ROOT):
    root=Path(root)
    result={}
    def load(path):
        return json.loads((root/path).read_text(encoding='utf-8-sig'))
    def rows(path):
        return list(csv.DictReader((root/path).open(encoding='utf-8-sig'),delimiter='\t' if path.endswith('.tsv') else ','))
    def array(path):
        return np.load(root/path,allow_pickle=False)
    def put(ids,value,source,operation):
        if isinstance(ids,str):ids=ids.split()
        for identifier in ids:
            if identifier in result:raise ValueError('Duplicate binding '+identifier)
            result[identifier]=dict(value=float(value),source=source,operation=operation)
    prefix='results/molecular/claim_sources/'

    # S-J: count each labelled physical atom once, even when motif matches overlap.
    source=prefix+'hydroxyl_atom_values.npz';a=array(source)
    put('SI-1769',a['carbon_actual'].mean(),source,'Mean actual neuron-293 activation over unique hydroxyl-labelled carbon atoms.')
    put('SI-1770',len(a['carbon_actual']),source,'Count unique (CID, carbon graph atom index) among hydroxyl matches.')
    put('SI-1771',a['oxygen_actual'].mean(),source,'Mean actual neuron-293 activation over unique hydroxyl-labelled oxygen atoms.')
    put('SI-1772',len(a['oxygen_actual']),source,'Count unique (CID, oxygen graph atom index) among hydroxyl matches.')
    source=prefix+'reconstruction_configuration.json';config=load(source)
    put('SI-1763 SI-1768',config['coordinate_ids'][0],source,'Design: first frozen physical coordinate identifier.')
    put('SI-1764',config['coordinate_ids'][1],source,'Design: second frozen physical coordinate identifier.')
    put('SI-1766 MS-0221',config['alpha_by_coordinate']['j293'],source,'Design: frozen radius-one/radius-zero shrinkage prior weight alpha; not an effect estimate.')

    # Earlier notation challenge: denominators come from all retained QA records.
    source=prefix+'notation_support.npz';a=array(source)
    n=len(a['n_atoms']);strict=a['full_offsets'].all(1)&~a['identity_lossy'].any(1)
    put('SI-1778 SI-2088',n,source,'Count accepted parent molecules in the common three-notation support.')
    put('SI-1779',a['n_atoms'].sum(),source,'Sum physical atom counts once per accepted parent, without multiplying by notations.')
    put('SI-1784 SI-2090',strict.sum(),source,'Count parents with no identity loss and complete offsets in all three notations.')
    put('SI-1785',a['identity_lossy'].sum(),source,'Count accepted molecule-notation inputs with recorded identity loss.')
    put('SI-1786',a['identity_lossy'].size,source,'Count all accepted molecule-notation inputs, including canonical strings.')
    put('SI-1787',np.count_nonzero(~a['full_offsets']),source,'Count accepted molecule-notation inputs with blocked complete offsets.')
    source=prefix+'notation_denominators.json';d=load(source)
    put('SI-1777',d['attempted_molecules'],source,'Design: fixed CID-hash sample size before mapping; no replacement of failures.')
    put('SI-1780',d['excluded_molecules'],source,'Archived mapping exclusion count, retaining all original failed cases.')
    source=prefix+'notation_summary.json';d=load(source)
    put('SI-1781',len(d['metric_names']),source,'Count every frozen metric name, including adverse outcomes.')
    put('SI-1783',d['bootstrap_replicates'],source,'Design: shared whole-molecule bootstrap replicate count.')
    source=prefix+'notation_interval_configuration.json';d=load(source)
    confidence=100*(d['upper_quantile']-d['lower_quantile'])
    put('SI-1782 SI-2087',confidence,source,'Design: 100 times the difference between frozen upper and lower percentile probabilities; marginal interval coverage.')

    # Positional verification differs from token identity/full-offset fidelity.
    source=prefix+'anchor_chemberta_mtr.npz';a=array(source)
    verified=np.count_nonzero(a['category']=='verified_unique_anchor_match')
    unresolved=np.count_nonzero(a['category']=='ambiguous/unverifiable')
    b=array(prefix+'anchor_chemberta_mlm.npz')
    assert np.array_equal(a['category'],b['category'])
    put('SI-1801',2,source+'; '+prefix+'anchor_chemberta_mlm.npz','Count the two archived tokenizer/model positional-audit records, both checked for identical categories.')
    put('SI-1802 SI-1808 SI-2067',verified,source,'Count unique positional-anchor matches; does not assert complete token identity.')
    put('SI-1803',len(a['category']),source,'Count original calibration molecules, retaining unresolved cases.')
    put('SI-1804 SI-2069',unresolved,source,'Count unresolved calibration molecules, without replacing the original cohort.')
    put('SI-2068',a['position_disagreements'].sum(),source,'Sum actual atom-position disagreements in the archived positional comparison.')
    source=prefix+'anchor_rank_tests.json';d=load(source)
    ps=[]
    for model in sorted(d):
        for test in ['J1','J2']:
            x=d[model][test]
            assert x['T_target']>x['T_competitors_max']
            ps.append(1/(1+x['n_competitors']))
    assert len(set(ps))==1
    put('SI-1809',ps[0],source,'Inclusive upper rank p = 1/(199+1), because all initialization statistics are below the trained statistic; all four model/criterion cases checked.')
    source=prefix+'standardization_development.json';d=load(source)
    put('SI-1810',d['planned_D'],source,'Count frozen independently standardized development molecules after the listed structure-overlap exclusions.')
    put('SI-1811',len(d['excluded_original_D_rows_for_H4_structure_overlap']),source,'Count explicitly listed original development rows excluded for structure overlap.')
    source=prefix+'standardization_rank_tests.json';d=load(source)
    ps=[d[m][t]['p'] for m in sorted(d) for t in ['J1','J2']]
    order=np.argsort(ps);adjusted=np.empty(len(ps));adjusted[order]=np.minimum(1,np.maximum.accumulate(np.asarray(ps)[order]*(len(ps)-np.arange(len(ps)))))
    put('SI-1812',len(ps),source,'Count the two models times two criteria in the new standardized-geometry Holm family.')
    assert np.ptp(adjusted)==0
    put('SI-1813',adjusted[0],source,'Recompute Holm adjustment over the complete four-test standardized-geometry family.')
    source=prefix+'scaffold_slice.json';d=load(source)
    put('SI-2066',d['n_molecules'],source,'Frozen development-unseen-scaffold subset molecule count; conditional descriptive slice.')

    # Dictionary configuration is a design quantity, distinct from fitted scores.
    source=prefix+'dictionary_training_config.json';d=load(source)
    for ids,field in [('SI-1862 SI-1894 SI-2112','width'),('SI-1863','k'),('SI-1864','learning_rate'),('SI-1865','batch_size'),('SI-1866 SI-1868','min_epochs'),('SI-1867','max_epochs'),('SI-1869','patience_epochs'),('SI-1870','relative_min_improvement')]:
        put(ids,d[field],source,'Design: frozen reconstruction-only selected training configuration field '+field+'.')
    source='results/molecular/dictionary/training_log_summary.tsv';data=rows(source)
    primary={x['network']:x for x in data if x['recipe']=='primary' and x['network'] in ['chemberta_mtr','chemberta_mlm']}
    normalizer=load(prefix+'dictionary_normalization.json')
    native_dimension=normalizer['dimension']
    assert native_dimension==array('data/molecular/calibration_dictionary_encoder.npz')['encoder_weight'].shape[1]
    put('SI-1861',native_dimension,prefix+'dictionary_normalization.json','Design: the frozen scale sqrt(SS/(N*dimension)) makes mean normalized squared norm equal dimension; checked against the decoder input width.')
    for identifier,recipe in [('SI-1882','half_k'),('SI-1884','double_k')]:
        subset=[x for x in data if x['network']=='chemberta_mtr' and x['recipe']==recipe]
        put(identifier,int(subset[0]['k']),source,'Design: active-feature count in the named frozen '+recipe+' sensitivity recipe.')
    put('SI-1879',100*float(primary['chemberta_mtr']['validation_fvu']),source,'MTR primary dictionary validation fraction of variance unexplained, multiplied by 100.')
    put('SI-1880',100*float(primary['chemberta_mlm']['validation_fvu']),source,'MLM primary dictionary validation fraction of variance unexplained, multiplied by 100.')
    put('SI-1886',len(data),source,'Count all completed fit summaries, including descriptive sensitivity recipes.')
    source='results/molecular/dictionary/per_concept_per_network_scores.tsv';data=rows(source)
    put('SI-1887',len(data),source,'Count every saved network/recipe/concept score row, rather than multiplying an assumed fit count.')
    put('SI-2111',len({x['concept_id'] for x in data}),source,'Count distinct scored primary concepts.')
    put('SI-1874 SI-2113',len({x['network'] for x in data if x['recipe']=='primary' and x['network'].startswith('init')}),source,'Count distinct initialization networks in the primary scored dictionaries.')
    def dictionary_fold(recipe):
        subset=[x for x in data if x['recipe']==recipe]
        target={x['concept_id']:np.log(float(x['dictionary_over_native'])) for x in subset if x['network']=='chemberta_mtr'}
        init={c:[] for c in target}
        for x in subset:
            if x['network'].startswith('init'):init[x['concept_id']].append(np.log(float(x['dictionary_over_native'])))
        return np.exp(np.median([target[c]-np.median(init[c]) for c in sorted(target)]))
    put('SI-1881',dictionary_fold('half_k'),source,'MTR k=8 T2 fold: exp median across concepts of target log(dictionary/native) minus initialization median log(dictionary/native).')
    put('SI-1883',dictionary_fold('double_k'),source,'MTR k=32 T2 fold: exp median across concepts of target log(dictionary/native) minus initialization median log(dictionary/native).')
    source='results/molecular/dictionary/all_recipe_estimates.tsv';data=rows(source)
    primary=[x for x in data if x['role']=='primary_four_family']
    sensitivity=[x for x in data if x['role']!='primary_four_family']
    put('SI-1877',len(primary),source,'Count the explicitly identified primary dictionary tests in the Holm family.')
    put('SI-1888',len(sensitivity),source,'Count explicitly identified descriptive sensitivity test rows outside the primary family.')
    put('SI-1893',len({x['test'] for x in primary}),source,'Design: count the two primary calibration definitions T1 and T2.')

    # Legacy figure data retains its own 39-reference family, separate from 199.
    art='results/artwork/supplement/'
    source=art+'figure_s_b1.json';d=load(source);h3=d['H3_readouts_all_concepts']
    put('SI-1964 SI-1978',h3[0]['source_log_effects']['J1']['n_competitors'],source,'Historical H3 initialization-reference count; separate from the later 199-reference calibration.')
    put('SI-1965 SI-1992',len(h3),source,'Count archived H3 model/readout rows.')
    put('SI-1991',len([k for k in h3[0]['source_log_effects'] if k in ['J1','J2']]),source,'Design: count the two named calibration criteria J1 and J2.')
    put('SI-1993',h3[0]['source_log_effects']['J1']['n_concepts'],source,'Number of scored concepts in the archived H3 family.')
    put('SI-1995',h3[0]['source_log_effects']['J1']['n_competitors']-1,source,'For one untrained network, compare it with the remaining historical untrained networks: 39 minus itself.')
    source=art+'figure_s_c1.json';d=load(source)
    put('SI-1968',d['vocab_size'],source,'Released tokenizer vocabulary row count, including unused and special tokens.')
    put('SI-1969 SI-2001',d['n_smiles'],source,'Number of molecule strings in the frozen pilot corpus.')
    put('SI-1970',d['init_expected_norm']['chemberta_mtr'],source,'Archived expected random-initialization embedding norm for the MTR architecture.')
    plotted_source='figures/native/supplement_tokenizer.py'
    plotted_examples=[node.iter.elts for node in ast.walk(ast.parse((root/plotted_source).read_text(encoding='utf8'))) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='s' and isinstance(node.iter,ast.Tuple)]
    assert len(plotted_examples)==1
    assert all(ast.literal_eval(x) in d['examples'] for x in plotted_examples[0])
    put('SI-1971',len(plotted_examples[0]),plotted_source+'; '+source,'Count only the explicit tokenizer strings rendered in panel c, rather than every archived candidate example.')
    for filename,ids in [('figure_s_c2.json','SI-1976'),('figure_s_c3.json','SI-1982 SI-1983')]:
        source=art+filename;d=load(source)
        put(ids,len(d['examples']),source,'Count fixed concept panels in the frozen activation-example specification.')
    source='data/artwork/molecular_example_depictions.json';depictions=load(source)
    for model,identifier in [('chemberta_mlm','SI-1974'),('chemeleon','SI-1987')]:
        counts=[len(x['cells']) for x in depictions if x['model']==model]
        assert counts and len(set(counts))==1
        put(identifier,counts[0],source,'Count rendered molecule cells within each '+model+' concept panel; distinct from the number of concepts.')
    source=art+'figure_s_c6.json';d=load(source);pilot=next(iter(d['pilot'].values()))
    put('SI-1999',len(pilot['initialization_dots_pct']),source,'Count initialization-network measurements in the earlier pilot; not the main rank-test family.')
    put('SI-2000',len(pilot['rotation_dots_pct']),source,'Count earlier pilot rotation measurements; not independent networks.')
    put('SI-2003',d['H3_examples'][0]['source']['top100_composition']['concept atoms'],source,'Count nitro-concept atoms in the MTR example top-100 composition.')
    put('SI-2004',d['H3_examples'][1]['source']['top100_composition']['concept atoms'],source,'Count ketone carbonyl-carbon atoms in the CheMeleon example top-100 composition.')
    put('SI-2005',d['H3_examples'][1]['source']['top100_composition']['sulfonyl S'],source,'Count sulfonyl sulfur atoms in the CheMeleon example top-100 composition.')

    # Sensitivity captions: scientific object identities and attained-dose design.
    source=art+'figure_s_e1.json';d=load(source);objects=d['objects']
    carbonyl=[x for x in objects if int(x['D_selected_unit'])==454]
    put('SI-2007 SI-2012',len(objects),source,'Count the three prespecified chemical objects in each interval family.')
    put('SI-2008',len(carbonyl),source,'Count the two carbonyl definitions sharing one physical coordinate.')
    put('SI-2009',int(carbonyl[0]['D_selected_unit']),source,'Design: frozen carbonyl coordinate identifier.')
    put('SI-2010',int(next(x for x in objects if x['exact_id']=='26')['D_selected_unit']),source,'Design: frozen aliphatic hydroxyl coordinate identifier.')
    put('SI-2013',d['bootstrap_replicates_each_H_or_rho'],source,'Design: fixed whole-molecule bootstrap replicate count for each chemical-object sensitivity.')
    put('SI-2014',int(objects[0]['T_positive_subset']),source,'Count frozen positive-stratum molecules for the first chemical object; all three objects use this denominator.')
    put('SI-2015',int(objects[0]['T_negative_subset']),source,'Count frozen all-negative-stratum molecules for the first chemical object; all three objects use this denominator.')
    source=prefix+'paired_dose_configuration.json';d=load(source);paired=d['objects']
    put('SI-2011 SI-2016 SI-2071',len(paired[0]['units'])-1,source,'Design: count matched control coordinates, excluding the selected coordinate in the frozen unit list.')
    for identifier,index in [('SI-2072',0),('SI-2073',1),('SI-2074',2)]:
        put(identifier,paired[0]['scales'][index],source,'Design: frozen fraction of the common attainable scalar dose.')
    put('SI-2075',100*(1-.05/len(paired)),source,'Design: separate three-object Bonferroni interval coverage, 100*(1-0.05/3); not a measured effect.')
    put('SI-2076',d['bootstrap']['replicates'],source,'Design: fixed paired-dose bootstrap draw count.')
    put('SI-2077',len(paired),source,'Design: number of chemical objects in each separate Bonferroni family.')
    put('SI-2080',2*len(paired[0]['units']),source,'Design: common scalar dose is bounded by positive and negative full-clamp doses for all selected/control coordinates.')
    put('SI-2082',len(config['coordinate_ids']),prefix+'reconstruction_configuration.json','Design: count the two frozen focal coordinate replacements.')
    for identifier,value in [('SI-2083',config['coordinate_ids'][0]),('SI-2085',config['coordinate_ids'][1])]:
        put(identifier,value,prefix+'reconstruction_configuration.json','Design: frozen focal coordinate identifier.')
    put('SI-2084',next(x['q'] for x in paired if x['concept_id']==26),source,'Design: frozen anonymous output index for the hydroxyl coordinate.')
    put('SI-2086',next(x['q'] for x in paired if x['concept_id']==24),source,'Design: frozen primary anonymous output index for the carbonyl coordinate.')

    # Matched panels and whole-layer conclusions: select scientific roles explicitly.
    source='results/molecular/reconstruction/panel_units.tsv';data=rows(source)
    controls={m:[x for x in data if x['model']==m and x['role']=='matched_control'] for m in ['chemberta_mtr','molformer']}
    selected={m:[x for x in data if x['model']==m and x['role']=='selected'] for m in controls}
    assert len(controls['chemberta_mtr'])==len(controls['molformer'])
    put('SI-2092 SI-2101',len(controls['chemberta_mtr']),source,'Count matched physical control coordinates per encoder, selected on development only.')
    put('SI-2100',len(selected['chemberta_mtr']),source,'Count distinct selected physical coordinates per encoder.')
    put('SI-2093',int(controls['chemberta_mtr'][0]['n_atoms']),source,'Confirmation atom denominator for each two-encoder panel row.')
    put('SI-2094',int(controls['chemberta_mtr'][0]['n_molecules']),source,'Confirmation molecule denominator for each two-encoder panel row.')
    for identifier,model in [('SI-2096','chemberta_mtr'),('SI-2097','molformer')]:
        put(identifier,np.mean([float(x['R2_local']) for x in controls[model]]),source,'Mean local-graph R-squared over the nine development-matched '+model+' control coordinates.')
    source='results/molecular/whole_layer/layer_reconstruction_8_networks.tsv';data=rows(source)
    network_field=next(k for k in ['network','network_id'] if k in data[0])
    counts=[len({x[network_field] for x in data if x['model']==model and x[network_field].startswith('init')}) for model in {x['model'] for x in data}]
    assert len(set(counts))==1
    put('SI-2105 SI-1847 SI-1858',counts[0],source,'Count distinct untrained references within each encoder, rather than combining the two encoder reference sets.')
    source='results/molecular/whole_layer/primary_12_family.tsv';data=rows(source)
    basic=[x for x in data if x['reference'] in ['development_mean','radius0']]
    wins=[x for x in basic if float(x['effect_reference_error_minus_local_error'])>0 and float(x['Holm_adjusted_p_12'])<=.05]
    perm=[x for x in data if x['reference']=='residual_permutation']
    perm_wins=[x for x in perm if float(x['effect_reference_error_minus_local_error'])<0 and float(x['Holm_adjusted_p_12'])<=.05]
    put('SI-2106 SI-1849',len(wins),source,'Count positive local-versus-mean/radius-zero fidelity advantages with Holm-adjusted p <=0.05 in the fixed twelve-test family.')
    put('SI-2107 SI-1850',len(perm_wins),source,'Count task/encoder contrasts where residual shuffling has smaller deviations and Holm-adjusted p <=0.05.')
    put('SI-2108 SI-1851',len(perm),source,'Count the two tasks times two encoders in the residual-permutation comparison.')

    # Calibration and application population denominators use their actual rows.
    source=prefix+'anchor_chemberta_mtr.npz';a=array(source)
    put('MS-0140',a['legacy_n_atoms'].sum(),source,'Sum original parseable heavy-atom counts over all 10,000 calibration molecules, retaining unresolved positional cases.')
    source=prefix+'calibration_concept_partitions.json';concepts=load(source)
    put('MS-0144',sum(x['partition'] in ['input','input_equivalent'] for x in concepts),source,'Count direct-input and input-equivalent descriptive concept definitions; excluded from the primary readable partition.')
    put('SI-1313',2*sum(x['partition']=='readable' for x in concepts),source,'Design: two polarities for every readable concept in the earlier exploratory matched-neuron screen.')
    source='results/molecular/reconstruction/focal_activation_predictions.npz';a=array(source)
    put('MS-0151',len(a['label_ids']),source,'Count all curated chemistry label definitions before development-only coverage filtering.')
    for identifier,task in [('MS-0166','bace'),('MS-0167','esol')]:
        source='results/molecular/whole_layer/chemberta_mtr_'+task+'.npz';a=array(source)
        put(identifier,len(a['sample_ids']),source,'Count retained, qualified '+task.upper()+' task structures with frozen predictions; no per-input measured targets are redistributed.')
    source='results/molecular/reconstruction/label_lookup_sufficient.npz';a=array(source)
    put('SI-1818 MS-0209',len(a['concept_ids']),source,'Count the fixed reconstruction panel concept associations, before duplicate physical coordinates are consolidated.')
    source='results/molecular/reconstruction/panel_units.tsv';data=rows(source)
    selected=[x for x in data if x['role']=='selected']
    by_model={m:[x for x in selected if x['model']==m] for m in {x['model'] for x in selected}}
    assert len({len(x) for x in by_model.values()})==1
    put('SI-1819',len(next(iter(by_model.values()))),source,'Count unique selected physical neurons within each encoder, consolidating duplicate concept associations.')
    put('SI-1823 MS-0163',len(by_model),source,'Count encoder populations in the reconstruction panel and its model-level family.')
    exceptions=[x for x in selected if float(x['R2_binary'])>float(x['R2_local'])]
    put('SI-1825',len(exceptions),source,'Count selected physical neurons where binary concept means have higher confirmation R-squared than the local rule, retaining negative comparisons.')
    source='results/molecular/reconstruction/panel_all_coordinate_stratum_metrics.tsv';data=rows(source)
    acyclic=[x for x in data if x['role']=='selected' and x['model']=='chemberta_mtr' and x['stratum']=='scaffold_empty_separate']
    by_coordinate={int(x['coordinate']):{} for x in acyclic}
    for x in acyclic:by_coordinate[int(x['coordinate'])][x['predictor']]=float(x['R2'])
    worst=min(by_coordinate,key=lambda k:by_coordinate[k]['local_graph']-by_coordinate[k]['binary_mean'])
    put('SI-1826',worst,source,'Identify the selected ChemBERTa coordinate with the most negative local-minus-binary R-squared in the separately retained acyclic/empty-scaffold stratum.')
    source='results/molecular/reconstruction/joint_expected.json';d=load(source);primary=d['primary']
    put('SI-1839',len(primary),source,'Count all selected-joint task fidelity contrasts in the frozen Holm family.')
    put('SI-1840',sum(float(x['effect_reference_error_minus_local_error'])>0 and float(x['Holm_adjusted_p_12'])<=.05 for x in primary),source,'Count selected-joint local-rule advantages surviving the complete twelve-test Holm family.')
    put('MS-0493',len({x['reference'] for x in primary}),source,'Count the three named references represented in the complete selected-joint fidelity family.')
    source=prefix+'carbonyl_label_lookup_ap.npz';a=array(source)
    order=np.argsort(a['scores'])[::-1];positive=a['positive_count'][order];total=positive+a['negative_count'][order]
    precision=np.cumsum(positive)/np.cumsum(total)
    put('SI-1326',np.sum(positive*precision)/positive.sum(),source,'Recompute average precision for carbonyl concept 24 from grouped exact float32 radius-one probabilities and all tied positive/negative counts.')
    source=prefix+'cohort_identity_intersections.json';d=load(source)
    put('SI-1754',d['canonical_nonisomeric_smiles']['D-T']['unique_identities'],source,'Count shared development/confirmation canonical identities only after stereochemistry removal; distinct from the zero isomeric-identity intersection.')

    # Earlier output-position exploration retains its different intervention scope.
    source=prefix+'fragment_readout_exploration.json';data=load(source)
    put('SI-1343',len({x['concept'] for x in data}),source,'Count distinct frozen fragment concepts in the separate all-token/atom-only readout-position exploration.')
    pyridine=[x for x in data if x['concept']=='frag:fr_pyridine' and x['layer']=='L1' and x['mode']=='atoms']
    assert len(pyridine)==1
    put('SI-1344',pyridine[0]['specificity_rank'],source,'Archived pyridine target-output specificity rank for block-two atom-only intervention; not a general rank across all layers/scopes.')
    terminal=[x['max_abs_change_any_output'] for x in data if x['layer']=='L2' and x['mode']=='atoms']
    put('SI-1345',max(terminal)*1e5,source,'Maximum saved normalized-output absolute change across final-block atom-only rows, expressed as the coefficient of 10^-5.')
    source=art+'figure_s_e1.json';d=load(source)
    put('SI-1322 SI-1335 SI-1337 SI-1351',len(d['objects']),source,'Count frozen development-selected chemical concept/output objects, retaining duplicate physical neurons as separate definitions.')
    sensitive=[x for x in d['objects'] if float(x['H_CI_lower'])>0 and float(x['oriented_rho_CI_lower'])>0]
    put('SI-1339',len(sensitive),source,'Count objects meeting both positive lower-bound sensitivity and oriented association criteria under the historical full replacement.')
    source=prefix+'paired_dose_configuration.json';d=load(source)
    negative=[x for x in d['objects'] if x['H_pair1']<0 and x['H_pair1_CI98_3333333333'][0]<=0<=x['H_pair1_CI98_3333333333'][1]]
    put('SI-1340',len(negative),source,'Count all common-dose sensitivity point estimates below zero with their prespecified intervals spanning zero.')
    put('SI-1336',len(d['objects'][0]['units'])-1,source,'Design: count frozen same-block matched controls, excluding the selected coordinate.')
    put('SI-1755',d['objects'][0]['pairs'],source,'Count fixed positive/negative artificial pairs per chemical object in the paired-dose archive.')
    put('SI-1756',2*len(d['objects'][0]['units']),source,'Design: two molecules times all six selected/control coordinates define twelve attainable full-clamp doses.')

    # Census diagnostics use only originally usable, determinate coordinates.
    source='results/molecular/rewrites/chemberta_census.tsv';data=rows(source)
    usable=[x for x in data if float(x['canonical_R2'])>=.5 and min(float(x['canonical_mean_low']),float(x['canonical_shuffled_low']))>0]
    states=lambda x:'retained' if min(float(x['rewritten_mean_low']),float(x['rewritten_shuffled_low']))>0 else ('failed' if float(x['rewritten_mean_high'])<0 else 'undetermined')
    determinate=[x for x in usable if states(x)!='undetermined']
    put('MS-0551',len(usable),source,'Count canonically usable physical neurons by both frozen positive control-advantage intervals and R-squared >=0.5.')
    put('SI-1901',sum(states(x)=='failed' for x in usable),source,'Count failed rewrites among the canonically usable neurons; unresolved neurons remain a separate category.')
    put('MS-0497',sum(states(x)=='retained' for x in usable)/len(usable),source,'Observed retention fraction 26/54, described approximately as half; not the 0.5 design threshold for activation reconstruction.')
    def auc(key,sign):
        pos=np.asarray([sign*float(x[key]) for x in determinate if states(x)=='failed'])
        neg=np.asarray([sign*float(x[key]) for x in determinate if states(x)=='retained'])
        return (np.sum(pos[:,None]>neg[None,:])+.5*np.sum(pos[:,None]==neg[None,:]))/(len(pos)*len(neg))
    put('SI-1903',auc('canonical_string_increment_T',1),source,'Mann-Whitney AUROC of the canonical-string increment for failed versus retained neurons, restricted to originally usable determinate cases.')
    put('SI-1904',auc('Pearson',-1),source,'Mann-Whitney AUROC of negative same-atom Pearson correlation for failed versus retained neurons, using the identical determinate cases.')
    put('MS-0554',len(data),source,'Count all physical neurons in the measured ChemBERTa layer, including unusable and unresolved coordinates.')
    source='results/molecular/bases/basis_summary.tsv';data=rows(source)
    put('MS-0555',len({x['basis'] for x in data if x['basis'].startswith('rot_')}),source,'Count distinct frozen random bases, rather than intervention coordinates or independent networks.')
    put('MS-0556',len({x['basis'] for x in data if x['basis'].startswith('sae_')}),source,'Count refitted dictionary identities, consolidating full and top-variance subsets of each fit.')
    source='results/molecular/edits/raw_pairs.npz';a=array(source)
    scale=array('results/molecular/edits/d_scale.npz')['std'];coordinate=[293,454]
    delta=a['controlled_product'][:,coordinate].astype(float)-a['source'][:,coordinate].astype(float)
    unchanged=[];counts=[]
    for kind in range(1,5):
        parts=[]
        for i in np.flatnonzero(a['main_pass']&(a['edit_classes']==kind)):
            lo,hi=a['offsets'][i:i+2];values=delta[lo:hi][~a['changed'][lo:hi]]
            if len(values):parts.append(np.abs(values).mean(0)/scale[coordinate])
        unchanged.extend(np.asarray(parts).mean(0));counts.append(len(parts))
    put('MS-0460',max(unchanged),source+'; results/molecular/edits/d_scale.npz','Largest molecule-equal mean absolute measured change, divided by development SD, across the eight focal controlled edit/neuron cells on unchanged environments; zero-atom molecules do not contribute zeros.')

    # Final historical definitions keep the pilot and later calibration separate.
    source=prefix+'calibration_concept_partitions.json';definitions=load(source)
    put('SI-1294',sum(x['partition']=='readable' and x['name'].startswith('ertl:') for x in definitions),source,'Count Ertl functional-group definitions in the primary readable calibration partition; not all 200 candidate definitions.')
    source=prefix+'historical_analysis_configuration.json';historical=load(source)
    put('SI-1296',historical['heavy_atom_minimum'],source,'Design: inclusive original largest-heavy-fragment minimum heavy-atom count.')
    put('SI-1297',historical['heavy_atom_maximum'],source,'Design: inclusive original largest-heavy-fragment maximum heavy-atom count.')
    put('SI-1314',historical['pilot_minimum_lift'],source,'Design: earlier exploratory pilot matching rule requires lift >=3 in addition to its null and BH conditions; not the later calibration rule.')
    put('SI-1315',historical['pilot_bh_q'],source,'Design: original Benjamini-Hochberg q for the exploratory matched-neuron screen, distinct from a network-rank p-value threshold.')
    put('MS-0184 SI-1308',historical['molformer_alpha'],source,'Design: separate MoLFormer gate level for all three prespecified rank p-values; no eight-model correction is applied to this separate family.')
    put('MS-0391',historical['calibration_upper_percentile']-historical['calibration_lower_percentile'],source,'Design: original calibrated-fold concept-bootstrap interval width in percentile probability, 97.5 minus 2.5; conditional on the fixed trained/reference networks.')
    put('MS-0528',100*historical['basis_high_R2_threshold'],source,'Design: 80% explained-variance threshold for per-direction R-squared counts, not a measured layer-wide fraction.')
    source=prefix+'reconstruction_configuration.json';d=load(source)
    put('SI-1317',d['coordinate_ids'][1],source,'Design: frozen post-GELU carbonyl reconstruction coordinate 454, distinct from the final-block pre-GELU illustrative coordinate.')
    source=prefix+'fragment_readout_exploration.json';data=load(source)
    magnitude=max(x['max_abs_change_any_output'] for x in data if x['layer']=='L2' and x['mode']=='atoms')
    exponent=int(np.floor(np.log10(magnitude)))
    assert exponent==-5
    put('SI-1346',10,source,'Scientific notation: radix 10 for the saved final-block atom-only maximum; exponent floor(log10(maximum))=-5 and coefficient maximum/10^-5=4.57763671875. The radix is notation, not an effect estimate.')
    source='results/molecular/whole_layer/chemberta_mtr_bace_conditions.json';d=load(source)
    primary_condition_names={'baseline','development_mean','radius0','local_graph'}
    observed={x['condition'] for x in d['conditions'] if x['condition'] in primary_condition_names}
    assert observed==primary_condition_names
    put('SI-1848',len(observed),source,'Count the four explicitly named activation states A, development mean, atom type and local graph in the actual frozen condition registry; sham, remainder and displacement permutations are separate controls.')
    return result

bindings=compute

if __name__=='__main__':
    print(json.dumps(compute(),ensure_ascii=True,indent=2))
