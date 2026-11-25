"""Compute application, rewrite-sensitivity and reference-rank evidence selectors."""
from pathlib import Path
import csv
import json
import sys


def values(root):
    root=Path(root)
    sys.path.insert(0,str(root/'src'))
    from nccal.rewrite_sensitivity import validate_archived_main_counts
    out={}
    def put(key,value,source,operation):
        out[key]=dict(value=float(value),source=source,operation=operation)
    def rows(source):
        with (root/source).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
    source='results/artwork/use/basis_rewrite_direction_inputs.tsv'
    counts=validate_archived_main_counts(root)
    for row in counts:
        group=row['basis']+'.'+('full' if row['subset']=='full' else 'variance_top464')
        for field in ['physical_n','canonical_usable_n','retained','retained_and_rewritten_R2_ge_05','retained_missing_rewritten_R2','retained_fraction_all_directions','dual_fraction_all_directions']:
            put('bases.'+group+'.'+field,row[field],source,'Archived per-direction recount: '+group+'; '+field+'. Main retention requires canonical_usable AND retained; the sensitivity also requires finite rewritten_R2 >= 0.5. Missing R2 values are preserved.')
    rotations=[r for r in counts if r['basis'].startswith('rot_') and r['subset']=='full']
    dictionaries=[r for r in counts if r['basis'].startswith('sae_') and r['subset']=='full']
    top=[r for r in counts if r['basis'].startswith('sae_') and r['subset']!='full']
    for group,selected in [('rotations',rotations),('dictionary_full',dictionaries),('dictionary_variance_top464',top)]:
        for field in ['physical_n','canonical_usable_n','retained','retained_and_rewritten_R2_ge_05','retained_fraction_all_directions','dual_fraction_all_directions']:
            for operation in ['min','max']:
                value={'min':min,'max':max}[operation](r[field] for r in selected)
                put('bases.'+group+'.'+field+'.'+operation,value,source,'Archived per-direction recount: '+group+'; '+operation+'('+field+'); full physical denominators are retained.')
    native=next(r for r in counts if r['basis']=='native' and r['subset']=='full')
    put('bases.dictionary_to_native_width_ratio',dictionaries[0]['physical_n']/native['physical_n'],source,'Full dictionary physical width divided by native neuron width: 928/464; all dictionary widths agree.')
    put('bases.dictionary_display_scale',native['physical_n']/dictionaries[0]['physical_n'],source,'Directions-per-464 display factor for full dictionary counts: 464/928.')
    for operation in ['min','max']:
        value={'min':min,'max':max}[operation](r['retained']/native['retained'] for r in dictionaries)
        put('bases.dictionary_retained_to_native_ratio.'+operation,value,source,'Full dictionary retained count divided by native retained count; '+operation+' across the three dictionaries. The displayed whole-number multiple is approximate.')
    source='results/artwork/use/5b_and_s_q1_census.tsv'
    census=rows(source)
    usable=[r for r in census if r['canonical_usable']=='True']
    retained=[r for r in usable if r['rewritten_status']=='retained']
    unresolved=[r for r in usable if r['rewritten_status']=='undetermined']
    tests={'canonical_usable':usable,'retained':retained,'retained_rewritten_R2_ge05':[r for r in retained if float(r['rewritten_R2'])>=.5],'retained_rewritten_R2_lt05':[r for r in retained if float(r['rewritten_R2'])<.5],'unresolved':unresolved,'unresolved_rewritten_R2_ge05':[r for r in unresolved if float(r['rewritten_R2'])>=.5]}
    for key,data in tests.items():put('census.'+key,len(data),source,'Frozen rewriting census: count canonical_usable rows in '+key+'. Retention is the write-back criterion; rewritten R2 is a separate sensitivity.')
    for row in retained:
        if float(row['rewritten_R2'])<.5:
            put('census.retained_unit_'+row['unit']+'.rewritten_R2',row['rewritten_R2'],source,'Frozen rewriting census: canonical_usable=True, rewritten_status=retained, unit='+row['unit']+'; rewritten_R2 from the original development-fitted rule.')
    source='results/artwork/use/6b_and_s_t1_public_labels.tsv'
    public=rows(source)
    for frame in ['T','NEW']:
        selected=[r for r in public if r['source']=='Hallee' and r['formal_beta_gamma_eligible']=='True' and r['frame']==frame]
        put('public.'+frame+'.labels',len(selected),source,'All formally eligible published labels in frame '+frame+'; retain unfavourable results.')
        for group,part in [('all',selected)]+[('model_'+name,[r for r in selected if r['model']==name]) for name in sorted({r['model'] for r in selected})]+[('class_'+name,[r for r in selected if r['annotation_category']==name]) for name in sorted({r['annotation_category'] for r in selected})]:
            put('public.'+frame+'.'+group+'.labels',len(part),source,'Count formally eligible published labels, frame='+frame+', '+group+'.')
            for metric in ['pair_accuracy','strong_mean']:
                for reference,K in [('rotation',9),('random',199)]:
                    field=reference+'_'+metric+'_upper_rank'
                    assert all(int(float(r[reference+'_K']))==K for r in part)
                    winners=[r for r in part if float(r[field])<=1/(K+1)+1e-9]
                    put('public.'+frame+'.'+group+'.'+metric+'.'+reference+'_wins',len(winners),source,'Published labels, frame='+frame+', '+group+'; count '+field+' <= 1/(K+1) with K='+str(K)+'. Ties count against the published unit. This descriptive rank is distinct from BH test flags.')
            put('public.'+frame+'.'+group+'.same_amino_acid_pass',sum(r['same_AA_flag']=='True' for r in part),source,'Published labels, frame='+frame+', '+group+'; count same_AA_flag=True.')
            put('public.'+frame+'.'+group+'.strong_increment_pass',sum(r['strong_increment_flag']=='True' for r in part),source,'Published labels, frame='+frame+', '+group+'; count strong_increment_flag=True. This set differs from the untrained-reference rank winners.')
        for reference,K in [('rotation',9),('random',199)]:
            put('public.'+frame+'.'+reference+'_reference_count',K,source,'Common '+reference+'_K across every formally eligible published label in frame '+frame+'; checked against the source records.')
        put('public.'+frame+'.not_ahead_all_rotations',sum(float(r['rotation_pair_accuracy_upper_rank'])>.1+1e-9 for r in selected),source,'Frame '+frame+': labels that do not strictly beat all nine selected cross-block rotated directions on paired accuracy; descriptive complement, not a layer-localisation claim.')
    source='results/ctp_case/Table_S-U1.tsv'
    table=rows(source);released=next(r for r in table if r['Record ID']=='release')
    field='_Both-win and both-top-five fraction';release_values=[float(released[c+field]) for c in ['JDM','Wells']]
    assert len({round(v*100,1) for v in release_values})==1
    put('ctp.recovery.release',min(release_values),source,'Released model: both JDM and Wells donor-median conditional pair-recovery fractions independently round to the same one-decimal percentage. Use the minimum as a conservative numeric witness; do not pool their donors or denominators.')
    for group,prefix in [('untrained','untrained_'),('rotations','rotation_')]:
        rate=[float(r[c+field]) for r in table if r['Record ID'].startswith(prefix) for c in ['JDM','Wells']]
        for operation in ['min','max']:
            put('ctp.recovery.'+group+'.'+operation,{'min':min,'max':max}[operation](rate),source,'Across each '+group+' basis and each JDM/Wells cohort, '+operation+' of the donor-median both-win-and-both-top-five fractions; basis-specific conditional pair denominators remain separate.')
    source='results/artwork/main/mechanism.json'
    mechanism=json.loads((root/source).read_text(encoding='utf-8'))
    target=[r for r in mechanism['objects'] if int(r['concept_id'])==26]
    assert len(target)==1
    target=target[0]
    comparisons=target['dose_comparison']
    conditions=target['conditions']
    full=[r for r in comparisons if conditions[r['condition_index']]['scope']=='atoms' and conditions[r['condition_index']]['alpha']==1]
    selected_rows=[r for r in full if conditions[r['condition_index']]['name']=='selected_atom_full']
    control_rows=[r for r in full if conditions[r['condition_index']]['name'].startswith('control') and conditions[r['condition_index']]['name'].endswith('_atom_full')]
    assert len(selected_rows)==1 and target['n_positive_subset']==200
    selected=selected_rows[0]['positive']['mean'];controls=[r['positive']['mean'] for r in control_rows]
    assert len(controls)==5
    for key,value in [('selected',selected),('control_min',min(controls)),('control_max',max(controls))]:
        put('hydroxyl.positive_mean_dose.'+key,value,source,'Hydroxyl concept 26, neuron 293, full atom-scope mean clamp alpha=1, 200 concept-positive confirmation molecules; '+key+' of per-molecule RMS activation displacement times outgoing FFN-column norm. Negative molecules are a separate stratum.')
    assert {int(r['concept_id']) for r in mechanism['objects']}=={24,25,26}
    put('chemical_output_contrasts.selected_objects',len(mechanism['objects']),source,'Count prespecified concept-output objects 24,25,26; this is not a count of distinct neuron indices.')
    for name,original in {
        'public.T.all.pair_accuracy.non_rotation_wins':'public.T.not_ahead_all_rotations',
        'public.reference.rotation_K':'public.T.rotation_reference_count',
        'public.reference.random_K':'public.T.random_reference_count',
        'bases.dictionary_full.width_ratio_to_native':'bases.dictionary_to_native_width_ratio',
        'bases.dictionary_full.retained_ratio_to_native.min':'bases.dictionary_retained_to_native_ratio.min',
    }.items():out[name]=dict(out[original])
    source='results/ctp_case/Table_S-U1.tsv'
    release=next(r for r in rows(source) if r['Record ID']=='release')
    rendered={format(float(release[c+'_Both-win and both-top-five fraction'])*100,'.1f') for c in ['JDM','Wells']}
    assert len(rendered)==1
    put('ctp.recovery.release.common_one_decimal_display_fraction',float(next(iter(rendered)))/100,source,'Released JDM and Wells donor-median conditional pair fractions independently produce the same one-decimal percentage rendering; return that shared rendering divided by100, without pooling either denominator.')
    source='results/ctp_case/Table_S-U5.tsv'
    unused={r['Cohort'] for r in rows(source) if r['Cohort identity']=='Previously unused cohort'}
    assert len(unused)==1
    put('ctp.scored_cohorts.previously_unused_count',len(unused),source,'Distinct scored cohorts explicitly labelled Previously unused cohort; excludes metadata-screening candidates and old cohorts scored after locking.')
    return out


def bindings(root):
    path=Path(root)/'verify/application_claim_specs.json'
    if not path.exists():return {}
    calculated=values(root)
    specifications=json.loads(path.read_text(encoding='utf-8'))
    result={};legacy={}
    for row in specifications:
        if 'legacy_id' in row:
            owner=row['legacy_owner']
            if owner not in legacy:
                from importlib import import_module
                legacy[owner]=import_module(owner+'_claim_bindings').bindings(root)
            result[row['id']]=legacy[owner][row['legacy_id']]
        else:result[row['id']]=calculated[row['selector']]
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    calculated=values(Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(calculated,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scientific_selectors=len(calculated),scope='Archived application evidence')))
