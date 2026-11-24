"""Count frozen design dimensions straight from study assets."""
import csv,json,gzip
from collections import defaultdict
import numpy as np
def design(root):
    a={};read=lambda rel:json.loads((root/rel).read_text(encoding='utf-8'))
    cm=read('results/artwork/main/concept_protein.json')
    for k,frame in [('D','protein.dev'),('T','protein.test')]:
        row=cm[k]
        for col in ['pairs','groups']:a[frame+'.'+col]=row[col]
        a[frame+'.min_len'],a[frame+'.max_len']=row['length_aa']
    a['label.panel']=len(cm['concept_rows']);a['label.neurons']=len({r['selected_coordinate'] for r in cm['concept_rows']})
    aa=read('results/artwork/main/activation_output.json')
    one=next(i for i in aa['C_saved_atom_values']['molecules'] if i['molecule']['coordinate']==293)['molecule']
    mol=next(m for m in read('data/artwork/molecules/fig4_mol2d.json')['molecules'] if m['smiles']==one['smiles'])
    elements={at['index']:at['element'] for at in mol['atoms']};positive={k for k,z in zip(one['graph_atom_ids'],one['observed']) if z>0}
    carbons={k for k,item in elements.items() if item=='C'}
    a['illustration.carbons']=len(carbons)
    a['illustration.hydroxyl_carbons']=len(carbons.intersection(one['target_graph_atom_ids']))
    a['illustration.other_nonpositive_carbons']=len(carbons.difference(positive).difference(one['target_graph_atom_ids']))
    focal=aa['historical_contrast_rows']
    a['clamp.objects']=len(focal)
    mechanism=read('results/artwork/main/mechanism.json')
    for row in mechanism['objects']:
        obj=row['D_object']
        for label,field in [('unit','unit'),('output','anonymous_output_q'),('concept','concept_id')]:
            a['clamp.'+str(obj['concept_id'])+'.'+label]=obj[field]
        a['clamp.'+str(obj['concept_id'])+'.controls']=len(obj['control_units'])
        a['clamp.'+str(obj['concept_id'])+'.positive']=row['n_positive_subset']
        a['clamp.'+str(obj['concept_id'])+'.negative']=row['n_negative_subset']
    dose=read('results/artwork/main/paired_dose.json')
    a['clamp.bootstrap']=dose['bootstrap_reused']['replicates']
    for ix,value in enumerate(dose['objects'][0]['scales']):a['dose.scale.'+str(ix)]=value
    for r in focal:a['clamp.'+str(r['concept_id'])+'.confidence']=r['original_ci_percent']
    fdir=root/'results/molecular/calibration'
    expectation=read('results/molecular/calibration/expected.json')
    a['calibration.family']=len(expectation);a['calibration.graphs']=sum(r['model'].startswith(('hu_','chemeleon','molclr')) for r in expectation)
    a['calibration.hu']=sum(r['model'].startswith('hu_') for r in expectation)
    a['calibration.chemberta']=sum(r['model'].startswith('chemberta_') for r in expectation)
    a['calibration.criteria']=len([k for k in expectation[0] if k.endswith('_p')])
    with np.load(fdir/'molformer_depth.npz') as g:
        a['calibration.networks'],a['calibration.layers'],a['calibration.rotations'],a['calibration.concepts']=g['rotated'].shape
        a['calibration.random_networks']=a['calibration.networks']-1
    with np.load(fdir/'transfer_profiles.npz') as g:
        a['transfer.axes']=g['point_ell'].shape[1]-1
        a['transfer.concepts']=len(g['concept_ids']);a['transfer.others']=len(g['point_ell'])-1
        a['transfer.bootstrap']=len(g['replicated_ell'])
    a['calibration.bonferroni']=.05/a['calibration.family']
    s=read('results/molecular/reconstruction/label_lookup_expected.json')
    with np.load(root/'results/molecular/reconstruction/label_lookup_sufficient.npz') as g:
        losses=g['squared_errors'].sum(axis=0)/g['atom_counts'].sum()
        a['label.lookup_relative_reduction']=float((losses[:,2]-losses[:,3]).mean()/losses[:,2].mean())
    with np.load(root/'results/molecular/edits/raw_pairs.npz') as g:
        a['edits.n_main_pairs']=int(g['main_pass'].sum())
        a['frozen_output.width']=g['baseline_outputs'].shape[-1]
    p=read('results/tables/model_study_design.json')
    a['study.model_rows']=len(p)
    ids=defaultdict(set)
    with gzip.open(root/'data/molecular_cohort_membership.tsv.gz','rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            for key in ('cid','canonical_identity_sha256'):
                ids[row['cohort'],key].add(row[key])
    for (cohort,key),values in ids.items():a['molecules.'+cohort+'.'+key]=len(values)
    for left,right in [('calibration','development'),('calibration','confirmation'),('development','confirmation')]:
        for key in ('cid','canonical_identity_sha256'):a['overlap.'+left+'.'+right+'.'+key]=len(ids[left,key]&ids[right,key])
    for label in ['8m','35m']:
        x=read('results/protein/protein_exposure/exposure/results/'+label+'/results.json')
        for key in ['rows','groups','axis_candidates','axis_zero_based']:a['protein.'+label+'.development.'+key]=x['selection'][key]
        for key in ['D_groups','T_groups']:a['protein.'+label+'.pairs.'+key]=x['buried_exposed_pairs'][key]
    src=root/'results/protein/protein_exposure/exposure/results/35m'
    evalgroup=src/'buried_exposed_group_errors.tsv'
    with evalgroup.open(encoding='utf-8',newline='') as f: a['exposure.pair_count']=sum(1 for _ in csv.DictReader(f,delimiter='\t'))
    return a
