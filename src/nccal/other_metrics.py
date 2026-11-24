"""Paper-level aggregates for observation counts, matching, perturbed molecules and exposure."""
from pathlib import Path
import csv,json
import numpy as np
def compute(root,metrics,reader):
    def tsv(p):
        with (root/p).open(encoding='utf-8',newline='') as s:return list(csv.DictReader(s,delimiter='\t'))
    for p in (root/'results/tables/trace_inputs').glob('*.json'):
        for key,v in json.loads(p.read_text()).items():metrics['trace.'+p.stem+'.'+key]=v
    edits=tsv('results/molecular/edits/primary_results.tsv')
    controlled=[r for r in edits if r['representation']=='controlled']
    edits=[r for r in controlled if r['edit_class'] in ('1','2','3','4')]
    focus=[r for r in edits if r['unit'] in ['293','454']]
    passes=lambda rows:sum(r['better_than_both']=='True' and r['unchanged_small']=='True' for r in rows)
    metrics.update({'edits.n_panel_cells':len(edits),'edits.n_focal_cells':len(focus),'edits.focal_pass':passes(focus),
                    'edits.panel_pass':passes(edits),'edits.focal_neurons':len(set(r['unit'] for r in focus)),
                    'edits.classes':len(set(r['edit_class'] for r in edits))})
    with np.load(root/'results/molecular/edits/raw_pairs.npz') as ar:
        if 'pair' in ar.files:metrics['edits.pairs']=len(np.unique(ar['pair']))
    if 'edits.pairs' not in metrics:
        grouped={}
        for r in controlled:grouped[r['edit_class']]=int(r['molecules'])
        metrics['edits.pairs']=sum(grouped.values())
    molecule=reader('layer_statistics')
    random=[float(v['VE_local_minus_radius0']) for v in molecule if v['model']=='chemberta_mtr' and v['network']!='trained']
    metrics['init.delta.low']=min(random);metrics['init.delta.high']=max(random);metrics['init.delta.networks']=len(random)
    for v in molecule:
        if v['network']=='trained':
            for col in ['n_atoms','n_molecules','width']:metrics[v['model']+'.panel.'+col]=int(v[col])
    cats=set(r['annotation_category'] for r in reader('published_subgroups'))
    panels=tsv('results/molecular/reconstruction/panel_all_coordinate_stratum_metrics.tsv')
    for model in ('chemberta_mtr','molformer'):
        controls=[float(r['R2']) for r in panels if r['model']==model and r['stratum']=='all_T2' and r['role']=='matched_control' and r['predictor']=='local_graph']
        metrics[model+'.control.mean_r2']=sum(controls)/len(controls)
        metrics[model+'.control.count']=len(controls)
    metrics['public.classes']=len(cats)
    stats=reader('basis_statistics');metrics['basis.all_rows']=len(stats)
    for r in reader('protein_losses'):
        metrics['exposure.'+r['model']+'.neuron']=int(r['axis'])
    base='results/protein/protein_exposure/exposure/results/35m/buried_exposed_group_errors.tsv'
    rs=tsv(base);metrics['exposure.pairs']=len(rs)
    for choice in ('sequence','augmented'):
        field=choice+'_probability'
        if field in rs[0]:
            correct=sum((float(r[field])>.5)==bool(int(r['later_residue_exposed'])) for r in rs)
            metrics['exposure.35m.'+choice+'.accuracy']=correct/len(rs)
    rec='results/protein/protein_reconstruction/neuron_814/results/'
    rs=tsv(rec+'reconstruction_groups_A_T.tsv')
    vals=lambda key:np.array([float(r[key]) for r in rs])
    den=vals('mean_z2').mean()-vals('mean_z').mean()**2
    for col in ('seq_mse','near_mse','full_mse'):metrics['814.R2.'+col]=1-vals(col).mean()/den
    rs=tsv(rec+'writeback_groups_T.tsv')
    for other in ('seq','swapped'):
        delta=vals(other+'_JSD')-vals('full_JSD')
        metrics['814.'+other+'.wins']=int((delta>0).sum())
        metrics['814.'+other+'.share10']=float(np.sort(delta)[-10:].sum()/delta.sum())
    metrics['814.writeback.inputs']=len(rs)*2
    return metrics
