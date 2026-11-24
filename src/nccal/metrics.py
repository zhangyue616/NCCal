"""Named estimands used for paragraph-level checks, from the specified study populations."""
from tables import read,build,ROOT
from collections import Counter
import csv,json,math,statistics
import numpy as np
def compute():
    t=build()
    out={}
    bs=read('basis_counts')
    for group,rows in {
        'native':[r for r in bs if r['basis']=='native'],
        'rotation':[r for r in bs if r['basis'].startswith('rot_')],
        'top':[r for r in bs if r['subset']!='full']}.items():
        for col in ['physical_n','R2_ge05_n','R2_ge08_n','canonical_usable_n','retained']:
            a=[int(r[col]) for r in rows]
            out[f'{group}.{col}.min']=min(a);out[f'{group}.{col}.max']=max(a)
        out[group+'.replicates']=len(rows)
    for r in read('two_neuron_metrics'):
        for k in ['estimate','ci95_lower','ci95_upper']:
            out['.'.join([r['kind'],r['coordinate'],r['condition'],k])]=float(r[k])
    for u in ['293','454']:
        out['output.'+u+'.reduction_pct']=100*(1-out['output.'+u+'.M2.estimate']/out['output.'+u+'.D_mean.estimate'])
    for r in read('edit_sites'):
        for k in ['molecules','changed_atoms','R2']:
            out['.'.join(['edit',r['unit'],r['edit_class'],k])]=float(r[k])
    for name,data in zip(['chem','bace','esol'],read('rewrite_status')):
        a=[r for r in data if r['canonical_usable']=='True']
        out[name+'.usable']=len(a);out[name+'.width']=len(data)
        for k in ['retained','failed','undetermined']:out[name+'.'+k]=sum(r['rewritten_status']==k for r in a)
    high=[r for r in read('atom_tracking') if r['canonical_usable']=='True' and float(r['Pearson'])>=.8]
    for k in ['retained','failed','undetermined']:out['tracking.'+k]=sum(r['rewritten_status']==k for r in high)
    out['tracking.total']=len(high)
    for r in read('atom_tracking'):
        if r['unit'] in ('293','454'):
            for k in ['canonical_R2','rewritten_R2','Pearson']:out['rewrite.'+r['unit']+'.'+k]=float(r[k])
    for row in t['SI8']:
        for name,v in zip(['eligible','same','strong'],row[1:]):
            out[('public.T.' if row[0]=='Confirmation' else 'public.NEW.')+name]=int(v)
    ps=read('published_subgroups')
    for cat in {r['annotation_category'] for r in ps}:
        rr=[r for r in ps if r['annotation_category']==cat]
        out['public.'+cat+'.eligible']=len(rr)
        out['public.'+cat+'.same']=sum(r['same_AA_flag']=='True' for r in rr)
    rr=[r for r in ps if r['auc_mean'] and float(r['auc_mean'])>=.7]
    out['public.auroc70.eligible']=len(rr)
    out['public.auroc70.same']=sum(r['same_AA_flag']=='True' for r in rr)
    for r in read('e1_heldout'):
        prefix='e1.'+r['unit0']+'.'
        for k in ['groups','strong_mean','strong_CI_lo','strong_CI_hi','random_K','rotation_K','pair_accuracy']:
            out[prefix+k]=float(r[k])
        out[prefix+'correct']=round(float(r['groups'])*float(r['pair_accuracy']))
        out[prefix+'block']=int(r['block0'])+1
    out['e1.count']=len(read('e1_heldout'))
    for r in read('protein_losses'):
        out['exposure.'+r['model']+'.correct']=int(r['pair_correct'])
    for i,v in enumerate(read('transfer_statistics')):
        for k,val in zip(['point','lo','hi'],v[1:4]):out['transfer.'+['preference','readability'][i]+'.'+k]=float(val)
    cal=read('calibration_statistics')
    out['calibration.total']=len(cal)
    out['calibration.passed']=sum(a[2] and b[2] for _,(a,b) in cal)
    pilot=json.loads((ROOT/'results/artwork/main/pilot.json').read_text(encoding='utf-8'))
    for k in ['native','initialization_mean']:out['pilot.chemeleon.'+k]=pilot['pilot']['chemeleon'][k]
    for r in read('layer_statistics'):
        if r['model']=='chemberta_mtr' and r['network']=='trained':out['layer.mtr.increment']=float(r['VE_local_minus_radius0'])
    art=ROOT/'results/artwork/main'
    artifact=lambda n:json.loads((art/(n+'.json')).read_text(encoding='utf-8'))
    cp=artifact('calibration')['panel_CD_frozen_source']['source_records']
    for scope,k1,k2 in [('raw','difference_Btrained_minus_BinitMedian','Gconcept'),('scaffold','Theta_point','G_init_point')]:
        rr=cp[scope]
        for label,k in [('axis',k1),('read',k2)]:
            vals=[float(r[k]) for r in rr if r[k] not in ('',None)]
            out[scope+'.'+label+'.positive']=sum(x>0 for x in vals)
            out[scope+'.'+label+'.average']=statistics.mean(vals)
            out[scope+'.'+label+'.count']=len(vals)
    for model,crit in cal:
        for j,(logfold,p,passed) in enumerate(crit):
            out['calib.'+model+'.'+str(j+1)+'.fold']=math.exp(logfold)
    coords=artifact('reconstruction_panel')['coordinates']
    win=total=0
    for model in ('chemberta_mtr','molformer'):
        for pred in ('Local graph','Binary concept means'):
            indices={a['coordinate']:a['point'] for a in coords if a['model']==model and a['role']=='selected' and a['predictor']==pred}
            out['coord.'+model+'.'+pred]=statistics.mean(indices.values())
            if pred=='Local graph':local=indices
            else:
                total+=len(indices);win+=sum(local[k]>indices[k] for k in indices)
        for role in ['selected','control']:
            rows=[a for a in coords if a['model']==model and a['role']==role and a['predictor']=='Local graph']
            if rows:out[model+'.'+role+'.mean_r2']=statistics.mean(a['point'] for a in rows)
    out['coord.wins']=win;out['coord.total']=total
    tasks=artifact('task_output_panel')
    for a in tasks['absolute']:
        if a['scope']=='selected/joint':out['task.'+a['model']+'.'+a['task']+'.'+a['condition']]=a['point']
    out['task.family']=len(tasks['paired'])
    out['task.significant']=sum(a['Holm12_p']<=.05 for a in tasks['paired'])
    counts=read('atom_counts')
    for frame,lens in counts.items():
        out['anchors.'+frame+'.molecules']=len(lens)
        out['anchors.'+frame+'.atoms']=sum(lens)
    with (ROOT/'data/molecule_ids.tsv').open(encoding='utf-8') as s:
        cohorts=Counter(r['cohort'] for r in csv.DictReader(s,delimiter='\t'))
    for n,v in cohorts.items():out['cohort.'+n]=v
    bins=read('reference_pair_counts')
    v=[int(a['pair_correct']) for a in bins if a['model']=='35m']
    out['exposure.35m.untrained.low']=min(v);out['exposure.35m.untrained.high']=max(v);out['exposure.35m.untrained.size']=len(v)
    with np.load(ROOT/'results/tables/alanine_pairs.npz',allow_pickle=False) as a:
        signed,delta=a['strand_minus_helix'],a['loss_reduction']
        for name,mask in [('higher',signed>0),('lower',signed<0)]:
            out['1246.'+name]=int(np.sum(mask))
            out['1246.'+name+'.improves']=int(np.sum(mask & (delta>0)))
            out['1246.'+name+'.worsens']=int(np.sum(mask & (delta<0)))
        out['1246.n']=len(signed)
    from other_metrics import compute as other_metrics
    from design_counts import design
    out.update(design(ROOT))
    return other_metrics(ROOT,out,read)

