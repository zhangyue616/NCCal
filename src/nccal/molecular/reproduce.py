"""Recalculate archived molecular results or run an offline real-model subset."""
from pathlib import Path
import argparse,csv,json,math,hashlib
import numpy as np
from .metrics import rank_test,transfer,pooled_r2,census_from_molecules,holm

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def compare(name,actual,expected,tolerance=1e-9):
    if isinstance(actual,(bool,str,int)) and isinstance(expected,type(actual)):ok=actual==expected
    else:ok=bool(np.allclose(actual,expected,rtol=0,atol=tolerance,equal_nan=True))
    return dict(name=name,actual=actual,expected=expected,tolerance=tolerance,passed=ok)

def calibration(root):
    d=root/'results/molecular/calibration';out=[]
    for expected in load(d/'expected.json'):
        z=np.load(d/(expected['model']+'.npz'));s=z['native'];r=z['rotated']
        a=rank_test(np.log(s));delta=np.log(s)-np.median(np.log(r),axis=1);b=rank_test(delta)
        for key,value in [('readability_fold',a['fold']),('axis_fold',b['fold']),('readability_p',a['p']),('axis_p',b['p'])]:out.append(compare(expected['model']+'/'+key,value,expected[key],1e-10))
    z=np.load(d/'transfer_profiles.npz');point=transfer(z['point_ell']);rep=transfer(z['replicated_ell']);expected=load(d/'transfer_expected.json')
    for key in point:
        out.append(compare('transfer/'+key,float(point[key]),expected[key]['point'],1e-10));out.append(compare('transfer/'+key+'/interval',np.quantile(rep[key],[.025,.975]).tolist(),expected[key]['ci95'],1e-10))
    z=np.load(d/'molformer_depth.npz');s=z['native'];delta=np.log(s)-np.median(np.log(z['rotated']),axis=2);axis=np.exp(delta);expected=load(d/'molformer_expected.json')
    def statistics(v):return np.stack([((v<x).sum(0)+.5*((v==x).sum(0)-1))/(len(v)-1) for x in v]).mean(1)
    t1=np.stack([statistics(s[:,layer]) for layer in range(12)],axis=1);t2=np.stack([statistics(axis[:,layer]) for layer in range(12)],axis=1);selected=11-np.argmax(t1[:,::-1],axis=1);free=11-np.argmax(t2[:,::-1],axis=1);i=np.arange(len(s));one=t1[i,selected];two=t2[i,selected];three=t2[i,free];target=expected['primary'];layer=int(selected[0])
    out.extend([compare('MoLFormer/selected_layer_zero_based',layer,int(target['layer'][1:])),compare('MoLFormer/J1/p',float(np.mean(one>=one[0])),target['J1']['p']),compare('MoLFormer/J2/p',float(np.mean(two>=two[0])),target['J2']['p']),compare('MoLFormer/J2/free_p',float(np.mean(three>=three[0])),target['J2']['p_free'])])
    for key,values in [('J1',np.log(s)),('J2',delta)]:out.append(compare('MoLFormer/'+key+'/selected_fold',float(np.exp(np.median(values[0,layer]-np.median(values[1:,layer],axis=0)))),target[key]['fold'],1e-10))
    for layer in range(12):
        target=expected['depth_profile_descriptive']['L'+str(layer)]
        for key,values in [('J1',np.log(s[:,layer])),('J2',delta[:,layer])]:
            summary=rank_test(values);out.extend([compare('MoLFormer/L'+str(layer)+'/'+key+'_fold',summary['fold'],target[key+'_fold'],1e-10),compare('MoLFormer/L'+str(layer)+'/'+key+'_p',summary['p'],target[key+'_p'],1e-12)])
    return out

def reconstruction(root):
    d=root/'results/molecular/reconstruction';out=[];z=np.load(d/'focal_activation_predictions.npz');y=z['actual'].astype(float);p=z['predictions'];expected=read(d/'activation_metrics.tsv')
    coordinate_ids=z['coordinate_ids'];model_names=z['model_names'].astype(str)
    molecule_index=z['mol_idx'];labels=z['target_labels'];positive=np.stack([np.bincount(molecule_index,weights=labels[:,i],minlength=10000)>0 for i in range(2)],axis=1)
    def molecule_mask(stratum,u):
        return positive[:,u] if stratum=='any_target_positive_molecule' else ~positive[:,u] if stratum=='all_target_negative_molecule' else np.ones(10000,bool)
    for r in expected:
        unit=int(r['coordinate']);u=int(np.flatnonzero(coordinate_ids==unit)[0]);condition=r['condition']
        aliases={'D_mean':0,'M1':1,'M2':2};k=aliases.get(condition)
        if k is None:continue
        stratum=r['stratum'];mask=molecule_mask(stratum,u)[molecule_index]
        if stratum.startswith('target_atom_label_'):mask=labels[:,u]==int(stratum[-1])
        if stratum.startswith('D_fixed_bin_'):mask=z['bins'][:,u]==int(stratum[-1])
        if not mask.any():continue
        if r['metric']=='pooled_atom_mse':actual=float(np.mean((y[mask,u]-p[mask,u,k])**2))
        elif 'r2' in r['metric'].lower():
            values=y[mask,u];sst=np.sum((values-values.mean())**2,dtype=np.float64);sse=np.sum((p[mask,u,k]-values)**2,dtype=np.float64);actual=float(1-sse/sst) if sst>0 else float('nan')
        else:continue
        expected_value=float(r['estimate']) if r['estimate'] else float('nan')
        # Nearly constant bins have huge negative R-squared. Preserve these
        # adverse values and allow only eight floating-point rounding units;
        # ordinary-scale scientific estimates retain the absolute tolerance.
        tolerance=max(1e-9,8*float(np.spacing(abs(expected_value)))) if np.isfinite(expected_value) else 1e-9
        record=compare(str(unit)+'/'+stratum+'/'+condition+'/'+r['metric'],actual,expected_value,tolerance)
        if tolerance>1e-9:record['rounding_rule']='At most eight float64 ULPs for an ill-conditioned negative R-squared; value not truncated.'
        out.append(record)
    values=np.load(d/'monitored_output_deltas.npz');delta=values['delta'];rms=values['rms'];endpoints=values['endpoints']
    for r in read(d/'output_metrics.tsv'):
        k={'D_mean':0,'M1':1,'M2':2,'M1_delta_perm':3,'M2_delta_perm':4}[r['condition']];u=int(np.flatnonzero(coordinate_ids==int(r['coordinate']))[0]);endpoint=int(np.flatnonzero(endpoints==int(r['endpoint']))[0]);mask=molecule_mask(r['stratum'],u);v=delta[u,k,mask,endpoint];metric=r['metric']
        if metric=='mae':actual=np.abs(v).mean()
        elif metric=='signed_mean_shift':actual=v.mean()
        elif metric=='mean_normalized199_rms':actual=rms[u,k,mask].mean()
        elif metric=='absolute_error_p95':actual=np.quantile(np.abs(v),.95)
        elif metric=='paired_mae_minus_D_mean':actual=(np.abs(v)-np.abs(delta[u,0,mask,endpoint])).mean()
        elif metric=='paired_mae_minus_own_delta_perm':actual=(np.abs(v)-np.abs(delta[u,{1:3,2:4}[k],mask,endpoint])).mean()
        else:continue
        out.append(compare('output/'+str(u)+'/'+r['stratum']+'/'+r['condition']+'/'+str(r['endpoint'])+'/'+metric,float(actual),float(r['estimate']),1e-9))
    panel=read(d/'panel_units.tsv')
    for model in ['chemberta_mtr','molformer']:
        selected=[r for r in panel if r['model']==model and r['role']=='selected']
        for key,value in [('R2_local',.603 if model=='chemberta_mtr' else .487),('R2_binary',.474 if model=='chemberta_mtr' else .242)]:out.append(compare(model+'/'+key+'/three_decimal_mean',round(float(np.mean([float(r[key]) for r in selected])),3),value,1e-12))
    out.append(compare('panel/local_better_than_binary_count',sum(float(r['R2_local'])>float(r['R2_binary']) for r in panel if r['role']=='selected'),15))
    z=np.load(d/'label_lookup_sufficient.npz');loss=z['squared_errors'].sum(0)/z['atom_counts'].sum();expected=load(d/'label_lookup_expected.json')
    out.extend([compare('label_lookup/per_concept_Brier',loss.tolist(),expected['Brier_per_concept_reference'],1e-12),compare('label_lookup/panel_loss_reduction',float(np.mean(loss[:,2]-loss[:,3])),expected['Delta_L_panel'],1e-12),compare('label_lookup/panel_interval',np.quantile(z['bootstrap'],[.025,.975]).tolist(),expected['panel_ci95']['ci'],1e-12),compare('label_lookup/relative_percent_display',round(float(100*(1-loss[:,3].mean()/loss[:,2].mean())),2),5.34,1e-12)])
    expected=load(d/'joint_expected.json');draws=np.load(d/'joint_bootstrap_counts.npz');pvalues=[]
    for row in expected['primary']:
        job=row['model']+'_'+row['task'];z=np.load(d/('joint_'+job+'.npz'));meta=load(d/('joint_'+job+'_metadata.json'));scope=[q['id'] for q in meta['scopes']].index('selected/joint');prediction=z['prediction'][:,scope];errors=abs(prediction-prediction[:,0,None]);local=errors[:,4];reference=errors[:,{'development_mean':2,'binary_mean':3}[row['reference']]] if row['reference']!='residual_permutation' else errors[:,5:].mean(1);effects=reference-local;point=float(effects.mean());weights=draws[row['task']+'_cluster_counts'][:,draws[row['task']+'_molecule_cluster_index']];rep=weights@effects/weights.sum(1);ci=np.quantile(rep,[.025,.975]);p=float((1+np.sum(abs(rep-point)>=abs(point)))/(len(rep)+1));pvalues.append(p)
        out.extend([compare(job+'/'+row['reference']+'/joint_effect',point,row['effect_reference_error_minus_local_error'],2e-12),compare(job+'/'+row['reference']+'/joint_interval',ci.tolist(),row['CI95_percentile'],2e-12),compare(job+'/'+row['reference']+'/joint_p',p,row['approximate_centered_bootstrap_p_two_sided_plus_one'],1e-12)])
    adjusted=holm(pvalues)
    for row,q in zip(expected['primary'],adjusted):out.append(compare(row['model']+'/'+row['task']+'/'+row['reference']+'/Holm12',float(q),row['Holm_adjusted_p_12'],1e-12))
    out.append(compare('joint/nine_of_twelve_adjusted_comparisons',int((adjusted<.05).sum()),9))
    jobs={}
    for row in load(d/'joint_fidelity_expected.json'):
        job=row['model']+'_'+row['task']
        if job not in jobs:
            z=np.load(d/('joint_'+job+'.npz'));meta=load(d/('joint_'+job+'_metadata.json'));jobs[job]=(z['prediction'],meta)
        prediction,meta=jobs[job];scope=[q['id'] for q in meta['scopes']].index(row['scope']);errors=abs(prediction[:,scope]-prediction[:,scope,0,None]);value=errors[:,5:].mean() if row['condition']=='permutation_summary' else errors[:,meta['conditions'].index(row['condition'])].mean();out.append(compare(job+'/'+row['scope']+'/'+row['condition']+'/output_deviation',float(value),row['expected'],2e-12))
    return out

def whole_layer(root):
    d=root/'results/molecular/whole_layer';out=[];curves=read(d/'curve_by_order.tsv')
    for path in sorted(d.glob('*_conditions.json')):
        job=path.stem.removesuffix('_conditions');model,task=job.rsplit('_',1);z=np.load(d/(job+'.npz'));pred=z['predictions'];performance=np.load(d/(job+'_performance.npz'));conditions=load(path)['conditions'];by={(x['order'],x['k'],x['condition']):x['column'] for x in conditions}
        for row in curves:
            if row['model']!=model or row['task']!=task:continue
            columns=json.loads(row['computed_columns']);deviations=np.abs(pred[:,columns]-pred[:,0,None]);actual=float(deviations.mean())
            out.append(compare(job+'/'+row['order']+'/'+row['k']+'/'+row['method']+'/deviation',actual,float(row['mean_absolute_output_deviation']),2e-12))
            j=performance['row_keys'].tolist().index(row['order']+'/'+row['k']+'/'+row['method'])
            for key in ['AUROC','AUPRC','Brier','RMSE','MAE']:
                if not row[key]:continue
                n=int(performance['n'])
                if key in ['AUROC','AUPRC']:
                    start,end=performance['offsets'][j:j+2];score,positive,negative=performance['bins'][start:end].T
                    if key=='AUROC':value=float(np.sum(positive*(np.cumsum(negative)-.5*negative))/(positive.sum()*negative.sum()))
                    else:
                        pos=positive[::-1];neg=negative[::-1];value=float(np.sum(pos/pos.sum()*np.cumsum(pos)/np.cumsum(pos+neg)))
                elif key=='Brier':value=float(performance['squared_error_sum'][j]/n)
                elif key=='RMSE':value=float(np.sqrt(performance['squared_error_sum'][j]/n))
                else:value=float(performance['absolute_error_sum'][j]/n)
                out.append(compare(job+'/'+row['order']+'/'+row['k']+'/'+row['method']+'/'+key,value,float(row[key]),3e-12))
    return out

def dictionaries(root):
    d=root/'results/molecular/dictionary';rows=read(d/'per_concept_per_network_scores.tsv');expected=read(d/'all_recipe_estimates.tsv');out=[]
    for target in expected:
        checkpoint=target['checkpoint'];recipe=target['recipe'];twins=sorted({r['network'] for r in rows if r['recipe']==recipe and r['network'].startswith('init')},key=lambda x:int(x[4:]));networks=[checkpoint]+twins
        data={n:sorted([r for r in rows if r['network']==n and r['recipe']==recipe],key=lambda r:int(r['concept_id'])) for n in networks}
        matrix=np.log([[float(r['best_dictionary_lift']) for r in data[n]] for n in networks])
        if target['test']=='T2':matrix-=np.log([[float(r['best_native_lift']) for r in data[n]] for n in networks])
        value=rank_test(matrix,target['test']=='T2')
        out.extend([compare(checkpoint+'/'+target['test']+'/fold',value['fold'],float(target['calibrated_fold']),1e-10),compare(checkpoint+'/'+target['test']+'/p',value['p'],float(target['p']),1e-12)])
        differences=matrix[0]-np.median(matrix[1:],axis=0);rng=np.random.default_rng(int(target['bootstrap_seed']));indices=rng.integers(0,len(differences),size=(2000,len(differences)));ci=np.exp(np.percentile(np.median(differences[indices],axis=1),[2.5,97.5]));out.append(compare(checkpoint+'/'+recipe+'/'+target['test']+'/interval',ci.tolist(),json.loads(target['CI95_concept_bootstrap']),1e-10))
    return out

def rewrites(root):
    d=root/'results/molecular/rewrites';out=[]
    for name in ['chemberta','molformer_bace','molformer_esol']:
        original=read(d/(name+'_census.tsv'));z=np.load(d/(name+'_per_molecule.npz'));r2=np.array([float(r['canonical_R2']) for r in original]);mean,ci,usable,status=census_from_molecules(z['advantages'],r2)
        for i,r in enumerate(original):
            out.extend([compare(name+'/'+r['unit']+'/usable',bool(usable[i]),r['canonical_usable']=='True'),compare(name+'/'+r['unit']+'/status',str(status[i]),r['rewritten_status'])])
            for g,prefix in enumerate(['canonical','rewritten']):
                for k,c in enumerate(['mean','shuffled']):
                    stem=prefix+'_'+c
                    for suffix,value in [('gain',mean[g,i,k]),('low',ci[0,g,i,k]),('high',ci[1,g,i,k])]:out.append(compare(name+'/'+r['unit']+'/'+stem+'_'+suffix,float(value),float(r[stem+'_'+suffix]),3e-12))
    return out

def edits(root):
    d=root/'results/molecular/edits';z=np.load(d/'raw_pairs.npz');offsets=z['offsets'];changed=z['changed'];expected=read(d/'per_unit_results.tsv');out=[]
    for representation,arr in [('controlled','controlled_product'),('canonical','canonical_product')]:
        actual=z[arr].astype(float)-z['source'].astype(float);prediction=z['predicted_delta']
        for kind in range(1,6):
            ids=np.flatnonzero((z['edit_classes']==kind)&(z['main_pass'] if representation=='controlled' else True)).tolist();ids.sort(key=lambda i:hashlib.sha256(('20261123/edit-fold/'+str(z['cids'][i])).encode()).hexdigest());ys=[];ps=[]
            for i in ids:
                sl=slice(offsets[i],offsets[i+1]);mask=changed[sl];ys.append(actual[sl][mask]);ps.append(prediction[sl][mask])
            # Preserve the frozen summation order and moment arithmetic, including
            # the ill-conditioned, very negative values of nearly constant units.
            am=np.asarray([a.mean(0) for a in ys]);a2=np.asarray([(a*a).mean(0) for a in ys]);mse=np.asarray([((a-p)**2).mean(0) for a,p in zip(ys,ps)]);var=a2.mean(0)-am.mean(0)**2
            values=1-mse.mean(0)/np.where(var>0,var,np.nan)
            for row in expected:
                if row['representation']==representation and int(row['edit_class'])==kind:
                    u=int(row['unit']);expected_value=float(row['R2']) if row['R2'] else float('nan');out.append(compare(representation+'/'+str(kind)+'/'+str(u)+'/R2',float(values[u]),expected_value,5e-8))
    return out

def bases(root):
    d=root/'results/molecular/bases';expected=read(d/'basis_summary.tsv');out=[]
    for row in expected:
        name=row['basis'];units=read(d/(name+'_units.tsv'))
        if row['subset']!='full':
            selected=set(np.load(d/(name+'_selection.npz'))['top_464'].tolist());units=[r for r in units if int(r['unit']) in selected]
        r2=np.asarray([float(r['canonical_R2']) for r in units]);usable=[r for r in units if r['canonical_usable']=='True']
        computed=dict(physical_n=len(units),finite_R2=int(np.isfinite(r2).sum()),R2_ge05_n=int((r2>=.5).sum()),R2_ge08_n=int((r2>=.8).sum()),median_R2=float(np.nanmedian(r2)),canonical_usable_n=len(usable))
        computed.update({s:sum(r['rewritten_status']==s for r in usable) for s in ['retained','failed','undetermined']})
        for key,value in computed.items():out.append(compare(name+'/'+row['subset']+'/'+key,value,float(row[key]),1e-10))
    return out

FUNCTIONS={'01_molecular_calibration':calibration,'02_molecular_reconstruction':reconstruction,'03_whole_layer':whole_layer,'04_sparse_dictionary':dictionaries,'05_molecular_rewrites':rewrites,'06_molecular_edits':edits,'07_basis_comparison':bases}
def main(experiment=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['verify','smoke'],default='verify');parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--model-dir',type=Path);parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path);parser.add_argument('--experiment',choices=list(FUNCTIONS),default=experiment)
    args=parser.parse_args()
    if not args.experiment:parser.error('--experiment is required')
    if args.mode=='verify':checks=FUNCTIONS[args.experiment](args.root)
    else:
        if not args.model_dir:parser.error('--model-dir is required for a real-model subset')
        from .smoke import run
        checks=run(args.root,args.experiment,args.model_dir,args.device)
    report=dict(experiment=args.experiment,mode=args.mode,checks=len(checks),failed=sum(not r['passed'] for r in checks),results=checks,status='PASS' if checks and all(r['passed'] for r in checks) else 'FAIL')
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    if report['status']!='PASS':
        print(json.dumps([r for r in checks if not r['passed']][:20],indent=2));raise SystemExit(1)
if __name__=='__main__':main()
