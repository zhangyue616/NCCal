"""Thin native-unit extraction; random-network full activations stay in RAM."""
from prepare import *
from inventory import jl,table
from model_runtime import Runtime,batches
import numpy as np,torch,time
from scipy.stats import rankdata
from collections import defaultdict
_GATE_CHECKED=None
def auc(p,n):
    a=np.r_[p,n];m=len(p);k=len(n)
    return float((rankdata(a,method='average')[:m].sum()-m*(m+1)/2)/(m*k))
def verify_gate():
    global _GATE_CHECKED
    gate=js(OUT.parent/'COMPUTE_GO.json');assert gate['status']=='GO' and gate['random_networks']==199
    freeze=js(OUT/'INPUT_FREEZE.json');assert freeze['status']=='FROZEN_BEFORE_EVALUATION'
    current=sha(OUT/'INPUT_FREEZE.json')
    if _GATE_CHECKED==current:return freeze
    for rel,expected in freeze['input_files'].items():
        if sha(OUT/rel)!=expected:raise ValueError('Input drift '+rel)
    _GATE_CHECKED=current
    return freeze
def metadata():
    records=jl(OUT/'inputs/sequences.jsonl');pairs=table(OUT/'inputs/pairs.tsv');labels=js(OUT/'inputs/LABELS.json');return records,pairs,labels
def save_tables(folder,units,records,pairs,labels,values,index,reference):
    fields=['unit_id','concept','group','accession','split','early0','late0','target','raw_late_minus_early','oriented_late_minus_early','reference','reference_id','source','model','block0','unit0','pole','family','original_family','sign','masked','readpoint','own_fresh_eligible']
    by_concept=defaultdict(list)
    for p in pairs:by_concept[p['concept']].append(p)
    recordmap={r['accession']:r for r in records};pairtemp=folder/'pairs.tsv.partial';aucrows=[]
    with pairtemp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
        for col,u in enumerate(units):
            for p in by_concept[u['concept']]:
                a=p['accession']
                if a not in index:continue
                offset=index[a]['offset'];early=int(p['early0']);late=int(p['late0']);diff=float(values[offset+late,col])-float(values[offset+early,col]);sign=float(u.get('sign',1))
                row={k:p[k] for k in ['concept','group','accession','split','early0','late0','target']};row.update({k:u[k] for k in ['unit_id','source','model','block0','unit0','pole']});row.update(raw_late_minus_early=diff/sign,oriented_late_minus_early=diff,reference=reference,reference_id=folder.name,family=u['family'],original_family=u.get('original_family',u['family']),sign=sign,masked=u.get('masked',False),readpoint=u.get('readpoint',u['family']),own_fresh_eligible=recordmap[a].get('own_fresh_eligible',True));w.writerow(row)
            if reference=='trained':
                for r in records:
                    a=r['accession'];lab=labels.get(a,{}).get(u['concept'])
                    if not lab or a not in index:continue
                    pos=lab['positive0'];neg=lab.get('negative0')
                    if neg is None:neg=sorted(set(range(len(r['sequence'])))-set(pos)-set(lab.get('unknown0',[])))
                    if not pos or not neg:continue
                    offset=index[a]['offset'];pp=values[offset+np.array(pos),col];nn=values[offset+np.array(neg),col]
                    if not np.isfinite(pp).all() or not np.isfinite(nn).all():continue
                    aucrows.append(dict(unit_id=u['unit_id'],concept=u['concept'],group=r['group'],accession=a,split=r['split'],auroc=auc(pp,nn),n_positive=len(pos),n_unannotated=len(neg),n_negative_reference=len(neg),negative_reference_meaning=lab.get('negative_interpretation','unannotated reference'),auc_scope='all_eligible_Ala_Helix_vs_Beta_centres_individually_masked' if u.get('masked') else 'all_eligible_observed_or_annotated_domain_residues' if u['concept']!='OWN_HELIX_START' else 'fixed_same_helix_pair_descriptive_only',full_protein_residue_auc=False if u.get('masked') or u['concept']=='OWN_HELIX_START' else True,own_fresh_eligible=r.get('own_fresh_eligible',True)))
    pairtemp.replace(folder/'pairs.tsv')
    if aucrows:
        with (folder/'auc.tsv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(aucrows[0]),delimiter='\t');w.writeheader();w.writerows(aucrows)
def extract_values(runtime,units,records,folder):
    folder.mkdir(parents=True,exist_ok=True);index={};offset=0
    for r in records:index[r['accession']]={'offset':offset,'length':len(r['sequence']),'group':r['group'],'split':r['split']};offset+=len(r['sequence'])
    value_dtype=np.float64 if any(u['family'].startswith('rotation_') for u in units) else np.float32
    values=np.lib.format.open_memmap(folder/'values.npy',mode='w+',dtype=value_dtype,shape=(offset,len(units)));values[:]=np.nan
    byblock=defaultdict(list)
    for col,u in enumerate(units):byblock[int(u['block0'])].append((col,u))
    active=[];checks=[]
    def capture(block,z):
        cols=[col for col,u in byblock[block]];thin=torch.stack([runtime.read_unit(z,u) for col,u in byblock[block]],dim=-1).detach().cpu().numpy()
        for i,r in enumerate(active):
            ix=index[r['accession']];values[ix['offset']:ix['offset']+ix['length'],cols]=thin[i,runtime.prefix:runtime.prefix+ix['length'],:]
        if not checks:
            with torch.inference_mode():
                probe=z[0,0].detach().float();test=byblock[block][0][1];j=int(test['unit0'])
                if test['family'].startswith('rotation_'):
                    from references import rotate,rotation_spec
                    base=probe if test['original_family']=='pre_pole' else runtime.raw_product(probe);array=base.detach().cpu().numpy()
                    value=rotate(array[None,:],rotation_spec(runtime.width,test['rotation_seed']))[0,j]*test['rotation_sign'];value=max(0.,value) if test['family']=='rotation_pole' else value
                    expected=torch.tensor(value,device=z.device)
                elif test['family']=='pre_pole':expected=torch.relu(probe[j] if test['pole']=='positive' else -probe[j])
                elif test['family']=='gate_tagged':
                    g,u=probe.chunk(2);product=torch.nn.functional.silu(g[j])*u[j];expected=product if g[j]>0 and u[j]>0 and test['pole']=='positive' else -product if g[j]>0 and u[j]<0 and test['pole']=='negative' else torch.tensor(0.,device=z.device)
                else:expected=runtime.raw_product(probe)[j]*float(test.get('sign',1))
                observed=runtime.read_unit(z[0,0],test);err=float((observed-expected).abs().cpu());assert err<=2e-5;checks.append(dict(block0=block,unit_id=test['unit_id'],max_abs_error=err))
    runtime.observe(capture,byblock.keys());start=time.time();done=0
    for active_batch in batches(records,runtime.name):
        active[:]=active_batch;runtime.forward(active_batch);done+=len(active_batch)
        if done%80<8:
            values.flush();write(str((folder/'PROGRESS.json').relative_to(OUT)),dict(done_chains=done,total_chains=len(records),elapsed_seconds=time.time()-start))
    values.flush();runtime.clear_hooks();assert np.isfinite(values).all()
    schema={'model':runtime.name,'units':units,'index':index,'shape':list(values.shape),'dtype':str(values.dtype),'oriented':True,'unit_readout_semantics_checks':checks,'hook_linear_identity_checks':runtime.identity_errors,'elapsed_seconds':time.time()-start,'runtime_seed':runtime.seed,'native_full_chains':True,'masking':False}
    write(str((folder/'VALUE_SCHEMA.json').relative_to(OUT)),schema);return values,index,schema
def trained(model):
    verify_gate();records,pairs,labels=metadata();units=[u for u in js(OUT/'UNIT_MANIFEST.json')['units'] if u['model']==model and not u.get('masked')];folder=OUT/'readouts'/model/'trained'
    if (folder/'COMPLETE.json').exists():print('already complete',model);return
    rt=Runtime(model);start=time.time();values,index,schema=extract_values(rt,units,records,folder);save_tables(folder,units,records,pairs,labels,values,index,'trained');rt.close()
    write(str((folder/'SELECTION.json').relative_to(OUT)),{'selection':'fixed author table or archived NCCal identity; no current T selection','units':units,'unit_manifest_sha256':sha(OUT/'UNIT_MANIFEST.json'),'runtime_model':rt.metadata})
    write(str((folder/'COMPLETE.json').relative_to(OUT)),{'status':'COMPLETE','model':model,'reference':'trained','rows':len(records),'units':len(units),'runtime_seconds':time.time()-start,'input_freeze_sha256':sha(OUT/'INPUT_FREEZE.json'),'pairs_sha256':sha(folder/'pairs.tsv'),'auc_sha256':sha(folder/'auc.tsv') if (folder/'auc.tsv').exists() else None,'D_groups_by_concept':{c:sorted({p['group'] for p in pairs if p['split']=='D' and p['concept']==c}) for c in {u['concept'] for u in units}},'value_schema_sha256':sha(folder/'VALUE_SCHEMA.json')})
    print('trained COMPLETE',model,time.time()-start,flush=True)
if __name__=='__main__':trained(sys.argv[1])
