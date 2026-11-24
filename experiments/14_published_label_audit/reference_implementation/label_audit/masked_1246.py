"""Preserve the original masked-centre observation for 1246 and its controls."""
from references import *
class Captured(Exception):pass
def masked_values(rt,units,records,pairs,labels,folder,all_label_sites):
    folder.mkdir(parents=True,exist_ok=True);index={};offset=0;wanted=defaultdict(set);byacc={r['accession']:r for r in records}
    for r in records:index[r['accession']]=dict(offset=offset,length=len(r['sequence']),group=r['group'],split=r['split']);offset+=len(r['sequence'])
    for p in pairs:
        if p['accession'] in byacc and p['concept'] in {u['concept'] for u in units}:wanted[p['accession']].update([int(p['early0']),int(p['late0'])])
    if all_label_sites:
        for r in records:
            for u in units:
                lab=labels.get(r['accession'],{}).get(u['concept'])
                if lab:wanted[r['accession']].update(lab['positive0']+lab.get('negative0',[]))
    jobs=[(r,p) for r in sorted(records,key=lambda r:(len(r['sequence']),r['accession'])) for p in sorted(wanted[r['accession']])]
    values=np.lib.format.open_memmap(folder/'values.npy',mode='w+',dtype=np.float64 if any(u['family'].startswith('rotation_') for u in units) else np.float32,shape=(offset,len(units)));values[:]=np.nan
    byblock=defaultdict(list)
    for col,u in enumerate(units):byblock[u['block0']].append((col,u))
    active=[];last=max(byblock)
    def capture(block,z):
        for i,(r,p) in enumerate(active):
            for col,u in byblock[block]:values[index[r['accession']]['offset']+p,col]=float(rt.read_unit(z[i,p+rt.prefix],u).cpu())
        if block==last:raise Captured()
    rt.observe(capture,byblock);start=time.time();done=0
    for k in range(0,len(jobs),8):
        active[:]=jobs[k:k+8]
        try:rt.forward([r for r,p in active],[p for r,p in active])
        except Captured:pass
        done+=len(active)
        if done%1000<8:values.flush();write(str((folder/'PROGRESS.json').relative_to(OUT)),dict(masked_sites=done,total_masked_sites=len(jobs),elapsed_seconds=time.time()-start))
    values.flush();rt.clear_hooks()
    write(str((folder/'VALUE_SCHEMA.json').relative_to(OUT)),dict(model=rt.name,units=units,index=index,shape=list(values.shape),dtype=str(values.dtype),oriented=True,masking=True,only_declared_sites_filled=True,masked_forward_sites=len(jobs),elapsed_seconds=time.time()-start))
    return values,index
def collect_masked_D(rt,records,pairs):
    byacc={r['accession']:r for r in records};wanted=defaultdict(set)
    for p in pairs:wanted[p['accession']].update([int(p['early0']),int(p['late0'])])
    jobs=[(byacc[a],p) for a in sorted(wanted) for p in sorted(wanted[a])];pointindex={(r['accession'],p):i for i,(r,p) in enumerate(jobs)}
    arrays=[np.empty((len(jobs),rt.width),np.float32) for _ in rt.mods];active=[]
    def capture(block,z):
        for i,(r,p) in enumerate(active):arrays[block][pointindex[r['accession'],p]]=z[i,p+rt.prefix].detach().cpu().numpy()
    rt.observe(capture);start=time.time()
    for k in range(0,len(jobs),8):
        active[:]=jobs[k:k+8];rt.forward([r for r,p in active],[p for r,p in active])
    rt.clear_hooks();return arrays,pointindex,dict(D_masked_sites=len(jobs),D_forward_seconds=time.time()-start,D_RAM_bytes=sum(a.nbytes for a in arrays))
def trained_masked():
    verify_gate();records,pairs,labels=metadata();units=[u for u in js(OUT/'UNIT_MANIFEST.json')['units'] if u.get('masked')];folder=OUT/'readouts/esm2_8m/trained_masked'
    if (folder/'COMPLETE.json').exists():return
    start=time.time();rt=Runtime('esm2_8m');values,index=masked_values(rt,units,records,pairs,labels,folder,True);save_tables(folder,units,records,pairs,labels,values,index,'trained');rt.close()
    write(str((folder/'SELECTION.json').relative_to(OUT)),dict(units=units,source='original masked-centre block3 post-GELU1246 with original D sign=-1'))
    write(str((folder/'COMPLETE.json').relative_to(OUT)),dict(status='COMPLETE',model='esm2_8m',reference='trained',masked=True,units=len(units),runtime_seconds=time.time()-start,input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json'),pairs_sha256=sha(folder/'pairs.tsv')))
    print('MASKED TRAINED COMPLETE',time.time()-start,flush=True)
def random_masked(index,reference='random'):
    verify_gate();records,pairs,labels=metadata();concept='OWN_ALA_HELIX_VS_BETA';dp=[p for p in pairs if p['concept']==concept and p['split']=='D'];folder=OUT/'readouts/esm2_8m'/(f'random_masked_{index:03d}' if reference=='random' else f'rotation_masked_{index:02d}')
    if (folder/'COMPLETE.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);start=time.time();seed=202611230000+index if reference=='random' else None;rt=Runtime('esm2_8m',seed)
    assert all(p['aa']=='A' for p in dp)
    arrays,pointindex,timing=collect_masked_D(rt,records,dp);scopes={'OWN_ALA_HELIX_VS_BETA|native_signed':dict(concept=concept,family='native_signed',masked=True,annotation_category='alanine_helix_vs_beta',D_pairs=dp)}
    selected=select(arrays,pointindex,scopes,rt,reference,index if reference=='rotation' else None)
    for u in selected:u['unit_id']=u['unit_id'].replace(reference+'|',folder.name+'|',1);u['masked']=True;u['reference_id']=folder.name
    drows=D_rows(selected,arrays,pointindex,pairs,rt,reference)
    for r in drows:r['masked']=True
    write(str((folder/'SELECTION.json').relative_to(OUT)),dict(status='SELECTED_D_BEFORE_HELDOUT_FORWARD',units=selected,seed=seed,masked=True,timing=timing,D_only=True))
    del arrays;gc.collect()
    if reference=='rotation':enable_rotation_readout(rt)
    held=[r for r in records if r['split']!='D'];values,ix=masked_values(rt,selected,held,pairs,labels,folder,False);save_tables(folder,selected,records,pairs,labels,values,ix,reference)
    with (folder/'pairs.tsv').open(encoding='utf-8',newline='') as f:fields=next(csv.reader(f,delimiter='\t'))
    with (folder/'pairs.tsv').open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields,delimiter='\t').writerows(drows)
    del values;gc.collect();(folder/'values.npy').unlink();rt.close()
    write(str((folder/'COMPLETE.json').relative_to(OUT)),dict(status='COMPLETE',model='esm2_8m',reference=reference,reference_id=folder.name,reference_index=index,index=index,masked=True,seed=seed,runtime_seconds=time.time()-start,input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json'),pairs_sha256=sha(folder/'pairs.tsv'),D_groups_by_concept={concept:[p['group'] for p in dp]},no_full_random_activations_on_disk=True))
    print('MASKED REFERENCE COMPLETE',reference,index,time.time()-start,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='trained':trained_masked()
    else:random_masked(int(sys.argv[2]),sys.argv[1])
