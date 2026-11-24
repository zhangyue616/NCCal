"""Same-budget selection, full D candidates in RAM, selected heldout units only."""
from extract import *
from scipy.special import erf,expit
from haar import matrix as haar_matrix
import gc
def rotation_spec(width,seed):
    return haar_matrix(width,seed)
def rotate(x,spec):
    return np.asarray(x,dtype=np.float64)@spec
def direction(width,j,spec):
    return np.asarray(spec[:,j],dtype=np.float64).copy()
def native(raw,kind):
    if kind=='esm2':return torch.nn.functional.gelu(torch.from_numpy(np.ascontiguousarray(raw))).numpy()
    if kind=='esmc':
        g,u=torch.from_numpy(np.ascontiguousarray(raw)).chunk(2,-1);return (torch.nn.functional.silu(g)*u).numpy()
    return raw
def poles(raw,kind):
    if kind=='esmc':
        g,u=np.split(raw,2,-1);v=native(raw,kind);return np.c_[np.where((g>0)&(u>0),v,0),np.where((g>0)&(u<0),-v,0)]
    return np.c_[np.maximum(raw,0),np.maximum(-raw,0)]
def scope_units(model,pairs):
    units=[u for u in js(OUT/'UNIT_MANIFEST.json')['units'] if u['model']==model and not u.get('masked')]
    requested={}
    for u in units:
        family=u['family'];key=u['concept']+'|'+family
        if key not in requested:requested[key]=dict(concept=u['concept'],family=family,masked=False,annotation_category=u['annotation_category'])
    eligible={}
    for key,u in requested.items():
        d=[p for p in pairs if p['concept']==u['concept'] and p['split']=='D'];t=[p for p in pairs if p['concept']==u['concept'] and p['split']=='T']
        if len(d)>=20 and len(t)>=20:eligible[key]=dict(u,D_pairs=d)
    return eligible,requested
def collect_D(rt,records,pairs,scopes):
    needed=defaultdict(set)
    for s in scopes.values():
        for p in s['D_pairs']:needed[p['accession']].update([int(p['early0']),int(p['late0'])])
    positions={a:sorted(ps) for a,ps in needed.items()};pointindex={};offset=0
    for a,ps in sorted(positions.items()):
        for p in ps:pointindex[a,p]=offset;offset+=1
    rawwidth=rt.width*(2 if rt.kind=='esmc' else 1);arrays=[np.empty((offset,rawwidth),np.float32) for _ in rt.mods];active=[]
    def capture(block,z):
        for i,r in enumerate(active):
            a=r['accession'];pp=positions[a];values=z[i,torch.tensor(pp,device=z.device)+rt.prefix,:].detach().cpu().numpy()
            ix=[pointindex[a,p] for p in pp];arrays[block][ix]=values
    rt.observe(capture);start=time.time();dd=[r for r in records if r['split']=='D' and r['accession'] in needed]
    for batch in batches(dd,rt.name):active[:]=batch;rt.forward(batch)
    rt.clear_hooks();return arrays,pointindex,dict(D_chains=len(dd),D_unique_positions=offset,D_full_native_residues=sum(len(r['sequence']) for r in dd),D_RAM_bytes=sum(a.nbytes for a in arrays),D_forward_seconds=time.time()-start,hook_linear_identity_checks=rt.identity_errors,trained_weight_load=rt.seed is None)
def select(arrays,pointindex,scopes,rt,reference,rotation_index=None):
    best={};selected=[]
    for block,raw in enumerate(arrays):
        cache={}
        for key,s in scopes.items():
            family=s['family'];rotation_seed=2026111800+(rotation_index or 0)
            if family not in cache:
                if reference=='rotation':
                    base=raw if family=='pre_pole' else native(raw,rt.kind);rot=rotate(base,rotation_spec(rt.width,rotation_seed));cache[family]=np.c_[np.maximum(rot,0),np.maximum(-rot,0)] if family in ['pre_pole','gate_tagged'] else np.c_[rot,-rot]
                elif family in ['pre_pole','gate_tagged']:cache[family]=poles(raw,rt.kind)
                else:v=native(raw,rt.kind);cache[family]=np.c_[v,-v]
            vals=cache[family];d=s['D_pairs'];early=np.array([pointindex[p['accession'],int(p['early0'])] for p in d]);late=np.array([pointindex[p['accession'],int(p['late0'])] for p in d]);y=np.array([2*int(p['target'])-1 for p in d],np.float32)
            diff=(vals[late]-vals[early])*y[:,None];score=(2*(diff>0).sum(0)+(diff==0).sum(0))
            mx=int(score.max());cols=np.flatnonzero(score==mx);col=min(cols,key=lambda c:(int(c)%rt.width,int(c)//rt.width))
            candidate=(mx,-block,-int(col%rt.width),-int(col//rt.width))
            if key not in best or candidate>best[key][0]:best[key]=(candidate,block,int(col))
    for key,s in scopes.items():
        rank,block,col=best[key];negative=col>=rt.width;j=col%rt.width
        family=s['family'];isrot=reference=='rotation';sign=-1 if negative else 1
        u=dict(unit_id=f"{reference}|{rt.name}|{key}",source='REFERENCE',model=rt.name,block0=block,unit0=j,pole='negative' if negative else 'positive',family=family,sign=sign if family=='native_signed' else 1,concept=s['concept'],annotation_category=s['annotation_category'],masked=False,
            D_groups=len(s['D_pairs']),D_score_twice_numerator=rank[0],D_score=rank[0]/(2*len(s['D_pairs'])),candidate_count=2*rt.width*len(arrays),D_groups_order=[p['group'] for p in s['D_pairs']],selection_scope='D only; exact integer score tie block/unit/positive first')
        if isrot:u.update(rotation_index=rotation_index,rotation_seed=2026111800+rotation_index,original_family=family,family='rotation_pole' if family in ['pre_pole','gate_tagged'] else 'rotation_signed',rotation_sign=sign,rotation_convention='Z@Q; inherited Gaussian QR sign-corrected Haar; same width/index Q across blocks')
        selected.append(u)
    return selected
def D_rows(selected,arrays,pointindex,pairs,rt,reference):
    result=[];byc=defaultdict(list)
    for p in pairs:
        if p['split']=='D':byc[p['concept']].append(p)
    for u in selected:
        raw=arrays[u['block0']]
        if reference=='rotation':
            family=u['original_family'];base=raw if family=='pre_pole' else native(raw,rt.kind);v=base@direction(rt.width,u['unit0'],rotation_spec(rt.width,u['rotation_seed']))*u['rotation_sign']
            if u['family']=='rotation_pole':v=np.maximum(v,0)
        else:
            if u['family'] in ['pre_pole','gate_tagged']:v=poles(raw,rt.kind)[:,u['unit0']+(rt.width if u['pole']=='negative' else 0)]
            else:v=native(raw,rt.kind)[:,u['unit0']]*u['sign']
        for p in byc[u['concept']]:
            if (p['accession'],int(p['early0'])) not in pointindex:continue
            d=float(v[pointindex[p['accession'],int(p['late0'])]])-float(v[pointindex[p['accession'],int(p['early0'])]])
            row={k:p[k] for k in ['concept','group','accession','split','early0','late0','target']};row.update({k:u[k] for k in ['unit_id','source','model','block0','unit0','pole']});row.update(raw_late_minus_early=d/u.get('sign',1),oriented_late_minus_early=d,reference=reference,reference_id=u['reference_id'],family=u['family'],original_family=u.get('original_family',u['family']),sign=u.get('sign',1),masked=False,readpoint=u['family'],own_fresh_eligible=False);result.append(row)
    return result
def enable_rotation_readout(rt):
    original=rt.read_unit
    vectors={}
    def read(z,u):
        if not u['family'].startswith('rotation_'):return original(z,u)
        key=(u['block0'],u['unit0'],u['rotation_seed'])
        if key not in vectors:vectors[key]=direction(rt.width,u['unit0'],rotation_spec(rt.width,u['rotation_seed']))
        base=z if u['original_family']=='pre_pole' else rt.raw_product(z);array=base.detach().cpu().numpy().astype(np.float64);v=array@vectors[key]*u['rotation_sign'];v=np.maximum(v,0) if u['family']=='rotation_pole' else v
        return torch.as_tensor(v,device=z.device,dtype=torch.float64)
    rt.read_unit=read;return original
def one(model,index,reference='random'):
    verify_gate();records,pairs,labels=metadata();scopes,requested=scope_units(model,pairs);folder=OUT/'readouts'/model/(f'random_{index:03d}' if reference=='random' else f'rotation_{index:02d}')
    if (folder/'COMPLETE.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    if not scopes:
        write(str((folder/'COMPLETE.json').relative_to(OUT)),dict(status='NO_EVALUABLE_CONCEPTS',model=model,reference=reference,index=index));return
    seed=202611230000+index if reference=='random' else None;start=time.time();rt=Runtime(model,seed);arrays,pointindex,timing=collect_D(rt,records,pairs,scopes)
    selected=select(arrays,pointindex,scopes,rt,reference,index if reference=='rotation' else None)
    for u in selected:u['reference_id']=folder.name;u['unit_id']=u['unit_id'].replace(reference+'|',folder.name+'|',1)
    drows=D_rows(selected,arrays,pointindex,pairs,rt,reference)
    write(str((folder/'SELECTION.json').relative_to(OUT)),{'status':'SELECTED_D_BEFORE_HELDOUT_FORWARD','model':model,'reference':reference,'index':index,'seed':seed,'units':selected,'requested_but_ineligible':[s for key,s in requested.items() if key not in scopes],'timing':timing,'runtime_model':rt.metadata,'input_freeze_sha256':sha(OUT/'INPUT_FREEZE.json'),'D_only':True})
    del arrays;gc.collect()
    if reference=='rotation':enable_rotation_readout(rt)
    held=[r for r in records if r['split']!='D'];values,ix,schema=extract_values(rt,selected,held,folder);save_tables(folder,selected,records,pairs,labels,values,ix,reference)
    with (folder/'pairs.tsv').open('r',encoding='utf-8',newline='') as f:fields=next(csv.reader(f,delimiter='\t'))
    with (folder/'pairs.tsv').open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writerows(drows)
    del values;gc.collect()
    temporary=folder/'values.npy';assert temporary.resolve().is_relative_to(OUT.resolve());temporary.unlink()
    rt.close()
    write(str((folder/'COMPLETE.json').relative_to(OUT)),{'status':'COMPLETE','model':model,'reference':reference,'reference_id':folder.name,'reference_index':index,'index':index,'seed':seed,'runtime_seconds':time.time()-start,'units':len(selected),'input_freeze_sha256':sha(OUT/'INPUT_FREEZE.json'),'pairs_sha256':sha(folder/'pairs.tsv'),'D_groups_by_concept':{u['concept']:u['D_groups_order'] for u in selected},'no_full_random_activations_on_disk':True,'temporary_selected_full_sequence_array_discarded':True})
    print('REFERENCE COMPLETE',model,reference,index,time.time()-start,flush=True)
if __name__=='__main__':one(sys.argv[1],int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'random')
