"""Outcome-blind natural beta-sheet topology panel. No model outputs imported."""
from pathlib import Path
import sys,os,json,csv,gzip,hashlib,math,itertools,argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'structure/runtime'))
import gemmi,numpy as np,biotite.structure as bst
from scipy.spatial import cKDTree
AA3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
EXPECTED={k:set(v.split()) for k,v in {'ALA':'N CA C O CB','ARG':'N CA C O CB CG CD NE CZ NH1 NH2','ASN':'N CA C O CB CG OD1 ND2','ASP':'N CA C O CB CG OD1 OD2','CYS':'N CA C O CB SG','GLN':'N CA C O CB CG CD OE1 NE2','GLU':'N CA C O CB CG CD OE1 OE2','GLY':'N CA C O','HIS':'N CA C O CB CG ND1 CD2 CE1 NE2','ILE':'N CA C O CB CG1 CG2 CD1','LEU':'N CA C O CB CG CD1 CD2','LYS':'N CA C O CB CG CD CE NZ','MET':'N CA C O CB CG SD CE','PHE':'N CA C O CB CG CD1 CD2 CE1 CE2 CZ','PRO':'N CA C O CB CG CD','SER':'N CA C O CB OG','THR':'N CA C O CB OG1 CG2','TRP':'N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2','TYR':'N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH','VAL':'N CA C O CB CG1 CG2'}.items()}
def utc():return datetime.now(timezone.utc).isoformat()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr),encoding='utf-8')
def js(p,r):Path(p).write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def tsv(p,rr,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rr)
def readjl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def val(x):return '' if x in (None,False,'.','?') else str(x)
def tab(b,c):
    d=b.get_mmcif_category(c);return [dict(zip(d,x)) for x in zip(*d.values())] if d else []
def coord(x):return np.array([float(x[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')])
def dist(a,b):return float(np.linalg.norm(a-b))
def angle(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def contact(a,b):
    if dist(a['CA'],b['CA'])>6.5:return False
    return any(dist(z['O'],w['N'])<=3.5 and angle(z['C']-z['O'],w['N']-z['O'])<=math.cos(math.radians(100)) for z,w in ((a,b),(b,a)))
def linked(bb,a,b):return a in bb and b in bb and 1.0<=dist(bb[a]['C'],bb[b]['N'])<=1.8 and 2.7<=dist(bb[a]['CA'],bb[b]['CA'])<=4.3
def parse_source(path):
    b=gemmi.cif.read_string(gzip.decompress(Path(path).read_bytes()).decode()).sole_block()
    rows=tab(b,'_atom_site.');by=defaultdict(list)
    for x in rows:
        p=val(x.get('label_seq_id'));c=val(x.get('label_asym_id'))
        if p.isdigit() and val(x.get('label_comp_id')) in AA3:by[(c,int(p))].append(x)
    bb={};bad={}
    for key,xx in by.items():
        z=[x for x in xx if x['label_atom_id'] in {'N','CA','C','O'}];why=[]
        if Counter(x['label_atom_id'] for x in z)!={'N':1,'CA':1,'C':1,'O':1}:why.append('missing_or_duplicate_backbone')
        if any(val(x.get('label_alt_id')) for x in z):why.append('alternate_backbone')
        if any(float(x['occupancy'])<.99 for x in z):why.append('partial_backbone_occupancy')
        if len({val(x.get('pdbx_PDB_model_num')) for x in xx})!=1:why.append('dependent_models')
        if len({x['label_comp_id'] for x in xx})!=1:why.append('ambiguous_identity')
        if why:bad[key]=why
        else:bb[key]={x['label_atom_id']:coord(x) for x in z}
    nodes=[];index={};sheets=defaultdict(list);orders=defaultdict(set);sense={};decl={}
    for x in tab(b,'_struct_sheet.'):
        try:decl[x['id']]=int(x['number_strands'])
        except:decl[x['id']]=-1
    for x in tab(b,'_struct_sheet_range.'):
        sid=x['sheet_id'];nid=f"{sid}:{x['id']}"
        try:s,e=int(x['beg_label_seq_id']),int(x['end_label_seq_id'])
        except:s,e=-1,-1
        n={'node_id':nid,'sheet':sid,'range_id':x['id'],'chain':x['beg_label_asym_id'],'end_chain':x['end_label_asym_id'],'start':s,'end':e,'length':e-s+1,'duplicate_id':nid in index}
        n['keys']=[(n['chain'],p) for p in range(s,e+1)] if 0<s<=e else []
        nodes.append(n);index[nid]=n;sheets[sid].append(n)
    for x in tab(b,'_struct_sheet_order.'):
        a=f"{x['sheet_id']}:{x['range_id_1']}";z=f"{x['sheet_id']}:{x['range_id_2']}";edge=tuple(sorted((a,z)))
        orders[x['sheet_id']].add(edge);sense[edge]=val(x.get('sense'))
    # Preserve the complete archive graph, never prefilter interchain edges.
    geom={};contacts={};adj=defaultdict(set)
    for a,z in itertools.combinations(nodes,2):
        if not a['keys'] or not z['keys']:continue
        pairs=[]
        for ka in a['keys']:
            if ka not in bb:continue
            for kz in z['keys']:
                if kz not in bb or (ka[0]==kz[0] and abs(ka[1]-kz[1])<=2):continue
                if contact(bb[ka],bb[kz]):pairs.append((ka[1],kz[1]))
        na=len({q[0] for q in pairs});nz=len({q[1] for q in pairs})
        if na>=2 and nz>=2 and min(na,nz)>=.3*min(a['length'],z['length']):
            edge=tuple(sorted((a['node_id'],z['node_id'])));geom[edge]={'contacts':pairs,'n1':na,'n2':nz};contacts[edge]=pairs
            adj[a['node_id']].add(z['node_id']);adj[z['node_id']].add(a['node_id'])
    # All deposited biological assembly operations, with label chains preserved.
    assemblies=[];st=gemmi.make_structure_from_block(b)
    for ass in st.assemblies:
        copies=[]
        for gen in ass.generators:
            chains=list(gen.subchains)
            if not chains and gen.chains:
                auth_to_label=defaultdict(set)
                for x in rows:auth_to_label[val(x.get('auth_asym_id'))].add(val(x.get('label_asym_id')))
                chains=sorted(set().union(*(auth_to_label[x] for x in gen.chains)))
            for op in gen.operators:
                mat=np.array(op.transform.mat.tolist());vec=np.array(op.transform.vec.tolist());tid=h(str(mat.tolist())+str(vec.tolist()))[:12]
                copies.extend((c,tid,mat,vec) for c in chains)
        assemblies.append((ass.name,copies))
    # SASA on observed ASU protein heavy atoms; not used to select primary pairs.
    atom_by={}
    for x in rows:
        if x['label_comp_id'] not in AA3 or x['type_symbol'] in {'H','D'} or val(x.get('pdbx_PDB_model_num')) not in {'','1'} or val(x.get('label_alt_id')) not in {'','A'}:continue
        if not val(x.get('label_seq_id')).isdigit():continue
        key=(x['label_asym_id'],int(x['label_seq_id']),x['label_atom_id'])
        if key not in atom_by or float(x['occupancy'])>float(atom_by[key]['occupancy']):atom_by[key]=x
    chosen=list(atom_by.items());arr=bst.AtomArray(len(chosen))
    arr.coord=np.array([coord(x) for k,x in chosen],dtype=np.float32)
    arr.chain_id=np.array([k[0] for k,x in chosen]);arr.res_id=np.array([k[1] for k,x in chosen]);arr.res_name=np.array([x['label_comp_id'] for k,x in chosen]);arr.atom_name=np.array([k[2] for k,x in chosen]);arr.element=np.array([x['type_symbol'] for k,x in chosen])
    areas=bst.sasa(arr,probe_radius=1.4,point_number=960,vdw_radii='ProtOr');sasa=defaultdict(float)
    for (k,x),a in zip(chosen,areas):
        if np.isfinite(a):sasa[k[:2]]+=float(a)
    for key,xx in by.items():
        name=xx[0]['label_comp_id'];names={x['label_atom_id'] for x in xx if x['type_symbol'] not in {'H','D'} and val(x.get('pdbx_PDB_model_num')) in {'','1'}}
        if not EXPECTED[name]<=names or any(val(x.get('label_alt_id')) or float(x['occupancy'])<.99 for x in xx if x['type_symbol'] not in {'H','D'}):sasa.pop(key,None)
    return {'block':b,'by':by,'bb':bb,'bad':bad,'nodes':nodes,'index':index,'sheets':sheets,'orders':orders,'sense':sense,'decl':decl,'geom':geom,'adj':adj,'assemblies':assemblies,'sasa':sasa}
def outside_extension(n,allkeys,bb,trans=None):
    """Detect >=2 residue ladder into any 8-residue outside window."""
    hits=defaultdict(list)
    for ka in n['keys']:
        if ka not in bb:continue
        for kz in allkeys:
            if trans is None and kz[0]==ka[0] and n['start']-2<=kz[1]<=n['end']+2:continue
            b=bb[kz] if trans is None else {a:trans[0]@v+trans[1] for a,v in bb[kz].items()}
            if contact(bb[ka],b):hits[kz[0]].append((ka[1],kz[1]))
    for c,pp in hits.items():
        for _,q in pp:
            win=[(a,z) for a,z in pp if q<=z<=q+7]
            if len({a for a,z in win})>=2 and len({z for a,z in win})>=2:return {'outside_chain':c,'outside_positions':sorted({z for a,z in win}),'strand_positions':sorted({a for a,z in win})}
    return None
def assess(r,source):
    bb=source['bb'];out=[];sites=[];ch=r['label_asym_id'];by=source['by'];off=r['native_offset0']
    for sid,nn in source['sheets'].items():
        if not any(n['chain']==ch for n in nn):continue
        reasons=[];ids={n['node_id'] for n in nn};edges=source['orders'][sid];geom={x for x in source['geom'] if x[0] in ids or x[1] in ids};deg=Counter(x for e in edges for x in e)
        if len(nn)<3:reasons.append('fewer_than_3_strands')
        if source['decl'].get(sid)!=len(nn):reasons.append('declared_count_mismatch')
        if any(n['duplicate_id'] or n['chain']!=n['end_chain'] or not n['keys'] for n in nn):reasons.append('malformed_range')
        if {n['chain'] for n in nn}!={ch}:reasons.append('interchain_archive_sheet')
        reached=set();todo=[next(iter(ids))]
        while todo:
            v=todo.pop()
            if v in reached:continue
            reached.add(v);todo.extend(z for e in edges if v in e for z in e if z!=v)
        if len(edges)!=len(nn)-1 or sorted(deg.get(i,0) for i in ids)!=[1,1]+[2]*(len(nn)-2) or reached!=ids:reasons.append('archive_not_connected_open_simple_path')
        if geom!=edges:reasons.append('geometry_archive_disagreement_or_extra_neighbor')
        for n in nn:
            # Reject ranges that overlap any separate deposited strand, even another sheet.
            if any(z is not n and z['chain']==n['chain'] and max(z['start'],n['start'])<=min(z['end'],n['end']) for z in source['nodes']):reasons.append('overlapping_strand_ranges')
            keys=[(n['chain'],p) for p in range(n['start']-1,n['end']+2)]
            if n['length']<3 or any(k not in bb for k in keys) or not all(linked(bb,keys[i],keys[i+1]) for i in range(len(keys)-1)):reasons.append('incomplete_strand_or_flanks')
            if n['chain']==ch:
                for k in n['keys']:
                    p=k[1]
                    if not 1<=p<=len(r['entity_sequence']) or k not in by or any(AA3.get(x['label_comp_id'])!=r['entity_sequence'][p-1] for x in by[k]):reasons.append('native_identity_mismatch')
            for x in tab(source['block'],'_struct_conf.'):
                if x['beg_label_asym_id']==n['chain'] and val(x.get('beg_label_seq_id')).isdigit() and val(x.get('end_label_seq_id')).isdigit() and max(int(x['beg_label_seq_id']),n['start'])<=min(int(x['end_label_seq_id']),n['end']):reasons.append('helix_overlap')
        for e in edges:
            if e[0] not in source['index'] or e[1] not in source['index']:continue
            a,z=[source['index'][k] for k in e]
            if all((n['chain'],p) in bb for n in (a,z) for p in (n['start'],n['end'])):
                cos=angle(bb[(a['chain'],a['end'])]['CA']-bb[(a['chain'],a['start'])]['CA'],bb[(z['chain'],z['end'])]['CA']-bb[(z['chain'],z['start'])]['CA'])
                expected=source['sense'].get(e,'')
                if expected not in {'parallel','anti-parallel'} or (expected=='parallel')!=(cos>0):reasons.append('deposited_sense_disagreement')
        extension=[];sym=[]
        if not reasons:
            sheetkeys={k for n in nn for k in n['keys']};outside=[k for k in bb if k not in sheetkeys]
            for n in nn:
                hit=outside_extension(n,outside,bb)
                if hit:extension.append(dict(hit,node_id=n['node_id']))
            if extension:reasons.append('unannotated_or_other_sheet_ASU_extension')
            containing=[(name,copies) for name,copies in source['assemblies'] if any(c==ch for c,tid,mat,vec in copies)]
            if not containing:reasons.append('no_deposited_biological_assembly_for_chain')
            # Work in target-copy coordinate frame; compare every other assembly copy.
            for name,copies in containing:
                targetops={(tid,tuple(mat.ravel()),tuple(vec)):(mat,vec) for c,tid,mat,vec in copies if c==ch}
                for (_,_,_),(tm,tv) in targetops.items():
                    for c,tid,mat,vec in copies:
                        rm=tm.T@mat;rv=tm.T@(vec-tv)
                        if np.allclose(rm,np.eye(3),atol=1e-5) and np.allclose(rv,0,atol=1e-5):continue
                        keys=[k for k in bb if k[0]==c]
                        # Cheap chain bounding-box/CA prune, then direct contacts.
                        if not keys:continue
                        co=np.array([rm@bb[k]['CA']+rv for k in keys]);tree=cKDTree(co)
                        for n in nn:
                            possible=set()
                            for k in n['keys']:
                                possible.update(tree.query_ball_point(bb[k]['CA'],6.5))
                            if len(possible)<2:continue
                            hit=outside_extension(n,[keys[i] for i in sorted(possible)],bb,(rm,rv))
                            if hit:sym.append(dict(hit,assembly=name,node_id=n['node_id'],operator=tid))
            if sym:reasons.append('biological_assembly_copy_extension')
        row={'record_id':r['record_id'],'pdb_id':r['pdb_id'],'group':r['group'],'split':r['split'],'sheet_id':f"{r['record_id']}:{sid}",'deposited_sheet_id':sid,'n_strands':len(nn),'archive_edges':[list(e) for e in sorted(edges)],'geometric_edges':[list(e) for e in sorted(geom)],'nodes':[{k:v for k,v in n.items() if k!='keys'} for n in nn],'eligible':not reasons,'failure_reasons':sorted(set(reasons)),'outside_extensions':extension,'assembly_extensions':sym}
        out.append(row)
        if reasons:continue
        for n in nn:
            neighbour=set(source['adj'][n['node_id']]);role='edge' if len(neighbour)==1 else 'central'
            for p in range(n['start']+1,n['end']):
                f=(p-n['start']+.5)/n['length']
                if not .25<=f<=.75:continue
                if not any(contact(bb[(ch,p)],bb[k]) for z in neighbour for k in source['index'][z]['keys'] if k in bb):continue
                sites.append({'record_id':r['record_id'],'chain_id':r['chain_id'],'group':r['group'],'split':r['split'],'sheet_id':row['sheet_id'],'deposited_sheet_id':sid,'n_strands':len(nn),'strand_id':n['node_id'],'role':role,'aa':r['entity_sequence'][p-1],'label_seq_id':p,'index0':off+p-1,'strand_length':n['length'],'strand_offset0':p-n['start'],'distance_to_strand_end':min(p-n['start'],n['end']-p),'strand_fraction':f,'sasa_A2':source['sasa'].get((ch,p)),'sequence_sha256':r['sequence_sha256']})
    return out,sites
def select(chains,sites):
    byid={r['record_id']:r for r in chains};bins=defaultdict(lambda:[[],[]]);candidates=[]
    for s in sites:bins[(s['record_id'],s['sheet_id'],s['aa'])][0 if s['role']=='edge' else 1].append(s)
    for (cid,sheet,aa),(pp,nn) in bins.items():
        r=byid[cid]
        for p,n in itertools.product(pp,nn):
            inds=sorted((p['index0'],n['index0']));digest=h('sheet-pair-v1|'+r['sequence_sha256']+'|'+str(inds)+'|'+sheet)
            row={'pair_id':f"sheet_edge_vs_central:{cid}:{p['index0']}-{n['index0']}",'record_id':cid,'chain_id':cid,'group':r['group'],'group_id':r['group'],'homology_group':r['group'],'split':r['split'],'task':'sheet_edge_vs_central','aa':aa,'index_positive':p['index0'],'index_negative':n['index0'],'positive_label_seq_id':p['label_seq_id'],'negative_label_seq_id':n['label_seq_id'],'positive_secondary':'E','negative_secondary':'E','positive_role':'edge','negative_role':'central','sheet_id':sheet,'deposited_sheet_id':p['deposited_sheet_id'],'n_sheet_strands':p['n_strands'],'sheet_ge4_bool':p['n_strands']>=4,'selection_digest':digest,'sequence_sha256':r['sequence_sha256']}
            for pref,s in (('positive',p),('negative',n)):
                for k in ('strand_id','strand_length','strand_offset0','distance_to_strand_end','strand_fraction','sasa_A2'):row[pref+'_'+k]=s[k]
                row[pref+'_strand_start_index0']=s['index0']-s['strand_offset0']
                row[pref+'_strand_end_index0']=s['index0']-s['strand_offset0']+s['strand_length']-1
            row['sasa_matched_bool']=p['sasa_A2'] is not None and n['sasa_A2'] is not None and abs(p['sasa_A2']-n['sasa_A2'])<=10
            row['position_length_matched_bool']=abs(p['strand_length']-n['strand_length'])<=2 and abs(p['strand_fraction']-n['strand_fraction'])<=.15
            row['confound_matched_bool']=row['sasa_matched_bool'] and row['position_length_matched_bool'];candidates.append(row)
    used=set();counts=Counter();selected=[]
    for p in sorted(candidates,key=lambda p:(p['group'],p['selection_digest'])):
        keys=[(p['group'],p['sequence_sha256'],p[k]) for k in ('index_positive','index_negative')]
        if counts[p['group']]>=16 or any(k in used for k in keys):continue
        selected.append(p);used.update(keys);counts[p['group']]+=1
    return sorted(selected,key=lambda p:(p['split'],p['group'],p['pair_id']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default=str(ROOT/'native_channels/cohort/chains.jsonl'));ap.add_argument('--fresh',action='store_true');ap.add_argument('--tag',default='');args=ap.parse_args()
    suffix='_'+args.tag if args.tag else ('_F' if args.fresh else '');chains=readjl(args.input);allrows=[];allsites=[];errors=[];cache={}
    for i,r in enumerate(chains):
        try:
            key=r['source_cif']
            if key not in cache:cache[key]=parse_source(key)
            a,s=assess(r,cache[key]);allrows.extend(a);allsites.extend(s)
        except Exception as e:
            import traceback;errors.append({'record_id':r['record_id'],'error':repr(e),'traceback':traceback.format_exc()});print('ERROR',r['record_id'],e,flush=True)
        if i%10==0:print('STRUCTURES',i,len(chains),'eligible_sheets',sum(x['eligible'] for x in allrows),'sites',len(allsites),flush=True)
    pairs=select(chains,allsites);used={p['record_id'] for p in pairs};retained=[r for r in chains if r['record_id'] in used]
    jl(HERE/f'chains{suffix}.jsonl',retained);tsv(HERE/f'pairs{suffix}.tsv',pairs);jl(HERE/f'sheet_graph_audit{suffix}.jsonl',allrows);tsv(HERE/f'sites{suffix}.tsv',allsites);jl(HERE/f'build_errors{suffix}.jsonl',errors)
    summary={'utc':utc(),'input_path':str(Path(args.input).resolve())}
    summary.update({'input_chains':len(chains),'input_sequences':len({r['sequence_sha256'] for r in chains}),'input_groups':len({r['group'] for r in chains}),'sheets_audited':len(allrows),'eligible_sheets':sum(x['eligible'] for x in allrows),'failure_reasons':dict(Counter(y for x in allrows for y in x['failure_reasons'])),'site_counts':dict(Counter(s['role'] for s in allsites)),'pairs':len(pairs),'retained_chains':len(retained),'groups':len({r['group'] for r in pairs}),'counts_by_split':{sp:{'pairs':sum(p['split']==sp for p in pairs),'groups':len({p['group'] for p in pairs if p['split']==sp}),'chains':len({p['record_id'] for p in pairs if p['split']==sp}),'sheet_ge4_pairs':sum(p['split']==sp and p['sheet_ge4_bool'] for p in pairs),'confound_matched_pairs':sum(p['split']==sp and p['confound_matched_bool'] for p in pairs)} for sp in (('F',) if args.fresh else ('D','T','E'))},'errors':len(errors),'model_scores_read':False,'sources':{r['source_cif']:r['source_cif_sha256'] for r in chains}})
    js(HERE/f'BUILD_SUMMARY{suffix}.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2),flush=True)
if __name__=='__main__':main()
