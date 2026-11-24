"""Outcome-blind experimental assembly RSA extension. Writes confined to this folder."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, csv, gzip, hashlib, time, argparse, traceback, threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT/'runtime'))
sys.dont_write_bytecode = True
import numpy as np
import requests

AA3 = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))
MAXASA = dict(zip('ARNDCQEGHILKMFPSTWYV', [129,274,195,193,167,225,223,104,224,197,201,236,224,240,159,155,172,285,263,174]))
SIDE = {'A':'CB','R':'CB CG CD NE CZ NH1 NH2','N':'CB CG OD1 ND2','D':'CB CG OD1 OD2','C':'CB SG','Q':'CB CG CD OE1 NE2','E':'CB CG CD OE1 OE2','G':'','H':'CB CG ND1 CD2 CE1 NE2','I':'CB CG1 CG2 CD1','L':'CB CG CD1 CD2','K':'CB CG CD CE NZ','M':'CB CG SD CE','F':'CB CG CD1 CD2 CE1 CE2 CZ','P':'CB CG CD','S':'CB OG','T':'CB OG1 CG2','W':'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2','Y':'CB CG CD1 CD2 CE1 CE2 CZ OH','V':'CB CG1 CG2'}
REQUIRED = {a:set(('N CA C O '+s).split()) for a,s in SIDE.items()}
def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(p)+f'.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def prepare():
    sources=[]; records=[]
    final=ROOT/'data/revision/protein_fresh/OLD_SELECTED_FINAL_GROUPS.json'
    old=ROOT/'data/revision/protein_metadata/OLD_SELECTED_UNION_GROUPS.json'
    getmap=lambda p:{(r['origin'],r['id']):r['current_known_union_group'] for r in read(p)['selected_pair_group_mapping']}
    gm=getmap(final); assert gm==getmap(old)
    for origin,n in [('pilot3a',176),('pilot4a',128),('pilot5a',128)]:
        p=ROOT/f'data/{origin}/proteins.jsonl'; r=rows(p); assert len(r)==n
        records.extend(dict(x,split='D',group=gm[(origin,x['id'])],origin=origin) for x in r)
        sources.append({'path':str(p),'n':n,'sha256':sha(p)})
    p=ROOT/'data/revision/protein_fresh/proteins.jsonl'; r=rows(p); assert len(r)==1304
    records.extend(dict(x,split='T',group=x['known_group_id'],origin='protein_fresh') for x in r)
    sources.append({'path':str(p),'n':1304,'sha256':sha(p)})
    cohort=[]
    for r in records:
        cs={}
        for key in ('h_site','s_site'):
            ev=r[key]['evidence']; raw=ev.get('raw_xref',{})
            props={z['key']:z['value'] for z in raw.get('properties',[])}
            import re
            m=re.search(r'[0-9]+(?:\.[0-9]+)?',props.get('Resolution',''))
            res=float(m.group()) if m else None
            pid=ev['pdb_id'].lower()
            c=cs.setdefault(pid,{'pdb_id':pid,'resolution':res,'method':ev.get('method',''),'old_source_chains':[]})
            c['old_source_chains'].append(ev['chain'])
        candidates=sorted(cs.values(),key=lambda c:(c['resolution'] if c['resolution'] is not None else 999,c['pdb_id']))
        for c in candidates: c['old_source_chains']=sorted(set(c['old_source_chains']))
        cohort.append({'split':r['split'],'group':r['group'],'accession':r['id'],'sequence':r['sequence'],'origin':r['origin'],'candidates':candidates})
    cohort.sort(key=lambda r:(r['split'],r['group'],r['accession']))
    assert len(cohort)==1736 and len({r['group'] for r in cohort if r['split']=='D'})==421
    assert len({r['group'] for r in cohort if r['split']=='T'})==1304
    p=OUT/'cohort_index.jsonl'
    txt=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in cohort)
    if p.exists(): assert p.read_text()==txt
    else: p.write_text(txt,encoding='utf-8')
    write(OUT/'input_receipt.json',{'created_utc':utc(),'cohort_records':len(cohort),'D_pairs':432,'D_groups':421,'T_groups':1304,'old_and_final_D_mapping_equal':True,'sources':sources,'final_group_source':str(final),'final_group_sha256':sha(final),'rule_lock_sha256':sha(OUT/'STRUCTURE_RULE_LOCK.md'),'no_predictions_read':True,'reserved_groups_read':False})
    return cohort

def fetch(url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0:return path
    last=None
    for attempt in range(2):
        try:
            r=requests.get(url,timeout=(15,50),headers={'User-Agent':'NCCal-structural-extension/1.0 (public structure analysis)'})
            r.raise_for_status(); content=r.content
            if url.endswith('.gz'): gzip.decompress(content)
            temp=Path(str(path)+f'.{threading.get_ident()}.tmp');temp.write_bytes(content);temp.replace(path)
            write(str(path)+'.source.json',{'url':url,'retrieved_utc':utc(),'status':r.status_code,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'last_modified':r.headers.get('Last-Modified')})
            return path
        except Exception as e:last=e
    raise RuntimeError(f'Download failed {url}: {last}')

def sifts_map(pid, accession):
    p=fetch(f'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pid}.xml.gz',OUT/f'cache/sifts/{pid}.xml.gz')
    root=ET.fromstring(gzip.decompress(p.read_bytes())); maps=defaultdict(dict); ambiguous=set()
    for res in root.iter():
        if res.tag.rsplit('}',1)[-1]!='residue':continue
        refs=[x.attrib for x in res if x.tag.rsplit('}',1)[-1]=='crossRefDb']
        ur=[x for x in refs if x.get('dbSource')=='UniProt' and x.get('dbAccessionId')==accession]
        pr=[x for x in refs if x.get('dbSource')=='PDB']
        if not ur or not pr:continue
        # All UniProt references to this residue must identify this exact canonical accession.
        if {x.get('dbAccessionId') for x in refs if x.get('dbSource')=='UniProt'}!={accession}:continue
        if len(ur)!=1 or len(pr)!=1:continue
        u,p=ur[0],pr[0]
        if p.get('dbResNum') in ('null','None',None):continue
        try:pos=int(u['dbResNum'])
        except ValueError:continue
        k=(p['dbChainId'],p['dbResNum'])
        v=(pos,u.get('dbResName'),p.get('dbResName'))
        if k in maps[p['dbChainId']] and maps[p['dbChainId']][k]!=v:ambiguous.add(k)
        maps[p['dbChainId']][k]=v
    for chain in maps:
        pc=Counter(v[0] for v in maps[chain].values())
        maps[chain]={k:v for k,v in maps[chain].items() if k not in ambiguous and pc[v[0]]==1}
    return maps

def process_accession(rec):
    import gemmi
    import biotite, biotite.structure as bs
    seq=rec['sequence']; acc=rec['accession']; attempts=[]
    for candidate in rec['candidates']:
        pid=candidate['pdb_id']
        try:
            if candidate['resolution'] is None or candidate['resolution']>3.0 or not ('x-ray' in candidate['method'].lower() or 'electron' in candidate['method'].lower() or candidate['method'].upper()=='EM'):
                attempts.append({'pdb_id':pid,'status':'source_method_or_resolution_ineligible'});continue
            cp=fetch(f'https://files.rcsb.org/download/{pid.upper()}.cif.gz',OUT/f'cache/mmcif/{pid}.cif.gz')
            block=gemmi.cif.read_string(gzip.decompress(cp.read_bytes()).decode()).sole_block()
            st=gemmi.make_structure_from_block(block)
            # Recheck experimental type and resolution against coordinate metadata.
            methods=list(block.find_values('_exptl.method'))
            if not any(('X-RAY' in s.upper() or 'ELECTRON MICROSCOPY' in s.upper()) for s in methods):
                attempts.append({'pdb_id':pid,'status':'coordinate_method_ineligible','method':methods});continue
            if st.resolution>3.0 or st.resolution<=0:
                attempts.append({'pdb_id':pid,'status':'coordinate_resolution_ineligible','resolution':st.resolution});continue
            if not st.assemblies:
                attempts.append({'pdb_id':pid,'status':'no_deposited_biological_assembly'});continue
            asm=next((a for a in st.assemblies if a.name=='1'),sorted(st.assemblies,key=lambda a:a.name)[0])
            model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
            mapping=sifts_map(pid,acc)
            coords=[]; elements=[]; chain_indices=[]; atom_names=[]; residue_keys=[]; residue_data={}; chain_data={}
            for ci,chain in enumerate(model):
                chain_data[ci]={'author_chain':chain.name,'residues':[]}
                for ri,res in enumerate(chain):
                    if res.entity_type!=gemmi.EntityType.Polymer or not gemmi.find_tabulated_residue(res.name).is_amino_acid():continue
                    key=(ci,ri); atoms=[a for a in res if a.element.name not in ('H','D')]
                    if not atoms:continue
                    rid=str(res.seqid.num)+(res.seqid.icode.strip() or '')
                    mp=mapping.get(chain.name,{}).get((chain.name,rid))
                    aa=AA3.get(res.name); pos=mp[0] if mp else None
                    exact=bool(mp and aa and 1<=pos<=len(seq) and seq[pos-1]==aa and mp[1]==aa and mp[2]==res.name)
                    names=[a.name for a in atoms]
                    quality=bool(aa and REQUIRED[aa].issubset(names) and len(names)==len(set(names)) and all(a.occ>=.95 and a.altloc=='\x00' for a in atoms) and np.mean([a.b_iso for a in atoms])<=60)
                    residue_data[key]={'aa':aa,'pos':pos,'exact':exact,'quality':quality,'author_residue':rid,'res_name':res.name,'mean_b':float(np.mean([a.b_iso for a in atoms])),'indices':[],'atoms':{a.name:np.array([a.pos.x,a.pos.y,a.pos.z]) for a in atoms}}
                    chain_data[ci]['residues'].append(key)
                    # Keep all conformers as occluders; such sites cannot pass quality.
                    for a in atoms:
                        residue_data[key]['indices'].append(len(coords));coords.append([a.pos.x,a.pos.y,a.pos.z]);elements.append(a.element.name);chain_indices.append(ci);atom_names.append(a.name);residue_keys.append(key)
            if len(coords)>150000:
                attempts.append({'pdb_id':pid,'status':'assembly_over_150000_atoms_resource_bound','atoms':len(coords)});continue
            choices=[]
            for ci,cd in chain_data.items():
                exact=[k for k in cd['residues'] if residue_data[k]['exact']]
                if exact:choices.append((-len(exact),cd['author_chain'],ci))
            if not choices:
                attempts.append({'pdb_id':pid,'status':'no_exact_canonical_SIFTS_chain'});continue
            _,chain,ci=min(choices)
            keys=chain_data[ci]['residues']; bypos={residue_data[k]['pos']:k for k in keys if residue_data[k]['exact']}
            if len(bypos)<30:
                attempts.append({'pdb_id':pid,'status':'fewer_than_30_exact_observed_residues'});continue
            arr=bs.AtomArray(len(coords));arr.coord=np.array(coords,dtype=np.float32);arr.element=np.array(elements);arr.atom_name=np.array(atom_names)
            arr.chain_id=np.array([str(c) for c in chain_indices]);arr.res_id=np.array([k[1]+1 for k in residue_keys]);arr.res_name=np.array([residue_data[k]['res_name'] for k in residue_keys])
            mask=np.array(chain_indices)==ci
            # atom_filter limits reported atoms, not occluders: all assembly
            # protein atoms remain in arr and contribute to target-chain SASA.
            sasa=bs.sasa(arr,atom_filter=mask,probe_radius=1.4,point_number=1000,point_distr='Fibonacci',vdw_radii='Single',ignore_ions=True)
            isolated=arr[mask]
            iso=bs.sasa(isolated,probe_radius=1.4,point_number=1000,point_distr='Fibonacci',vdw_radii='Single',ignore_ions=True)
            iso_index=np.where(mask)[0];iso_lookup=dict(zip(iso_index,iso))
            sse=bs.annotate_sse(isolated)
            starts=bs.get_residue_starts(isolated)
            sse_by_key={residue_keys[iso_index[s]]:str(v) for s,v in zip(starts,sse)}
            # P-SEA values are a,b,c in pinned Biotite; normalize to H/E/C.
            norm={'a':'H','b':'E','c':'C','H':'H','E':'E','C':'C','PSeaElement.ALPHA_HELIX':'H','PSeaElement.BETA_SHEET':'E','PSeaElement.COIL':'C'}
            features=[]
            for pos,key in sorted(bypos.items()):
                rd=residue_data[key];aa=rd['aa']
                if not rd['quality']:continue
                ids=rd['indices'];sa=float(np.sum(sasa[ids]));si=float(sum(iso_lookup[i] for i in ids))
                if not np.isfinite(sa+si):continue
                rsa=sa/MAXASA[aa];ri=si/MAXASA[aa]
                neighborhood=pos>=16 and pos<=len(seq)-15 and all(p in bypos for p in range(pos-15,pos+16))
                immediate=all(p in bypos and residue_data[bypos[p]]['quality'] for p in range(pos-2,pos+3))
                # Also reject chain breaks in the local window using adjacent Calpha distance.
                continuity=neighborhood and all('CA' in residue_data[bypos[p]]['atoms'] for p in range(pos-15,pos+16)) and all(np.linalg.norm(residue_data[bypos[p]]['atoms']['CA']-residue_data[bypos[p+1]]['atoms']['CA'])<=4.5 for p in range(pos-15,pos+15))
                central=rd['atoms']['CA' if aa=='G' else 'CB'];count=0;denom=0
                for p,k in bypos.items():
                    rr=residue_data[k];atom='CA' if rr['aa']=='G' else 'CB'
                    if abs(pos-p)>=24 and atom in rr['atoms']:
                        denom+=1;count+=bool(np.linalg.norm(central-rr['atoms'][atom])<=8.0)
                interface=bool(si-sa>1.0 and ri-rsa>.01)
                secondary=norm.get(sse_by_key.get(key,''),'U')
                local2_continuous=immediate and all(np.linalg.norm(residue_data[bypos[p]]['atoms']['CA']-residue_data[bypos[p+1]]['atoms']['CA'])<=4.5 for p in range(pos-2,pos+2))
                features.append({'group':rec['group'],'split':rec['split'],'accession':acc,'pos':pos,'aa':aa,'secondary':secondary,'rsa':rsa,'remote_contacts':int(count),'pdb_id':pid,'assembly_id':asm.name,'chain':chain,'assembly_chain':ci,'rsa_isolated':ri,'interface':interface,'sequence_index':acc,'sasa_A2':sa,'sasa_isolated_A2':si,'remote_eligible_partners':denom,'local15_exact_observed':bool(neighborhood),'local2_complete_quality':bool(immediate),'local15_CA_continuous':bool(continuity),'main_clean':bool(neighborhood and immediate and continuity and not interface and secondary!='U'),'center2_clean_sensitivity':bool(immediate and local2_continuous and not interface and secondary!='U' and pos>=16 and pos<=len(seq)-15),'author_residue':rd['author_residue'],'mean_b':rd['mean_b']})
            # Structural eligibility is fixed before checking whether any RSA pair exists.
            if not features:
                attempts.append({'pdb_id':pid,'status':'no_quality_eligible_standard_residues'});continue
            def select_pair(flag):
                eligible=[f for f in features if f[flag]];pairs=[]
                for f1 in eligible:
                    if f1['rsa']>.10:continue
                    for f2 in eligible:
                        if f2['rsa']<.40 or f1['aa']!=f2['aa'] or f1['secondary']!=f2['secondary'] or abs(f1['pos']-f2['pos'])<=30:continue
                        salt=f"nccal-assembly-rsa-v1|{rec['group']}|{pid}|{chain}|{f1['pos']}|{f2['pos']}"
                        pairs.append((hashlib.sha256(salt.encode()).hexdigest(),f1,f2))
                if not pairs:return None
                digest,a,b=min(pairs,key=lambda x:x[0])
                return {'group':rec['group'],'split':rec['split'],'accession':acc,'pos1':a['pos'],'pos2':b['pos'],'aa':a['aa'],'secondary':a['secondary'],'pdb_id':pid,'assembly_id':asm.name,'chain':chain,'assembly_chain':ci,'rsa1':a['rsa'],'rsa2':b['rsa'],'remote_contacts1':a['remote_contacts'],'remote_contacts2':b['remote_contacts'],'selection_digest':digest}
            eligible=[f for f in features if f['main_clean']]
            pair=select_pair('main_clean');pair_sensitivity=select_pair('center2_clean_sensitivity')
            attempts.append({'pdb_id':pid,'status':'quality_structure_selected','residues':len(features),'main_clean':len(eligible),'matched_pair':pair is not None})
            return {'status':'features_computed','split':rec['split'],'group':rec['group'],'accession':acc,'pdb_id':pid,'assembly_id':asm.name,'chain':chain,'assembly_chain':ci,'assembly_protein_atoms':len(coords),'mapped_exact_observed':len(bypos),'residues':features,'pair':pair,'pair_center2_sensitivity':pair_sensitivity,'attempts':attempts,'software':{'gemmi':gemmi.__version__,'biotite':biotite.__version__,'numpy':np.__version__},'computed_utc':utc(),'experimental_methods':methods,'source_record_resolution':candidate['resolution'],'coordinate_resolution':st.resolution}
        except Exception as e:
            attempts.append({'pdb_id':pid,'status':'error','error':str(e),'traceback':traceback.format_exc(limit=4)})
            # Unknown eligibility of a better-ranked candidate must not silently
            # change structure selection; resume this group after fixing/retrying.
            break
    status='attempt_error' if any(a['status']=='error' for a in attempts) else 'no_eligible_structure'
    return {'status':status,'split':rec['split'],'group':rec['group'],'accession':acc,'attempts':attempts,'computed_utc':utc()}

def process_group(group_records,force=False):
    r=group_records[0];p=OUT/f"groups/{r['split']}/{r['group']}.json"
    if p.exists():
        prior=read(p)
        eligible_em={(rec['accession'],c['pdb_id']) for rec in group_records for c in rec['candidates'] if c['method'].upper()=='EM' and c['resolution'] is not None and c['resolution']<=3.0}
        em_alias_bug=any((aa['accession'],a['pdb_id']) in eligible_em and a['status']=='source_method_or_resolution_ineligible' for aa in prior['accession_attempts'] for a in aa['attempts'])
        if not force and not em_alias_bug and not any(a['status']=='error' for aa in prior['accession_attempts'] for a in aa['attempts']):return prior
        write(OUT/f"implementation_errors/{r['split']}_{r['group']}.json",prior)
    tried=[]
    for rec in sorted(group_records,key=lambda x:x['accession']):
        result=process_accession(rec);tried.append({'accession':rec['accession'],'attempts':result['attempts']})
        if result['status'] in ('features_computed','attempt_error'):break
    result['accession_attempts']=tried
    write(p,result)
    return result

def export(cohort):
    found={};features=[];pairs=[];sensitivity=[]
    length_by_acc={r['accession']:len(r['sequence']) for r in cohort}
    for p in sorted((OUT/'groups').glob('*/*.json')):
        r=read(p);found[(r['split'],r['group'])]=r
        for f in r.get('residues',[]):
            f['native_length']=length_by_acc[f['accession']]
            f['mapped_chain_fraction']=r['mapped_exact_observed']/f['native_length']
        features.extend(r.get('residues',[]))
        if r.get('pair'):pairs.append(r['pair'])
        if r.get('pair_center2_sensitivity'):sensitivity.append(r['pair_center2_sensitivity'])
    def tsv(p,rr,fields=None):
        if fields is None:fields=list(rr[0]) if rr else []
        temp=Path(str(p)+f'.{threading.get_ident()}.tmp')
        with temp.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rr)
        temp.replace(p)
    if features:tsv(OUT/'residue_features.tsv',features)
    if pairs:tsv(OUT/'matched_pairs.tsv',pairs)
    if sensitivity:tsv(OUT/'matched_pairs_center2_sensitivity.tsv',sensitivity)
    seqs={r['accession']:{'sequence_index':r['accession'],'accession':r['accession'],'split':r['split'],'group':r['group'],'sequence':r['sequence']} for r in cohort}
    (OUT/'sequence_index.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in seqs.values()),encoding='utf-8')
    groups=sorted({(r['split'],r['group']) for r in cohort})
    ledger=[]
    for key in groups:
        r=found.get(key,{})
        ledger.append({'split':key[0],'group':key[1],'status':r.get('status','unattempted'),'accession':r.get('accession',''),'pdb_id':r.get('pdb_id',''),'quality_residues':len(r.get('residues',[])),'main_clean_residues':sum(x['main_clean'] for x in r.get('residues',[])),'center2_clean_residues':sum(x['center2_clean_sensitivity'] for x in r.get('residues',[])),'matched_pair':bool(r.get('pair')),'matched_pair_center2_sensitivity':bool(r.get('pair_center2_sensitivity'))})
    tsv(OUT/'group_coverage.tsv',ledger)
    accession_ledger=[]
    for rec in cohort:
        gr=found.get((rec['split'],rec['group']),{})
        attempt=next((a for a in gr.get('accession_attempts',[]) if a['accession']==rec['accession']),None)
        accession_ledger.append({'split':rec['split'],'group':rec['group'],'accession':rec['accession'],'status':('selected_group_accession' if gr.get('status')=='features_computed' and gr.get('accession')==rec['accession'] else 'attempted_not_selected' if attempt else 'not_attempted_after_group_selection' if gr.get('status')=='features_computed' else 'unattempted'),'source_candidate_PDBs':','.join(c['pdb_id'] for c in rec['candidates']),'group_selected_accession':gr.get('accession','') if gr.get('status')=='features_computed' else ''})
    tsv(OUT/'accession_coverage.tsv',accession_ledger)
    summary={'updated_utc':utc(),'population':{'D_pairs':432,'D_groups':421,'T_groups':1304},'source_scope':'old evidence PDB candidate IDs only; not exhaustive accession-PDB coverage','by_split':{},'no_model_predictions_read':True,'no_reserved_groups_used':True,'feature_file':'residue_features.tsv','pair_file':'matched_pairs.tsv','ledger_file':'group_coverage.tsv'}
    for split in ['D','T']:
        ll=[r for r in ledger if r['split']==split];ff=[r for r in features if r['split']==split]
        summary['by_split'][split]={'population_groups':len(ll),'attempted_groups':sum(r['status']!='unattempted' for r in ll),'quality_structure_groups':sum(r['status']=='features_computed' for r in ll),'matched_pair_groups':sum(r['matched_pair'] for r in ll),'matched_pair_center2_sensitivity_groups':sum(r['matched_pair_center2_sensitivity'] for r in ll),'quality_residues':len(ff),'main_clean_residues':sum(r['main_clean'] for r in ff),'center2_clean_sensitivity_residues':sum(r['center2_clean_sensitivity'] for r in ff),'interface_residues':sum(r['interface'] for r in ff),'status_counts':dict(Counter(r['status'] for r in ll))}
    write(OUT/'coverage_summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--limit-per-split',type=int,default=64);a.add_argument('--split',choices=['D','T','both'],default='both');a.add_argument('--export-only',action='store_true');a.add_argument('--prepare-only',action='store_true');args=a.parse_args()
    cohort=rows(OUT/'cohort_index.jsonl') if args.export_only and (OUT/'cohort_index.jsonl').exists() else prepare()
    if args.prepare_only:print('Prepared 432 D accessions / 421 D groups and 1304 T groups',flush=True);sys.exit()
    if not args.export_only:
        gr=defaultdict(list)
        for r in cohort:gr[(r['split'],r['group'])].append(r)
        keys=[]
        for split in ['D','T']:
            if args.split in [split,'both']:keys.extend(sorted(k for k in gr if k[0]==split)[:args.limit_per_split])
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(process_group,gr[k]):k for k in keys}
            for finished,future in enumerate(as_completed(pending),1):
                r=future.result();print(f"{r['split']} {r['group']} {r['status']} {r.get('pdb_id','')} residues={len(r.get('residues',[]))} pair={bool(r.get('pair'))}",flush=True)
                if finished%64==0:export(cohort)
    export(cohort)
