"""Fixed, outcome-blind independent experimental natural-protein structure tranche."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,csv,gzip,hashlib,time,traceback,importlib.util,argparse
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCHIVE=ROOT/'structure'
sys.path.insert(0,str(ARCHIVE/'runtime'))
sys.dont_write_bytecode=True
import requests,numpy as np,gemmi
SPEC=importlib.util.spec_from_file_location('archived_geometry',ARCHIVE/'build_structure_extension.py')
BASE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BASE)
BASE.OUT=HERE
TAXIDS=[9606,10090,559292,4932,3702,4530,7227,6239,7955,224308,1423,83333,511145,300852,224324,243274]
AA=set('ACDEFGHIKLMNPQRSTVWY')
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):return BASE.write(p,x)
def sha(p):return BASE.sha(p)
def jl(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def savejl(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf-8')
def tsv(p,rr,fields=None):
    if not fields:fields=list(rr[0]) if rr else []
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rr)
def download(url,path,params=None):
    path=Path(path)
    if path.exists():return path
    for i in range(3):
        try:
            r=requests.get(url,params=params,timeout=(15,90));r.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
            write(str(path)+'.source.json',{'url':r.url,'retrieved_utc':utc(),'status':r.status_code,'bytes':len(r.content),'sha256':sha(path),'last_modified':r.headers.get('Last-Modified')})
            return path
        except Exception:
            if i==2:raise
            time.sleep(1)

def exclusions():
    dest=HERE/'exclusion_references.jsonl'
    if dest.exists():return list(jl(dest))
    refs=[]
    for r in jl(ARCHIVE/'cohort_index.jsonl'):
        refs.append({'accession':r['accession'],'group':r['group'],'split':r['split'],'sequence':r['sequence'],'pdb_ids':sorted(c['pdb_id'].upper() for c in r['candidates'])})
    fresh=ROOT/'data/revision/protein_fresh'
    reserved={r['id']:r for r in jl(fresh/'unused_component_inventory.jsonl')}
    for p in sorted((fresh/'raw').glob('page*.json')):
        for r in read(p)['results']:
            acc=r['primaryAccession']
            if acc in reserved:
                z=reserved.pop(acc)
                refs.append({'accession':acc,'group':z['known_group_id'],'split':'RESERVED','sequence':r['sequence']['value'],'pdb_ids':sorted({x['id'].upper() for x in r.get('uniProtKBCrossReferences',[]) if x['database']=='PDB'})})
    if reserved:raise RuntimeError(f'Unresolved reserved sequences: {len(reserved)}')
    assert len(refs)==1930 and sum(r['split']=='RESERVED' for r in refs)==194
    savejl(dest,refs)
    write(HERE/'EXCLUSION_INPUT_RECEIPT.json',{'records':len(refs),'counts':dict(Counter(r['split'] for r in refs)),'sources':[{'path':str(p),'sha256':sha(p)} for p in [ARCHIVE/'cohort_index.jsonl',fresh/'unused_component_inventory.jsonl']],'raw_reserved_source':'same archived 28 UniProt source pages; only 194 unused representative native sequence and PDB metadata projected','model_outputs_read':False})
    return refs

def frame():
    nodes=[]
    for att,op,val in [('exptl.method','exact_match','X-RAY DIFFRACTION'),('rcsb_entry_info.resolution_combined','less_or_equal',2.0),('rcsb_accession_info.initial_release_date','range',{'from':'2025-01-01','to':'2026-11-01'}),('rcsb_entity_source_organism.taxonomy_lineage.id','in',list(map(str,TAXIDS))),('entity_poly.rcsb_sample_sequence_length','range',{'from':96,'to':512})]:
        nodes.append({'type':'terminal','service':'text','parameters':{'attribute':att,'operator':op,'value':val}})
    q={'query':{'type':'group','logical_operator':'and','nodes':nodes},'return_type':'polymer_entity','request_options':{'return_all_hits':True,'results_content_type':['experimental']}}
    write(HERE/'SEARCH_QUERY.json',q)
    response=read(download('https://search.rcsb.org/rcsbsearch/v2/query',HERE/'sources/search_response.json',{'json':json.dumps(q)}))
    ids=sorted(r['identifier'] for r in response['result_set']);assert len(ids)==response['total_count']
    clusters=download('https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt',HERE/'sources/clusters-by-entity-30.txt')
    entity_cluster={};cluster_members={};pdb_clusters=defaultdict(set)
    for line in clusters.read_text().splitlines():
        members=sorted(line.split());cid=hashlib.sha256(' '.join(members).encode()).hexdigest()
        cluster_members[cid]=members
        for member in members:
            entity_cluster[member]=cid;pdb_clusters[member.split('_')[0]].add(cid)
    refs=exclusions();blocked=set();missing=[]
    for r in refs:
        cs=set().union(*(pdb_clusters[p] for p in r['pdb_ids'])) if r['pdb_ids'] else set()
        r['source_cluster_ids']=sorted(cs)
        if not cs:missing.append(r)
        blocked.update(cs)
    savejl(HERE/'exclusion_coverage.jsonl',refs)
    candidates=defaultdict(list);excluded=[]
    for eid in ids:
        cid=entity_cluster.get(eid)
        if cid is None:excluded.append({'entity_id':eid,'reason':'no_official_cluster'})
        elif cid in blocked:excluded.append({'entity_id':eid,'reason':'old_or_reserved_source_PDB_cluster'})
        else:candidates[cid].append(eid)
    chosen=sorted(candidates,key=lambda c:hashlib.sha256(('nccal-independent-experimental-20261122-v1|'+c).encode()).hexdigest())[:256]
    groups=[{'group':'external30-'+c,'cluster_id':c,'candidate_entity_ids':candidates[c],'cluster_members':cluster_members[c]} for c in chosen]
    savejl(HERE/'frozen_groups.jsonl',groups);savejl(HERE/'search_exclusions.jsonl',excluded)
    frozen_utc=read(HERE/'FRAME_RECEIPT.json')['frozen_utc'] if (HERE/'FRAME_RECEIPT.json').exists() else utc()
    write(HERE/'FRAME_RECEIPT.json',{'frozen_utc':frozen_utc,'protocol_sha256':sha(HERE/'PROTOCOL.md'),'query_sha256':sha(HERE/'SEARCH_QUERY.json'),'search_response_sha256':sha(HERE/'sources/search_response.json'),'official_cluster_sha256':sha(clusters),'search_entities':len(ids),'nonblocked_clusters':len(candidates),'retained_clusters':len(groups),'exclusion_reference_records':len(refs),'references_with_no_source_cluster':len(missing),'missing_reference_accessions':[r['accession'] for r in missing],'search_entity_exclusions':dict(Counter(r['reason'] for r in excluded)),'result_blind':True,'model_outcomes_read':False})
    print('FRAME',json.dumps(read(HERE/'FRAME_RECEIPT.json')),flush=True)
    return groups,missing

def graphql_batch(job):
    index,ids=job;p=HERE/f'sources/entity_batch_{index:03d}.json'
    query='''query{polymer_entities(entity_ids:IDS){rcsb_id entity_poly{pdbx_seq_one_letter_code_can rcsb_mutation_count rcsb_conflict_count rcsb_non_std_monomer_count rcsb_entity_polymer_type} rcsb_polymer_entity{pdbx_mutation} rcsb_polymer_entity_container_identifiers{uniprot_ids} rcsb_entity_source_organism{ncbi_taxonomy_id taxonomy_lineage{id name}}} entries(entry_ids:PDBS){rcsb_id rcsb_entry_info{resolution_combined}}}'''.replace('IDS',json.dumps(ids)).replace('PDBS',json.dumps(sorted({s.split('_')[0] for s in ids})))
    result=read(download('https://data.rcsb.org/graphql',p,{'query':query}))
    if result.get('errors'):raise RuntimeError(result['errors'])
    return result['data']

def native_sequence(acc):
    r=read(download('https://rest.uniprot.org/uniprotkb/'+acc+'.json',HERE/f'sources/uniprot/{acc}.json'))
    seq=r['sequence']['value']
    return seq,r

def local_homology(seq,missing):
    if not missing:return None
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal,SubstitutionMatrix
    a=ProteinSequence(seq);m=SubstitutionMatrix.std_protein_matrix()
    for ref in missing:
        al=align_optimal(a,ProteinSequence(ref['sequence']),m,gap_penalty=(-10,-1),local=True,max_number=1)[0]
        t=al.trace;both=t[(t[:,0]>=0)&(t[:,1]>=0)]
        if len(both)==0:continue
        identity=sum(seq[i]==ref['sequence'][j] for i,j in both)/len(both)
        shorter=min(len(seq),len(ref['sequence']))
        coverage=min(len(set(t[t[:,0]>=0,0])),len(set(t[t[:,1]>=0,1])))/shorter
        if identity>=.30 and coverage>=.80:return {'reference':ref['accession'],'identity':identity,'coverage_shorter':coverage,'alignment_score':int(al.score)}
    return None

def source_candidates(groups,missing):
    ids=sorted({i for g in groups for i in g['candidate_entity_ids']})
    jobs=[(j,ids[i:i+80]) for j,i in enumerate(range(0,len(ids),80))]
    entities={};resolution={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for data in pool.map(graphql_batch,jobs):
            entities.update((x['rcsb_id'],x) for x in data['polymer_entities'] if x)
            resolution.update((x['rcsb_id'],min(x['rcsb_entry_info'].get('resolution_combined') or [999])) for x in data['entries'] if x)
    old=exclusions();old_acc={r['accession'] for r in old};old_seq={r['sequence'] for r in old}
    out=[];ledger=[]
    for g in groups:
        candidates=[]
        for eid in sorted(g['candidate_entity_ids'],key=lambda i:(resolution.get(i.split('_')[0],999),i)):
            e=entities.get(eid);reason=None
            if e is None:reason='metadata_unavailable'
            else:
                p=e['entity_poly'];org=e.get('rcsb_entity_source_organism') or []
                source_allowed=bool(org) and all(set(TAXIDS)&{int(t['id']) for t in z.get('taxonomy_lineage',[])} for z in org)
                refs=e['rcsb_polymer_entity_container_identifiers'].get('uniprot_ids') or []
                if p['rcsb_entity_polymer_type']!='Protein' or not source_allowed:reason='not_allowed_source_protein'
                elif any(p.get(k,0) for k in ['rcsb_mutation_count','rcsb_conflict_count','rcsb_non_std_monomer_count']):reason='annotated_mutation_conflict_or_nonstandard'
                elif len(refs)!=1 or '-' in refs[0]:reason='not_single_canonical_UniProt'
                elif refs[0] in old_acc:reason='old_or_reserved_accession'
                elif resolution.get(eid.split('_')[0],999)>2:reason='resolution_ineligible'
                else:candidates.append({'entity_id':eid,'pdb_id':eid.split('_')[0].lower(),'accession':refs[0],'resolution':resolution[eid.split('_')[0]],'source_organism':org})
            ledger.append({'group':g['group'],'entity_id':eid,'status':reason or 'metadata_eligible'})
        out.append(dict(g,candidates=candidates))
    savejl(HERE/'candidate_metadata_ledger.jsonl',ledger);savejl(HERE/'ranked_candidates.jsonl',out)
    print('METADATA',len(ids),'entities',sum(bool(g['candidates']) for g in out),'eligible clusters',flush=True)
    return out

def counts_for_result(r,seq):
    block=gemmi.cif.read_string(gzip.decompress((HERE/f"cache/mmcif/{r['pdb_id']}.cif.gz").read_bytes()).decode()).sole_block()
    st=gemmi.make_structure_from_block(block);asm=next(a for a in st.assemblies if a.name==r['assembly_id'])
    model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
    mp=BASE.sifts_map(r['pdb_id'],r['accession']);bypos={}
    chain=model[r['assembly_chain']]
    for res in chain:
        if res.entity_type!=gemmi.EntityType.Polymer:continue
        rid=str(res.seqid.num)+(res.seqid.icode.strip() or '');ref=mp.get(chain.name,{}).get((chain.name,rid));aa=BASE.AA3.get(res.name)
        if ref is None or aa is None:continue
        pos,mapped_aa,mapped_name=ref
        if not (1<=pos<=len(seq) and seq[pos-1]==aa and mapped_aa==aa and mapped_name==res.name):continue
        atoms={a.name:np.array([a.pos.x,a.pos.y,a.pos.z]) for a in res if a.element.name not in ('H','D')}
        name='CA' if aa=='G' else 'CB'
        if name in atoms:bypos[pos]=atoms[name]
    positions=np.array(sorted(bypos));xyz=np.array([bypos[p] for p in positions])
    for f in r['residues']:
        distance=np.linalg.norm(xyz-bypos[f['pos']],axis=1);gap=np.abs(positions-f['pos'])
        f['total']=int(np.count_nonzero((distance<=8)&(gap>=3)))
        f['remote']=int(np.count_nonzero((distance<=8)&(gap>=24)))
        if f['remote']!=f['remote_contacts']:raise ValueError('Reconstructed remote contact mismatch')
    return r

def process_group(job):
    g,missing=job;dest=HERE/f"groups/{g['group']}.json"
    if dest.exists():
        prior=read(dest)
        if prior.get('status')=='features_computed' and prior.get('contact_counter_version')!=2:
            previous=[(f['total'],f['remote']) for f in prior['residues']]
            prior=counts_for_result(prior,prior['sequence']);prior['contact_counter_version']=2
            prior['polymer_partner_filter_update_changed_sites']=sum(a!=(f['total'],f['remote']) for a,f in zip(previous,prior['residues']))
            write(dest,prior)
        return prior
    old=exclusions();old_seq={z['sequence'] for z in old};attempts=[]
    for c in g['candidates']:
        try:
            seq,u=native_sequence(c['accession'])
            if not 96<=len(seq)<=512 or not set(seq)<=AA:attempts.append(dict(c,status='native_sequence_length_or_nonstandard'));continue
            if seq in old_seq:attempts.append(dict(c,status='old_or_reserved_exact_native_sequence'));continue
            hom=local_homology(seq,missing)
            if hom:attempts.append(dict(c,status='missing_reference_local_homology',homology=hom));continue
            rec={'group':g['group'],'split':'EXTERNAL','accession':c['accession'],'sequence':seq,'candidates':[dict(c,method='X-RAY DIFFRACTION')]}
            r=BASE.process_accession(rec)
            attempts.append(dict(c,status=r['status'],structure_attempts=r['attempts']))
            if r['status']=='attempt_error':break
            if r['status']=='features_computed':
                r=counts_for_result(r,seq);r['contact_counter_version']=2;r['sequence']=seq;r['source_candidate']=c;r['cluster_id']=g['cluster_id'];r['all_candidate_attempts']=attempts
                write(dest,r);print('STRUCTURE',g['group'][:20],c['accession'],len(r['residues']),sum(f['main_clean'] for f in r['residues']),flush=True);return r
        except Exception as ex:
            attempts.append(dict(c,status='source_or_compute_error',error=str(ex),traceback=traceback.format_exc(limit=4)));break
    r={'group':g['group'],'status':'error' if any(x['status'] in ('source_or_compute_error','attempt_error') for x in attempts) else 'no_eligible_structure','all_candidate_attempts':attempts}
    write(dest,r);return r

def export(groups,results):
    sequences=[];sites=[];all_features=[];coverage=[];seen_accessions=set();seen_sequences=set()
    for g,r in zip(groups,results):
        status=r['status'];include='sequence' in r
        if include and (r['accession'] in seen_accessions or r['sequence'] in seen_sequences):
            status='external_sequence_duplicate';include=False
        if include and (not 0<float(r['coordinate_resolution'])<=2.0 or 'X-RAY DIFFRACTION' not in [m.strip("'\"") for m in r['experimental_methods']]):
            status='coordinate_frame_rule_ineligible';include=False
        ff=r.get('residues',[]) if include else [];clean=[f for f in ff if f['main_clean'] and not f['interface']]
        if include:
            seen_accessions.add(r['accession']);seen_sequences.add(r['sequence'])
            sequences.append({'accession':r['accession'],'id':r['accession'],'group':g['group'],'sequence':r['sequence'],'pdb_id':r['pdb_id'],'chain':r['chain'],'assembly_id':r['assembly_id'],'assembly_chain':r['assembly_chain'],'cluster_id':g['cluster_id']})
            for f in ff:
                f['native_length']=len(r['sequence']);f['mapped_chain_fraction']=r['mapped_exact_observed']/len(r['sequence'])
            for f in clean:
                sites.append(dict(f,id=f['accession'],position=f['pos'],AA=f['aa'],P_SEA=f['secondary'],RSA=f['rsa']))
            all_features.extend(ff)
        coverage.append({'group':g['group'],'status':status,'candidate_entities':len(g['candidate_entity_ids']),'metadata_eligible_candidates':len(g['candidates']),'accession':r.get('accession',''),'quality_sites':len(ff),'main_clean_sites':len(clean),'pdb_id':r.get('pdb_id',''),'coordinate_resolution':r.get('coordinate_resolution','')})
    savejl(HERE/'sequences.jsonl',sequences);tsv(HERE/'sites.tsv',sites,['accession','group','pos','aa','secondary','rsa','total','remote','pdb_id','chain','assembly_id','assembly_chain','author_residue','main_clean','interface','rsa_isolated','sasa_A2','remote_eligible_partners','mean_b','native_length','mapped_chain_fraction','id','position','AA','P_SEA','RSA'])
    tsv(HERE/'all_quality_features.tsv',all_features);tsv(HERE/'group_coverage.tsv',coverage)
    summary={'completed_utc':utc(),'status':'COMPLETE_WITH_SOURCE_ERRORS' if any(r['status']=='error' for r in results) else 'COMPLETE','frozen_clusters':len(groups),'selected_structures':len(sequences),'groups_with_main_clean_sites':sum(r['main_clean_sites']>0 for r in coverage),'main_clean_sites':len(sites),'remote_positive_main_sites':sum(f['remote']>0 for f in sites),'remote_zero_main_sites':sum(f['remote']==0 for f in sites),'all_quality_sites':len(all_features),'status_counts':dict(Counter(r['status'] for r in coverage)),'metadata_filter_counts':dict(Counter(r['status'] for r in jl(HERE/'candidate_metadata_ledger.jsonl'))),'source_or_quality_attempt_status_counts':dict(Counter(a['status'] for r in results for a in r.get('all_candidate_attempts',[]))),'old_reserved_references_with_no_PDB_cluster':read(HERE/'FRAME_RECEIPT.json')['references_with_no_source_cluster'],'model_activations_or_results_read':False,'old_files_written':False,'geometry_definition':'same-chain CB/Gly-CA <=8A; total gap>=3; remote gap>=24; main-clean non-interface only in sites.tsv','independence_limit':'Source-PDB30 cluster screen over 1930 old/reserved representatives plus accession/exact native sequence exclusion; no exhaustive all-pairs or remote-homology independence claim','files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [HERE/'sequences.jsonl',HERE/'sites.tsv',HERE/'all_quality_features.tsv',HERE/'group_coverage.tsv',HERE/'PROTOCOL.md',HERE/'TECHNICAL_AMENDMENT.md',HERE/'build_external.py']}}
    write(HERE/'DELIVERY.json',summary);print('DELIVERY',json.dumps(summary),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-only',action='store_true');args=ap.parse_args()
    groups,missing=frame();groups=source_candidates(groups,missing)
    if args.source_only:return
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(process_group,[(g,missing) for g in groups]))
    export(groups,results)
if __name__=='__main__':main()
