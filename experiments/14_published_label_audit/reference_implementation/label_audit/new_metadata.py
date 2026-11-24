"""RCSB metadata-only feasibility sample from a predeclared source frame."""
from prepare import *
import requests
from concurrent.futures import ThreadPoolExecutor
def main():
    fresh=ROOT/'structural_readout_validation/fresh';response=js(fresh/'search_response.json')
    used=set()
    # Every locally acquired PDB is conservatively prior-exposed.
    caches=[ROOT/'structure/cache/mmcif',ROOT/'cross_architecture/external/cache/mmcif',ROOT/'nccal_evidence_extension/cohort/cache/mmcif',ROOT/'native_channels/cohort/cache/mmcif',fresh/'sources/mmcif']
    for folder in caches:
        if folder.exists():
            for p in folder.glob('*.cif*'):used.add(p.name.split('.')[0].upper())
    ids=sorted((r['identifier'].upper() for r in response['result_set'] if r['identifier'].upper() not in used),key=lambda x:hashlib.sha256(('label-audit-20261123|'+x).encode()).hexdigest())[:1000]
    write('new_source/SOURCE_FRAME.json',{'input':str(fresh/'search_response.json'),'input_sha256':sha(fresh/'search_response.json'),'query_sha256':sha(fresh/'search_query.json'),'original_hits':response['total_count'],'prior_PDB_count':len(used),'metadata_frame_ids':ids,'metadata_sample_first_n':100,'model_activation_read':False,'scope':'exact prior source dates, X-ray<=3A, host taxa; no new PDB source query expansion'})
    def fetch(offset):
        names=ids[offset:offset+25]
        query='{ entries(entry_ids:'+json.dumps(names)+') { rcsb_id polymer_entities { rcsb_id entity_poly { pdbx_seq_one_letter_code_can } rcsb_polymer_entity { pdbx_description pdbx_mutation } rcsb_polymer_entity_container_identifiers { uniprot_ids } rcsb_entity_source_organism { ncbi_taxonomy_id scientific_name } } } }'
        out=OUT/'new_source'/f'metadata_{offset:04d}.json'
        if out.exists():return
        r=requests.get('https://data.rcsb.org/graphql',params={'query':query},timeout=45);r.raise_for_status();j=r.json()
        if j.get('errors'):raise RuntimeError(j['errors'])
        write(f'new_source/metadata_{offset:04d}.json',{'url':'https://data.rcsb.org/graphql','query':query,'utc':datetime.now(timezone.utc).isoformat(),'response':j})
        print(offset,len(j['data']['entries']),flush=True)
    list(ThreadPoolExecutor(max_workers=4).map(fetch,range(0,len(ids),25)))
    rows=[]
    for p in (OUT/'new_source').glob('metadata_*.json'):
        if not p.stem.removeprefix('metadata_').isdigit():continue
        for entry in js(p)['response']['data']['entries']:
            if not entry:continue
            for entity in entry['polymer_entities'] or []:
                seq=(entity.get('entity_poly') or {}).get('pdbx_seq_one_letter_code_can','').replace('\n','')
                ids=(entity.get('rcsb_polymer_entity_container_identifiers') or {}).get('uniprot_ids') or []
                desc=(entity.get('rcsb_polymer_entity') or {}).get('pdbx_description') or ''
                mutation=(entity.get('rcsb_polymer_entity') or {}).get('pdbx_mutation')
                taxa=[int(x['ncbi_taxonomy_id']) for x in entity.get('rcsb_entity_source_organism') or [] if x.get('ncbi_taxonomy_id')]
                eligible=60<=len(seq)<=1000 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY') and bool(ids) and not mutation and bool(taxa) and all(t in [9606,10090,4932,559292,3702] for t in taxa) and not any(x in desc.lower() for x in ['antibody','immunoglobulin','fusion','synthetic','designed'])
                rows.append(dict(pdb_id=entry['rcsb_id'],entity_id=entity['rcsb_id'],sequence=seq,length=len(seq),uniprot_ids=';'.join(ids),taxids=';'.join(map(str,taxa)),description=desc,metadata_eligible=eligible,exact_native_and_homology='PENDING_NOT_FINAL_QUEUE'))
    p=OUT/'new_source/METADATA_CANDIDATES.tsv'
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    eligible=[r for r in rows if r['metadata_eligible']]
    write('new_source/METADATA_SUMMARY.json',{'sampled_entries':len(ids),'entities':len(rows),'metadata_eligible_entities':len(eligible),'actual_entity_residues':sum(r['length'] for r in eligible),'max_length':max([r['length'] for r in eligible],default=0),'no_structure_download':True,'no_model_forward':True,'final_fresh_groups':'UNKNOWN; exact canonical native mapping and same legacy homology exclusion must precede final freeze','continuation':'complete frozen remaining metadata frame under COMPUTE_GO; no result-based sampling'})
    print('eligible',len(eligible),sum(r['length'] for r in eligible))
if __name__=='__main__':main()
