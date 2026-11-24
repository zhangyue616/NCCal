"""Apply a task-scope metadata gate without reading RSA or model predictions."""
from repeat_structure import *

def native(acc):
    p=HERE/f'cache/uniprot/{acc}.json'
    if not p.exists():
        opts=[CROSS/f'external/sources/uniprot/{acc}.json',ROOT/f'nccal_evidence_extension/cohort/public_sources/uniprot/{acc}.json']
        src=next((x for x in opts if x.exists()),None)
        if src:
            p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
            write(str(p)+'.source.json',{'local_source':str(src),'sha256':sha(p),'copied_utc':utc()})
        else:ORIGINAL_FETCH(f'https://rest.uniprot.org/uniprotkb/{acc}.json',p)
    return read(p)

def inspect(pid,target,target_acc,target_canonical):
    b=block_for(pid)
    ids={e for e,t in table(b,['_entity_poly.entity_id','_entity_poly.type']) if 'polypeptide' in t}
    es=[e for e in entities(b) if e['entity'] in ids]
    fail=[];verified=[]
    refs=defaultdict(set)
    for eid,db,acc in table(b,['_struct_ref.entity_id','_struct_ref.db_name','_struct_ref.pdbx_db_accession']):
        if db in ('UNP','UNIPROT') and acc and '-' not in acc:refs[eid].add(acc)
    for e in es:
        why=None;acc=None
        if e['sequence']==target:acc=target_acc;seq=target_canonical
        elif len(refs[e['entity']])==1:
            acc=next(iter(refs[e['entity']]))
            try:
                u=native(acc);seq=u['sequence']['value']
                if u['organism']['taxonId'] not in ALLOWED:why='canonical_taxonomy_outside_scope'
                elif DANGER.search(native_name(u)):why='canonical_excluded_name'
            except Exception as ex:why='canonical_reference_unknown';seq=''
        else:why='companion_canonical_reference_unknown';seq=''
        why=why or valid_entity(e,seq)
        if why:fail.append(dict(entity=e['entity'],name=e['name'],reason=why,source_taxa=e['source_taxa']))
        else:verified.append(dict(entity=e['entity'],name=e['name'],accession=acc,native_length=len(seq),deposited_length=len(e['sequence']),source_taxa=e['source_taxa']))
    return {'pdb_id':pid,'eligible':bool(es) and not fail,'polypeptide_entities':len(es),'failures':fail,'verified_native_entities':verified}

def main():
    out=[]
    for z in jl(HERE/'frame.jsonl'):
        p=HERE/f"groups/{z['accession']}.json"
        if not p.exists():raise RuntimeError('execution not complete')
        # Project chosen identity only; no residue labels or predictor outcomes read.
        r=read(p);chosen=r.get('chosen')
        a=inspect(z['original_pdb'],z['original_entity']['sequence'],z['accession'],z['sequence'])
        b=inspect(chosen['pdb_id'],z['original_entity']['sequence'],z['accession'],z['sequence']) if chosen else None
        out.append({'group':z['group'],'accession':z['accession'],'original':a,'second':b,'eligible':a['eligible'] and bool(b and b['eligible'])})
    savejl(HERE/'safety_screen.jsonl',out)
    write(HERE/'safety_freeze.json',{'frozen_utc':utc(),'original_freeze_sha256':sha(HERE/'freeze.json'),'amendment_sha256':sha(HERE/'SAFETY_AMENDMENT.md'),'script_sha256':sha(Path(__file__)),'metadata_screen_sha256':sha(HERE/'safety_screen.jsonl'),'retained':sum(r['eligible'] for r in out),'new_RSAPrediction_aggregate_read':False,'note':'added during finite-list execution before statistical summaries'})
    print(json.dumps(read(HERE/'safety_freeze.json'),indent=2))

if __name__=='__main__':main()
