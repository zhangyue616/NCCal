"""Reuse the existing frozen physical annotation code with output redirected.
No existing project file/cache is written. Only new RESERVE/FRESH inputs.
"""
from prepare import *
from inventory import table
import importlib.util,re,shutil
from concurrent.futures import ThreadPoolExecutor
def main():
    archive=ROOT/'structure';source=archive/'build_structure_extension.py';spec=importlib.util.spec_from_file_location('exp2_physical_source',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    local=OUT/'physical';local.mkdir(exist_ok=True);mod.OUT=local
    oldfetch=mod.fetch
    def fetch(url,path):
        path=Path(path)
        if not path.exists():
            rel=path.relative_to(local);prior=archive/rel
            if prior.exists():path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior,path)
        return oldfetch(url,path)
    mod.fetch=fetch
    by={r['primaryAccession']:r for p in (OUT/'annotations').glob('uniprot_*.json') for r in js(p)['response']['results']};records=[]
    for r in table(OUT/'RESERVE_194_INVENTORY.tsv'):
        a=r['id'];u=by[a];refs={x['id'].lower():x for x in u.get('uniProtKBCrossReferences',[]) if x['database']=='PDB'};pdbs=set()
        for f in u.get('features',[]):
            if f['type'] in ['Helix','Beta strand']:
                for e in f.get('evidences',[]):
                    if e.get('source')=='PDB' and e.get('id'):pdbs.add(e['id'].lower())
        candidates=[]
        for pdb in pdbs:
            x=refs.get(pdb);props={p['key']:p['value'] for p in x.get('properties',[])} if x else {};m=re.search(r'[0-9]+(?:\.[0-9]+)?',props.get('Resolution',''))
            chains=sorted({c for part in props.get('Chains','').split(',') for c in part.split('=')[0].strip().split('/') if c})
            candidates.append(dict(pdb_id=pdb,resolution=float(m.group()) if m else None,method=props.get('Method',''),old_source_chains=chains))
        candidates.sort(key=lambda c:(c['resolution'] if c['resolution'] is not None else 999,c['pdb_id']))
        records.append(dict(accession=a,group=r['known_group_id'],sequence=r['sequence'],split='RESERVE',origin='original_unused194',candidates=candidates))
    for r in js(OUT/'new_source/STRUCTURE_IDENTITY.json')['entries']:
        if r['status']!='NATURAL_POLYMER_AND_SIFTS_IDENTITY_VERIFIED':continue
        records.append(dict(accession=r['accession'],group=r['group'],sequence=r['sequence'],split='FRESH',origin='new12_exact_native_SIFTS',candidates=[dict(pdb_id=r['representative_pdb'],resolution=3.0,method='X-ray',old_source_chains=r['mapping_chains'])]))
    write('physical/INPUTS.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_code':str(source),'source_sha256':sha(source),'inherited_rule_lock_sha256':sha(archive/'STRUCTURE_RULE_LOCK.md'),'source_population':'original194 UniProt evidence PDBs plus new12 validated canonical entities; select first quality structure before seeing pair existence','records':records,'activation_read':False,'FRESH_recorded_resolution':'3A is query upper bound; actual coordinate resolution independently checked'})
    (local/'cohort_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    def run(r):
        result=mod.process_group([r]);print(r['split'],r['accession'],result['status'],flush=True);return result
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,records))
    mod.export(records)
    write('physical/COMPLETE.json',{'utc':datetime.now(timezone.utc).isoformat(),'records':len(records),'status_counts':dict(__import__('collections').Counter(r['status'] for r in results)),'selected_structures':sum(r['status']=='features_computed' for r in results),'model_activations_read':False,'output_dir':str(local),'input_sha256':sha(local/'INPUTS.json')})
if __name__=='__main__':main()
