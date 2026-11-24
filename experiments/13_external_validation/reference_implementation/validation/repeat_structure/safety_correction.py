"""Correct canonical-field masking of modified monomers; retain initial output."""
from repeat_structure import *

def main():
    archive=HERE/'v1_before_raw_polymer_correction';archive.mkdir(exist_ok=True)
    for name in ('summary.json','REPORT_zh.md','ACCEPTANCE.json','group_funnel.tsv','pair_scores.tsv','fixed_sites.tsv','analysis.log'):
        if not (archive/name).exists():shutil.copyfile(HERE/name,archive/name)
    rows=jl(HERE/'safety_screen.jsonl');hits=[]
    for z in rows:
        for key in ('original','second'):
            q=z[key]
            if not q:continue
            b=block_for(q['pdb_id'])
            for eid,kind,raw in table(b,['_entity_poly.entity_id','_entity_poly.type','_entity_poly.pdbx_seq_one_letter_code']):
                if 'polypeptide' not in kind:continue
                raw=''.join(raw.split())
                if not raw or not set(raw)<=set('ACDEFGHIKLMNPQRSTVWY'):
                    failure={'entity':eid,'reason':'nonstandard_raw_polymer_monomer','codes':re.findall(r'\([^)]*\)',raw)}
                    q['failures'].append(failure);q['eligible']=False;hits.append(dict(accession=z['accession'],pdb_id=q['pdb_id'],entry_role=key,**failure))
        z['eligible']=z['original']['eligible'] and bool(z['second'] and z['second']['eligible'])
    savejl(HERE/'safety_screen_corrected.jsonl',rows)
    write(HERE/'safety_correction_freeze.json',{'corrected_utc':utc(),'prior_safety_freeze_sha256':sha(HERE/'safety_freeze.json'),'corrected_screen_sha256':sha(HERE/'safety_screen_corrected.jsonl'),'correction_script_sha256':sha(Path(__file__)),'retained':sum(r['eligible'] for r in rows),'raw_polymer_hits':hits,'first_summaries_had_been_seen':True,'reason':'concrete implementation defect: canonical polymer field normalizes modified monomers, contrary to already specified no-nonstandard rule','new_scientific_thresholds_or_predictors':False,'replacement_entries_or_pairs':False})
    print(json.dumps(read(HERE/'safety_correction_freeze.json'),indent=2))

if __name__=='__main__':main()
