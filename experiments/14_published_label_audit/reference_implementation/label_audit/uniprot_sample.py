"""Resolve the frozen RCSB metadata sample to public natural sequences/annotations."""
from prepare import *
import requests
from collections import Counter
def main():
    with (OUT/'new_source/METADATA_CANDIDATES.tsv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    ids=sorted({a for r in rows if r['metadata_eligible']=='True' for a in r['uniprot_ids'].split(';') if a})
    paths=[]
    for offset in range(0,len(ids),100):
        subset=ids[offset:offset+100];params={'query':' OR '.join('accession:'+a for a in subset),'format':'json','size':500,'fields':'accession,organism_id,sequence,xref_pfam,xref_pdb,ft_helix,ft_strand,ft_turn,ft_disulfid,ft_signal,ft_transmem,ft_binding,ft_act_site,ft_mod_res,ft_region'}
        p=OUT/'new_source'/f'UNIPROT_FULL_{offset:04d}.json'
        if not p.exists():
            r=requests.get('https://rest.uniprot.org/uniprotkb/search',params=params,timeout=60);r.raise_for_status();write(f'new_source/UNIPROT_FULL_{offset:04d}.json',{'url':r.url,'utc':datetime.now(timezone.utc).isoformat(),'response':r.json()})
        paths.append(p)
    records=[r for p in paths for r in js(p)['response']['results']];by={r['primaryAccession']:r for r in records};eligible={};failure=Counter()
    for r in rows:
        if r['metadata_eligible']!='True':continue
        accs=r['uniprot_ids'].split(';');matches=[]
        for acc in accs:
            u=by.get(acc)
            if not u:continue
            seq=u['sequence']['value'];tax=u.get('organism',{}).get('taxonId')
            if 60<=len(seq)<=1000 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY') and seq.count(r['sequence'])==1 and tax in [9606,10090,4932,559292,3702]:matches.append((acc,u))
        if len(matches)!=1:failure['nonunique_or_missing_exact_native_mapping']+=1;continue
        acc,u=matches[0];eligible.setdefault(acc,{'accession':acc,'sequence':u['sequence']['value'],'length':u['sequence']['length'],'taxid':u['organism']['taxonId'],'feature_types':dict(Counter(f['type'] for f in u.get('features',[]))),'pfam':[x['id'] for x in u.get('uniProtKBCrossReferences',[]) if x['database']=='Pfam'],'entities':[]})['entities'].append(r['entity_id'])
    write('new_source/EXACT_NATIVE_SAMPLE.json',{'entries':list(eligible.values()),'unique_accessions':len(eligible),'actual_native_residues':sum(r['length'] for r in eligible.values()),'requested_accessions':len(ids),'resolved_accessions':len(by),'failure_counts':dict(failure),'homology_status':'PENDING_NOT_CONFIRMATION_READY','activation_read':False})
    print('requested',len(ids),'resolved',len(by),'exact_native',len(eligible),'actual_residues',sum(r['length'] for r in eligible.values()))
if __name__=='__main__':main()
