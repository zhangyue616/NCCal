"""Public UniProt metadata/annotation binding; never reads activations."""
from prepare import *
from inventory import jl,table
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import requests
FIELDS='accession,organism_id,sequence,xref_pfam,xref_pdb,ft_helix,ft_strand,ft_turn,ft_disulfid,ft_signal,ft_transmem,ft_binding,ft_act_site,ft_mod_res,ft_region'
def fetch(item):
    number,accs=item;p=OUT/'annotations'/f'uniprot_{number:03d}.json'
    if p.exists():return p
    query=' OR '.join('accession:'+a for a in accs)
    r=requests.get('https://rest.uniprot.org/uniprotkb/search',params={'query':query,'format':'json','size':500,'fields':FIELDS},timeout=60);r.raise_for_status()
    write(f'annotations/uniprot_{number:03d}.json',{'url':r.url,'requested_accessions':accs,'utc':datetime.now(timezone.utc).isoformat(),'response':r.json()});return p
def main():
    S=ROOT/'structure';source={r['accession']:r for r in jl(S/'sequence_index.jsonl')};keep=set()
    for r in table(S/'residue_features.tsv'):
        if r['main_clean']=='True' and r['interface']=='False':keep.add(r['accession'])
    for r in table(OUT/'RESERVE_194_INVENTORY.tsv'):
        if r['previously_used']=='False':source[r['id']]=dict(accession=r['id'],group=r['known_group_id'],sequence=r['sequence'],split='RESERVE');keep.add(r['id'])
    ids=sorted(keep);paths=list(ThreadPoolExecutor(max_workers=4).map(fetch,enumerate([ids[i:i+100] for i in range(0,len(ids),100)])))
    by={r['primaryAccession']:r for p in paths for r in js(p)['response']['results']};rows=[]
    for a in ids:
        old=source[a];r=by.get(a);seq=r.get('sequence',{}).get('value') if r else None;tax=r.get('organism',{}).get('taxonId') if r else None
        rows.append(dict(accession=a,group=old['group'],split=old['split'],length=len(old['sequence']),taxid=tax,canonical_sequence_exact=seq==old['sequence'],retrieved=r is not None,scope_eligible=tax in [9606,10090,4932,559292,3702],features=len(r.get('features',[])) if r else 0))
    with (OUT/'annotations/ACCESSION_BINDINGS.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    summary={'requested':len(ids),'retrieved':sum(r['retrieved'] for r in rows),'canonical_exact':sum(r['canonical_sequence_exact'] for r in rows),'within_existing_host_scope':sum(r['scope_eligible'] for r in rows),'per_split':{s:dict(Counter('eligible_exact' if r['scope_eligible'] and r['canonical_sequence_exact'] else 'metadata_or_scope_excluded' for r in rows if r['split']==s)) for s in ['D','T','RESERVE']},'activation_read':False,'missing_annotation_is_not_negative':True,'legacy_mother_frame_preserved':1099}
    write('annotations/SUMMARY.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
