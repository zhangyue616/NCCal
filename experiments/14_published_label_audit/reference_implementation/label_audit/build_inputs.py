"""Compile label and pair definitions without reading model activations."""
from prepare import *
from inventory import jl,table,category
from annotation_inventory import positive_mask
from collections import defaultdict,Counter
import numpy as np
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def choose_pair(sequence,positive,negative,group,concept,min_gap=30):
    # Stable seed defines independent site orderings; first compatible pair.
    L=len(sequence);positive=[p for p in positive if 15<=p<L-15];negative=[p for p in negative if 15<=p<L-15]
    pp=sorted(positive,key=lambda p:digest(f'2026112302|{concept}|{group}|positive|{p}'))
    nn=defaultdict(list)
    for p in sorted(negative,key=lambda p:digest(f'2026112302|{concept}|{group}|negative|{p}')):nn[sequence[p]].append(p)
    for p in pp:
        for q in nn[sequence[p]]:
            if abs(p-q)<=min_gap:continue
            a,b=sorted([p,q]);return dict(early0=a,late0=b,target=int(p==b),aa=sequence[p],positive0=p,negative0=q)
    return None
def compile_inputs():
    S=ROOT/'structure';seqmap={r['accession']:r for r in jl(S/'sequence_index.jsonl')}
    bindings=table(OUT/'annotations/ACCESSION_BINDINGS.tsv');res={r['id']:r for r in table(OUT/'RESERVE_194_INVENTORY.tsv')}
    source={r['primaryAccession']:r for p in (OUT/'annotations').glob('uniprot_*.json') for r in js(p)['response']['results']}
    records=[]
    recent={r['accession']:r['unseen_by_identity_and_recent_homology'] for r in js(OUT/'RESERVE_RECENT_HOMOLOGY.json')['rows']} if (OUT/'RESERVE_RECENT_HOMOLOGY.json').exists() else {}
    for r in bindings:
        if r['canonical_sequence_exact']!='True':continue
        a=r['accession'];old=seqmap.get(a,res.get(a));seq=old['sequence'];split=r['split'];group=r['group']
        records.append(dict(accession=a,group=group,sequence=seq,split=split,annotation=source[a],source='original_archive' if split!='RESERVE' else 'original194_unused_identity',own_fresh_eligible=bool(split=='RESERVE' and recent.get(a,False))))
    if (OUT/'new_source/STRUCTURE_IDENTITY.json').exists():
        fresh=[r for r in js(OUT/'new_source/STRUCTURE_IDENTITY.json')['entries'] if r['status']=='NATURAL_POLYMER_AND_SIFTS_IDENTITY_VERIFIED'];newann={r['primaryAccession']:r for p in (OUT/'new_source').glob('UNIPROT_FULL_*.json') for r in js(p)['response']['results']}
        for r in fresh:records.append(dict(accession=r['accession'],group=r['group'],sequence=r['sequence'],split='FRESH',annotation=newann[r['accession']],source='new_RCSB_exact_native_homology_screened',own_fresh_eligible=True))
    observed=defaultdict(set);exposed=defaultdict(set);buried=defaultdict(set)
    for path in [S/'residue_features.tsv',OUT/'physical/residue_features.tsv']:
        if not path.exists():continue
        for row in table(path):
            if row['main_clean']=='True' and row['interface']=='False':
                a=row['accession'];p=int(row['pos'])-1;observed[a].add(p)
                if float(row['rsa'])>=.4:exposed[a].add(p)
                if float(row['rsa'])<=.1:buried[a].add(p)
    concepts=sorted({r['concept'] for r in table(OUT/'PUBLIC_SELECTED.tsv')});pairs=[];labels={};counts=[]
    for concept in concepts:
        for r in records:
            a=r['accession'];mask,unknown=positive_mask(r['annotation'],concept)
            if mask is None:continue
            positive=np.flatnonzero(mask&~unknown).tolist();negative=np.flatnonzero(~mask&~unknown).tolist()
            if category(concept) in ['alpha_helix','beta_strand','turn']:
                positive=[p for p in positive if p in observed[a]];negative=[p for p in negative if p in observed[a]]
            if not positive or not negative:continue
            labels.setdefault(a,{})[concept]={'positive0':positive,'negative0':negative,'unknown0':np.flatnonzero(unknown).tolist(),'negative_interpretation':'not annotated, not verified absence'}
            pair=choose_pair(r['sequence'],positive,negative,r['group'],concept)
            if pair:pairs.append(dict(accession=a,group=r['group'],split=r['split'],concept=concept,**pair))
    # Own exposure uses the inherited matched structural pairs and RSA cutoffs.
    for path in [S/'matched_pairs.tsv',OUT/'physical/matched_pairs.tsv']:
        if not path.exists():continue
        for r in table(path):
            if r['accession'] not in {z['accession'] for z in records}:continue
            a,b=sorted([int(r['pos1'])-1,int(r['pos2'])-1]);rec=next(z for z in records if z['accession']==r['accession']);pairs.append(dict(accession=r['accession'],group=r['group'],split=r['split'],concept='OWN_EXPOSURE',early0=a,late0=b,target=int(b==int(r['pos2'])-1),aa=rec['sequence'][a],positive0=int(r['pos2'])-1,negative0=int(r['pos1'])-1))
    for r in records:
        acc=r['accession']
        if exposed[acc] and buried[acc]:labels.setdefault(acc,{})['OWN_EXPOSURE']={'positive0':sorted(exposed[acc]),'negative0':sorted(buried[acc]),'unknown0':[],'negative_interpretation':'RSA<=0.10 versus >=0.40 under inherited assembly rule'}
        h,hu=positive_mask(r['annotation'],'Helix');b,bu=positive_mask(r['annotation'],'Beta strand')
        hp=[int(p) for p in np.flatnonzero(h&~hu) if p in observed[acc]];bp=[int(p) for p in np.flatnonzero(b&~bu) if p in observed[acc]]
        if hp and bp:
            labels.setdefault(acc,{})['OWN_HELIX_VS_BETA']={'positive0':hp,'negative0':bp,'unknown0':[],'negative_interpretation':'experimentally annotated Beta strand; does not include coil'}
            pair=choose_pair(r['sequence'],hp,bp,r['group'],'OWN_HELIX_VS_BETA')
            if pair:pairs.append(dict(accession=acc,group=r['group'],split=r['split'],concept='OWN_HELIX_VS_BETA',**pair))
        ah=[p for p in hp if r['sequence'][p]=='A'];ab=[p for p in bp if r['sequence'][p]=='A']
        if ah and ab:
            labels.setdefault(acc,{})['OWN_ALA_HELIX_VS_BETA']={'positive0':ah,'negative0':ab,'unknown0':[],'negative_interpretation':'Ala in Beta strand versus Ala in Helix; original masked1246 label'}
            pair=choose_pair(r['sequence'],ah,ab,r['group'],'OWN_ALA_HELIX_VS_BETA',min_gap=0)
            if pair:pairs.append(dict(accession=acc,group=r['group'],split=r['split'],concept='OWN_ALA_HELIX_VS_BETA',**pair))
        # Strict same-long-helix start/interior; choose one pair per group.
        options=[]
        for feature in r['annotation'].get('features',[]):
            if feature['type']!='Helix':continue
            loc=feature.get('location',{});aa=loc.get('start',{});bb=loc.get('end',{})
            if aa.get('modifier')!='EXACT' or bb.get('modifier')!='EXACT':continue
            a,b=aa.get('value',0)-1,bb.get('value',0)
            if a<0 or b>len(r['sequence']) or b-a<10:continue
            if not all(p in observed[acc] for p in range(a-1,b+1)):continue
            pair=choose_pair(r['sequence'],list(range(a,a+2)),list(range(a+3,b-3)),r['group'],'OWN_HELIX_START',min_gap=0)
            if pair:options.append(pair)
        if options:
            pair=min(options,key=lambda p:digest(f"2026112302|{r['group']}|{p['early0']}|{p['late0']}"));pairs.append(dict(accession=r['accession'],group=r['group'],split=r['split'],concept='OWN_HELIX_START',**pair))
            labels.setdefault(acc,{})['OWN_HELIX_START']={'positive0':[pair['positive0']],'negative0':[pair['negative0']],'unknown0':[],'negative_interpretation':'same-chain same-AA interior of same fully observed long helix; AUC descriptive on the fixed pair'}
    assert len({(p['group'],p['concept']) for p in pairs})==len(pairs)
    for p in pairs:
        r=next(x for x in records if x['accession']==p['accession']);assert r['sequence'][p['early0']]==p['aa']==r['sequence'][p['late0']]
    for r in records:
        r.pop('annotation');r['sequence_sha256']=digest(r['sequence'])
    (OUT/'inputs').mkdir(exist_ok=True)
    with (OUT/'inputs/sequences.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write('inputs/LABELS.json',labels)
    with (OUT/'inputs/pairs.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]),delimiter='\t');w.writeheader();w.writerows(pairs)
    summary={'records_by_split':dict(Counter(r['split'] for r in records)),'pairs_by_concept_split':dict(Counter(p['concept']+'|'+p['split'] for p in pairs)),'activation_read':False,'new_structure_SIFTS_status':'must be complete before FRESH scoring; other input splits separately frozen','label_negative_interpretation':'unannotated reference, not verified absence','pair_rule':'seeded independent positive/negative site order then first same-AA distance>30 with complete±15; OWN_HELIX_START retains strict same-helix definition and distance>0'}
    write('inputs/SUMMARY.json',summary);print(summary['records_by_split'],len(pairs),flush=True)
if __name__=='__main__':compile_inputs()
