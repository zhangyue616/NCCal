"""Count endpoint eligibility without opening any model readouts."""
from prepare import *
from inventory import table,category
from collections import defaultdict,Counter
import numpy as np
def positive_mask(record,concept):
    aliases={'Helix':'Helix','Beta strand':'Beta strand','Turn':'Turn','Disulfide bond':'Disulfide bond','Lipidation':'Lipidation'}
    if concept in aliases:kind=aliases[concept];description=None
    elif '_' in concept:
        prefix,description=concept.split('_',1);kind={'Signal peptide':'Signal','Glycosylation':'Glycosylation','Modified residue':'Modified residue','Active site':'Active site','Binding site':'Binding site','Region':'Region'}.get(prefix)
        if description=='any':description=None
    else:return None,None
    if not kind:return None,None
    n=record['sequence']['length'];mask=np.zeros(n,bool);unknown=np.zeros(n,bool)
    for f in record.get('features',[]):
        if f['type']!=kind or (description is not None and f.get('description')!=description):continue
        loc=f.get('location',{});a=loc.get('start',{});b=loc.get('end',{})
        if not isinstance(a.get('value'),int) or not isinstance(b.get('value'),int):continue
        aa,bb=a['value']-1,b['value']
        if not (0<=aa<bb<=n):continue
        if a.get('modifier')!='EXACT' or b.get('modifier')!='EXACT':unknown[aa:bb]=True;continue
        if kind=='Disulfide bond':mask[aa]=True;mask[bb-1]=True
        else:mask[aa:bb]=True
    return mask,unknown
def main():
    by={r['primaryAccession']:r for p in (OUT/'annotations').glob('uniprot_*.json') for r in js(p)['response']['results']}
    bindings=[r for r in table(OUT/'annotations/ACCESSION_BINDINGS.tsv') if r['canonical_sequence_exact']=='True']
    concepts=sorted({r['concept'] for r in table(OUT/'PUBLIC_SELECTED.tsv')});rows=[]
    for concept in concepts:
        for split in ['D','T','RESERVE']:
            eligible_auc=eligible_pair=annotated=0;positive_residues=0;count=0
            for bind in bindings:
                if bind['split']!=split:continue
                count+=1;r=by[bind['accession']];mask,unknown=positive_mask(r,concept)
                if mask is None:continue
                positive_residues+=int(mask.sum())
                if not mask.any():continue
                annotated+=1;seq=r['sequence']['value'];valid=~unknown
                # Pair windows must be fully within the natural chain. Masks
                # describe annotation-vs-unannotated, not verified negatives.
                if (mask&valid).any() and ((~mask)&valid).any():eligible_auc+=1
                valid[:15]=False;valid[-15:]=False;ok=False
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    pos=[p for p,a in enumerate(seq) if a==aa and valid[p] and mask[p]]
                    neg=[p for p,a in enumerate(seq) if a==aa and valid[p] and not mask[p]]
                    if pos and neg and (max(pos)-min(neg)>30 or max(neg)-min(pos)>30):ok=True;break
                eligible_pair+=ok
            rows.append(dict(concept=concept,category=category(concept),split=split,mother_scope_chains=count,chains_with_positive_annotation=annotated,positive_annotated_residues=positive_residues,auc_eligible_chains=eligible_auc,same_AA_pair_eligible_chains=eligible_pair,confirmation_group_threshold20=eligible_pair>=20,negative_meaning='unannotated reference, not verified absence',scope='annotation-only eligibility; structural observability may further reduce; one archived chain per homology group'))
    with (OUT/'ANNOTATION_ELIGIBILITY.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    lookup={(r['concept'],r['split']):r for r in rows};selected=table(OUT/'PUBLIC_SELECTED.tsv');full=[]
    for r in selected:
        d=lookup[r['concept'],'D'];t=lookup[r['concept'],'T'];f=lookup[r['concept'],'RESERVE']
        full.append(dict(canonical_model=r['canonical_model'],block0=r['block0'],unit0=r['unit0'],pole=r['pole'],concept=r['concept'],category=r['annotation_category'],D_pair_groups=d['same_AA_pair_eligible_chains'],T_pair_groups=t['same_AA_pair_eligible_chains'],reserve_pair_groups=f['same_AA_pair_eligible_chains'],D_and_T_at_least20=d['same_AA_pair_eligible_chains']>=20 and t['same_AA_pair_eligible_chains']>=20))
    with (OUT/'PUBLIC_ELIGIBILITY_PRELIMINARY.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(full[0]),delimiter='\t');w.writeheader();w.writerows(full)
    write('ANNOTATION_ELIGIBILITY_SUMMARY.json',{'selected_units':len(full),'provisionally_D_T_pair_evaluable':sum(r['D_and_T_at_least20'] for r in full),'per_category':dict(Counter(r['category'] for r in full if r['D_and_T_at_least20'])),'unavailable_is_not_negative':True,'activation_read':False,'newly_recruited_metadata_sample_not_included':True,'structural_observability_and_group_alignment_pending':True})
    print(js(OUT/'ANNOTATION_ELIGIBILITY_SUMMARY.json'))
if __name__=='__main__':main()
