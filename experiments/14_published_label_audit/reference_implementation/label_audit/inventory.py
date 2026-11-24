"""Result-blind source sampling and natural-sequence inventory."""
from prepare import *
sys.path.insert(0,str(ROOT/'disulfide/vendor'))
import pandas as pd,numpy as np,re
from collections import Counter,defaultdict
def jl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def category(label):
    s=label.lower()
    if s=='helix' or s in ['alpha helix','α helix']:return 'alpha_helix'
    if s in ['beta strand','β strand','beta sheet']:return 'beta_strand'
    if s=='turn':return 'turn'
    if s=='disulfide bond':return 'disulfide'
    if s.startswith('signal peptide'):return 'signal_peptide'
    if s.startswith('transmembrane'):return 'transmembrane'
    if s.startswith('binding site_') and any(x in s for x in ['zinc','iron','copper','metal','magnesium','manganese','calcium','cobalt','nickel']):return 'metal_binding'
    if s.startswith(('active site_','binding site_')):return 'active_or_binding'
    if s.startswith(('modified residue_','glycosylation_')) or s=='lipidation':return 'ptm'
    if s in ['region_disordered','disordered region']:return 'disorder'
    return None
def sources():
    tables=[('esm2_8m','retest__esm2_8m__e1_per_concept.parquet',1280),('esm2_650m','retest__esm2_650m__e1_per_concept.parquet',5120),('esmc_600','retest_esmc__esmc_600_depth__e13_per_concept.parquet',3072)]
    records=[];excluded=[]
    for model,file,width in tables:
        df=pd.read_parquet(OUT/'sources'/file)
        for row in df.to_dict('records'):
            cat=category(row['concept']);arm=row['arm']
            accepted=arm in ['mlp_poles','mlp_positive'] if model.startswith('esm2') else arm=='mlp_native'
            reason='not_native_target_arm' if not accepted else 'UNMAPPABLE' if cat is None else 'AUTHOR_NO_VALID_FEATURE' if not (0<=row['best_feature']<2*width) else None
            rec=dict(row,source='Hallee',canonical_model=model,annotation_category=cat,source_table=file)
            if reason:excluded.append(dict(rec,reason=reason));continue
            feature=int(row['best_feature']);assert 0<=feature<2*width
            rec.update(unit0=feature%width,pole='positive' if feature<width else 'negative',block0=int(row['layer'])-int(model.startswith('esm2')),
                original_granularity='residue annotations / domain-adjusted F1',readpoint='pre-GELU sign pole' if model.startswith('esm2') else 'post-SwiGLU gate-tagged pole')
            records.append(rec)
    df=pd.DataFrame(records).sort_values(['best_test_f1','concept','block0','unit0','pole'],ascending=[False,True,True,True,False],kind='stable')
    df=df.drop_duplicates(['source','canonical_model','annotation_category','block0','unit0','pole'])
    selected=df.groupby(['source','canonical_model','annotation_category'],sort=True).head(10).copy()
    selected=selected.sort_values(['canonical_model','annotation_category','best_test_f1','block0','unit0'],ascending=[True,True,False,True,True])
    selected.to_csv(OUT/'PUBLIC_SELECTED.tsv',sep='\t',index=False)
    df.to_csv(OUT/'PUBLIC_ALL_MAPPABLE.tsv',sep='\t',index=False)
    pd.DataFrame(excluded).to_csv(OUT/'PUBLIC_EXCLUDED.tsv',sep='\t',index=False)
    b=[]
    for model in ['8M','35M']:
        bd=pd.read_csv(OUT/'sources'/f'banerjee_{model}.csv');bd['canonical_model']='esm2_'+model.lower();bd['reason']='whole-sequence description of block hidden coordinate; pole and author ranking unavailable; no residue-level truth';b.append(bd)
    pd.concat(b).to_csv(OUT/'BANERJEE_EXCLUDED.tsv',sep='\t',index=False)
    info={'selected_rows':len(selected),'selected_by_model_category':selected.groupby(['canonical_model','annotation_category']).size().to_dict(),
      'needed_blocks':selected.groupby('canonical_model')['block0'].apply(lambda x:sorted(set(x))).to_dict(),'banerjee_excluded_rows':sum(len(x) for x in b),'activation_read':False,
      'source_gate':'ESM2 original hook and sign formula locally bound; ESMC gate-tagged formula bound to author bases.py/poles.py. ESMC implementation parity and checkpoint alias mapping pending; no scoring permitted yet.'}
    info['selected_by_model_category']={'|'.join(k):v for k,v in info['selected_by_model_category'].items()};write('SOURCE_SUMMARY.json',info);print(json.dumps(info,indent=2))
def cohorts():
    src=ROOT/'data/revision/protein_fresh';raw={}
    for p in (src/'raw').glob('*.json'):
        for r in js(p)['results']:raw[r['primaryAccession']]=r
    reserve=jl(src/'unused_component_inventory.jsonl')
    prior=js(ROOT/'structural_readout_validation/fresh/PRIOR_EXCLUSION_SCOPE.json')['files']
    extra=[ROOT/'structural_readout_validation/fresh/chains.jsonl',ROOT/'structural_readout_validation/fresh/native_candidate_chains.jsonl',ROOT/'structural_readout_validation/fresh/prior_native_sequences.jsonl']
    used_accessions=set();used_sequences=set();used_groups=set();paths=[]
    # This inventory counts prior exposure including candidate/discovery queues.
    # Source-only original fresh raw pages are not themselves experimental use.
    for p in [Path(r['path']) for r in prior]+extra:
        if not p.exists():continue
        paths.append(str(p))
        for r in jl(p):
            for k in ['accession','uniprot','id']:
                v=r.get(k)
                if isinstance(v,str):used_accessions.add(v)
            if isinstance(r.get('sequence'),str):used_sequences.add(r['sequence'])
            for k in ['known_group_id','group','original_group','homology_group']:
                v=r.get(k)
                if isinstance(v,str):used_groups.add(v)
    rows=[]
    for r in reserve:
        rr=raw.get(r['id']);seq=rr['sequence']['value'] if rr else ''
        used=r['id'] in used_accessions or seq in used_sequences or r['known_group_id'] in used_groups
        rows.append(dict(r,length=len(seq),sequence=seq,previously_used=used,status='PREVIOUSLY_USED' if used else 'PROVISIONAL_NOT_USED_IDENTITY_ONLY',taxonomy='UNKNOWN',exact_prior_checked=True,full_homology_rescreened=False))
    pd.DataFrame(rows).to_csv(OUT/'RESERVE_194_INVENTORY.tsv',sep='\t',index=False)
    S=ROOT/'structure';seqmap={r['accession']:r for r in jl(S/'sequence_index.jsonl')};acc=defaultdict(set);res=Counter()
    for r in table(S/'residue_features.tsv'):
        if r['main_clean']=='True' and r['interface']=='False':acc[r['split']].add(r['accession']);res[r['split']]+=1
    legacy={s:{'chains':len(a),'groups':len({seqmap[x]['group'] for x in a}),'full_sequence_residues':sum(len(seqmap[x]['sequence']) for x in a),'selected_structure_residues':res[s],
       'length_quantiles':dict(zip(['q0','q25','q50','q75','q90','q100'],np.quantile([len(seqmap[x]['sequence']) for x in a],[0,.25,.5,.75,.9,1]).tolist()))} for s,a in acc.items()}
    lengths=[r['length'] for r in rows if not r['previously_used']]
    info={'legacy':legacy,'reserve_mother_frame':len(reserve),'reserve_sequence_recovered':sum(r['length']>0 for r in rows),'reserve_previously_used_identity':sum(r['previously_used'] for r in rows),
      'reserve_provisional_unused':len(lengths),'reserve_provisional_residues':sum(lengths),'reserve_length_quantiles':dict(zip(['q0','q25','q50','q75','q90','q100'],np.quantile(lengths,[0,.25,.5,.75,.9,1]).tolist())) if lengths else {},
      'prior_identity_sources':paths,'taxonomy_and_fresh_homology_status':'PENDING; do not call remaining194 fully new until prior complete-scope group links and taxonomy pass','new_structure_queue_not_selected':True,'activation_read':False}
    write('COHORT_INVENTORY.json',info);print(json.dumps(info,indent=2))
if __name__=='__main__':sources();cohorts()
