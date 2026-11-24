"""Consolidated data acceptance; coordinate audit independent of stored site flags."""
from pathlib import Path
import sys,json,csv,math
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_structure_panel as B
HERE=B.HERE
def rows(name):return list(csv.DictReader((HERE/name).open(encoding='utf-8'),delimiter='\t'))
def truth(x):return x is True or x=='True'
def main():
    failures=[]
    def check(test,message):
        if not test:failures.append(message)
    chains=B.readjl(HERE/'chains.jsonl');pairs=rows('pairs.tsv');sites=rows('residue_annotations.tsv');segments=rows('helix_segments.tsv');freeze=json.loads((HERE/'COHORT_FREEZE.json').read_text())
    cr={r['record_id']:r for r in chains};si={(r['record_id'],int(r['seq_index0'])):r for r in sites};hs={r['segment_id']:r for r in segments}
    check(len(cr)==len(chains),'duplicate record ID');check(len({r['pair_id'] for r in pairs})==len(pairs),'duplicate pair ID')
    seqsplit=defaultdict(set);groupsplit=defaultdict(set);oldsplit=defaultdict(set)
    for r in chains:
        seqsplit[r['sequence_sha256']].add(r['split']);groupsplit[r['group']].add(r['split']);oldsplit[r['original_group']].add(r['split'])
        check(r['split']==r['original_split'],f"reassigned split {r['record_id']}")
        check(r['sequence_sha256']==B.digest(r['sequence']),f"sequence hash {r['record_id']}")
        check(r['sequence_length']==len(r['sequence']) and 60<=len(r['sequence'])<=1000,f"native length {r['record_id']}")
        seq=r['entity_sequence'];off=r['native_offset0'];native=r['sequence']
        check(native.find(seq)==off and native.find(seq,off+1)<0,f"nonunique/nonexact native mapping {r['record_id']}")
        check(set(map(int,r['taxid'].split(';')))<=B.SAFE and not B.BAD.search(r['protein_name']),f"source boundary {r['record_id']}")
        check(B.sha(r['source_cif'])==r['source_cif_sha256'],f"source hash {r['record_id']}")
    check(all(len(v)==1 for v in seqsplit.values()),'exact sequence split leakage');check(all(len(v)==1 for v in groupsplit.values()),'component split leakage');check(all(len(v)==1 for v in oldsplit.values()),'original group split leakage')
    align={frozenset((r['sequence_sha256_1'],r['sequence_sha256_2'])):r for r in rows('sequence_similarity_audit.tsv')};seqs=sorted(seqsplit);cross=0
    for i,a in enumerate(seqs):
        for b in seqs[:i]:
            if seqsplit[a]==seqsplit[b]:continue
            cross+=1;r=align.get(frozenset((a,b)));check(r is not None,f"unaudited cross-split sequence {a[:8]} {b[:8]}")
            if r:check(not truth(r['meets_cutoff']) and not(float(r['identity'])>=.30 and float(r['coverage_shorter'])>=.80),f"cross-split homology {a[:8]} {b[:8]}")
    raw={};source_paths={}
    for i,r in enumerate(chains):
        cid=r['record_id'];b=B.pdbblock(Path(r['source_cif']));ch=r['label_asym_id'];L=len(r['entity_sequence']);pos=defaultdict(list)
        source_paths[r['source_cif']]=r['source_cif_sha256']
        for a in B.table(b,'_atom_site.'):
            if B.val(a.get('label_asym_id'))==ch and B.val(a.get('label_seq_id')).isdigit():pos[int(a['label_seq_id'])].append(a)
        labels=defaultdict(set)
        for cat in ('_struct_conf.','_struct_sheet_range.'):
            for a in B.table(b,cat):
                if B.val(a.get('beg_label_asym_id'))!=ch or B.val(a.get('end_label_asym_id'))!=ch:continue
                try:s,e=int(a['beg_label_seq_id']),int(a['end_label_seq_id'])
                except:continue
                if not 1<=s<=e<=L:continue
                if B.AA3.get(B.val(a.get('beg_label_comp_id')))!=r['entity_sequence'][s-1] or B.AA3.get(B.val(a.get('end_label_comp_id')))!=r['entity_sequence'][e-1]:continue
                label='E' if cat=='_struct_sheet_range.' else 'H' if (B.val(a.get('conf_type_id'))=='HELX_P' and B.val(a.get('pdbx_PDB_helix_class'))=='1') or B.val(a.get('conf_type_id'))=='HELX_RH_AL_P' else 'other_helix'
                for p in range(s,e+1):labels[p].add(label)
        bb={}
        for p,aa in pos.items():
            x=[a for a in aa if B.val(a.get('label_atom_id')) in {'N','CA','C','O'}]
            if Counter(B.val(a.get('label_atom_id')) for a in x)!={'N':1,'CA':1,'C':1,'O':1}:continue
            if any(B.val(a.get('label_alt_id')) or B.num(a.get('occupancy'))<.99 for a in x):continue
            if not 1<=p<=L or any(B.AA3.get(B.val(a.get('label_comp_id')))!=r['entity_sequence'][p-1] for a in aa):continue
            if len({B.val(a.get('pdbx_PDB_model_num')) for a in aa})!=1:continue
            bb[p]={B.val(a['label_atom_id']):tuple(B.num(a[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')) for a in x}
        def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        links={p for p in bb if p+1 in bb and 1.0<=distance(bb[p]['C'],bb[p+1]['N'])<=1.8 and 2.7<=distance(bb[p]['CA'],bb[p+1]['CA'])<=4.3}
        raw[cid]=(bb,links,labels)
        if i%40==0:print('RAW_ACCEPTANCE',i,len(chains),flush=True)
    used=defaultdict(set);counts=Counter();strict=Counter()
    for p in pairs:
        cid=p['record_id'];r=cr[cid];a,n=int(p['index_positive']),int(p['index_negative']);kp=(p['group'],p['task']);ids={(r['sequence_sha256'],a),(r['sequence_sha256'],n)}
        check(a!=n and r['sequence'][a]==r['sequence'][n]==p['aa'],f"same chain/AA {p['pair_id']}")
        check(p['group']==r['group'] and p['split']==r['split'],f"pair group/split {p['pair_id']}")
        check(not (ids&used[kp]),f"site reuse {p['pair_id']}");used[kp]|=ids;counts[kp]+=1
        bb,links,labels=raw[cid]
        for index,key in ((a,'positive'),(n,'negative')):
            q=int(p[f'{key}_label_seq_id']);site=si[(cid,index)]
            check(index==r['native_offset0']+q-1 and site['aa']==p['aa'],f"index mapping {p['pair_id']} {key}")
            check(q in bb and q-1 in links and q in links,f"raw endpoint continuity {p['pair_id']} {key}")
            target='H' if key=='positive' or p['task']=='helix_start_vs_internal' else 'E'
            check(labels[q]=={target} and site['secondary']==target,f"raw secondary {p['pair_id']} {key}")
            if p['task']=='helix_start_vs_internal':
                seg=hs[p[f'{key}_helix_segment_id']];s,e=int(seg['start_label_seq_id']),int(seg['end_label_seq_id'])
                check(e-s+1>=10 and 1<s<=q<=e<len(r['entity_sequence']),f"long helix positions {p['pair_id']} {key}")
                check(all(j in bb for j in range(s-1,e+2)) and all(j in links for j in range(s-1,e+1)),f"full raw helix and flanks {p['pair_id']} {key}")
                check(all(labels[j]=={'H'} for j in range(s,e+1)) and 'H' not in labels[s-1] and 'H' not in labels[e+1],f"maximal alpha boundary {p['pair_id']} {key}")
                check(q-s in (0,1) if key=='positive' else q-s>=3 and e-q>=3,f"start/internal definition {p['pair_id']} {key}")
        same=bool(p['positive_helix_segment_id'] and p['positive_helix_segment_id']==p['negative_helix_segment_id']);check(same==truth(p['same_helix_segment']),f"same segment flag {p['pair_id']}")
        if same:strict[p['split']]+=1
    check(max(counts.values())<=16,'group pair cap')
    # Regenerate only pair ordering from stored annotations; no coordinate/label or homology rebuild.
    for s in sites:
        for k in ('seq_index0','label_seq_id'):s[k]=int(s[k])
        for k in ('backbone_quality','local_backbone_continuous','long_helix_complete'):s[k]=truth(s[k])
    regenerated,_=B.select_pairs(chains,sites)
    check([p['pair_id'] for p in regenerated]==[p['pair_id'] for p in pairs],'deterministic pair priority/order')
    check(all(B.sha(HERE/f)==z['sha256'] for f,z in freeze['files'].items()),'existing freeze hashes')
    cp=B.EXT/'sources/clusters-by-entity-30.txt'
    if not cp.exists():cp=B.CYS/'public_sources/clusters-by-entity-30.txt'
    sources=[B.ARCH/'cohort_index.jsonl',B.EXT/'sequences.jsonl',B.CYS/'chains.jsonl',B.CYS/'COHORT_FREEZE.json',cp]
    B.write(HERE/'SOURCE_PROVENANCE.json',provenance)
    for f in ('PROTOCOL.md','SCHEMA.md','validate_panel.py','SOURCE_PROVENANCE.json','INVENTORY.json','inventory_funnel.jsonl','candidate_chains.jsonl','annotated_candidate_chains.jsonl','chain_selection_exclusions.jsonl','additional_homology_edges.jsonl'):
        freeze['files'][f]={'sha256':B.sha(HERE/f),'bytes':(HERE/f).stat().st_size}
    freeze['final_acceptance_utc']=B.utc();B.write(HERE/'COHORT_FREEZE.json',freeze)
    report={'checked_utc':B.utc(),'status':'PASS' if not failures else 'FAIL','failures':failures,'chains':len(chains),'distinct_native_sequences':len(seqsplit),'groups':len(groupsplit),'pairs':len(pairs),'cross_split_native_sequence_pairs_explicitly_below_cutoff':cross,'same_helix_start_internal_pairs':dict(strict),'source_coordinates_checked':len(source_paths),'scope':'Final input/source hashes; exact unique native mapping; canonical host taxa and names; every pair same-chain same-AA indices; group/original-group/exact-sequence split isolation; complete declared-cutoff cross-split similarity evidence; independent raw backbone, peptide continuity and deposited H/E annotations; entire long-helix plus flanking missing-boundary and role checks; pair caps, site reuse, same-segment flags and deterministic same-segment-first pairing.','model_activations_or_scores_read':False,'cohort_freeze_sha256':B.sha(HERE/'COHORT_FREEZE.json')}
    B.write(HERE/'ACCEPTANCE.json',report)
    if failures:print(json.dumps(report,indent=2));raise SystemExit(1)
    B.write(HERE/'READY.json',{'ready_utc':B.utc(),'status':'READY_FOR_SCORING','cohort_freeze_sha256':report['cohort_freeze_sha256'],'acceptance_sha256':B.sha(HERE/'ACCEPTANCE.json'),'chains_sha256':B.sha(HERE/'chains.jsonl'),'pairs_sha256':B.sha(HERE/'pairs.tsv'),'index_origin':'zero-based full-native sequence','split_roles':freeze['split_roles'],'model_results_read':False})
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
