"""One consolidated acceptance and score-blind freeze, per legacy/F release."""
from pathlib import Path
import json,csv,hashlib,argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');ap.add_argument('--tag',default='');args=ap.parse_args();s='_'+args.tag if args.tag else ('_F' if args.fresh else '');utc=datetime.now(timezone.utc).isoformat()
    pp=H/f'pairs{s}.tsv';pairs=list(csv.DictReader(pp.open(encoding='utf-8'),delimiter='\t'));chains=read(H/f'chains{s}.jsonl');by={r['record_id']:r for r in chains};sites=H/f'sites{s}.tsv'
    for p in pairs:
        for pre in ('positive','negative'):
            start=int(p['index_'+pre])-int(p[pre+'_strand_offset0']);p[pre+'_strand_start_index0']=str(start);p[pre+'_strand_end_index0']=str(start+int(p[pre+'_strand_length'])-1)
    with pp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(pairs[0]) if pairs else ['pair_id','record_id','chain_id','group','homology_group','split','task','aa','index_positive','index_negative'],delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(pairs)
    graph=read(H/f'sheet_graph_audit{s}.jsonl');gg={r['sheet_id']:r for r in graph};spot=read(H/f'blind_spot_check{s}.jsonl');errors=read(H/f'build_errors{s}.jsonl');summary=json.loads((H/f'BUILD_SUMMARY{s}.json').read_text());checks={};fail=[]
    upstream=summary.get('input_path') or (str(H.parent/'fresh/chains.jsonl') if args.fresh else str(H.parents[1]/'native_channels/cohort/chains.jsonl'))
    sourcechains=read(upstream);(H/f'source_chains{s}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sourcechains),encoding='utf-8')
    checks['no_build_exceptions']=not errors;checks['unique_pair_ids']=len({p['pair_id'] for p in pairs})==len(pairs)
    used=set();counts=Counter();splits=defaultdict(set);valid=True
    for p in pairs:
        r=by[p['record_id']];a,b=int(p['index_positive']),int(p['index_negative']);g=gg[p['sheet_id']]
        valid &= a!=b and 0<=a<len(r['sequence']) and 0<=b<len(r['sequence']) and r['sequence'][a]==r['sequence'][b]==p['aa'] and g['eligible'] and p['group']==r['group'] and p['split']==r['split'] and p['chain_id']==r['chain_id'] and p['positive_strand_id']!=p['negative_strand_id']
        for i in (a,b):
            key=(p['group'],p['sequence_sha256'],i)
            if key in used:valid=False
            used.add(key)
        for pre in ('positive','negative'):
            ls=int(p[pre+'_label_seq_id']);valid &= int(p['index_'+pre])==int(r['native_offset0'])+ls-1
            valid &= int(p[pre+'_strand_start_index0'])<=int(p['index_'+pre])<=int(p[pre+'_strand_end_index0'])
        deg=Counter(x for e in g['archive_edges'] for x in e);valid &= deg[p['positive_strand_id']]==1 and deg[p['negative_strand_id']]==2
        counts[p['group']]+=1;splits[p['group']].add(p['split'])
    checks['same_chain_sheet_AA_native_mapping_topology_no_reuse']=bool(valid);checks['cap_16_per_group']=max(counts.values(),default=0)<=16;checks['groups_disjoint_across_splits']=all(len(v)==1 for v in splits.values())
    # Source hashes bound by upstream cohort; calculate each distinct file exactly once.
    checks['source_sha_matches']=all(sha(path)==expected for path,expected in summary['sources'].items())
    checks['eligible_blind_spot_graphs_exact']=all(r['archive_energy_graph_exact'] for r in spot if r['eligible'])
    checks['blind_spot_has_eligible_and_failures']=any(r['eligible'] for r in spot) and any(not r['eligible'] for r in spot)
    status='PASS' if all(checks.values()) else 'FAIL';accept={'utc':utc,'status':status,'checks':checks,'counts_by_split':summary['counts_by_split'],'pairs':len(pairs),'groups':len(counts),'retained_chains':len(chains),'blind_spot_sheets':len(spot),'blind_spot_eligible':sum(r['eligible'] for r in spot),'failed_spot_disagreements':sum(not r['eligible'] and not r['archive_energy_graph_exact'] for r in spot),'model_results_read':False,'scientific_conclusion':'No model or neuron effect assessed; structure/data acceptance only.'}
    write(H/f'ACCEPTANCE{s}.json',accept)
    names=['PROTOCOL.md','SCHEMA.md','build_sheet.py','spotcheck.py','finalize.py',f'source_chains{s}.jsonl',f'chains{s}.jsonl',f'pairs{s}.tsv',f'sites{s}.tsv',f'sheet_graph_audit{s}.jsonl',f'blind_spot_check{s}.jsonl',f'BUILD_SUMMARY{s}.json',f'build_errors{s}.jsonl',f'ACCEPTANCE{s}.json']
    freeze={'utc':utc,'state':'FROZEN_BEFORE_MODEL_SCORES' if status=='PASS' else 'NOT_RELEASED','files':{n:sha(H/n) for n in names},'source_cif_hashes':summary['sources'],'upstream_interface':upstream,'model_scores_read':False,'endpoint':'same-chain same-AA same-open-sheet edge(1 neighbor) versus central(2)','split_roles':{'F':'new_external_confirmation'} if args.fresh else {'D':'development_selection','T':'historical_confirmation','E':'historical_external_confirmation'}}
    write(H/f'COHORT_FREEZE{s}.json',freeze)
    if status=='PASS':write(H/f'READY{s}.json',{'utc':utc,'status':'READY_FOR_SCORING','freeze_sha256':sha(H/f'COHORT_FREEZE{s}.json'),'pairs_sha256':sha(pp),'chains_sha256':sha(H/f'chains{s}.jsonl'),'counts':summary['counts_by_split'],'index_origin':'zero-based full native sequence','positive':'edge','negative':'central','model_scores_read':False})
    print(json.dumps(accept,indent=2))
if __name__=='__main__':main()
