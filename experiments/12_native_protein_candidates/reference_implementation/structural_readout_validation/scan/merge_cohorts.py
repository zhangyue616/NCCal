"""Merge only explicitly supplied frozen labels/sequences. No model or activation imports."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,hashlib,time,io

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def merge(base,dev_add,holdout,out):
    sources=[('inherited',Path(base),None)]
    if dev_add:sources.append(('D_add',Path(dev_add),'D'))
    if holdout:sources.append(('H',Path(holdout),'H'))
    fields=[];pairs=[];chains={};bindings=[];pair_ids=set();group_splits=defaultdict(set);sequence_splits=defaultdict(set)
    for role,root,expected in sources:
        pf=root/'pairs.tsv';cf=root/'chains.jsonl'
        with open(pf,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,delimiter='\t');rows=list(reader)
            for field in reader.fieldnames:
                if field not in fields:fields.append(field)
        seqs=[json.loads(line) for line in cf.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        by_id={r['record_id']:r for r in seqs};assert len(by_id)==len(seqs)
        assert rows and all(r['task']=='sheet_edge_vs_central' for r in rows)
        assert set(r['split'] for r in rows)<={'D','T','E'} if expected is None else all(r['split']==expected for r in rows)
        for r in rows:
            assert r['pair_id'] not in pair_ids,('duplicate pair_id',r['pair_id']);pair_ids.add(r['pair_id'])
            c=by_id[r['record_id']];seq=c['sequence']
            assert seq[int(r['index_positive'])]==seq[int(r['index_negative'])]==r['aa']
            assert c['split']==r['split'] and c['group']==r['group']
            if r['record_id'] in chains:assert chains[r['record_id']]==c,('record collision',r['record_id'])
            chains[r['record_id']]=c
            group_splits[r['group']].add(r['split']);sequence_splits[seq].add(r['split'])
            pairs.append(r)
        bindings.append(dict(role=role,root=str(root.resolve()),pairs_sha256=sha(pf),chains_sha256=sha(cf),pairs=len(rows),groups=len({r['group'] for r in rows}),sequences=len({by_id[r['record_id']]['sequence'] for r in rows}),expected_split=expected))
    assert all(len(v)==1 for v in group_splits.values()),'Homology group crosses a split'
    assert all(len(v)==1 for v in sequence_splits.values()),'Exact native sequence crosses a split'
    # No columns or values of inherited T/E rows are changed by the merge.
    out=Path(out).resolve()
    assert out.is_relative_to(Path(__file__).resolve().parent),'Write only within scan/'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'READY.json').exists(),'Never overwrite frozen merge'
    with open(out/'pairs.tsv','x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(pairs)
    with open(out/'chains.jsonl','x',encoding='utf-8') as f:
        for rid in sorted(chains):f.write(json.dumps(chains[rid],ensure_ascii=False)+'\n')
    counts={sp:dict(pairs=sum(r['split']==sp for r in pairs),groups=len({r['group'] for r in pairs if r['split']==sp}),chains=len({r['record_id'] for r in pairs if r['split']==sp})) for sp in ['D','T','E','H']}
    receipt=dict(created_unix=time.time(),role='explicit labels/sequences merge; no model import',model_scores_read=False,sources=bindings,counts_by_split=counts,pairs_sha256=sha(out/'pairs.tsv'),chains_sha256=sha(out/'chains.jsonl'),checks=dict(original_TE_row_values_preserved=True,all_pair_ids_unique=True,homology_groups_single_split=True,exact_native_sequences_single_split=True,all_pair_aa_identical=True),homology_note='Homology assignments are inherited from independently frozen sources; this merge does not redo alignment-based exclusions.')
    with open(out/'READY.json','x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--D-add');ap.add_argument('--H');ap.add_argument('--out',required=True);a=ap.parse_args()
    merge(a.base,a.D_add,a.H,a.out)
