"""Materialize D_add/H inputs from accepted extra release; preserve every row/ID."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ready=json.loads((H/'READY_D_extra.json').read_text());assert ready['status']=='READY_FOR_SCORING'
assert sha(H/'pairs_D_extra.tsv')==ready['pairs_sha256']
assert sha(H/'chains_D_extra.jsonl')==ready['chains_sha256']
with (H/'pairs_D_extra.tsv').open(encoding='utf-8',newline='') as f:
    rd=csv.DictReader(f,delimiter='\t');fields=list(rd.fieldnames);pairs=list(rd)
chains=[json.loads(x)for x in (H/'chains_D_extra.jsonl').read_text(encoding='utf-8').splitlines()if x.strip()]
assert set(p['split']for p in pairs)<={'D','H'}
receipts={}
for split,name in [('D','D_add'),('H','H')]:
    dest=H/name;dest.mkdir(exist_ok=True);rr=[p for p in pairs if p['split']==split];ids={p['record_id']for p in rr};cc=[c for c in chains if c['record_id']in ids]
    assert all(c['split']==split for c in cc)
    assert not(dest/'READY.json').exists(),'Do not overwrite accepted subset'
    with (dest/'pairs.tsv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rr)
    (dest/'chains.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n'for c in cc),encoding='utf-8')
    receipt={'role':name,'split':split,'pairs':len(rr),'groups':len({p['group']for p in rr}),'chains':len(cc),'source':str(H/'pairs_D_extra.tsv'),'source_pairs_sha256':ready['pairs_sha256'],'source_chains_sha256':ready['chains_sha256'],'source_freeze_sha256':ready['freeze_sha256'],'pairs_sha256':sha(dest/'pairs.tsv'),'chains_sha256':sha(dest/'chains.jsonl'),'checks':{'all_rows_exact_original_subset':True,'all_ids_preserved':True,'only_one_split':True},'model_scores_read':False}
    (dest/'READY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');receipts[name]=receipt
assert len(pairs)==sum(x['pairs']for x in receipts.values())
assert len(chains)==sum(x['chains']for x in receipts.values())
(H/'EXTRA_SUBSET_RECEIPT.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts,indent=2))
