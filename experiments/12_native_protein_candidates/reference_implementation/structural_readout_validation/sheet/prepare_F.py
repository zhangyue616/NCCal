"""Explicit E_new -> F display alias; fresh frozen source is never edited."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
src=H.parent/'fresh/chains.jsonl'
rows=[json.loads(x) for x in src.read_text(encoding='utf-8').splitlines() if x.strip()]
assert all(r['split']=='E_new' for r in rows),'Only the finalized separately acquired E_new frame may map to F'
for r in rows:
    r['fresh_original_split']=r['split'];r['split']='F'
dest=H/'input_chains_F.jsonl'
dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
record={'utc':datetime.now(timezone.utc).isoformat(),'original_fresh_file':str(src),'original_fresh_sha256':sha(src),'mapped_file':str(dest),'mapped_sha256':sha(dest),'rows':len(rows),'mapping':'split E_new becomes F only; original_split retained, fresh_original_split added; no source/group/sequence/coordinate changes','model_scores_read':False,'script_sha256':sha(__file__)}
(H/'F_ORIGIN_FREEZE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
