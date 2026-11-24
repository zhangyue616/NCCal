"""Read-only, outcome-blind native-host eligibility inventory. Writes only here."""
import sys, os
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,csv,gzip,re,hashlib
from datetime import datetime,timezone
from collections import defaultdict,Counter
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'protein_extension'
sys.path.insert(0,str(OLD/'structure/runtime'))
import gemmi
SAFE={9606,10090,10116,9913,559292,4932,3702,4530,7227,6239,7955}
BAD=re.compile(r'toxin|venom|virulen|viral|virus|pathogen|defensin|antimicrobial|interferon|interleukin|cytokine|chemokine|growth factor|hemagglutinin|haemagglutinin|engineered|designed|chimer',re.I)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig'),delimiter='\t'))
def tab(b,c):
 z=b.get_mmcif_category(c)
 return [dict(zip(z,r)) for r in zip(*z.values())] if z else []
def val(v):return '' if v in (None,False,'.','?') else str(v)
def main():
 pairs=rows(OLD/'structure/D_frozen/matched_pairs.tsv')
 seq={x['accession']:x for x in map(json.loads,(OLD/'structure/D_frozen/sequence_index.jsonl').read_text().splitlines())}
 accepted=[];excluded=[];bindings={}
 for p in pairs:
  acc=p['accession'];s=seq[acc]['sequence'];f=OLD/'structure/cache/mmcif'/f"{p['pdb_id'].lower()}.cif.gz"
  if not f.exists():excluded.append({'accession':acc,'reason':'missing_structure'});continue
  b=gemmi.cif.read_string(gzip.decompress(f.read_bytes()).decode()).sole_block()
  ent={str(x['id']):x for x in tab(b,'_entity.')};src=defaultdict(set);refs=defaultdict(set)
  for cat,key in [('_entity_src_gen.','pdbx_gene_src_ncbi_taxonomy_id'),('_entity_src_nat.','pdbx_ncbi_taxonomy_id')]:
   for x in tab(b,cat):
    try:src[str(x['entity_id'])].add(int(x[key]))
    except:pass
  for x in tab(b,'_struct_ref.'):
   if val(x.get('db_name')).upper() in {'UNP','UNIPROT'}:refs[str(x['entity_id'])].add(val(x.get('pdbx_db_accession')))
  eligible=[]
  for x in tab(b,'_entity_poly.'):
   eid=str(x['entity_id']);e=ent[eid];tax=src[eid];native=re.sub(r'\s+','',val(x.get('pdbx_seq_one_letter_code_can')))
   if not tax or not tax<=SAFE or BAD.search(val(e.get('pdbx_description'))) or val(e.get('pdbx_mutation')):continue
   if not native or not 60<=len(s)<=1000 or set(s)-set('ACDEFGHIKLMNPQRSTVWY'):continue
   off=s.find(native)
   if off<0 or s.find(native,off+1)>=0 or (refs[eid] and acc not in refs[eid]):continue
   strands={v.strip() for v in val(x.get('pdbx_strand_id')).split(',')}
   if p['chain'] not in strands:continue
   eligible.append((eid,sorted(tax),val(e.get('pdbx_description')),len(native),off))
  if not eligible:excluded.append({'accession':acc,'reason':'not_allowed_unmodified_native_host_entity'});continue
  cache=OLD/'rsa/activations/35m/D-all'/f'{acc}.npy'
  if not cache.exists():excluded.append({'accession':acc,'reason':'missing_full_layer'});continue
  a=np.load(cache,mmap_mode='r',allow_pickle=False)
  assert a.shape==(len(s),1920)
  early,late=sorted([int(p['pos1']),int(p['pos2'])]);eid,tax,name,clen,off=sorted(eligible)[0]
  accepted.append({**p,'early':early,'late':late,'target':int(int(p['pos2'])==late),'length':len(s),'entity_id':eid,'taxids':','.join(map(str,tax)),'protein_name':name,'native_entity_length':clen,'native_entity_offset0':off,'activation_path':str(cache),'structure_path':str(f),'sequence_sha256':hashlib.sha256(s.encode()).hexdigest()})
  bindings[str(f)]=sha(f)
 assert len({r['group'] for r in accepted})==len(accepted)
 # A new, explicit benchmark split within historical D; never relabel as original external T.
 accepted.sort(key=lambda r:hashlib.sha256(('native-host-benchmark-split-v1|'+r['group']).encode()).digest())
 n=len(accepted)//2
 for i,r in enumerate(accepted):r['benchmark_split']='development' if i<n else 'confirmation'
 with (OUT/'cohort.tsv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,list(accepted[0]),delimiter='\t');w.writeheader();w.writerows(accepted)
 result={'utc':datetime.now(timezone.utc).isoformat(),'original_D_matched_groups':len(pairs),'eligible_native_host_groups':len(accepted),'split_counts':dict(Counter(x['benchmark_split'] for x in accepted)),'host_taxids':dict(Counter(x['taxids'] for x in accepted)),'excluded':excluded,'complete_width':1920,'layer':'ESM2-35M block6 post-GELU','activation_association_scoring_performed':False,'source_bindings':{str(OLD/'structure/D_frozen/matched_pairs.tsv'):sha(OLD/'structure/D_frozen/matched_pairs.tsv'),str(OLD/'structure/D_frozen/sequence_index.jsonl'):sha(OLD/'structure/D_frozen/sequence_index.jsonl'),**bindings}}
 (OUT/'INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('excluded','source_bindings')}))
if __name__=='__main__':main()
