from pathlib import Path
import sys,os,json,csv,gzip,hashlib,re,time,subprocess,importlib.util
from collections import defaultdict,Counter,deque
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
P=Path(__file__).parent;ROOT=P.parents[1];OLD=ROOT/'native_channels/cohort'
spec=importlib.util.spec_from_file_location('oldcohort',OLD/'build_structure_panel.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.sha;digest=old.digest;readjl=old.readjl;jl=old.jl;write=old.write;tsv=old.tsv;table=old.table;val=old.val
for x in ('sources/mmcif','sources/uniprot','sources/metadata'): (P/x).mkdir(parents=True,exist_ok=True)
BADPLUS=re.compile(r'antibod|immunoglobulin|nanobod|fusion|synthetic|designer|biosynthetic',re.I)
def fetch(url,path):
 path=Path(path)
 if path.exists() and path.stat().st_size>0:return path
 for k in range(3):
  r=subprocess.run(['curl.exe','--silent','--show-error','--fail','--location','--max-time','65','--retry','0',url,'-o',str(path)+'.part'],capture_output=True)
  if r.returncode==0:
   Path(str(path)+'.part').replace(path);return path
  time.sleep(k+1)
 raise RuntimeError('public_fetch_failed '+url+' '+r.stderr.decode(errors='replace')[-150:])
def prior():
 roots=['protein_extension','cross_architecture','nccal_evidence_extension','native_channels','protein_case_extension','neuron_814','scale']
 names={'chains.jsonl','candidate_chains.jsonl','annotated_candidate_chains.jsonl','cohort_index.jsonl','sequence_index.jsonl','sequences.jsonl','D_sequences.jsonl','E_sequences.jsonl'}
 fs=sorted({p for d in roots for p in (ROOT/d).rglob('*.jsonl') if p.name in names and not any(x in p.parts for x in ('manuscript','v1_before_sulfur_correction','D_frozen'))})
 seqs={};acc=set();pdb=set();entities=set();groups=set();fr=[]
 for f in fs:
  rr=readjl(f);n=0
  for r in rr:
   if not r.get('sequence'):continue
   s=r['sequence'];acc.add(r.get('accession',''));seqs[s]=r;n+=1
   if r.get('group'):groups.add(r['group'])
   if r.get('pdb_id'):pdb.add(r['pdb_id'].upper())
   if r.get('pdb_id') and r.get('entity_id'):entities.add(r['pdb_id'].upper()+'_'+str(r['entity_id']))
   for z in r.get('candidates',[]):pdb.add(z['pdb_id'].upper())
  fr.append({'path':str(f),'rows_with_sequence':n,'sha256':sha(f)})
 write(P/'PRIOR_EXCLUSION_SCOPE.json',{'files':fr,'native_sequences':len(seqs),'accessions':len(acc- {''}),'pdb_ids':len(pdb),'entity_ids_explicit':len(entities),'historical_groups':len(groups),'scope_limit':'Explicit sequence JSONLs under the seven named experiment roots. Manuscript copies and frozen duplicates excluded; no inference of model pretraining contents.'})
 jl(P/'prior_native_sequences.jsonl',[{'sequence':s,'sequence_sha256':digest(s),'accession':r.get('accession','')} for s,r in sorted(seqs.items())])
 return seqs,acc,pdb,entities
def raw_entities(pid):
 path=P/f'sources/mmcif/{pid.lower()}.cif.gz'
 try:fetch(f'https://files.rcsb.org/download/{pid}.cif.gz',path)
 except Exception:fetch(f'https://www.ebi.ac.uk/pdbe/entry-files/download/{pid.lower()}.cif.gz',path)
 b=old.pdbblock(path);methods=[val(x.get('method')) for x in table(b,'_exptl.')]
 dates=[val(x.get('revision_date')) for x in table(b,'_pdbx_audit_revision_history.') if val(x.get('ordinal'))=='1']
 if methods!=['X-RAY DIFFRACTION'] or not dates or not '2025-11-14'<=dates[0]<='2026-11-23':return [],{'pdb_id':pid,'reason':'method_or_initial_date'}
 res=min([old.num(x.get('ls_d_res_high'),99) for x in table(b,'_refine.')]or[99])
 if res>3:return [],{'pdb_id':pid,'reason':'resolution'}
 ent={val(x['id']):x for x in table(b,'_entity.')};src=defaultdict(set);names=defaultdict(set);accs=defaultdict(set);chains=defaultdict(list)
 for cat,key,nk in [('_entity_src_gen.','pdbx_gene_src_ncbi_taxonomy_id','pdbx_gene_src_scientific_name'),('_entity_src_nat.','pdbx_ncbi_taxonomy_id','pdbx_organism_scientific')]:
  for x in table(b,cat):
   try:src[val(x['entity_id'])].add(int(x[key]));names[val(x['entity_id'])].add(val(x.get(nk)))
   except:pass
 for x in table(b,'_struct_ref.'):
  if val(x.get('db_name')).upper() in {'UNP','UNIPROT'}:accs[val(x['entity_id'])].add(val(x.get('pdbx_db_accession')))
 for x in table(b,'_struct_asym.'):chains[val(x['entity_id'])].append(val(x['id']))
 out=[];reasons=Counter()
 for poly in table(b,'_entity_poly.'):
  eid=val(poly['entity_id']);e=ent.get(eid,{});name=val(e.get('pdbx_description'));tax=src[eid];seq=re.sub(r'\s+','',val(poly.get('pdbx_seq_one_letter_code_can')))
  reason=''
  if not tax or not tax<=old.SAFE:reason='source_not_allowed'
  elif old.BAD.search(name) or BADPLUS.search(name):reason='description_excluded'
  elif val(e.get('pdbx_mutation')):reason='declared_mutation'
  elif not 60<=len(seq)<=1000 or set(seq)-set(old.AA):reason='entity_length_or_noncanonical'
  elif len(accs[eid])!=1:reason='uniprot_accession_not_unique'
  if reason:reasons[reason]+=1;continue
  for ch in sorted(chains[eid])[:1]:
   out.append(dict(record_id=f'{pid.lower()}_{ch}',chain_id=f'{pid.lower()}_{ch}',pdb_id=pid.lower(),label_asym_id=ch,entity_id=eid,accession=next(iter(accs[eid])),entity_sequence=seq,source_cif=str(path),source_cif_sha256=sha(path),taxid=';'.join(map(str,sorted(tax))),organism=';'.join(sorted(names[eid])),protein_name=name,resolution_A=res,structure_initial_release_date=dates[0],public_cif_url=f'https://files.rcsb.org/download/{pid}.cif',public_structure_url=f'https://www.rcsb.org/structure/{pid}'))
 return out,dict(pdb_id=pid,reason='eligible_raw' if out else 'no_allowed_entity',raw_chains=len(out),entity_exclusions=dict(reasons))
def native(acc):
 path=P/f'sources/uniprot/{acc}.json'
 fetch('https://rest.uniprot.org/uniprotkb/'+acc+'.json',path)
 return json.loads(path.read_text(encoding='utf8'))
def acquire():
 seqs,accs,oldp,olde=prior();response=json.loads((P/'search_response.json').read_text(encoding='utf8'));ids=[r['identifier'] for r in response['result_set']]
 ids=sorted([p for p in ids if p.upper() not in oldp],key=lambda p:digest('fresh-20261123|'+p.upper()))[:1000]
 write(P/'ACQUISITION_FREEZE.json',{'utc':old.utc(),'search_total':response['total_count'],'entries_after_prior_pdb_exclusion':len([p for p in response['result_set'] if p['identifier'].upper() not in oldp]),'capped_download_pdb_ids':ids,'query_sha256':sha(P/'search_query.json'),'protocol_sha256':sha(P/'LOCKED_PROTOCOL.md'),'model_scores_read':False})
 rr=[];ff=[]
 with ThreadPoolExecutor(max_workers=6)as pool:
  jobs={pool.submit(raw_entities,p):p for p in ids}
  for i,f in enumerate(as_completed(jobs),1):
   try:a,b=f.result();rr.extend(a);ff.append(b)
   except Exception as e:ff.append(dict(pdb_id=jobs[f],reason='download_parse_error',error=str(e)))
   if i%50==0:print('PDB',i,len(ids),'raw',len(rr),flush=True);jl(P/'acquisition_funnel.jsonl',sorted(ff,key=lambda x:x['pdb_id']))
 jl(P/'raw_candidate_chains.jsonl',sorted(rr,key=lambda r:r['record_id']));jl(P/'acquisition_funnel.jsonl',sorted(ff,key=lambda x:x['pdb_id']))
 aa=sorted({r['accession'] for r in rr if r['accession'] not in accs});nat={};nf=[]
 with ThreadPoolExecutor(max_workers=6)as pool:
  jobs={pool.submit(native,a):a for a in aa}
  for i,f in enumerate(as_completed(jobs),1):
   a=jobs[f]
   try:nat[a]=f.result()
   except Exception as e:nf.append(dict(accession=a,error=str(e)))
   if i%50==0:print('UNIPROT',i,len(aa),flush=True)
 selected=[];ex=[]
 for r in sorted(rr,key=lambda r:r['record_id']):
  reason='';a=r['accession'];u=nat.get(a,{})
  if a in accs:reason='prior_accession'
  elif not u:reason='uniprot_unavailable'
  else:
   s=u.get('sequence',{}).get('value','');tax=u.get('organism',{}).get('taxonId');off=s.find(r['entity_sequence'])
   if tax not in old.SAFE or str(tax) not in r['taxid'].split(';'):reason='uniprot_source_mismatch'
   elif not 60<=len(s)<=1000 or set(s)-set(old.AA):reason='native_length_or_noncanonical'
   elif off<0 or s.find(r['entity_sequence'],off+1)>=0:reason='no_unique_exact_native_mapping'
   elif s in seqs:reason='prior_full_sequence'
   else:
    dates=u.get('entryAudit',{});r.update(sequence=s,sequence_length=len(s),sequence_sha256=digest(s),native_offset0=off,uniprot_primary_accession=u.get('primaryAccession'),uniprot_entry_created=dates.get('firstPublicDate'),uniprot_sequence_version=dates.get('sequenceVersion'),uniprot_json=str(P/f'sources/uniprot/{a}.json'),uniprot_json_sha256=sha(P/f'sources/uniprot/{a}.json'),split='E_new',original_split='E_new');selected.append(r)
  if reason:ex.append(dict(record_id=r['record_id'],accession=a,reason=reason))
 jl(P/'native_candidate_chains.jsonl',selected);jl(P/'native_mapping_exclusions.jsonl',ex);jl(P/'uniprot_errors.jsonl',nf)
 fetch('https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt',P/'sources/clusters-by-entity-30.txt')
 print('NATIVE_CANDIDATES',len(selected),'sequences',len({r['sequence']for r in selected}),flush=True)
if __name__=='__main__':acquire()
