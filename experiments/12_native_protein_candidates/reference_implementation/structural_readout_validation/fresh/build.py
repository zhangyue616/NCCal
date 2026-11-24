from pathlib import Path
import sys,os,json,csv,time,gzip
from collections import defaultdict,Counter,deque
from concurrent.futures import ProcessPoolExecutor
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from acquire import P,ROOT,OLD,old,sha,digest,readjl,jl,write,tsv
import numpy as np
def init_worker(prior):
 global PRIOR,MAT,ProteinSequence,align_optimal
 from biotite.sequence import ProteinSequence
 from biotite.sequence.align import align_optimal,SubstitutionMatrix
 MAT=SubstitutionMatrix.std_protein_matrix();PRIOR=[(h,s,ProteinSequence(s)) for h,s in prior]
def align_candidate(r):
 s=r['sequence'];ps=ProteinSequence(s);hit=[];audit=[]
 for h,t,pt in PRIOR:
  a=align_optimal(ps,pt,MAT,gap_penalty=(-10,-1),local=True,max_number=1)[0];trace=a.trace;both=trace[(trace[:,0]>=0)&(trace[:,1]>=0)]
  ident=float(np.mean(ps.code[both[:,0]]==pt.code[both[:,1]]))if len(both)else 0.
  cov=min(len(set(trace[trace[:,0]>=0,0])),len(set(trace[trace[:,1]>=0,1])))/min(len(s),len(t))
  yes=ident>=.30 and cov>=.80
  audit.append((h,ident,cov,yes))
  if yes:hit.append(h)
 return r,audit,hit
def cluster_map():
 scope=json.loads((P/'PRIOR_EXCLUSION_SCOPE.json').read_text(encoding='utf8'));oldp=set();olde=set()
 for f in scope['files']:
  for r in readjl(f['path']):
   if r.get('pdb_id'):
    if r.get('entity_id'):olde.add(r['pdb_id'].upper()+'_'+str(r['entity_id']))
    else:oldp.add(r['pdb_id'].upper())
   for c in r.get('candidates',[]):oldp.add(c['pdb_id'].upper())
 m={};blocked=set();needed={r['pdb_id'].upper()+'_'+r['entity_id'] for r in readjl(P/'native_candidate_chains.jsonl')}
 for n,line in enumerate((P/'sources/clusters-by-entity-30.txt').read_text(encoding='utf8').splitlines()):
  toks=line.split();ids=needed.intersection(toks)
  if not ids:continue
  gid='rcsb30-'+digest(line.strip())[:24]
  for x in ids:m[x]=gid
  if any(x in olde or x.split('_')[0]in oldp for x in toks):blocked.add(gid)
 write(P/'RCSB30_SCOPE.json',{'source_sha256':sha(P/'sources/clusters-by-entity-30.txt'),'prior_explicit_entities':len(olde),'prior_PDBs_without_explicit_entities_conservative_all_entities':len(oldp),'new_entities':len(needed),'new_entities_in_clusters':len(m),'blocked_new_clusters':len(blocked),'missing_cluster_policy':'Use singleton entity identifier and complete full-native sequence alignment; disclose count.'})
 return m,blocked
def build():
 m,blocked=cluster_map();rr=readjl(P/'native_candidate_chains.jsonl');exclude=[];byseq=defaultdict(list)
 for r in rr:
  if len(r['taxid'].split(';'))!=1:
   exclude.append(dict(record_id=r['record_id'],reason='multiple_source_taxa_not_unambiguous'));continue
  b=old.pdbblock(Path(r['source_cif']));polys=[x for x in old.table(b,'_entity_poly.')if old.val(x.get('entity_id'))==r['entity_id']]
  literal=__import__('re').sub(r'\s+','',old.val(polys[0].get('pdbx_seq_one_letter_code')))if polys else''
  if literal!=r['entity_sequence']:
   exclude.append(dict(record_id=r['record_id'],reason='modified_or_noncanonical_deposited_polymer_literal'));continue
  eid=r['pdb_id'].upper()+'_'+r['entity_id'];g=m.get(eid,'unclustered-'+eid)
  if g in blocked:exclude.append(dict(record_id=r['record_id'],reason='RCSB30_prior_component',rcsb30_group=g));continue
  r['original_group']=g;r['rcsb30_group']=g;byseq[r['sequence']].append(r)
 reps=[min(rs,key=lambda r:(r['resolution_A'],r['pdb_id'],r['entity_id'],r['label_asym_id']))for rs in byseq.values()]
 # One representative per public RCSB30 component before more expensive native audit.
 bygroup=defaultdict(list)
 for r in reps:bygroup[r['original_group']].append(r)
 reps=[min(rs,key=lambda r:digest('fresh-native-20261123|'+r['sequence']))for rs in bygroup.values()]
 taxorder=['9606','10090','4932','3702'];queues={t:deque(sorted([r for r in reps if ('4932'if r['taxid']=='559292'else r['taxid'])==t],key=lambda r:digest('fresh-native-20261123|'+r['sequence'])))for t in taxorder}
 candidates=[]
 while any(queues.values())and len(candidates)<300:
  for t in taxorder:
   if queues[t]and len(candidates)<300:candidates.append(queues[t].popleft())
 jl(P/'homology_screen_candidates.jsonl',candidates);jl(P/'homology_pre_exclusions.jsonl',exclude)
 prior=[(r['sequence_sha256'],r['sequence']) for r in readjl(P/'prior_native_sequences.jsonl')]
 print('ALIGN_START',len(candidates),'vs',len(prior),flush=True);t=time.time();eligible=[];hx=[]
 with gzip.open(P/'prior_full_native_similarity_audit.tsv.gz','wt',encoding='utf8',newline='')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['new_sequence_sha256','prior_sequence_sha256','identity','coverage_shorter','meets_cutoff'])
  with ProcessPoolExecutor(max_workers=6,initializer=init_worker,initargs=(prior,))as pool:
   for i,(r,audit,hits)in enumerate(pool.map(align_candidate,candidates,chunksize=1),1):
    w.writerows((r['sequence_sha256'],*x)for x in audit)
    if hits:hx.append(dict(record_id=r['record_id'],reason='full_native_identity30_coverage80_to_prior',prior_sequence_sha256=hits))
    else:eligible.append(r)
    print('ALIGNED',i,len(candidates),'eligible',len(eligible),'sec',round(time.time()-t,1),flush=True)
 jl(P/'homology_alignment_exclusions.jsonl',hx)
 # Full-native align all retained candidates to earlier retained new sequences; no scores.
 from biotite.sequence import ProteinSequence
 from biotite.sequence.align import align_optimal,SubstitutionMatrix
 mat=SubstitutionMatrix.std_protein_matrix();keep=[];newaudit=[];newex=[];u=old.Union(len(eligible));pss=[ProteinSequence(r['sequence'])for r in eligible]
 for i,r in enumerate(eligible):
  ps=pss[i]
  for j,k in enumerate(eligible[:i]):
   pt=ProteinSequence(k['sequence']);a=align_optimal(ps,pt,mat,gap_penalty=(-10,-1),local=True,max_number=1)[0];tr=a.trace;both=tr[(tr[:,0]>=0)&(tr[:,1]>=0)]
   ident=float(np.mean(ps.code[both[:,0]]==pt.code[both[:,1]]))if len(both)else 0.;cov=min(len(set(tr[tr[:,0]>=0,0])),len(set(tr[tr[:,1]>=0,1])))/min(len(ps),len(pt));hit=ident>=.3 and cov>=.8
   newaudit.append(dict(sequence_sha256_1=r['sequence_sha256'],sequence_sha256_2=k['sequence_sha256'],identity=ident,coverage_shorter=cov,meets_cutoff=hit))
   if hit:u.join(i,j)
 components=defaultdict(list)
 for i,r in enumerate(eligible):components[u.root(i)].append((i,r))
 representatives=[]
 for component in components.values():
  i,r=min(component,key=lambda z:z[0]);representatives.append((i,r))
  for j,k in component:
   if i!=j:newex.append(dict(record_id=k['record_id'],reason='new_homology_component_representative',related_to=[r['sequence_sha256']]))
 for i,r in sorted(representatives):
  if len(keep)>=100:newex.append(dict(record_id=r['record_id'],reason='target100_group_cap'));continue
  r.update(group='fresh-hg-'+r['sequence_sha256'][:24],homology_group='fresh-hg-'+r['sequence_sha256'][:24]);keep.append(r)
 tsv(P/'new_sequence_similarity_audit.tsv',newaudit);jl(P/'new_group_exclusions.jsonl',newex)
 sites=[];segs=[]
 for r in keep:
  a,b=old.structure_chain(r);sites.extend(a);segs.extend(b)
 # Keep all eligible chains for beta testing even if no helix-start pair exists.
 pairs,n=old.select_pairs(keep,sites);keep.sort(key=lambda r:r['record_id']);jl(P/'chains.jsonl',keep);tsv(P/'residue_annotations.tsv',sites);tsv(P/'helix_segments.tsv',segs);tsv(P/'pairs.tsv',pairs)
 example=min([r for r in pairs if r['task']=='helix_start_vs_internal'and r['same_helix_segment']],key=lambda r:(r['selection_digest'],r['pair_id']),default=None)
 write(P/'EXAMPLE_SELECTION.json',{'rule':'lowest selection_digest among strict primary pairs, activation-blind','pair':example})
 counts={}
 for task in ('helix_start_vs_internal','helix_vs_strand'):
  rows=[r for r in pairs if r['task']==task];strict=[r for r in rows if r['same_helix_segment']]
  counts[task]={'pairs':len(rows),'groups':len({r['group']for r in rows}),'strict_pairs':len(strict),'strict_groups':len({r['group']for r in strict})}
 files=['LOCKED_PROTOCOL.md','PRE_SCORE_IMPLEMENTATION_NOTE.md','acquire.py','build.py','chains.jsonl','residue_annotations.tsv','helix_segments.tsv','pairs.tsv','EXAMPLE_SELECTION.json','PRIOR_EXCLUSION_SCOPE.json','prior_native_sequences.jsonl','RCSB30_SCOPE.json','CLUSTER_SOURCE.json','sources/clusters-by-entity-30.txt','prior_full_native_similarity_audit.tsv.gz','new_sequence_similarity_audit.tsv','ACQUISITION_FREEZE.json']
 meta={'frozen_utc':old.utc(),'status':'FROZEN_BEFORE_ANY_NEW_MODEL_FORWARD','model_activations_read':False,'native_sequences':len(keep),'homology_groups':len({r['group']for r in keep}),'species':dict(Counter(r['taxid']for r in keep)),'task_counts':counts,'matched_candidate_pairs':n,'native_candidates_after_RCSB30':len(byseq),'screened_candidates':len(candidates),'date_policy':'initial structure release 2025-11-14..2026-11-23; sequence pretraining novelty UNKNOWN','files':{f:sha(P/f)for f in files}}
 ext=[OLD/'build_structure_panel.py',ROOT/'native_channels/e1/e1_runtime.py',ROOT/'native_channels/e1/trained/selection_lock.json',ROOT/'native_channels/e1/trained/model_metadata.json']
 meta['external_dependencies']={str(p):sha(p)for p in ext}
 write(P/'COHORT_FREEZE.json',meta);print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':build()
