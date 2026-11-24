from pathlib import Path
import sys,os,json,csv,hashlib,time,itertools,gzip
from collections import defaultdict,Counter
sys.dont_write_bytecode=True
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
P=Path(__file__).parent;ROOT=P.parents[1];E=ROOT/'native_channels/e1'
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def table(p):return list(csv.DictReader(Path(p).open(encoding='utf8'),delimiter='\t'))
def tsv(p,rows):
 with Path(p).open('w',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])if rows else['status'],delimiter='\t');w.writeheader();w.writerows(rows)
def stat(rows):
 bg=defaultdict(list)
 for r in rows:bg[r['group']].append(r['win'])
 gs=np.array([np.mean(v)for k,v in sorted(bg.items())]);n=len(gs)
 if not n:return {'n_pairs':0,'n_groups':0,'status':'NOT_EVALUABLE'}
 rng=np.random.default_rng(41624);boot=gs[rng.integers(0,n,(9999,n))].mean(1);delta=gs-.5;obs=delta.mean();nz=delta[delta!=0]
 if not len(nz):p=1.
 elif len(nz)<=18:
  perm=np.array(list(itertools.product((-1,1),repeat=len(nz))),dtype=np.int8)@nz/n;p=float(np.mean(perm>=obs-1e-15))
 else:
  rng=np.random.default_rng(41623);ge=0
  for k in range(0,99999,1000):
   a=rng.choice(np.array([-1,1],dtype=np.int8),size=(min(1000,99999-k),len(nz)))@nz/n;ge+=int(np.sum(a>=obs-1e-15))
  p=(ge+1)/100000
 return dict(n_pairs=len(rows),n_groups=n,n_sequences=len({r['sequence_sha256']for r in rows}),n_chains=len({r['record_id']for r in rows}),wins=sum(r['win']==1 for r in rows),ties=sum(r['win']==.5 for r in rows),losses=sum(r['win']==0 for r in rows),group_equal_accuracy=float(gs.mean()),ci95_group_bootstrap=[float(v)for v in np.quantile(boot,[.025,.975])],one_sided_group_signflip_p=p,status='DESCRIPTIVE_LT5_GROUPS'if len(nz)<5 else'EVALUABLE_FIXED_EXTERNAL_TEST',group_means={k:float(np.mean(v))for k,v in sorted(bg.items())})
def main():
 freeze=json.loads((P/'COHORT_FREEZE.json').read_text(encoding='utf8'))
 for f,h in freeze['files'].items():assert sha(P/f)==h,f
 for f,h in freeze['external_dependencies'].items():assert sha(f)==h,f
 chains=[json.loads(l)for l in (P/'chains.jsonl').read_text(encoding='utf8').splitlines()if l];pairs=table(P/'pairs.tsv');byid={r['record_id']:r for r in chains};seqs={r['sequence_sha256']:r['sequence']for r in chains}
 priorcount=sum(bool(l.strip())for l in(P/'prior_native_sequences.jsonl').read_text(encoding='utf8').splitlines());audits=Counter()
 with gzip.open(P/'prior_full_native_similarity_audit.tsv.gz','rt',encoding='utf8')as f:
  for row in csv.DictReader(f,delimiter='\t'):
   if row['new_sequence_sha256']in seqs:
    assert row['meets_cutoff']=='False';audits[row['new_sequence_sha256']]+=1
 assert set(audits)==set(seqs)and all(n==priorcount for n in audits.values())
 annotations={(r['record_id'],int(r['seq_index0'])):r for r in table(P/'residue_annotations.tsv')}
 for r in chains:
  assert sha(r['source_cif'])==r['source_cif_sha256']
  assert sha(r['uniprot_json'])==r['uniprot_json_sha256']
 for r in pairs:
  c=byid[r['record_id']];a,b=int(r['index_positive']),int(r['index_negative']);assert a!=b and c['sequence'][a]==c['sequence'][b]==r['aa'];assert r['split']=='E_new'
  assert c['group']==r['group']==r['homology_group']
  assert a==c['native_offset0']+int(r['positive_label_seq_id'])-1 and b==c['native_offset0']+int(r['negative_label_seq_id'])-1
  ap,an=annotations[(r['record_id'],a)],annotations[(r['record_id'],b)]
  assert all(z['backbone_quality']=='True'and z['local_backbone_continuous']=='True'for z in(ap,an))
  if r['task']=='helix_start_vs_internal':
   assert all(z['secondary']=='H'and z['long_helix_complete']=='True'for z in(ap,an))
   assert ap['helix_role']=='start'and an['helix_role']=='internal'
   assert 0<=int(ap['helix_offset0'])<2 and int(an['helix_offset0'])>=3 and int(an['helix_distance_to_end'])>=3
   assert (r['same_helix_segment']=='True')==(ap['helix_segment_id']==an['helix_segment_id'])
  else:assert(ap['secondary'],an['secondary'])==('H','E')
 sys.path.insert(0,str(E));import e1_runtime as rt;import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device='cuda'if torch.cuda.is_available()and torch.cuda.mem_get_info()[0]>3*1024**3 else'cpu';model=rt.get_model(trained=True,device=device);ctx={}
 def capture(key,ch):
  def hook(mod,args):
   a=args[0];assert a.shape[0]==1 and a.shape[-1]==2304;ctx[key]=a[0,2:-2,ch].detach().float().cpu().numpy().copy()
  return hook
 handles=[model.model.layers[14].ffn.mlp.w2.register_forward_pre_hook(capture('2192',2192)),model.model.layers[19].ffn.mlp.w2.register_forward_pre_hook(capture('1900',1900))];v={k:{}for k in('2192','1900')};coverage=[];t=time.time()
 for i,(h,s)in enumerate(sorted(seqs.items()),1):
  with torch.inference_mode():out=model.model(**rt.get_batch(s,device=device))
  assert torch.isfinite(out.last_hidden_state).all()
  for k in v:
   a=ctx.pop(k);assert a.shape==(len(s),)and np.isfinite(a).all();v[k][h]=a
  coverage.append(dict(sequence_sha256=h,sequence_length=len(s),forwards=1,device=device,record_ids=';'.join(r['record_id']for r in chains if r['sequence_sha256']==h)))
  if i%10==0:print('FORWARD',i,len(seqs),'sec',round(time.time()-t,1),device,flush=True)
 for h in handles:h.remove()
 for k in v:np.savez_compressed(P/f'unit{k}_per_native_sequence.npz',**v[k])
 tsv(P/'forward_coverage.tsv',coverage);scored=[]
 for r in pairs:
  row=dict(r);k='2192'if r['task']=='helix_start_vs_internal'else'1900';sign=-1 if k=='2192'else 1;a,b=int(r['index_positive']),int(r['index_negative']);x,y=(float(v[k][r['sequence_sha256']][z])for z in(a,b));delta=sign*(x-y)
  row.update(channel0=int(k),layer0=14 if k=='2192'else 19,sign=sign,value_positive=x,value_negative=y,signed_delta=delta,win=1. if delta>0 else 0. if delta<0 else.5);scored.append(row)
 tsv(P/'pair_results.tsv',scored)
 strict=[r for r in scored if r['task']=='helix_start_vs_internal'and r['same_helix_segment']=='True'];allstart=[r for r in scored if r['task']=='helix_start_vs_internal'];he=[r for r in scored if r['task']=='helix_vs_strand'];stats={'primary_2192_same_helix':stat(strict),'sensitivity_2192_all_pairs':stat(allstart),'secondary_1900_helix_vs_strand':stat(he)}
 tests=['primary_2192_same_helix','secondary_1900_helix_vs_strand'];order=sorted(tests,key=lambda k:stats[k].get('one_sided_group_signflip_p',1));last=0.
 for i,k in enumerate(order):last=max(last,min(1.,(len(order)-i)*stats[k].get('one_sided_group_signflip_p',1)));stats[k]['holm_p_two_external_fixed_tests']=last
 dump(P/'RESULTS.json',dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),newly_recruited_structures=True,pretraining_sequence_overlap='UNKNOWN',cohort_freeze_sha256=sha(P/'COHORT_FREEZE.json'),n_total_chains=len(chains),n_total_native_sequences=len(seqs),fixed_channels={'2192':{'layer0':14,'sign':-1},'1900':{'layer0':19,'sign':1}},stats=stats))
 ex=json.loads((P/'EXAMPLE_SELECTION.json').read_text(encoding='utf8'));pair=ex['pair'];row=next((r for r in scored if pair and r['pair_id']==pair['pair_id']),None)
 if row:
  c=byid[row['record_id']];ann={(r['record_id'],r['seq_index0']):r for r in table(P/'residue_annotations.tsv')};dump(P/'REAL_EXAMPLE.json',{'selection':ex,'result':row,'chain':c,'positive_annotation':ann[(row['record_id'],row['index_positive'])],'negative_annotation':ann[(row['record_id'],row['index_negative'])],'public_uniprot_url':'https://www.uniprot.org/uniprotkb/'+c['accession']+'/entry'})
 # One consolidated acceptance of the complete assay.
 assert len(coverage)==len(seqs)and all(r['forwards']==1 for r in coverage)
 assert all(r['sequence_sha256']in v['2192']for r in chains)
 for r in chains:assert hashlib.sha256(r['sequence'].encode()).hexdigest()==r['sequence_sha256']and r['sequence'].find(r['entity_sequence'])==r['native_offset0']
 for f,h in freeze['files'].items():assert sha(P/f)==h,f
 modelmeta=json.loads((E/'trained/model_metadata.json').read_text(encoding='utf8'));weightsha=sha(E/'weights/model.safetensors');assert weightsha==modelmeta['weight_sha256']
 dump(P/'ACCEPTANCE.json',{'status':'PASS','coverage_sequences':len(seqs),'pairs':len(scored),'frozen_inputs_unchanged':True,'same_chain_same_AA_and_native_indices':True,'finite_full_sequence_channels':True,'weight_sha256':weightsha,'source_revision':modelmeta['source_revision'],'weight_revision':modelmeta['weight_revision'],'runtime_python':sys.executable,'device':device,'note':'Engineering acceptance; date alone does not prove unseen model pretraining sequence.'})
 print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
