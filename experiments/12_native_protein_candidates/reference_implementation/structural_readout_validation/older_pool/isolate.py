"""Final label-blind old/F homology isolation and salted D/H assignment."""
from pathlib import Path
import os,sys,json,gzip,csv,hashlib,argparse,time,subprocess
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
P=Path(__file__).resolve().parent;ROOT=P.parents[1];H=P.parent/'sheet'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(H));import build_sheet as sh
def init(prior):
    global PRIOR,MAT,PS,ALIGN
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal,SubstitutionMatrix
    PS=ProteinSequence;ALIGN=align_optimal;MAT=SubstitutionMatrix.std_protein_matrix();PRIOR=[(h,s,PS(s),origin)for h,s,origin in prior]
def compare(s,t):
    a=ALIGN(s,t,MAT,gap_penalty=(-10,-1),local=True,max_number=1)[0];tr=a.trace;both=tr[(tr[:,0]>=0)&(tr[:,1]>=0)]
    ident=float((s.code[both[:,0]]==t.code[both[:,1]]).mean())if len(both)else 0.;cov=min(len(set(tr[tr[:,0]>=0,0])),len(set(tr[tr[:,1]>=0,1])))/min(len(s),len(t));return ident,cov,ident>=.3 and cov>=.8
def alignone(r):
    ps=PS(r['sequence']);audit=[];hits=[]
    for h,s,pt,origin in PRIOR:
        ident,cov,hit=compare(ps,pt);audit.append((h,origin,ident,cov,hit))
        if hit:hits.append((h,origin))
    return r,audit,hits
def main():
    a=argparse.ArgumentParser();a.add_argument('--limit',type=int,default=600);args=a.parse_args();fresh=P.parent/'fresh/chains.jsonl'
    assert fresh.exists()and(P.parent/'fresh/COHORT_FREEZE.json').exists(),'Final F must freeze before old-pool isolation'
    if (P/'BACKUP_ACQUISITION_FREEZE.json').exists():
        subprocess.run([sys.executable,str(P/'physical.py'),'--limit',str(args.limit),'--workers','6'],check=True)
    rr=sh.readjl(P/f'physical_candidates_{args.limit}.jsonl');nphysical=len(rr);prior=sh.readjl(P.parent/'fresh/prior_native_sequences.jsonl');fr=sh.readjl(fresh);scope=json.loads((P.parent/'fresh/PRIOR_EXCLUSION_SCOPE.json').read_text());oldent=set();oldp=set()
    for f in scope['files']:
        for r in sh.readjl(f['path']):
            if r.get('pdb_id'):
                if r.get('entity_id'):oldent.add(r['pdb_id'].upper()+'_'+str(r['entity_id']))
                else:oldp.add(r['pdb_id'].upper())
            for c in r.get('candidates',[]):oldp.add(c['pdb_id'].upper())
    fent={r['pdb_id'].upper()+'_'+str(r['entity_id'])for r in fr};needed={r['pdb_id'].upper()+'_'+str(r['entity_id']):r['group']for r in rr};blocked={};cf=Path(json.loads((P/'ACQUISITION_FREEZE.json').read_text())['cluster_file'])
    for line in cf.read_text().splitlines():
        toks=line.split();hit=needed.keys()&set(toks)
        if not hit:continue
        reasons=[]
        if any(x in oldent or x.split('_')[0]in oldp for x in toks):reasons.append('RCSB30_prior')
        if any(x in fent for x in toks):reasons.append('RCSB30_F')
        for x in hit:
            if reasons:blocked[needed[x]]=reasons
    ex=[{'record_id':r['record_id'],'reason':';'.join(blocked[r['group']])}for r in rr if r['group']in blocked];candidates=[r for r in rr if r['group']not in blocked]
    priord={r['sequence_sha256']:(r['sequence'],'old')for r in prior}
    for r in fr:priord[r['sequence_sha256']]=(r['sequence'],'F')
    plist=[(h,s,o)for h,(s,o)in sorted(priord.items())];keep=[];t=time.time();print('ISOLATE_START',len(candidates),'vs',len(plist),flush=True)
    with gzip.open(P/f'full_native_prior_F_audit_{args.limit}.tsv.gz','wt',encoding='utf-8',newline='')as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['candidate_sha256','prior_sha256','origin','identity','shorter_coverage','hit'])
        with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(plist,))as pool:
            for i,(r,audit,hits)in enumerate(pool.map(alignone,candidates,chunksize=1),1):
                w.writerows((r['sequence_sha256'],*x)for x in audit)
                if hits:ex.append({'record_id':r['record_id'],'reason':'full_native_30_80_to_prior_or_F','hits':hits})
                else:keep.append(r)
                if i%5==0:print('ISOLATED',i,len(candidates),'keep',len(keep),'sec',round(time.time()-t,1),flush=True)
    init([]);parents=list(range(len(keep)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    audits=[];pp=[PS(r['sequence'])for r in keep]
    for i,r in enumerate(keep):
        for j,z in enumerate(keep[:i]):
            ident,cov,hit=compare(pp[i],pp[j]);audits.append({'sha1':r['sequence_sha256'],'sha2':z['sequence_sha256'],'identity':ident,'shorter_coverage':cov,'hit':hit})
            if hit:parents[root(i)]=root(j)
    cc=defaultdict(list)
    for i,r in enumerate(keep):cc[root(i)].append(r)
    rank={g['cluster']:i for i,g in enumerate(json.loads((P/'ACQUISITION_FREEZE.json').read_text())['groups'])};final=[];graphs=[];sites=[];roster=[]
    for rr in cc.values():
        ids=sorted(r['original_group']for r in rr);seqids=sorted(r['sequence_sha256']for r in rr);gid='older-hg-'+sh.h('|'.join(ids+seqids))[:24];split='H'if int(sh.h('older-sheet-split-20261123|'+gid),16)%3==0 else'D';r=min(rr,key=lambda z:rank[z['original_group']]);oldg=r['original_group'];r.update(group=gid,homology_group=gid,split=split,original_split=split)
        final.append(r);roster.append({'group':gid,'split':split,'rcsb30_groups':ids,'native_sequence_sha256':seqids,'chosen_record_id':r['record_id']})
        cached=json.loads((P/f'physical_groups_v2/{oldg}.json').read_text())
        for g in cached['graphs']:graphs.append(dict(g,group=gid,split=split))
        for s in cached['sites']:sites.append(dict(s,group=gid,split=split))
    final.sort(key=lambda r:r['record_id']);pairs=sh.select(final,sites);sh.jl(P/'chains.jsonl',final);sh.jl(P/'group_roster.jsonl',roster);sh.jl(P/f'homology_exclusions_{args.limit}.jsonl',ex);sh.tsv(P/f'new_components_audit_{args.limit}.tsv',audits)
    # Reuse existing physical labels and exact source coordinates; do not remeasure unchanged structures.
    tag='_D_extra';used={p['record_id']for p in pairs};sh.jl(H/f'chains{tag}.jsonl',[r for r in final if r['record_id']in used]);sh.tsv(H/f'pairs{tag}.tsv',pairs);sh.tsv(H/f'sites{tag}.tsv',sites);sh.jl(H/f'sheet_graph_audit{tag}.jsonl',graphs);sh.jl(H/f'build_errors{tag}.jsonl',[])
    counts={sp:{'pairs':sum(p['split']==sp for p in pairs),'groups':len({p['group']for p in pairs if p['split']==sp}),'chains':len({p['record_id']for p in pairs if p['split']==sp}),'sheet_ge4_pairs':sum(p['split']==sp and p['sheet_ge4_bool']for p in pairs),'confound_matched_pairs':sum(p['split']==sp and p['confound_matched_bool']for p in pairs)}for sp in ('D','H')}
    summary={'utc':sh.utc(),'input_path':str(P/'chains.jsonl'),'input_chains':len(final),'input_sequences':len(final),'input_groups':len(final),'sheets_audited':len(graphs),'eligible_sheets':sum(g['eligible']for g in graphs),'pairs':len(pairs),'groups':len({p['group']for p in pairs}),'counts_by_split':counts,'sources':{r['source_cif']:r['source_cif_sha256']for r in final},'limit':args.limit,'physical_candidates':nphysical,'after_rcsb30_block':len(candidates),'after_full_native_prior_F':len(keep),'prior_F_input_hashes':{str(fresh):sh.sha(fresh),str(P.parent/'fresh/prior_native_sequences.jsonl'):sh.sha(P.parent/'fresh/prior_native_sequences.jsonl')},'model_scores_read':False}
    sh.js(H/f'BUILD_SUMMARY{tag}.json',summary);sh.js(P/f'ISOLATION_SUMMARY_{args.limit}.json',summary);print(json.dumps({k:v for k,v in summary.items()if k not in ('sources',)},indent=2),flush=True)
if __name__=='__main__':main()
