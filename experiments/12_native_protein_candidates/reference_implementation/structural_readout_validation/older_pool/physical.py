"""Download and physical/natural-label screen; entirely result-blind."""
from pathlib import Path
import sys,os,json,gzip,re,csv,hashlib,argparse,time,subprocess,importlib.util
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(P.parent/'sheet'));import build_sheet as sh
sys.path.insert(0,str(P));from download import source
OLD=ROOT/'native_channels/cohort'
spec=importlib.util.spec_from_file_location('old',OLD/'build_structure_panel.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BADPLUS=re.compile(r'antibod|immunoglobulin|nanobod|fusion|synthetic|designer|biosynthetic',re.I)
def get_native(acc):
    dest=P/f'sources/uniprot/{acc}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    # Reuse cached public native reference JSON if available.
    fresh=P.parent/f'fresh/sources/uniprot/{acc}.json'
    if fresh.exists():return json.loads(fresh.read_text()),str(fresh)
    if not dest.exists():
        part=Path(str(dest)+'.part.'+str(os.getpid()));r=subprocess.run(['curl.exe','--silent','--show-error','--fail','--location','--max-time','45','--retry','1',f'https://rest.uniprot.org/uniprotkb/{acc}.json','-o',str(part)],capture_output=True)
        if r.returncode:raise RuntimeError('native_fetch_failed '+acc)
        part.replace(dest)
    return json.loads(dest.read_text()),str(dest)
def raw_entity(path,entity):
    b=old.pdbblock(Path(path));pid=Path(path).name[:4].lower();eid=entity.split('_')[1]
    dates=[old.val(x.get('revision_date'))for x in old.table(b,'_pdbx_audit_revision_history.')if old.val(x.get('ordinal'))=='1'];methods=[old.val(x.get('method'))for x in old.table(b,'_exptl.')]
    if methods!=['X-RAY DIFFRACTION'] or not dates or not '2010-01-01'<=dates[0]<='2024-12-31':return None,'method_or_initial_date'
    res=min([old.num(x.get('ls_d_res_high'),99)for x in old.table(b,'_refine.')]or[99])
    if res>2.5:return None,'resolution'
    ent={old.val(x['id']):x for x in old.table(b,'_entity.')};e=ent.get(eid,{});name=old.val(e.get('pdbx_description'));tax=set();names=set();accs=set()
    for cat,key,nk in [('_entity_src_gen.','pdbx_gene_src_ncbi_taxonomy_id','pdbx_gene_src_scientific_name'),('_entity_src_nat.','pdbx_ncbi_taxonomy_id','pdbx_organism_scientific')]:
        for x in old.table(b,cat):
            if old.val(x.get('entity_id'))!=eid:continue
            try:tax.add(int(x[key]));names.add(old.val(x.get(nk)))
            except:pass
    for x in old.table(b,'_struct_ref.'):
        if old.val(x.get('entity_id'))==eid and old.val(x.get('db_name')).upper() in {'UNP','UNIPROT'}:accs.add(old.val(x.get('pdbx_db_accession')))
    polys=[x for x in old.table(b,'_entity_poly.')if old.val(x.get('entity_id'))==eid]
    if not polys:return None,'no_polymer'
    poly=polys[0];seq=re.sub(r'\s+','',old.val(poly.get('pdbx_seq_one_letter_code_can')));literal=re.sub(r'\s+','',old.val(poly.get('pdbx_seq_one_letter_code')))
    if not tax or not tax<=old.SAFE or len(tax)!=1:return None,'source_not_unique_allowed'
    if old.BAD.search(name)or BADPLUS.search(name):return None,'description_excluded'
    if old.val(e.get('pdbx_mutation')):return None,'declared_mutation'
    if not 60<=len(seq)<=1000 or set(seq)-set(old.AA)or literal!=seq:return None,'entity_length_modified_or_noncanonical'
    if len(accs)!=1:return None,'uniprot_accession_not_unique'
    chains=sorted(old.val(x['id'])for x in old.table(b,'_struct_asym.')if old.val(x.get('entity_id'))==eid)
    if not chains:return None,'no_chain'
    ch=chains[0]
    ranges=[x for x in old.table(b,'_struct_sheet_range.')if old.val(x.get('beg_label_asym_id'))==ch]
    if not ranges:return None,'no_sheet_annotation'
    return dict(record_id=f'{pid}_{ch}',chain_id=f'{pid}_{ch}',pdb_id=pid,label_asym_id=ch,entity_id=eid,accession=next(iter(accs)),entity_sequence=seq,sequence=seq,sequence_length=len(seq),sequence_sha256=old.digest(seq),native_offset0=0,source_cif=path,source_cif_sha256=old.sha(path),taxid=str(next(iter(tax))),organism=';'.join(sorted(names)),protein_name=name,resolution_A=res,structure_initial_release_date=dates[0],public_cif_url=f'https://files.rcsb.org/download/{pid}.cif.gz'),''
def worker(group):
    cache=P/f"physical_groups_v2/{group['cluster']}.json";cache.parent.mkdir(exist_ok=True)
    priorcache=P/f"physical_groups/{group['cluster']}.json"
    previous=json.loads(cache.read_text()) if cache.exists() else (json.loads(priorcache.read_text()) if priorcache.exists() else None)
    if previous:
        allowed=set(group['entities']);chosen=previous.get('chain',{});chosen_id=chosen.get('pdb_id','').upper()+'_'+str(chosen.get('entity_id',''))
        if previous['status']=='physically_paired_natural' and chosen_id not in allowed:previous=None
        elif previous['status']!='physically_paired_natural':previous={'cluster':group['cluster'],'status':'failed','attempts':[a for a in previous['attempts']if a['entity']in allowed]}
    if previous and (previous['status']=='physically_paired_natural' or {a['entity'] for a in previous['attempts']}>=set(group['entities'])):
        cache.write_text(json.dumps(previous,ensure_ascii=False)+'\n',encoding='utf-8');return previous
    result=previous or {'cluster':group['cluster'],'attempts':[],'status':'failed'}
    tried={a['entity'] for a in result['attempts']}
    for entity in group['entities']:
        if entity in tried:continue
        try:
            path=source(entity.split('_')[0]);r,why=raw_entity(path,entity)
            if r is None:result['attempts'].append({'entity':entity,'reason':why});continue
            r.update(group=group['cluster'],homology_group=group['cluster'],original_group=group['cluster'],split='D',original_split='D')
            src=sh.parse_source(path);graphs,sites=sh.assess(r,src);pairs=sh.select([r],sites)
            if not pairs:result['attempts'].append({'entity':entity,'reason':'no_pair_after_physical_sheet_gates','sheets_audited':len(graphs),'eligible_sheets':sum(x['eligible']for x in graphs)});continue
            u,up=get_native(r['accession']);seq=u.get('sequence',{}).get('value','');tax=u.get('organism',{}).get('taxonId');off=seq.find(r['entity_sequence'])
            if tax not in old.SAFE or str(tax)!=r['taxid']or not 60<=len(seq)<=1000 or set(seq)-set(old.AA)or off<0 or seq.find(r['entity_sequence'],off+1)>=0:
                result['attempts'].append({'entity':entity,'reason':'no_unique_exact_full_native_host_mapping'});continue
            r.update(sequence=seq,sequence_length=len(seq),sequence_sha256=old.digest(seq),native_offset0=off,uniprot_json=up,uniprot_json_sha256=old.sha(up),auth_asym_id=';'.join(sorted({old.val(x.get('auth_asym_id'))for x in old.table(src['block'],'_atom_site.')if old.val(x.get('label_asym_id'))==r['label_asym_id']})))
            # Label geometry is unchanged by translating deposited positions to native indices.
            for s in sites:s.update(index0=off+s['label_seq_id']-1,sequence_sha256=r['sequence_sha256'])
            pairs=sh.select([r],sites)
            result.update(status='physically_paired_natural',chain=r,graphs=graphs,sites=sites,pairs=pairs);result['attempts'].append({'entity':entity,'reason':'accepted_physical_native','pairs':len(pairs)});break
        except Exception as e:result['attempts'].append({'entity':entity,'reason':'download_parse_error','error':repr(e)})
    cache.write_text(json.dumps(result,ensure_ascii=False)+'\n',encoding='utf-8');return result
def main():
    a=argparse.ArgumentParser();a.add_argument('--limit',type=int,default=600);a.add_argument('--workers',type=int,default=6);args=a.parse_args();freeze=json.loads((P/'ACQUISITION_FREEZE.json').read_text())
    if (P/'BACKUP_ACQUISITION_FREEZE.json').exists():
        backup=json.loads((P/'BACKUP_ACQUISITION_FREEZE.json').read_text())
        if backup['query_sha256']==freeze['query_sha256']:freeze=backup
    groups=freeze['groups'][:args.limit];out=[];t=time.time()
    with ProcessPoolExecutor(max_workers=args.workers)as pool:
        for i,r in enumerate(pool.map(worker,groups,chunksize=1),1):
            out.append(r)
            if i%10==0:print('PHYSICAL',i,len(groups),'paired',sum(x['status']=='physically_paired_natural'for x in out),'sec',round(time.time()-t,1),flush=True)
    rr=[x['chain']for x in out if 'chain'in x];sh.jl(P/f'physical_candidates_{args.limit}.jsonl',rr);sh.jl(P/f'physical_flow_{args.limit}.jsonl',[{k:v for k,v in x.items()if k in ('cluster','status','attempts')}for x in out]);print('PHYSICAL_DONE',len(groups),len(rr),flush=True)
if __name__=='__main__':main()
