"""Natural experimental host-protein structural contrasts; never reads model output."""
from pathlib import Path
import sys, os, json, csv, gzip, hashlib, re, math
from collections import defaultdict, Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCH=ROOT/'structure'
EXT=ROOT/'cross_architecture/external'
CYS=ROOT/'nccal_evidence_extension/cohort'
sys.path.insert(0,str(ARCH/'runtime'))
import gemmi, numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
AA3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
SAFE={9606,10090,4932,559292,3702}
BAD=re.compile(r'toxin|venom|virulen|viral|virus|pathogen|defensin|antimicrobial|interferon|interleukin|cytokine|chemokine|growth factor|hemagglutinin|haemagglutinin|engineered|designed|chimer',re.I)
def utc():return datetime.now(timezone.utc).isoformat()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def jl(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr),encoding='utf-8')
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def tsv(p,rr,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rr)
def table(b,cat):
    z=b.get_mmcif_category(cat)
    return [dict(zip(z,row)) for row in zip(*z.values())] if z else []
def val(x):return '' if x in (None,False,'.','?') else str(x)
def num(x,default=0.):
    try:return float(x)
    except:return default
def pdbblock(path):return gemmi.cif.read_string(gzip.decompress(path.read_bytes()).decode()).sole_block()

def references():
    refs=readjl(ARCH/'cohort_index.jsonl')
    for r in readjl(EXT/'sequences.jsonl'):
        refs.append(dict(r,split='E',candidates=[{'pdb_id':r['pdb_id']}]))
    # Supplement native references use the already-frozen Cys cohort; only if no old sequence match.
    oldseq={r['sequence'] for r in refs}
    for r in readjl(CYS/'chains.jsonl'):
        if r['sequence'] in oldseq:continue
        refs.append(dict(r,group=r['homology_group'],split='D' if r['split'] in 'AB' else 'T',original_cys_split=r['split'],candidates=[{'pdb_id':r['pdb_id']}]))
    byp=defaultdict(list)
    for r in refs:
        for c in r['candidates']:byp[c['pdb_id'].lower()].append(r)
    paths={p.name[:4].lower():p for d in (ARCH/'cache/mmcif',EXT/'cache/mmcif',CYS/'public_sources/mmcif') if d.exists() for p in d.glob('*.cif.gz')}
    return refs,byp,paths

def inventory_one(pid,path,refs):
    b=pdbblock(path)
    methods=[val(x.get('method')) for x in table(b,'_exptl.')]
    if methods!=['X-RAY DIFFRACTION']:return [],{'pdb_id':pid,'reason':'not_single_Xray'}
    res=min([num(x.get('ls_d_res_high'),99) for x in table(b,'_refine.')] or [99])
    if res>3:return [],{'pdb_id':pid,'reason':'resolution_gt3A'}
    ent={val(x['id']):x for x in table(b,'_entity.')}
    src=defaultdict(set);names=defaultdict(set)
    for cat,key,namekey in [('_entity_src_gen.','pdbx_gene_src_ncbi_taxonomy_id','pdbx_gene_src_scientific_name'),('_entity_src_nat.','pdbx_ncbi_taxonomy_id','pdbx_organism_scientific')]:
        for x in table(b,cat):
            try:src[val(x['entity_id'])].add(int(x[key]))
            except:continue
            names[val(x['entity_id'])].add(val(x.get(namekey)))
    accs=defaultdict(set)
    for x in table(b,'_struct_ref.'):
        if val(x.get('db_name')).upper() in {'UNP','UNIPROT'}:accs[val(x['entity_id'])].add(val(x.get('pdbx_db_accession')))
    chains=defaultdict(list)
    for x in table(b,'_struct_asym.'):chains[val(x['entity_id'])].append(val(x['id']))
    keep=[]
    for poly in table(b,'_entity_poly.'):
        eid=val(poly['entity_id']);e=ent.get(eid,{});tax=src[eid];name=val(e.get('pdbx_description'))
        if not tax or not tax<=SAFE or BAD.search(name) or val(e.get('pdbx_mutation')):continue
        seq=re.sub(r'\s+','',val(poly.get('pdbx_seq_one_letter_code_can')))
        if not 60<=len(seq)<=1000 or set(seq)-set(AA):continue
        matches=[]
        for r in refs:
            s=r['sequence'];off=s.find(seq)
            if 60<=len(s)<=1000 and not set(s)-set(AA) and off>=0 and s.find(seq,off+1)<0 and (not accs[eid] or r['accession'] in accs[eid]):matches.append((r,off))
        if not matches or len({r['sequence'] for r,o in matches})!=1:continue
        # Original D precedes T precedes E; any conflicting historical split is audited later.
        r,off=sorted(matches,key=lambda x:({'D':0,'T':1,'E':2}[x[0]['split']],x[0]['group'],x[0]['accession']))[0]
        for ch in sorted(chains[eid]):
            keep.append({'record_id':f'{pid}_{ch}','chain_id':f'{pid}_{ch}','pdb_id':pid,'label_asym_id':ch,'entity_id':eid,'accession':r['accession'],'sequence':r['sequence'],'sequence_sha256':digest(r['sequence']),'entity_sequence':seq,'native_offset0':off,'original_group':r['group'],'original_split':r['split'],'original_cys_split':r.get('original_cys_split',''),'source_cif':str(path),'source_cif_sha256':sha(path),'taxid':';'.join(map(str,sorted(tax))),'organism':';'.join(sorted(names[eid])),'protein_name':name,'resolution_A':res})
    return keep,{'pdb_id':pid,'reason':'eligible' if keep else 'no_allowed_exact_native_host_entity','chains':len(keep)}

def inventory():
    refs,byp,paths=references();rr=[];funnel=[]
    jobs=[(p,paths[p],r) for p,r in sorted(byp.items()) if p in paths]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(inventory_one,*z):z[0] for z in jobs}
        for i,f in enumerate(as_completed(futures)):
            try:a,b=f.result();rr.extend(a);funnel.append(b)
            except Exception as e:funnel.append({'pdb_id':futures[f],'reason':'parse_error','error':str(e)})
            if i%200==0:print('INVENTORY',i,len(jobs),'eligible',len(rr),flush=True)
    rr.sort(key=lambda r:r['record_id']);funnel.sort(key=lambda r:r['pdb_id'])
    jl(HERE/'candidate_chains.jsonl',rr);jl(HERE/'inventory_funnel.jsonl',funnel)
    write(HERE/'INVENTORY.json',{'utc':utc(),'candidates':len(rr),'sequences':len({r['sequence'] for r in rr}),'original_groups':len({r['original_group'] for r in rr}),'by_split':dict(Counter(r['original_split'] for r in rr)),'files_scanned':len(jobs),'activation_or_model_scores_read':False})

class Union:
    def __init__(self,n):self.p=list(range(n))
    def root(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,i,j):self.p[self.root(j)]=self.root(i)

def homology(chains):
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal,SubstitutionMatrix
    seqs=sorted({r['sequence'] for r in chains});ix={s:i for i,s in enumerate(seqs)};u=Union(len(seqs));edges=[];audit=[]
    byold=defaultdict(list);byent=defaultdict(set)
    for r in chains:byold[r['original_group']].append(ix[r['sequence']]);byent[f"{r['pdb_id'].upper()}_{r['entity_id']}"].add(ix[r['sequence']])
    for ii in byold.values():
        for j in ii[1:]:u.join(ii[0],j)
    cp=EXT/'sources/clusters-by-entity-30.txt'
    if not cp.exists():cp=CYS/'public_sources/clusters-by-entity-30.txt'
    for line in cp.read_text().splitlines():
        ii=sorted(set().union(*(byent.get(x,set()) for x in line.split())))
        for j in ii[1:]:u.join(ii[0],j)
    pp=[ProteinSequence(s) for s in seqs];matrix=SubstitutionMatrix.std_protein_matrix()
    for i,s in enumerate(seqs):
        for j in range(i):
            if u.root(i)==u.root(j):continue
            a=align_optimal(pp[i],pp[j],matrix,gap_penalty=(-10,-1),local=True,max_number=1)[0]
            t=a.trace;both=t[(t[:,0]>=0)&(t[:,1]>=0)]
            ident=sum(s[k]==seqs[j][l] for k,l in both)/len(both) if len(both) else 0
            coverage=min(len(set(t[t[:,0]>=0,0])),len(set(t[t[:,1]>=0,1])))/min(len(s),len(seqs[j]))
            hit=ident>=.30 and coverage>=.80
            audit.append({'sequence_sha256_1':digest(s),'sequence_sha256_2':digest(seqs[j]),'identity':ident,'coverage_shorter':coverage,'meets_cutoff':hit})
            if hit:u.join(i,j);edges.append(audit[-1])
        if i%25==0:print('HOMOLOGY',i,len(seqs),flush=True)
    cc=defaultdict(list)
    for r in chains:cc[u.root(ix[r['sequence']])].append(r)
    retained=[];excluded=[];roster=[]
    for rr in cc.values():
        originals=sorted({r['original_group'] for r in rr});splits={r['original_split'] for r in rr}
        group=originals[0] if len(originals)==1 else 'structure-union-'+digest('|'.join(originals))[:20]
        priority=min(splits,key=lambda s:{'D':0,'T':1,'E':2}[s])
        for r in rr:
            r['group']=group;r['homology_group']=group;r['split']=r['original_split'];r['sequence_length']=len(r['sequence'])
            if r['original_split']!=priority:excluded.append(dict(r,reason='homology_to_earlier_split_quarantined_not_reassigned'))
            else:retained.append(r)
        roster.append({'group':group,'original_groups':';'.join(originals),'original_splits':';'.join(sorted(splits)),'retained_split':priority,'retained_sequences':len({r['sequence'] for r in rr if r['original_split']==priority}),'quarantined_chains':sum(r['original_split']!=priority for r in rr)})
    jl(HERE/'homology_quarantine.jsonl',excluded);tsv(HERE/'sequence_similarity_audit.tsv',audit);tsv(HERE/'group_roster.tsv',roster);jl(HERE/'additional_homology_edges.jsonl',edges)
    return retained

def structure_chain(r):
    b=pdbblock(Path(r['source_cif']));ch=r['label_asym_id'];seq=r['entity_sequence'];off=r['native_offset0'];L=len(seq)
    atoms=defaultdict(list)
    for x in table(b,'_atom_site.'):
        if val(x.get('label_asym_id'))==ch and val(x.get('label_seq_id')).isdigit():atoms[int(x['label_seq_id'])].append(x)
    coords={};audit=[];alllabels=defaultdict(set)
    for p in range(1,L+1):
        aa=seq[p-1];xx=atoms.get(p,[]);bb=[x for x in xx if val(x.get('label_atom_id')) in {'N','CA','C','O'}]
        counts=Counter(val(x.get('label_atom_id')) for x in bb);why=[]
        if counts!={'N':1,'CA':1,'C':1,'O':1}:why.append('missing_or_duplicate_backbone')
        if any(val(x.get('label_alt_id')) for x in bb):why.append('backbone_altloc')
        if any(num(x.get('occupancy'))<.99 for x in bb):why.append('backbone_occupancy_lt099')
        if any(AA3.get(val(x.get('label_comp_id')))!=aa for x in xx):why.append('residue_identity_mismatch')
        if len({val(x.get('pdbx_PDB_model_num')) for x in xx})>1:why.append('multiple_models')
        if not why:coords[p]={val(x['label_atom_id']):np.array([num(x[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')]) for x in bb}
        audit.append({'record_id':r['record_id'],'seq_index0':off+p-1,'label_seq_id':p,'aa':aa,'backbone_quality':not why,'exclusion_reason':';'.join(why),'auth_seq_id':val(xx[0].get('auth_seq_id')) if xx else '','auth_asym_id':val(xx[0].get('auth_asym_id')) if xx else '','ins_code':val(xx[0].get('pdbx_PDB_ins_code')) if xx else ''})
    connected={}
    for p in range(1,L):
        connected[p]=p in coords and p+1 in coords and 1.0<=float(np.linalg.norm(coords[p]['C']-coords[p+1]['N']))<=1.8 and 2.7<=float(np.linalg.norm(coords[p]['CA']-coords[p+1]['CA']))<=4.3
    segs=[]
    for cat in ('_struct_conf.','_struct_sheet_range.'):
        for x in table(b,cat):
            if val(x.get('beg_label_asym_id'))!=ch or val(x.get('end_label_asym_id'))!=ch:continue
            try:s=int(x['beg_label_seq_id']);e=int(x['end_label_seq_id'])
            except:continue
            if not 1<=s<=e<=L:continue
            if AA3.get(val(x.get('beg_label_comp_id')))!=seq[s-1] or AA3.get(val(x.get('end_label_comp_id')))!=seq[e-1]:continue
            if cat=='_struct_sheet_range.':label='E'
            else:label='H' if (val(x.get('conf_type_id'))=='HELX_P' and val(x.get('pdbx_PDB_helix_class'))=='1') or val(x.get('conf_type_id'))=='HELX_RH_AL_P' else 'other_helix'
            segs.append({'record_id':r['record_id'],'annotation_id':val(x.get('id')),'secondary':label,'start_label_seq_id':s,'end_label_seq_id':e,'annotation_category':cat})
            for p in range(s,e+1):alllabels[p].add(label)
    # Merge touching/overlapping alpha annotations into maximal consecutive alpha ranges.
    hranges=[]
    for x in sorted([x for x in segs if x['secondary']=='H'],key=lambda x:x['start_label_seq_id']):
        s,e=x['start_label_seq_id'],x['end_label_seq_id']
        if hranges and s<=hranges[-1][1]+1:hranges[-1][1]=max(hranges[-1][1],e)
        else:hranges.append([s,e])
    helixinfo={}
    longsegments=[]
    for n,(s,e) in enumerate(hranges,1):
        length=e-s+1
        complete=s>1 and e<L and all(p in coords for p in range(s-1,e+2)) and all(connected.get(p,False) for p in range(s-1,e+1)) and all(alllabels[p]=={'H'} for p in range(s,e+1)) and 'H' not in alllabels[s-1] and 'H' not in alllabels[e+1]
        sid=f"{r['record_id']}:alpha{s}-{e}"
        longsegments.append({'record_id':r['record_id'],'segment_id':sid,'start_label_seq_id':s,'end_label_seq_id':e,'start_index0':off+s-1,'end_index0':off+e-1,'length':length,'boundary_and_full_backbone_continuous':complete,'eligible_long_helix':complete and length>=10})
        for p in range(s,e+1):helixinfo[p]=(sid,s,e,length,complete)
    for row in audit:
        p=row['label_seq_id'];labels=alllabels[p];ss=next(iter(labels)) if len(labels)==1 else 'ambiguous' if labels else 'unassigned'
        row.update({'secondary':ss,'local_backbone_continuous':bool(connected.get(p-1) and connected.get(p)),'group':r['group'],'split':r['split'],'helix_segment_id':'','helix_start_label_seq_id':'','helix_end_label_seq_id':'','helix_length':'','helix_offset0':'','helix_distance_to_end':'','long_helix_complete':False,'helix_role':''})
        if p in helixinfo:
            sid,s,e,length,complete=helixinfo[p]
            role='start' if p-s<2 else 'internal' if p-s>=3 and e-p>=3 else 'other'
            row.update({'helix_segment_id':sid,'helix_start_label_seq_id':s,'helix_end_label_seq_id':e,'helix_length':length,'helix_offset0':p-s,'helix_distance_to_end':e-p,'long_helix_complete':complete and length>=10,'helix_role':role})
    auth={x['auth_asym_id'] for x in audit if x['auth_asym_id']};r['auth_asym_id']=';'.join(sorted(auth))
    return audit,longsegments

def select_pairs(chains,sites):
    byc=defaultdict(list);byid={r['record_id']:r for r in chains};candidates=[]
    for r in sites:byc[r['record_id']].append(r)
    for cid,rr in byc.items():
        cr=byid[cid]
        for task in ('helix_vs_strand','helix_start_vs_internal'):
            strata=defaultdict(lambda:[[],[]])
            for s in rr:
                if not s['backbone_quality'] or not s['local_backbone_continuous']:continue
                if task=='helix_vs_strand':label=1 if s['secondary']=='H' else 0 if s['secondary']=='E' else None
                else:label=1 if s['long_helix_complete'] and s['helix_role']=='start' else 0 if s['long_helix_complete'] and s['helix_role']=='internal' else None
                if label is not None:strata[s['aa']][label].append(s)
            for aa,(negative,positive) in strata.items():
                # Same position hash for either role; reversing label orientation keeps unordered pair identities.
                key=lambda s:digest(f"structure-site-v1|{cr['sequence_sha256']}|{s['seq_index0']}")
                matched=[]
                if task=='helix_start_vs_internal':
                    consumed=set()
                    for sid in sorted({s['helix_segment_id'] for s in positive+negative}):
                        pp=sorted([s for s in positive if s['helix_segment_id']==sid],key=key)
                        nn=sorted([s for s in negative if s['helix_segment_id']==sid],key=key)
                        for p,n in zip(pp,nn):matched.append((p,n));consumed.update((p['seq_index0'],n['seq_index0']))
                    positive=[s for s in positive if s['seq_index0'] not in consumed];negative=[s for s in negative if s['seq_index0'] not in consumed]
                matched.extend(zip(sorted(positive,key=key),sorted(negative,key=key)))
                for p,n in matched:
                    lo,hi=sorted((p['seq_index0'],n['seq_index0']));h=digest(f"structure-pair-v1|{task}|{cr['sequence_sha256']}|{lo}|{hi}|{cid}")
                    candidates.append({'pair_id':f'{task}:{cid}:{lo}-{hi}','record_id':cid,'chain_id':cid,'group':cr['group'],'homology_group':cr['group'],'split':cr['split'],'task':task,'aa':aa,'index_positive':p['seq_index0'],'index_negative':n['seq_index0'],'positive_label_seq_id':p['label_seq_id'],'negative_label_seq_id':n['label_seq_id'],'positive_secondary':p['secondary'],'negative_secondary':n['secondary'],'positive_helix_segment_id':p['helix_segment_id'],'negative_helix_segment_id':n['helix_segment_id'],'same_helix_segment':bool(p['helix_segment_id'] and p['helix_segment_id']==n['helix_segment_id']),'positive_helix_offset0':p['helix_offset0'],'negative_helix_offset0':n['helix_offset0'],'positive_helix_length':p['helix_length'],'negative_helix_length':n['helix_length'],'negative_helix_distance_to_end':n['helix_distance_to_end'],'selection_digest':h,'sequence_sha256':cr['sequence_sha256']})
    selected=[];used=defaultdict(set);counts=Counter()
    for p in sorted(candidates,key=lambda p:(0 if p['task']=='helix_start_vs_internal' and p['same_helix_segment'] else 1,p['selection_digest'])):
        k=(p['group'],p['task']);ids={(p['sequence_sha256'],p['index_positive']),(p['sequence_sha256'],p['index_negative'])}
        if counts[k]>=16 or ids&used[k]:continue
        selected.append(p);counts[k]+=1;used[k]|=ids
    return sorted(selected,key=lambda p:(p['split'],p['task'],p['group'],p['pair_id'])),len(candidates)

def build():
    if '--reuse-homology' in sys.argv:
        roster=list(csv.DictReader((HERE/'group_roster.tsv').open(encoding='utf-8'),delimiter='\t'));gm={old:r for r in roster for old in r['original_groups'].split(';')};chains=[]
        for r in readjl(HERE/'candidate_chains.jsonl'):
            g=gm[r['original_group']]
            if r['original_split']!=g['retained_split']:continue
            r.update(group=g['group'],homology_group=g['group'],split=r['original_split'],sequence_length=len(r['sequence']));chains.append(r)
    else:chains=homology(readjl(HERE/'candidate_chains.jsonl'))
    selected=[];excluded=[];seen=set();perseq=defaultdict(set)
    for r in sorted(chains,key=lambda r:(r['resolution_A'],r['pdb_id'],r['entity_id'],r['label_asym_id'])):
        k=(r['pdb_id'],r['entity_id']);s=r['sequence_sha256']
        if k in seen:excluded.append({'record_id':r['record_id'],'reason':'duplicate_PDB_entity_chain'});continue
        if len(perseq[s])>=3 and r['pdb_id'] not in perseq[s]:excluded.append({'record_id':r['record_id'],'reason':'sequence_max3_structures'});continue
        seen.add(k);perseq[s].add(r['pdb_id']);selected.append(r)
    sites=[];segments=[]
    for i,r in enumerate(selected):
        a,b=structure_chain(r);sites.extend(a);segments.extend(b)
        if i%25==0:print('ANNOTATION',i,len(selected),flush=True)
    jl(HERE/'annotated_candidate_chains.jsonl',selected)
    pairs,n=select_pairs(selected,sites);used={r['record_id'] for r in pairs}
    selected=[r for r in selected if r['record_id'] in used]
    sites=[r for r in sites if r['record_id'] in used];segments=[r for r in segments if r['record_id'] in used]
    selected.sort(key=lambda r:r['record_id'])
    jl(HERE/'chains.jsonl',selected);tsv(HERE/'pairs.tsv',pairs);tsv(HERE/'residue_annotations.tsv',sites);tsv(HERE/'helix_segments.tsv',segments);jl(HERE/'chain_selection_exclusions.jsonl',excluded)
    stats={'frozen_utc':utc(),'chains':len(selected),'distinct_native_sequences':len({r['sequence'] for r in selected}),'groups':len({r['group'] for r in selected}),'pairs':len(pairs),'candidate_matched_pairs_before_group_cap':n,'model_activations_or_scores_read':False,'split_roles':{'D':'training/development','T':'primary_confirmation','E':'external_confirmation'},'homology_cutoff':'RCSB30 entity clusters union original groups plus full-native local BLOSUM62 affine(-10,-1) alignment identity>=0.30 and coverage_shorter>=0.80; connected components. Later split members quarantined, never reassigned. Remote homology still possible.','native_length_range':[60,1000],'selection':'same-chain same-amino-acid; task2 same-helix matches prioritized within chain before cross-helix leftovers and again at group cap; label-direction-independent site/pair SHA ordering within priority, no site reuse per group/task, max16 pairs/group/task, max3 experimental structures/full-native sequence','tasks':{}}
    for task in ('helix_vs_strand','helix_start_vs_internal'):
        stats['tasks'][task]={s:{'pairs':sum(r['task']==task and r['split']==s for r in pairs),'groups':len({r['group'] for r in pairs if r['task']==task and r['split']==s}),'chains':len({r['record_id'] for r in pairs if r['task']==task and r['split']==s}),'same_helix_pairs':sum(r['task']==task and r['split']==s and r['same_helix_segment'] for r in pairs),'same_helix_groups':len({r['group'] for r in pairs if r['task']==task and r['split']==s and r['same_helix_segment']})} for s in 'DTE'}
    files=['chains.jsonl','pairs.tsv','residue_annotations.tsv','helix_segments.tsv','group_roster.tsv','sequence_similarity_audit.tsv','homology_quarantine.jsonl','build_structure_panel.py']
    stats['files']={f:{'sha256':sha(HERE/f),'bytes':(HERE/f).stat().st_size} for f in files}
    write(HERE/'COHORT_FREEZE.json',stats);print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    if '--build' in sys.argv:build()
    else:inventory()
