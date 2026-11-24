"""Frozen, bounded repeat-deposition test. No writes outside HERE."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,csv,gzip,hashlib,re,shutil,importlib.util,traceback
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CROSS=ROOT/'cross_architecture'
ARCH=ROOT/'structure'
sys.path.insert(0,str(ARCH/'runtime'))
import numpy as np,gemmi
sp=importlib.util.spec_from_file_location('archived_geometry',ARCH/'build_structure_extension.py')
BASE=importlib.util.module_from_spec(sp);sp.loader.exec_module(BASE);BASE.OUT=HERE
ALLOWED={9606,10090,4932,559292}
DANGER=re.compile(r'toxin|venom|viral|virus|pathogen|virulence|antimicrobial|defensin',re.I)

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):BASE.write(p,x)
def jl(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def savejl(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def tsv(p,rows,fields):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)
def rt(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline=''),delimiter='\t'))
def native_name(u):return json.dumps(u.get('proteinDescription',{}),ensure_ascii=False)
def cstr(v):return gemmi.cif.as_string(v)
def table(b,tags):return [[cstr(x) for x in r] for r in b.find(tags)]

def cached(pid,kind='mmcif'):
    name=pid+('.cif.gz' if kind=='mmcif' else '.xml.gz')
    roots=[CROSS/'external/cache',ROOT/'nccal_evidence_extension/cohort/public_sources',ARCH/'cache']
    return next((p/kind/name for p in roots if (p/kind/name).exists()),None)

ORIGINAL_FETCH=BASE.fetch
def fetch(url,path):
    path=Path(path)
    if path.exists():return path
    kind='sifts' if '.xml.gz' in url else 'mmcif'
    src=cached(path.name.split('.')[0].lower(),kind)
    if src:
        path.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,path)
        write(str(path)+'.source.json',{'local_source':str(src),'sha256':sha(path),'copied_utc':utc()})
        return path
    return ORIGINAL_FETCH(url,path)
BASE.fetch=fetch

def block_for(pid):
    p=fetch('https://files.rcsb.org/download/'+pid.upper()+'.cif.gz',HERE/f'cache/mmcif/{pid}.cif.gz')
    return gemmi.cif.read_string(gzip.decompress(p.read_bytes()).decode()).sole_block()

def entities(b):
    names={e:n for e,n in table(b,['_entity.id','_entity.pdbx_description'])}
    mut={e:m for e,m in table(b,['_entity.id','_entity.pdbx_mutation'])}
    org=defaultdict(set)
    for tags in [['_entity_src_gen.entity_id','_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id'],['_entity_src_nat.entity_id','_entity_src_nat.pdbx_ncbi_taxonomy_id']]:
        for e,t in table(b,tags):
            if t.isdigit():org[e].add(int(t))
    out=[]
    for e,s,ch in table(b,['_entity_poly.entity_id','_entity_poly.pdbx_seq_one_letter_code_can','_entity_poly.pdbx_strand_id']):
        out.append({'entity':e,'sequence':''.join(s.split()),'chains':ch.split(','),'source_taxa':sorted(org[e]),'name':names.get(e,''),'mutation':mut.get(e,'')})
    return out

def valid_entity(e,seq):
    if not e['source_taxa'] or not set(e['source_taxa'])<=ALLOWED:return 'source_taxonomy_not_allowed'
    if DANGER.search(e['name']):return 'excluded_name'
    if e['mutation'].strip().lower() not in ('','?','.','no','none','n/a'):return 'annotated_mutation'
    if not e['sequence'] or not set(e['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY'):return 'nonstandard_deposited_sequence'
    if e['sequence'] not in seq:return 'deposited_sequence_not_exact_native_fragment'
    return None

def prepare():
    seqs={s['accession']:s for s in jl(CROSS/'external/sequences.jsonl')}
    frame=[];screen=[]
    for pair in rt(CROSS/'cohorts/E_exposure.tsv'):
        acc=pair['accession'];s=seqs[acc];u=read(CROSS/f'external/sources/uniprot/{acc}.json')
        z=dict(pair);z.update(sequence=s['sequence'],original_pdb=s['pdb_id'],original_chain=s['chain'],taxid=u['organism']['taxonId'],protein_name=native_name(u))
        why=None
        if z['taxid'] not in ALLOWED:why='taxonomy_outside_scope'
        elif DANGER.search(z['protein_name']):why='excluded_name'
        elif u['sequence']['value']!=s['sequence']:why='native_sequence_drift'
        if why:z['status']=why;screen.append(z);continue
        try:
            b=block_for(s['pdb_id']);es=[e for e in entities(b) if s['chain'] in e['chains']]
            if len(es)!=1:raise ValueError('original_chain_entity_ambiguous')
            z['original_entity']=es[0];why=valid_entity(es[0],s['sequence'])
            if why:z['status']='original_'+why;screen.append(z);continue
        except Exception as ex:z['status']='original_metadata_error';z['error']=str(ex);screen.append(z);continue
        candidates=[]
        for x in u.get('uniProtKBCrossReferences',[]):
            if x['database']!='PDB' or x['id'].lower()==s['pdb_id']:continue
            p={r['key']:r['value'] for r in x.get('properties',[])}
            if p.get('Method')!='X-ray':continue
            m=re.match(r'([0-9.]+)',p.get('Resolution',''))
            if not m or float(m[1])>3:continue
            pid=x['id'].lower();candidates.append({'pdb_id':pid,'resolution':float(m[1]),'method':'X-RAY DIFFRACTION','cached':cached(pid) is not None})
        z['candidates']=sorted(candidates,key=lambda c:(not c['cached'],c['resolution'],c['pdb_id']))[:12]
        z['status']='framed';frame.append(z);screen.append({k:v for k,v in z.items() if k not in ('sequence','candidates')})
    savejl(HERE/'frame.jsonl',frame);savejl(HERE/'screen.jsonl',screen)
    inputs=[CROSS/'cohorts/E_exposure.tsv',CROSS/'external/sequences.jsonl',CROSS/'predictions/source_814.tsv',CROSS/'fits/source.npz',CROSS/'DEVELOPMENT_LOCK.json',ARCH/'build_structure_extension.py']
    obj={'frozen_utc':utc(),'protocol_sha256':sha(HERE/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'frame_sha256':sha(HERE/'frame.jsonl'),'screen_sha256':sha(HERE/'screen.jsonl'),'inputs':{str(p):sha(p) for p in inputs},'input_groups':len(screen),'screen':dict(Counter(r['status'] for r in screen)),'candidate_entry_attempt_cap_per_group':12,'new_rsa_read':False,'model_loss_summaries_computed':False}
    write(HERE/'freeze.json',obj);print(json.dumps(obj,indent=2),flush=True)

def atom_map(b,pid,acc,chain):
    st=gemmi.make_structure_from_block(b);mp=BASE.sifts_map(pid,acc).get(chain,{})
    ret={}
    for res in st[0][chain]:
        rid=str(res.seqid.num)+(res.seqid.icode.strip() or '')
        m=mp.get((chain,rid))
        if m:ret[m[0]]={'author_residue':rid,'aa':BASE.AA3.get(res.name),'atoms':sorted(a.name for a in res if a.element.name not in ('H','D'))}
    return ret

def process(z):
    dest=HERE/f"groups/{z['accession']}.json"
    if dest.exists():return read(dest)
    original=read(CROSS/f"external/groups/{z['group']}.json")
    attempts=[];chosen=None
    for c in z['candidates']:
        try:
            b=block_for(c['pdb_id']);st=gemmi.make_structure_from_block(b)
            methods=[cstr(x) for x in b.find_values('_exptl.method')]
            if 'X-RAY DIFFRACTION' not in methods or not 0<st.resolution<=3:attempts.append(dict(c,status='method_or_resolution'));continue
            ee=[e for e in entities(b) if e['sequence']==z['original_entity']['sequence'] and valid_entity(e,z['sequence']) is None]
            if not ee:attempts.append(dict(c,status='no_identical_native_deposited_entity'));continue
            # Assembly chain selection duplicates original implementation, before labels.
            if not st.assemblies:attempts.append(dict(c,status='no_biological_assembly'));continue
            asm=next((a for a in st.assemblies if a.name=='1'),sorted(st.assemblies,key=lambda a:a.name)[0])
            model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
            mp=BASE.sifts_map(c['pdb_id'],z['accession']);choices=[]
            for ci,chain in enumerate(model):
                n=0
                for res in chain:
                    if res.entity_type!=gemmi.EntityType.Polymer:continue
                    rid=str(res.seqid.num)+(res.seqid.icode.strip() or '');m=mp.get(chain.name,{}).get((chain.name,rid));aa=BASE.AA3.get(res.name)
                    if m and aa and 1<=m[0]<=len(z['sequence']) and z['sequence'][m[0]-1]==m[1]==aa and res.name==m[2]:n+=1
                if n:choices.append((-n,chain.name,ci))
            if not choices:attempts.append(dict(c,status='no_exact_mapped_chain'));continue
            n,chain,ci=min(choices)
            if -n<30 or not any(chain in e['chains'] for e in ee):attempts.append(dict(c,status='selected_chain_not_identical_native_entity'));continue
            # Entry identity, sequence, first-model/assembly and mapping are fixed now.
            chosen=dict(c,chain=chain,assembly_chain=ci,entity=next(e for e in ee if chain in e['chains']),mapped=-n)
            attempts.append(dict(c,status='selected_before_RSA'));break
        except Exception as ex:
            attempts.append(dict(c,status='retrieval_or_metadata_error',error=str(ex)));break
    result={k:z[k] for k in ('group','accession','original_pdb','original_chain','early','late','target','taxid')}
    result.update(attempts=attempts,original_entity=z['original_entity'],status='no_eligible_second_entry')
    if chosen:
        result['chosen']=chosen
        rec={k:z[k] for k in ('group','accession','sequence')};rec.update(split='REPEAT',candidates=[chosen])
        second=BASE.process_accession(rec)
        result['second']=second;result['status']=second['status']
        if second['status']=='features_computed':
            if (second['chain'],second['assembly_chain'])!=(chosen['chain'],chosen['assembly_chain']):raise ValueError('pre-RSA chain selection changed')
            oldmap=atom_map(block_for(z['original_pdb']),z['original_pdb'],z['accession'],z['original_chain'])
            newmap=atom_map(block_for(chosen['pdb_id']),chosen['pdb_id'],z['accession'],chosen['chain'])
            orig={r['pos']:r for r in original['residues']};new={r['pos']:r for r in second['residues']}
            result['fixed_sites']=[]
            for p in map(int,(z['early'],z['late'])):
                a,b=oldmap.get(p),newmap.get(p)
                result['fixed_sites'].append({'pos':p,'original':orig.get(p),'second':new.get(p),'original_atom_map':a,'second_atom_map':b,'atoms_match':bool(a and b and a['aa']==b['aa'] and a['atoms']==b['atoms'])})
    write(dest,result)
    print(z['accession'],result['status'],chosen['pdb_id'] if chosen else '',flush=True)
    return result

def run():
    f=read(HERE/'freeze.json')
    for p,k in [('PROTOCOL.md','protocol_sha256'),('frame.jsonl','frame_sha256')]:assert sha(HERE/p)==f[k]
    assert sha(Path(__file__))==f['script_sha256']
    for p,h in f['inputs'].items():assert sha(p)==h,p
    rows=jl(HERE/'frame.jsonl')
    with ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(process,rows))
    write(HERE/'execution_receipt.json',{'completed_utc':utc(),'groups':len(out),'statuses':dict(Counter(r['status'] for r in out)),'freeze_sha256':sha(HERE/'freeze.json')})

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
