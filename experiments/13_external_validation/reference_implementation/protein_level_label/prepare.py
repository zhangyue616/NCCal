"""Prepare matched experimental-structure and sequence descriptors without opening a model."""
from pathlib import Path
import gzip, json, csv, re, hashlib, sys, importlib.util
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'cross_architecture/external'
OLD=ROOT/'structure'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD/'runtime'))
sys.path.insert(0,str(HERE/'runtime'))
import biotite.structure as bs
import gemmi
from Bio.SeqUtils.ProtParam import ProteinAnalysis
spec=importlib.util.spec_from_file_location('archived_sifts',OLD/'build_structure_extension.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.OUT=DATA

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def whole_sse(r):
    seq,chain_name,pdb,acc=r['sequence'],r['chain'],r['pdb_id'],r['accession']
    sifts=old.sifts_map(pdb,acc)[chain_name]
    block=gemmi.cif.read_string(gzip.decompress((DATA/f'cache/mmcif/{pdb}.cif.gz').read_bytes()).decode()).sole_block()
    st=gemmi.make_structure_from_block(block)
    asm=next(a for a in st.assemblies if a.name==r['assembly_id'])
    chain=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)[r['assembly_chain']]
    assert chain.name==chain_name
    # Compute the P-SEA assignment on native-aligned, sequentially contiguous runs;
    # do not bridge a missing residue or an interrupted CA trace.
    coords={}
    for res in chain:
        if res.entity_type!=gemmi.EntityType.Polymer:continue
        key=str(res.seqid.num)+res.seqid.icode.strip(); ref=sifts.get((chain.name,key))
        if ref is None:continue
        pos,aa,name=ref
        if not (1<=pos<=len(seq) and old.AA3.get(res.name)==aa==seq[pos-1] and name==res.name):continue
        cas=sorted((a for a in res if a.name=='CA' and a.occ>0), key=lambda a:(-a.occ,str(a.altloc)))
        if cas:coords[pos]=[cas[0].pos.x,cas[0].pos.y,cas[0].pos.z]
    runs=[];run=[]
    for pos in sorted(coords):
        if run and (pos!=run[-1]+1 or np.linalg.norm(np.subtract(coords[pos],coords[run[-1]]))>4.5):
            runs.append(run);run=[]
        run.append(pos)
    if run:runs.append(run)
    aa3={a:n for n,a in old.AA3.items()}; counts=Counter()
    for run in runs:
        arr=bs.AtomArray(len(run));arr.coord=np.array([coords[i] for i in run]);arr.atom_name[:]='CA';arr.res_id=np.array(run);arr.res_name=np.array([aa3[seq[i-1]] for i in run]);arr.chain_id[:]='A'
        if len(run)<5:continue
        counts.update(map(str,bs.annotate_sse(arr)))
    total=sum(counts[k] for k in 'abc')
    return len(coords),total,counts['b']

def main():
    frozen=read(HERE/'FREEZE.json')
    assert frozen['protocol']==hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest().upper()
    recs=[json.loads(line) for line in (DATA/'sequences.jsonl').read_text().splitlines() if line]
    tax={9606,10090,559292,4932,3702,4530,39947,7227,6239,7955,224308,1423,83333,511145,300852,224324,243274}
    out=[];ledger=[]
    for r in recs:
        seq=r['sequence'];u=read(DATA/f"sources/uniprot/{r['accession']}.json"); meta=u['organism']
        stamp=dict(group=r['group'],accession=r['accession'],pdb_id=r['pdb_id'],chain=r['chain'])
        tags=' '.join([str(u.get('proteinDescription','')),str(u.get('keywords',''))])
        if meta['taxonId'] not in tax or re.search(r'\b(toxin|neurotoxin|cytotoxin|venom|virulence|pathogen)\b',tags,re.I):
            ledger.append(dict(**stamp,reason='organism or annotation outside the scope'));continue
        if not set(seq)<=set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('nonstandard native sequence')
        matched,total,beta=whole_sse(r)
        if matched/len(seq)<.70 or total<50:
            ledger.append(dict(**stamp,reason='structure coverage',mapped=matched,assigned=total,length=len(seq)));continue
        analyzer=ProteinAnalysis(seq)
        rec=dict(**stamp,sequence=seq,length=len(seq),organism=meta['scientificName'],mapped=matched,assigned=total,beta_positions=beta,sheet_fraction=beta/total,propensity=analyzer.secondary_structure_fraction()[2],charge74=analyzer.charge_at_pH(7.4),charge70=analyzer.charge_at_pH(7.0),fold=int.from_bytes(hashlib.sha256(('NCCal_204_fold|'+r['group']).encode()).digest()[:8],'big')%5)
        out.append(rec);ledger.append(dict(**stamp,reason='eligible'))
    assert len(set(r['group'] for r in out))==len(out)
    out.sort(key=lambda x:x['group']);save(HERE/'cohort.json',out);save(HERE/'cohort_funnel.json',ledger)
    print('fixed',len(recs),'eligible',len(out),'excluded',Counter(x['reason'] for x in ledger),flush=True)
if __name__=='__main__':main()
