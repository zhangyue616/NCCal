"""One concentrated geometry/interface acceptance; never reads model results."""
from pathlib import Path
import sys,json,csv,hashlib,gzip,xml.etree.ElementTree as ET
from collections import defaultdict,Counter
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parents[1]/'structure'
sys.path.insert(0,str(ARCHIVE/'runtime'));sys.dont_write_bytecode=True
import gemmi,numpy as np
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def tab(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seqs=jl(HERE/'sequences.jsonl');sites=tab(HERE/'sites.tsv');coverage=tab(HERE/'group_coverage.tsv')
frame=jl(HERE/'frozen_groups.jsonl');refs=jl(HERE/'exclusion_coverage.jsonl')
byacc={r['accession']:r for r in seqs};frameids={r['group'] for r in frame}
assert len(seqs)==len(byacc)==len({r['sequence'] for r in seqs})
assert len(coverage)==len(frame)==256 and {r['group'] for r in coverage}==frameids
assert not set(byacc)&{r['accession'] for r in refs}
assert not {r['sequence'] for r in seqs}&{r['sequence'] for r in refs}
blocked={c for r in refs for c in r['source_cluster_ids']}
assert not blocked&{r['cluster_id'] for r in seqs}
assert all(r['source_cluster_ids'] for r in refs)
keys=set();bys=defaultdict(list)
for r in sites:
    s=byacc[r['accession']];pos=int(r['pos']);total=int(r['total']);remote=int(r['remote'])
    key=(r['accession'],pos);assert key not in keys;keys.add(key)
    assert r['group']==s['group'] and s['sequence'][pos-1]==r['aa']
    assert r['chain']==s['chain'] and r['pdb_id']==s['pdb_id']
    assert r['main_clean']=='True' and r['interface']=='False' and r['secondary'] in ('H','E','C')
    assert 0<=remote<=total and float(r['rsa'])>=0 and np.isfinite(float(r['rsa']))
    bys[r['group']].append(r)
potential=[]
for s in seqs:
    rr=bys[s['group']];strata=defaultdict(list)
    for r in rr:strata[(r['aa'],r['secondary'],int(np.floor(float(r['rsa'])*10)),int(r['total']))].append(r)
    n=0
    for block in strata.values():
        positive=[r for r in block if int(r['remote'])>0];zero=[r for r in block if int(r['remote'])==0]
        n+=sum(abs(int(a['pos'])-int(b['pos']))>30 for a in positive for b in zero)
    buried=[r for r in rr if float(r['rsa'])<=.10];exposed=[r for r in rr if float(r['rsa'])>=.40]
    control=sum(a['aa']==b['aa'] and a['secondary']==b['secondary'] and abs(int(a['pos'])-int(b['pos']))>30 for a in buried for b in exposed)
    potential.append({'group':s['group'],'accession':s['accession'],'main_clean_sites':len(rr),'primary_condition_compatible_pairs':n,'exposure_control_compatible_pairs':control})
aa3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
sample=[seqs[i] for i in sorted({0,len(seqs)//2,len(seqs)-1})]
sample_checks=[]
for s in sample:
    pid=s['pdb_id'];acc=s['accession'];root=ET.fromstring(gzip.decompress((HERE/f'cache/sifts/{pid}.xml.gz').read_bytes()))
    mapping=defaultdict(list)
    for res in root.iter():
        if res.tag.rsplit('}',1)[-1]!='residue':continue
        rr=[x.attrib for x in res if x.tag.rsplit('}',1)[-1]=='crossRefDb']
        u=[x for x in rr if x.get('dbSource')=='UniProt'];p=[x for x in rr if x.get('dbSource')=='PDB']
        if len(u)!=1 or len(p)!=1 or u[0].get('dbAccessionId')!=acc:continue
        if p[0].get('dbChainId')!=s['chain']:continue
        try:position=int(u[0]['dbResNum'])
        except ValueError:continue
        mapping[p[0]['dbResNum']].append((position,u[0].get('dbResName'),p[0].get('dbResName')))
    st=gemmi.make_structure_from_block(gemmi.cif.read_string(gzip.decompress((HERE/f'cache/mmcif/{pid}.cif.gz').read_bytes()).decode()).sole_block())
    assert 0<st.resolution<=2.0
    asm=next(a for a in st.assemblies if a.name==s['assembly_id']);model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
    chain=model[s['assembly_chain']];assert chain.name==s['chain']
    raw=[]
    for res in chain:
        if res.entity_type!=gemmi.EntityType.Polymer:continue
        rid=str(res.seqid.num)+(res.seqid.icode.strip() or '');m=mapping.get(rid,[]);aa=aa3.get(res.name)
        if len(m)!=1 or aa is None:continue
        pos,ma,mn=m[0]
        if not (1<=pos<=len(s['sequence']) and s['sequence'][pos-1]==aa==ma and mn==res.name):continue
        atoms={a.name:np.array([a.pos.x,a.pos.y,a.pos.z]) for a in res if a.element.name not in ('H','D')};name='CA' if aa=='G' else 'CB'
        if name in atoms:raw.append((pos,atoms[name]))
    freq=Counter(p for p,_ in raw);points={p:x for p,x in raw if freq[p]==1}
    positions=np.array(list(points));xyz=np.array(list(points.values()));checked=0
    for f in bys[s['group']]:
        p=int(f['pos']);within=np.sum((xyz-points[p])**2,axis=1)<=64.0;gap=np.abs(positions-p)
        assert int(np.sum(within&(gap>=3)))==int(f['total'])
        assert int(np.sum(within&(gap>=24)))==int(f['remote']);checked+=1
    sample_checks.append({'accession':acc,'group':s['group'],'pdb_id':pid,'sites_independently_recounted':checked,'all_total_remote_match':True})
receipt={'status':'PASS','coverage_groups':len(coverage),'sequences':len(seqs),'sites':len(sites),'old_reserved_exclusion_records':len(refs),'old_reserved_missing_source_clusters':0,'sequence_and_site_identity_unique':True,'no_old_or_reserved_accession_exact_sequence_or_source_PDB_cluster_overlap':True,'all_sites_main_clean_noninterface':True,'all_contact_counts_nonnegative_and_remote_le_total':True,'primary_pair_coverage_groups':sum(r['primary_condition_compatible_pairs']>0 for r in potential),'primary_condition_compatible_pairs':sum(r['primary_condition_compatible_pairs'] for r in potential),'exposure_control_coverage_groups':sum(r['exposure_control_compatible_pairs']>0 for r in potential),'exposure_control_compatible_pairs':sum(r['exposure_control_compatible_pairs'] for r in potential),'sampled_independent_coordinate_checks':sample_checks,'model_outcomes_read':False,'archived_builder_sha256':digest(ARCHIVE/'build_structure_extension.py'),'archived_rule_lock_sha256':digest(ARCHIVE/'STRUCTURE_RULE_LOCK.md'),'files':{p.name:digest(p) for p in [HERE/'sequences.jsonl',HERE/'sites.tsv',HERE/'group_coverage.tsv',HERE/'PROTOCOL.md',HERE/'TECHNICAL_AMENDMENT.md']}}
(HERE/'PAIRING_COVERAGE_ONLY.json').write_text(json.dumps(potential,indent=2)+'\n',encoding='utf-8')
(HERE/'ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
