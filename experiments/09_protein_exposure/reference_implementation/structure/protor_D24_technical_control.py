"""Prespecified D-only 24-group radius convention comparison, separate from main."""
import os,sys,json,csv,gzip,hashlib,argparse,collections,ctypes
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;DEST=OUT/'protor_D24_control';DEST.mkdir(exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(OUT/'runtime'));sys.dont_write_bytecode=True
import gemmi,numpy as np,biotite.structure as bs
from scipy.stats import spearmanr
SALT='nccal-protor-technical-D24-v1'
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exposure(x):return 'B' if x<=.10 else 'E' if x>=.40 else 'I'
def rho(a,b):
    if len(a)<3 or len(set(a))<2 or len(set(b))<2:return None
    x=float(spearmanr(a,b).statistic);return x if np.isfinite(x) else None
def prepare():
    eligible=[]
    for p in sorted((OUT/'groups/D').glob('*.json')):
        r=json.loads(p.read_text())
        if any(f['main_clean'] for f in r.get('residues',[])):
            eligible.append({'group':r['group'],'accession':r['accession'],'pdb_id':r['pdb_id'],'assembly_id':r['assembly_id'],'chain':r['chain'],'assembly_chain':r['assembly_chain'],'hash_rank_key':hashlib.sha256(f"{SALT}|{r['group']}".encode()).hexdigest(),'strict_main_site_count':sum(f['main_clean'] for f in r['residues'])})
    chosen=sorted(eligible,key=lambda r:(r['hash_rank_key'],r['group']))[:24]
    assert len(eligible)==325 and len(chosen)==24
    selection={'locked_utc':utc(),'scope':'D only; no T or model outcomes','eligible_groups':len(eligible),'selected_groups':chosen,'salt':SALT,'selection_rule':'all frozen D groups with >=1 strict main_clean site; ascending SHA256(salt|group), first 24; exact tie lexical group','main_endpoint_unchanged':True,'frozen_D_unchanged':True}
    p=DEST/'SELECTION_LOCK.json'
    if p.exists():assert json.loads(p.read_text())['selected_groups']==chosen
    else:write(p,selection)
    rule=DEST/'RULE_LOCK.md'
    if not rule.exists():rule.write_text('# D-only ProtOr radius convention control\n\nLocked before any ProtOr results. Eligible population: the 325 frozen D groups with at least one strict main_clean site. Sort ascending SHA256("nccal-protor-technical-D24-v1|" + group), take first 24; lexical group breaks exact ties. SELECTION_LOCK.json stores the complete chosen list. No T data, model predictions, axis selection, endpoint changes or pair reselection.\n\nReuse the same experimental structure, assembly, chain copy, protein-heavy-atom coordinates and all alternate-conformer occluders as the D freeze. Calculate exactly the same target atoms, 1.4 A probe and 1000 Fibonacci points; only replace Single element radii with Biotite 1.6.0 standard ProtOr implicit-hydrogen radii. Normalize with the same Tien maxASA. No explicit hydrogen building. Unrecognized-atom ProtOr fallbacks, if any, are counted and reported.\n\nExport all frozen quality sites with both values and fixed main_clean flags. Separately summarize all valid quality sites and the existing strict main_clean subset: pooled Spearman correlation; within-group Spearman where at least three sites and nonconstant values; three-class exact agreement B<=0.10, E>=0.40, intermediate I otherwise; full 3x3 transition counts; retention of the old extreme (B/E) assignments. Do not apply a new radius-based interface filter or select new pairs. Technical control, no hypothesis-testing claims.\n\nOne low-priority CPU process, no GPU, no network, no effect on the two-worker T queue. Primary frozen outputs are read-only.\n',encoding='utf-8')
    return chosen
def process(c):
    from biotite.structure.info import vdw_radius_protor
    r=json.loads((OUT/f"groups/D/{c['group']}.json").read_text());ff=r['residues'];pid=r['pdb_id']
    st=gemmi.make_structure_from_block(gemmi.cif.read_string(gzip.decompress((OUT/f'cache/mmcif/{pid}.cif.gz').read_bytes()).decode()).sole_block())
    asm=next(a for a in st.assemblies if a.name==r['assembly_id']);model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
    atoms=[];keys=[]
    for ci,chain in enumerate(model):
        for ri,res in enumerate(chain):
            if res.entity_type!=gemmi.EntityType.Polymer or not gemmi.find_tabulated_residue(res.name).is_amino_acid():continue
            rid=str(res.seqid.num)+(res.seqid.icode.strip() or '')
            for a in res:
                if a.element.name not in ('H','D'):atoms.append((ci,ri,res.name,a));keys.append((ci,rid))
    arr=bs.AtomArray(len(atoms));arr.coord=np.array([[a.pos.x,a.pos.y,a.pos.z] for ci,ri,rn,a in atoms],dtype=np.float32);arr.element=np.array([a.element.name for ci,ri,rn,a in atoms]);arr.atom_name=np.array([a.name for ci,ri,rn,a in atoms]);arr.res_name=np.array([rn for ci,ri,rn,a in atoms]);arr.res_id=np.array([ri+1 for ci,ri,rn,a in atoms]);arr.chain_id=np.array([str(ci) for ci,ri,rn,a in atoms])
    mask=np.array([ci==r['assembly_chain'] for ci,ri,rn,a in atoms]);fallback=collections.Counter()
    for ci,ri,rn,a in atoms:
        if vdw_radius_protor(rn,a.name) is None:fallback[f'{rn}:{a.name}']+=1
    pro=bs.sasa(arr,atom_filter=mask,probe_radius=1.4,point_number=1000,point_distr='Fibonacci',vdw_radii='ProtOr',ignore_ions=True)
    bykey=collections.defaultdict(list)
    for i,k in enumerate(keys):bykey[k].append(i)
    maxasa=dict(zip('ARNDCQEGHILKMFPSTWYV',[129,274,195,193,167,225,223,104,224,197,201,236,224,240,159,155,172,285,263,174]))
    result=[]
    for f in ff:
        ids=bykey[(r['assembly_chain'],f['author_residue'])];v=float(np.sum(pro[ids]))/maxasa[f['aa']]
        if not np.isfinite(v):continue
        result.append({'group':r['group'],'split':'D','accession':r['accession'],'pos':f['pos'],'aa':f['aa'],'secondary':f['secondary'],'main_clean_frozen':f['main_clean'],'rsa_heavy_single':f['rsa'],'rsa_protor':v,'class_heavy_single':exposure(f['rsa']),'class_protor':exposure(v),'pdb_id':pid,'assembly_id':r['assembly_id'],'chain':r['chain'],'assembly_chain':r['assembly_chain']})
    return result,{'group':r['group'],'requested_quality_sites':len(ff),'valid_quality_sites':len(result),'ProtOr_fallback_atom_counts':dict(fallback),'assembly_atoms':len(arr),'target_atoms':int(mask.sum())}
def summarize(rr):
    matrix={a:{b:0 for b in ['B','I','E']} for a in ['B','I','E']}
    for r in rr:matrix[r['class_heavy_single']][r['class_protor']]+=1
    n=len(rr);oldext=[r for r in rr if r['class_heavy_single'] in ['B','E']]
    gg=collections.defaultdict(list)
    for r in rr:gg[r['group']].append(r)
    within={g:rho([r['rsa_heavy_single'] for r in ss],[r['rsa_protor'] for r in ss]) for g,ss in gg.items()}
    validrho=[v for v in within.values() if v is not None]
    return {'valid_sites':n,'pooled_spearman_rho':rho([r['rsa_heavy_single'] for r in rr],[r['rsa_protor'] for r in rr]),'three_class_agreement':sum(r['class_heavy_single']==r['class_protor'] for r in rr)/n if n else None,'B_I_E_transition_counts':matrix,'old_extreme_sites':len(oldext),'old_extreme_assignment_retention':sum(r['class_heavy_single']==r['class_protor'] for r in oldext)/len(oldext) if oldext else None,'within_group_spearman':within,'within_group_median_rho':float(np.median(validrho)) if validrho else None}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();chosen=prepare()
    if args.prepare_only:print(json.dumps({'eligible_groups':325,'selected_groups':[r['group'] for r in chosen]}));sys.exit()
    if os.name=='nt':ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x40)
    results=[];audit=[]
    for c in chosen:
        rr,aa=process(c);results.extend(rr);audit.append(aa);print(c['group'],len(rr),flush=True)
    with (DEST/'site_comparison.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter='\t');w.writeheader();w.writerows(results)
    summary={'completed_utc':utc(),'selected_groups':24,'all_quality':summarize(results),'frozen_strict_main':summarize([r for r in results if r['main_clean_frozen']]),'group_audit':audit,'same_occluders_and_targets':True,'only_radii_replaced':True,'D_frozen_modified':False,'main_endpoint_changed':False,'T_read':False,'model_outcomes_read':False}
    write(DEST/'SUMMARY.json',summary);print(json.dumps(summary))
