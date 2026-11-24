"""Descriptive alternate-conformer influence flags; never edits frozen features."""
import sys,json,csv,gzip,argparse
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'runtime'));sys.dont_write_bytecode=True
import gemmi,numpy as np
from scipy.spatial import cKDTree
from biotite.structure.info import vdw_radius_single
a=argparse.ArgumentParser();a.add_argument('--split',choices=['D','T'],required=True);args=a.parse_args()
out=[];groups=0;with_alt=0;strict_near=0;strict_n=0;pair_near=0;pair_n=0
for gp in sorted((OUT/f'groups/{args.split}').glob('*.json')):
    r=json.loads(gp.read_text());ff=r.get('residues',[])
    if not ff:continue
    groups+=1;pid=r['pdb_id']
    block=gemmi.cif.read_string(gzip.decompress((OUT/f'cache/mmcif/{pid}.cif.gz').read_bytes()).decode()).sole_block()
    st=gemmi.make_structure_from_block(block);asm=next(x for x in st.assemblies if x.name==r['assembly_id'])
    model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
    alt=[];ar=[]
    for chain in model:
        for res in chain:
            if res.entity_type!=gemmi.EntityType.Polymer or not gemmi.find_tabulated_residue(res.name).is_amino_acid():continue
            for atom in res:
                if atom.element.name not in ('H','D') and atom.altloc!='\x00':
                    alt.append([atom.pos.x,atom.pos.y,atom.pos.z]);ar.append(vdw_radius_single(atom.element.name.upper()))
    if alt:
        with_alt+=1;alt=np.array(alt);ar=np.array(ar);tree=cKDTree(alt)
    chain=model[r['assembly_chain']]
    residues={str(res.seqid.num)+(res.seqid.icode.strip() or ''):res for res in chain}
    flags={}
    for f in ff:
        nearby=set()
        if len(alt):
            for atom in residues[f['author_residue']]:
                if atom.element.name in ('H','D'):continue
                xyz=np.array([atom.pos.x,atom.pos.y,atom.pos.z]);rad=vdw_radius_single(atom.element.name.upper())
                candidates=tree.query_ball_point(xyz,rad+float(ar.max())+2.8)
                for i in candidates:
                    if np.linalg.norm(xyz-alt[i])<rad+ar[i]+2.8:nearby.add(i)
        flag=bool(nearby);flags[f['pos']]=flag
        strict_n+=f['main_clean'];strict_near+=f['main_clean'] and flag
        out.append({'group':r['group'],'split':r['split'],'accession':r['accession'],'pos':f['pos'],'pdb_id':pid,'assembly_id':r['assembly_id'],'chain':r['chain'],'assembly_chain':r['assembly_chain'],'assembly_altloc_heavy_atoms':len(alt),'possible_altloc_sasa_influence':flag,'near_altloc_atom_count':len(nearby),'main_clean_original':f['main_clean']})
    if r.get('pair'):
        pair_n+=1;pair_near+=flags[r['pair']['pos1']] or flags[r['pair']['pos2']]
p=OUT/f'{args.split}_altloc_flags.tsv'
with p.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]),delimiter='\t');w.writeheader();w.writerows(out)
summary={'created_utc':datetime.now(timezone.utc).isoformat(),'scope_split':args.split,'source_groups':groups,'groups_with_any_altloc_protein_heavy_atoms':with_alt,'quality_sites':len(out),'quality_sites_with_possible_influence':sum(x['possible_altloc_sasa_influence'] for x in out),'strict_main_sites':strict_n,'strict_main_sites_with_possible_influence':strict_near,'strict_main_pairs':pair_n,'strict_main_pairs_with_either_site_possible_influence':pair_near,'definition':'Conservative possible influence: any residue heavy-atom SASA sphere overlaps an alternate-conformer protein-polymer heavy-atom exclusion sphere; distance < vdW_i+vdW_j+2*1.4 A, same element radii as SASA. This marks possible rather than measured numerical change.','no_SASA_recomputation':True,'frozen_outputs_modified':False,'main_endpoint_modified':False,'file':p.name}
(OUT/f'{args.split}_altloc_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
