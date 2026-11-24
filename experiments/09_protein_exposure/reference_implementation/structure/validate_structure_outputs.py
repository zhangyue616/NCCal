"""Consolidated structural acceptance; no model outcomes and no recalculation of SASA."""
import sys,json,gzip,collections,argparse
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'runtime'));sys.dont_write_bytecode=True
import gemmi
cohort=[json.loads(s) for s in (OUT/'cohort_index.jsonl').read_text().splitlines()]
seq={r['accession']:r['sequence'] for r in cohort}
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['D','T','both'],default='both');args=parser.parse_args()
checks=collections.Counter();failures=[];rescan=[];nonpolymer=[];errors=[]
def need(ok,why,group):
    checks[why]+=1
    if not ok:failures.append({'group':group,'check':why})
for p in sorted((OUT/'groups').glob('*/*.json')):
    r=json.loads(p.read_text());g=r['group'];ff=r.get('residues',[])
    if args.split not in ['both',r['split']]:continue
    if r['status']=='features_computed':
        need(0<r['coordinate_resolution']<=3.,'coordinate_resolution',g)
        need(len({f['pos'] for f in ff})==len(ff),'unique_position',g)
        lookup={f['pos']:f for f in ff}
        for f in ff:
            need(seq[f['accession']][f['pos']-1]==f['aa'],'canonical_sequence_identity',g)
            need(f['rsa']>=0 and f['rsa_isolated']+1e-6>=f['rsa'],'assembly_SASA_monotonicity',g)
            need(0<=f['remote_contacts']<=f['remote_eligible_partners'],'remote_count_bounds',g)
            need(f['secondary'] in ['H','E','C','U'],'PSEA_category',g)
            if f['main_clean']:
                need(f['local15_exact_observed'] and f['local2_complete_quality'] and f['local15_CA_continuous'] and not f['interface'],'strict_main_flags',g)
            if f['center2_clean_sensitivity']:
                need(f['local2_complete_quality'] and not f['interface'],'center2_sensitivity_flags',g)
        for pair_name,flag in [('pair','main_clean'),('pair_center2_sensitivity','center2_clean_sensitivity')]:
            pp=r.get(pair_name)
            if pp:
                a,b=lookup[pp['pos1']],lookup[pp['pos2']]
                need(a[flag] and b[flag] and a['aa']==b['aa'] and a['secondary']==b['secondary'] and a['rsa']<=.10 and b['rsa']>=.40 and abs(a['pos']-b['pos'])>30,'valid_'+pair_name,g)
                need(all(a[k]==b[k]==pp[k] for k in ['group','split','accession','pdb_id','assembly_id','chain','assembly_chain']),'same_chain_'+pair_name,g)
        # Only inspect cached coordinate classification; do not rerun SASA.
        pid=r['pdb_id'];cp=OUT/f'cache/mmcif/{pid}.cif.gz'
        st=gemmi.make_structure_from_block(gemmi.cif.read_string(gzip.decompress(cp.read_bytes()).decode()).sole_block())
        asm=next(a for a in st.assemblies if a.name==r['assembly_id'])
        model=gemmi.make_assembly(asm,st[0],gemmi.HowToNameCopiedChain.Dup)
        aa_nonpoly=[(c.name,str(rr.seqid),rr.name) for c in model for rr in c if gemmi.find_tabulated_residue(rr.name).is_amino_acid() and rr.entity_type!=gemmi.EntityType.Polymer]
        expected_atoms=sum(a.element.name not in ('H','D') for c in model for rr in c if rr.entity_type==gemmi.EntityType.Polymer and gemmi.find_tabulated_residue(rr.name).is_amino_acid() for a in rr)
        need(r['assembly_protein_atoms']==expected_atoms,'protein_polymer_atom_count',g)
        if aa_nonpoly and r['assembly_protein_atoms']!=expected_atoms:nonpolymer.append({'group':g,'pdb_id':pid,'residues':aa_nonpoly});rescan.append(str(p))
    for aa in r.get('accession_attempts',[]):
        for at in aa['attempts']:
            if at['status']=='error':errors.append({'group':g,'accession':aa['accession'],'pdb_id':at['pdb_id'],'error':at['error']})
report={'scope_split':args.split,'checks':dict(checks),'failures':failures,'nonpolymer_aminoacid_affected_groups':nonpolymer,'implementation_error_attempts':errors,'status':'PASS' if not failures and not nonpolymer and not errors else 'REPAIR_OR_EXPLICIT_LIMIT_REQUIRED','SASA_not_recomputed':True,'source_checks':'cached experimental coordinates, group features, canonical sequence identities, source resolution, strict/sensitivity pair invariants','limitations':['No residue-level electron-density validation was requested or performed.','P-SEA categories are not DSSP.','RSA uses element radii and Tien theoretical normalization, not DSSP numerical identity.','The candidate universe is only the old evidence PDB indices, not all PDB links for each UniProt accession.']}
name='STRUCTURE_ACCEPTANCE.json' if args.split=='both' else f'{args.split}_FINAL_ACCEPTANCE.json'
(OUT/name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':dict(checks),'failures':len(failures),'nonpolymer_affected_groups':len(nonpolymer),'error_attempts':len(errors),'errors':errors[:5]}))
