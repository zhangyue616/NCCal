"""Deterministic pre-activation edit-class and site selection."""
import hashlib

def choose_edit(cid,canonical_smiles):
    from rdkit import Chem
    settings=Chem.SmilesParserParams();settings.removeHs=False
    molecule=Chem.MolFromSmiles(canonical_smiles,settings)
    if molecule is None:raise ValueError('Invalid molecular structure.')
    hydroxyl=Chem.MolFromSmarts('[CX4;+0:1]-[OX2H1;+0:2]')
    ketone=Chem.MolFromSmarts('[#6:3]-[CX3H0;+0:1](=[OX1;+0:2])-[#6:4]')
    acid=Chem.MolFromSmarts('[CX3;+0:1](=[OX1;+0:2])-[OX2H1;+0:3]')
    def safe(indices):
        for i in indices:
            atom=molecule.GetAtomWithIdx(i)
            if atom.GetIsotope() or atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:return False
        for bond in molecule.GetBonds():
            if bond.GetBeginAtomIdx() in indices and bond.GetEndAtomIdx() in indices and bond.GetStereo()!=Chem.BondStereo.STEREONONE:return False
        return True
    oh=sorted(set((x[0],x[1]) for x in molecule.GetSubstructMatches(hydroxyl)))
    co=sorted(set((x[1],x[2]) for x in molecule.GetSubstructMatches(ketone)))
    ac=sorted(set(tuple(x) for x in molecule.GetSubstructMatches(acid)))
    target={i for site in oh+co+ac for i in site}
    choices={1:[x for x in oh if safe(set(x))],2:[x for x in oh if safe(set(x))],3:[x for x in co if safe(set(x))],4:[x for x in ac if safe(set(x))],5:[]}
    if target:
        distance=Chem.GetDistanceMatrix(molecule)
        choices[5]=[(a.GetIdx(),) for a in molecule.GetAtoms() if a.GetAtomicNum()==6 and a.GetIsAromatic() and a.GetTotalNumHs()==1 and min(distance[a.GetIdx(),i] for i in target)>=4 and safe({a.GetIdx()})]
    classes=[k for k,v in choices.items() if v]
    if not classes:return dict(cid=int(cid),edit_class=None,site=None,candidate_counts={str(k):len(v) for k,v in choices.items()})
    digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
    kind=min(classes,key=lambda k:digest(f'20261123/edit-class/{int(cid)}/{k}'))
    site=min(choices[kind],key=lambda x:digest(f'20261123/edit-site/{int(cid)}/'+','.join(map(str,x))))
    return dict(cid=int(cid),edit_class=kind,site=list(site),candidate_counts={str(k):len(v) for k,v in choices.items()})
