"""Frozen molecular qualification; no acquisition or model loading."""
from rdkit import Chem
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
CHIRAL_OK = {Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW}
BOND_OK = {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
           Chem.rdchem.BondType.AROMATIC}
DIR_OK = {Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT}
PEPTIDE_SMARTS = "[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)"
NUCLEOSIDE_SMARTS = "[n][CX4;r5]1[OX2;r5][CX4;r5][CX4;r5][CX4;r5]1"
PEPTIDE, NUCLEOSIDE = Chem.MolFromSmarts(PEPTIDE_SMARTS), Chem.MolFromSmarts(NUCLEOSIDE_SMARTS)

def standardize(smi):
    """Exact received v1 standardization; return canonical SMILES or reason."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "parse_fail"
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    best, best_n = None, -1
    for f in frags:
        n = f.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = f, n
    m = Chem.RemoveHs(best)
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in m.GetAtoms()):
        return None, "element"
    if not 5 <= m.GetNumHeavyAtoms() <= 50 or any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        return None, "heavy_atoms"
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in m.GetAtoms()):
        return None, "radical_or_isotope"
    if any(a.GetChiralTag() not in CHIRAL_OK for a in m.GetAtoms()):
        return None, "chiral_tag"
    if any(b.GetBondType() not in BOND_OK or b.GetBondDir() not in DIR_OK for b in m.GetBonds()):
        return None, "bond_type_or_dir"
    if m.HasSubstructMatch(PEPTIDE) or m.HasSubstructMatch(NUCLEOSIDE):
        return None, "bio_structure"
    can = Chem.MolToSmiles(m)
    m2 = Chem.MolFromSmiles(can)
    if m2 is None or m2.GetNumAtoms() != m.GetNumAtoms():
        return None, "roundtrip"
    if any(a.GetChiralTag() not in CHIRAL_OK for a in m2.GetAtoms()) or \
            any(b.GetBondType() not in BOND_OK or b.GetBondDir() not in DIR_OK for b in m2.GetBonds()):
        return None, "roundtrip"
    return can, None


def parsed(text):
    settings=Chem.SmilesParserParams();settings.removeHs=False
    molecule=Chem.MolFromSmiles(text,settings)
    if molecule is None:raise ValueError('parse')
    return molecule
