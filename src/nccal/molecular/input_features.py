"""Frozen finite input tables; no target SMARTS or model fitting is used as input.

The radius-one/j refinement distributes the original category's prior mass
over observed bins. Consequently a constant j reproduces the base table.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

ALPHA = 20.0
QUANTILES = (0.2, 0.4, 0.6, 0.8)


def key(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def atom_signature(atom):
    return (atom.GetAtomicNum(), atom.GetFormalCharge(), bool(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), str(atom.GetHybridization()))


def input_keys(smiles, anchor_token_ids):
    """Construct exactly one row per RDKit atom, in molecule/atom order."""
    if len(smiles) != len(anchor_token_ids):
        raise ValueError("Molecule and anchor lists differ")
    rows = []
    for text, tokens in zip(smiles, anchor_token_ids):
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() != len(tokens):
            raise ValueError("Exact graph and previously validated anchor count differ")
        for atom, token in zip(mol.GetAtoms(), tokens):
            r0 = (atom.GetAtomicNum(), atom.GetFormalCharge(), bool(atom.GetIsAromatic()))
            neighbors = sorted((str(bond.GetBondType()), atom_signature(bond.GetOtherAtom(atom)))
                               for bond in atom.GetBonds())
            rows.append({"token": str(int(token)), "r0": key(r0),
                         "r1": key((atom_signature(atom), neighbors))})
    return rows

