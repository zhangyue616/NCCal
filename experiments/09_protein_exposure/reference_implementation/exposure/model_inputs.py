"""Map the published cohort's eligible structural residues to a sequence design matrix."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from extract_axes import HERE, roster

STRUCTURE = HERE.parent / "structure"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = AMINO_ACIDS + "X^"
SLOTS = {symbol: index for index, symbol in enumerate(ALPHABET)}
SEED = 20261118

def read_rows(split, structural_dir=STRUCTURE):
    cohort = roster()
    rows = []
    unique = set()
    with (structural_dir / "residue_features.tsv").open(encoding="utf-8-sig", newline="") as file:
        for r in csv.DictReader(file, delimiter="\t"):
            if r["split"] != split or r["main_clean"] != "True" or r["interface"] != "False":
                continue
            acc, pos = r["accession"], int(r["pos"])
            stored_split, group, sequence = cohort[acc]
            if (acc,pos) in unique or stored_split != split or group != r["group"] or sequence[pos-1] != r["aa"]:
                raise ValueError(f"Repeated or misaligned residue {acc}:{pos} {r['group']}")
            unique.add((acc,pos))
            rows.append((acc,pos,group,r["aa"],r["secondary"],float(r["rsa"]), int(r["remote_contacts"])))
    rows.sort(key=lambda row: (row[2],row[0],row[1]))
    if not rows: raise ValueError("No eligible residues in split "+split)
    # The structural rule selects one representative protein per known group.
    mapping = {(g,acc) for acc,pos,g,aa,ss,rsa,rc in rows}
    if len(mapping) != len({row[2] for row in rows}):
        raise ValueError("Multiple representative proteins in one known group")
    return rows, cohort

def design_matrix(rows, cohort):
    width = 31 * len(ALPHABET)
    ix, values, starts = [], [], [0]
    for acc,pos,_,aa,_,_,_ in rows:
        seq = cohort[acc][2]
        for k in range(31):
            site = pos - 1 + k - 15
            symbol = seq[site] if 0 <= site < len(seq) else "^"
            ix.append(k*len(ALPHABET) + SLOTS.get(symbol, SLOTS["X"]))
            values.append(1.0)
        rel = (pos-1) / max(1,len(seq)-1)
        ix.extend(range(width,width+4))
        values.extend((rel,rel*rel,len(seq)/256,1.0))
        starts.append(len(ix))
    return sparse.csr_matrix((values,ix,starts),shape=(len(rows),width+4))

def activations(rows, model, split, axis=None, all_axes=False):
    dest = "all" if split=="D" else str(axis)
    folder = HERE/"activations"/model/(split+"-"+dest)
    width = {"8m":1280,"35m":1920}[model]
    out = np.empty((len(rows),width) if all_axes else (len(rows),),dtype=np.float32)
    by_protein = {}
    for i,row in enumerate(rows): by_protein.setdefault(row[0],[]).append((i,row[1]-1))
    for acc,batch in by_protein.items():
        a = np.load(folder/(acc+".npy"),mmap_mode="r",allow_pickle=False)
        ordering,positions = map(np.array,zip(*batch))
        if all_axes: out[ordering] = a[positions,:]
        elif split=="D": out[ordering] = a[positions,axis]
        else: out[ordering] = a[positions]
    if not np.isfinite(out).all(): raise ValueError("Nonfinite model activations")
    return out

def grouped_folds(groups):
    uniq=np.unique(groups)
    if len(uniq)<4: raise ValueError(f"Four development folds require at least four distinct groups, found {len(uniq)}")
    assignments={g:i%4 for i,g in enumerate(np.random.default_rng(SEED).permutation(uniq))}
    return np.array([assignments[g] for g in groups])

def weights(groups):
    freq=Counter(groups)
    return np.array([1.0 / (freq[g]*len(freq)) for g in groups])

def group_means(values, groups):
    _, ix = np.unique(groups,return_inverse=True)
    return np.bincount(ix,weights=values)/np.bincount(ix)

def interval(values):
    values=np.asarray(values,dtype=np.float64)
    if len(values)<2: return [None,None]
    gen=np.random.default_rng(SEED)
    m=[values[gen.integers(0,len(values),size=(100,len(values)))].mean(axis=1) for _ in range(20)]
    return np.quantile(np.concatenate(m),[.025,.975]).tolist()
