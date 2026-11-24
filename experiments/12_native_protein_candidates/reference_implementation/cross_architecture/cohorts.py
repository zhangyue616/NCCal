"""Sample manifests. Structure determines which pairs exist; models never see these fields."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARCHIVE = PROJECT / "protein_extension" / "structure"
OUT = ROOT / "cohorts"
FIELDS = ("group", "accession", "early", "late", "target")

def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source, delimiter="\t")

def write_tsv(path, fields, records):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp=path.with_suffix(path.suffix+".partial")
    with temp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)

def sequences(path):
    with path.open(encoding="utf-8-sig") as f:
        return {r["accession"]: r for r in (json.loads(line) for line in f if line.strip())}

def stable_pair(task, group, accession, a, b):
    return hashlib.sha256(f"{task}|{group}|{accession}|{a}|{b}".encode()).digest()

def make_pairs(rows, task):
    strata = defaultdict(lambda: [[], []])
    for r in rows:
        k = (r["group"], r["accession"], r["aa"], r["secondary"])
        if task == "contact":
            k += (min(10, int(10*float(r["rsa"]))), int(r["total"]))
            positive = int(r["remote"]) > 0
        else:
            rsa = float(r["rsa"])
            if 0.1 < rsa < 0.4:
                continue
            positive = rsa >= 0.4
        strata[k][positive].append(int(r["pos"]))
    best = {}
    eligible_pairs = 0
    for (group, accession, *_), (neg, pos) in strata.items():
        for lo, hi in product(neg, pos):
            if abs(hi-lo) <= 30:
                continue
            eligible_pairs += 1
            a, b = sorted((lo, hi))
            digest = stable_pair(task, group, accession, a, b)
            if group not in best or digest < best[group][0]:
                best[group] = (digest, dict(group=group, accession=accession, early=a, late=b, target=int(hi == b)))
    return [best[g][1] for g in sorted(best)], eligible_pairs

def prepare(split):
    OUT.mkdir(exist_ok=True)
    if split == "D":
        seq = sequences(ARCHIVE/"D_frozen"/"sequence_index.jsonl")
        sites = sorted((r for r in read_tsv(ARCHIVE/"D_frozen"/"residue_features.tsv")
                       if r["main_clean"] == "True" and r["interface"] == "False"),
                       key=lambda r: (r["group"], r["accession"], int(r["pos"])))
        source = PROJECT/"neuron_814"/"results"/"geometry_D.npz"
        with np.load(source, allow_pickle=False) as g:
            if not (np.array_equal([r["accession"] for r in sites],g["acc"]) and
                    np.array_equal([int(r["pos"]) for r in sites],g["pos"]) and
                    np.array_equal([r["group"] for r in sites],g["group"])):
                raise ValueError("Development geometry row identities differ")
            remote = g["counts"][:,40:].sum(1)
            total = g["counts"].sum(1)
            for r, n, m in zip(sites, remote, total):
                if int(r["remote_contacts"]) != n: raise ValueError("Remote contact counts differ")
                r["remote"],r["total"] = int(n),int(m)
    else:
        seq = sequences(ROOT/"external"/"sequences.jsonl")
        sites = list(read_tsv(ROOT/"external"/"sites.tsv"))
        excluded = sequences(ROOT/"external"/"exclusion_references.jsonl")
        prior_sequences = {r["sequence"] for r in excluded.values()}
        if set(seq) & set(excluded) or any(r["sequence"] in prior_sequences for r in seq.values()):
            raise ValueError("External accessions or identical sequences overlap historical and reserve proteins")
        if len({r["sequence"] for r in seq.values()}) != len(seq):
            raise ValueError("Repeated canonical sequence in external groups")
    if not seq or len({(r["accession"],int(r["pos"])) for r in sites}) != len(sites):
        raise ValueError("Empty sequence source or duplicate positions")
    for r in sites:
        s=seq[r["accession"]]
        if s["sequence"][int(r["pos"])-1] != r["aa"] or s["group"] != r["group"]:
            raise ValueError("A residue identity or group did not match the sequence manifest")
    contact, possibilities = make_pairs(sites,"contact")
    exp, exp_possibilities = make_pairs(sites,"exposure")
    if split == "D":
        if len(contact) != 77 or possibilities != 117:
            raise ValueError(f"Long-range pairs differed from the structure-only preflight: {len(contact)}, {possibilities}")
        exp = []
        for r in read_tsv(ARCHIVE/"D_frozen"/"matched_pairs.tsv"):
            buried,exposed = int(r["pos1"]),int(r["pos2"])
            a,b = sorted((buried,exposed))
            exp.append(dict(group=r["group"],accession=r["accession"],early=a,late=b,target=int(exposed==b)))
        exp.sort(key=lambda r:r["group"])
        if len(exp) != len({r["group"] for r in exp}):
            raise ValueError("Repeated archived exposure development group")
    if len({r["group"] for r in seq.values()}) == 0: raise ValueError("No groups")
    for task,data in (("contact",contact),("exposure",exp)):
        write_tsv(OUT/f"{split}_{task}.tsv",FIELDS,data)
    with (OUT/f"{split}_sequences.jsonl").open("w",encoding="utf-8") as f:
        for acc in sorted(seq):
            r=seq[acc];f.write(json.dumps({k:r[k] for k in ("accession","sequence","group")})+"\n")
    if split == "D":
        grid = []
        for acc, r in sorted(seq.items()):
            for pos in sorted({max(1,k*len(r["sequence"])//9) for k in range(1,9)}):
                grid.append(dict(accession=acc,group=r["group"],pos=pos))
        write_tsv(OUT/"D_grid.tsv",("group","accession","pos"),grid)
    info = {"sequences":len(seq),"sequence_groups":len({r["group"] for r in seq.values()}),
            "sites":len(sites),"structure_groups":len({r["group"] for r in sites}),
            "contact_candidate_pairs":possibilities,"contact_groups":len(contact),
            "exposure_candidate_pairs_by_current_geometry":exp_possibilities,"exposure_groups":len(exp)}
    (OUT/f"{split}_coverage.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info))
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("split",choices=("D","E"))
    prepare(parser.parse_args().split)
