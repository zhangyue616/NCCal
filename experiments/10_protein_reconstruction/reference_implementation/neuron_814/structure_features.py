"""Reconstruct frozen physical-contact inputs from local coordinate/SIFTS caches.

No model activations or outcomes are read. The row filter/order is the archived
rsa.model_inputs.read_rows contract; its roster is the frozen structural
sequence_index.jsonl, so this program needs no inputs outside that archive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "protein_extension" / "structure"
sys.path.insert(0, str(ARCHIVE / "runtime"))
import gemmi
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))
EXPECTED = {"D": (20974, 325), "T": (173859, 1099)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_frozen_rows(split):
    cohort = {}
    with (ARCHIVE / "sequence_index.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            record = json.loads(line)
            acc = record["accession"]
            if acc in cohort:
                raise ValueError(f"Duplicate sequence {acc}")
            cohort[acc] = record
    rows, seen = [], set()
    with (ARCHIVE / "residue_features.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            if (record["split"] != split or record["main_clean"] != "True"
                    or record["interface"] != "False"):
                continue
            acc, pos = record["accession"], int(record["pos"])
            reference = cohort[acc]
            if ((acc, pos) in seen or reference["split"] != split
                    or reference["group"] != record["group"]
                    or reference["sequence"][pos - 1] != record["aa"]):
                raise ValueError(f"Repeated or misaligned centre {acc}:{pos}")
            seen.add((acc, pos))
            record["pos"] = pos
            rows.append(record)
    rows.sort(key=lambda r: (r["group"], r["accession"], r["pos"]))
    representatives = {(r["group"], r["accession"]) for r in rows}
    if len(representatives) != len({r["group"] for r in rows}):
        raise ValueError("More than one representative per group")
    if (len(rows), len(representatives)) != EXPECTED[split]:
        raise ValueError(f"Frozen cohort shape differs: {split} {len(rows)} {len(representatives)}")
    return rows, cohort


def read_sifts(pid, accession):
    # Exact copy of the archived residue-map semantics, with a local file read
    # instead of the builder's fetch() call. No requests/network code is used.
    path = ARCHIVE / "cache" / "sifts" / (pid + ".xml.gz")
    root = ET.fromstring(gzip.decompress(path.read_bytes()))
    maps, ambiguous = defaultdict(dict), set()
    for residue in root.iter():
        if residue.tag.rsplit("}", 1)[-1] != "residue":
            continue
        refs = [x.attrib for x in residue if x.tag.rsplit("}", 1)[-1] == "crossRefDb"]
        uniprot = [r for r in refs if r.get("dbSource") == "UniProt"
                   and r.get("dbAccessionId") == accession]
        pdb = [r for r in refs if r.get("dbSource") == "PDB"]
        if not uniprot or not pdb:
            continue
        if {r.get("dbAccessionId") for r in refs if r.get("dbSource") == "UniProt"} != {accession}:
            continue
        if len(uniprot) != 1 or len(pdb) != 1:
            continue
        u, p = uniprot[0], pdb[0]
        if p.get("dbResNum") in ("null", "None", None):
            continue
        try:
            pos = int(u["dbResNum"])
        except ValueError:
            continue
        key = (p["dbChainId"], p["dbResNum"])
        value = (pos, u.get("dbResName"), p.get("dbResName"))
        if key in maps[p["dbChainId"]] and maps[p["dbChainId"]][key] != value:
            ambiguous.add(key)
        maps[p["dbChainId"]][key] = value
    for chain in maps:
        frequencies = Counter(v[0] for v in maps[chain].values())
        maps[chain] = {k: v for k, v in maps[chain].items()
                       if k not in ambiguous and frequencies[v[0]] == 1}
    return maps


def exact_chain(chain, mapping, sequence):
    bypos = {}
    exact_count = 0
    for residue in chain:
        if (residue.entity_type != gemmi.EntityType.Polymer
                or not gemmi.find_tabulated_residue(residue.name).is_amino_acid()):
            continue
        atoms = [a for a in residue if a.element.name not in ("H", "D")]
        if not atoms:
            continue
        rid = str(residue.seqid.num) + (residue.seqid.icode.strip() or "")
        reference = mapping.get(chain.name, {}).get((chain.name, rid))
        aa = AA3.get(residue.name)
        if reference is None or aa is None:
            continue
        pos, mapped_aa, mapped_name = reference
        if not (1 <= pos <= len(sequence) and sequence[pos - 1] == aa
                and mapped_aa == aa and mapped_name == residue.name):
            continue
        exact_count += 1
        # The original builder keeps the last deposited atom of each name,
        # including at non-quality-filtered distant partner residues.
        coordinates = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z], dtype=np.float64)
                       for a in atoms}
        bypos[pos] = (aa, coordinates, rid)
    return exact_count, bypos


def extract_group(job):
    indices, rows, sequence = job
    first = rows[0]
    provenance = (first["pdb_id"], first["assembly_id"], first["chain"],
                  int(first["assembly_chain"]))
    if any((r["pdb_id"], r["assembly_id"], r["chain"], int(r["assembly_chain"]))
           != provenance for r in rows):
        raise ValueError(f"Inconsistent frozen provenance: {first['group']}")
    pid, assembly_id, author_chain, chain_index = provenance
    path = ARCHIVE / "cache" / "mmcif" / (pid + ".cif.gz")
    block = gemmi.cif.read_string(gzip.decompress(path.read_bytes()).decode()).sole_block()
    structure = gemmi.make_structure_from_block(block)
    assembly = next((a for a in structure.assemblies if a.name == "1"),
                    sorted(structure.assemblies, key=lambda a: a.name)[0])
    if assembly.name != assembly_id:
        raise ValueError(f"Assembly differs from archive for {pid}")
    model = gemmi.make_assembly(assembly, structure[0], gemmi.HowToNameCopiedChain.Dup)
    mapping = read_sifts(pid, first["accession"])
    choices, selected = [], None
    for ci, chain in enumerate(model):
        exact_count, bypos = exact_chain(chain, mapping, sequence)
        if exact_count:
            choices.append((-exact_count, chain.name, ci))
        if ci == chain_index:
            if chain.name != author_chain:
                raise ValueError(f"Author chain differs for {pid}/{chain_index}")
            selected = bypos
    if not choices or min(choices)[1:] != (author_chain, chain_index) or selected is None:
        raise ValueError(f"Deterministic chain selection differs for {pid}")
    points = []
    for pos, (aa, atoms, rid) in selected.items():
        name = "CA" if aa == "G" else "CB"
        if name in atoms:
            points.append((pos, AA.index(aa), atoms[name]))
    partner_pos = np.array([p[0] for p in points], dtype=np.int64)
    partner_aa = np.array([p[1] for p in points], dtype=np.int64)
    partner_xyz = np.array([p[2] for p in points], dtype=np.float64)
    output = np.zeros((len(rows), 80), dtype=np.uint16)
    remote_contacts_sum = 0
    for i, row in enumerate(rows):
        pos = row["pos"]
        aa, atoms, rid = selected[pos]
        if aa != row["aa"] or rid != row["author_residue"]:
            raise ValueError(f"Centre identity differs: {first['accession']}:{pos}")
        xyz = atoms["CA" if aa == "G" else "CB"]
        distance = np.linalg.norm(partner_xyz - xyz, axis=1)
        gap = np.abs(partner_pos - pos)
        remote = gap >= 24
        count = int(np.count_nonzero(remote & (distance <= 8.0)))
        if count != int(row["remote_contacts"]):
            raise ValueError(f"Remote contacts differ: {first['accession']}:{pos} {count} vs {row['remote_contacts']}")
        if int(remote.sum()) != int(row["remote_eligible_partners"]):
            raise ValueError(f"Remote partner denominator differs: {first['accession']}:{pos}")
        for gap_bin, mask_gap in enumerate(((gap >= 3) & (gap <= 23), remote)):
            for radius_bin, mask_radius in enumerate((distance <= 6.0, (distance > 6.0) & (distance <= 8.0))):
                counts = np.bincount(partner_aa[mask_gap & mask_radius], minlength=20)
                if counts.max() > np.iinfo(np.uint16).max:
                    raise OverflowError("A physical count exceeds uint16")
                start = (gap_bin * 2 + radius_bin) * 20
                output[i, start:start + 20] = counts
        if int(output[i, 40:].sum()) != count:
            raise ValueError("Remote count does not equal the last 40 feature sum")
        remote_contacts_sum += count
    evidence = {
        "group": first["group"], "accession": first["accession"], "pdb_id": pid,
        "assembly_id": assembly_id, "model": 1, "author_chain": author_chain,
        "assembly_chain": chain_index, "centres": len(rows),
        "exact_observed_positions": len(selected), "coordinate_partners": len(points),
        "remote_contacts_sum": remote_contacts_sum,
        "row_start": indices[0], "row_stop_exclusive": indices[-1] + 1,
        "deterministic_chain_selection_matches": True,
        "centre_author_residues_match": True, "all_remote_counts_match": True,
        "all_remote_partner_denominators_match": True,
        "mmcif": str(path), "sifts": str(ARCHIVE / "cache" / "sifts" / (pid + ".xml.gz")),
    }
    return indices, output, evidence


def run_split(split, workers):
    started = time.monotonic()
    rows, cohort = read_frozen_rows(split)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group"]].append((index, row))
    jobs = [([i for i, r in group], [r for i, r in group],
             cohort[group[0][1]["accession"]]["sequence"]) for group in groups.values()]
    counts = np.zeros((len(rows), 80), dtype=np.uint16)
    evidence = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, (indices, values, receipt) in enumerate(pool.map(extract_group, jobs), 1):
            counts[indices] = values
            evidence.append(receipt)
            if completed % 50 == 0 or completed == len(jobs):
                print(f"{split}: {completed}/{len(jobs)} groups, {time.monotonic() - started:.1f}s", flush=True)
    acc = np.array([r["accession"] for r in rows], dtype=str)
    pos = np.array([r["pos"] for r in rows], dtype=np.int64)
    group = np.array([r["group"] for r in rows], dtype=str)
    archived_remote = np.array([int(r["remote_contacts"]) for r in rows], dtype=np.int64)
    if not np.array_equal(counts[:, 40:].sum(axis=1), archived_remote):
        raise ValueError("Final remote counts differ from archive")
    dest = HERE / "results" / f"geometry_{split}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + f".{os.getpid()}.partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, acc=acc, pos=pos, group=group, counts=counts)
    with np.load(temp, allow_pickle=False) as saved:
        for name, expected in (("acc", acc), ("pos", pos), ("group", group), ("counts", counts)):
            if not np.array_equal(saved[name], expected):
                raise ValueError(f"Saved {name} failed roundtrip")
    temp.replace(dest)
    check_path = HERE / "GEOMETRY_CHECK.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
    check.update({
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "archived_residue_features_sha256": sha256(ARCHIVE / "residue_features.tsv"),
        "archived_sequence_index_sha256": sha256(ARCHIVE / "sequence_index.jsonl"),
        "alphabet": AA, "feature_order": "gap [3..23, >=24] x radius [<=6, (6,8]] x amino_acid",
        "counts_transform": "raw integer counts; downstream model applies log1p",
        "partner_rule": "exact canonical SIFTS, selected same assembly/model/author chain/copy, Cbeta or Gly Calpha present; no partner quality filter",
        "row_order": "archived model_inputs.read_rows filter and (group, accession, position) sort",
        "local_only": True, "model_outcomes_read": False,
        "software": {"numpy": np.__version__, "gemmi": gemmi.__version__},
    })
    check.setdefault("splits", {})[split] = {
        "status": "PASS", "path": str(dest), "sha256": sha256(dest),
        "shape": list(counts.shape), "groups": len(groups), "count_dtype": str(counts.dtype),
        "centres_remote_count_checked": len(rows), "remote_count_mismatches": 0,
        "remote_partner_denominator_mismatches": 0, "roundtrip_identical": True,
        "raw_counts_nonnegative": True, "remote_contacts_sum": int(archived_remote.sum()),
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "provenance_by_group": evidence,
    }
    check["status"] = "PASS" if set(check["splits"]) == {"D", "T"} else "D_READY" if "D" in check["splits"] else "T_READY"
    atomic_json(check_path, check)
    print(f"ATOMIC_DELIVERY {split}: {dest} {counts.shape} remote mismatches=0", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("D", "T", "both"), default="both")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    for split in ("D", "T") if args.split == "both" else (args.split,):
        run_split(split, args.workers)


if __name__ == "__main__":
    main()
