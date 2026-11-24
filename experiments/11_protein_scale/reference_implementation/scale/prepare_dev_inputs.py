"""Compute fixed development-only amino-acid exposure means for the scale protocol."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "structure/residue_features.tsv"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def main():
    by_residue = defaultdict(list)
    groups = set()
    seen = set()
    with SOURCE.open(encoding="utf-8-sig",newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["split"] != "D" or row["main_clean"] != "True" or row["interface"] != "False":
                continue
            key = row["accession"], int(row["pos"])
            if key in seen or row["aa"] not in ALPHABET:
                raise ValueError("Noncanonical or duplicate development observation")
            seen.add(key)
            groups.add(row["group"])
            by_residue[row["aa"]].append(float(row["rsa"]))
    counts = [len(by_residue[aa]) for aa in ALPHABET]
    if (sum(counts),len(groups)) != (20974,325) or min(counts) < 1:
        raise ValueError("Development population mismatch")
    with SOURCE.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {"alphabet":ALPHABET, "counts":counts,
              "values":[math.fsum(by_residue[aa])/len(by_residue[aa]) for aa in ALPHABET],
              "total_sites":sum(counts),"groups":len(groups),
              "weighting":"Arithmetic mean of all eligible D residues of that identity, each site once.",
              "source":str(SOURCE),"source_sha256":source_hash}
    target = HERE / "RSA_PROPENSITY.json"
    contents = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf8") != contents:
            raise ValueError("Pre-existing development reference differs")
        return
    target.write_text(contents,encoding="utf8")


if __name__ == "__main__":
    main()
