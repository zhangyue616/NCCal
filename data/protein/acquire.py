"""Explicit public-input retrieval; no request occurs at import time."""
from __future__ import annotations
import argparse,json,hashlib,urllib.request
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--kind",choices=["forward-sequences","public-label-tables"],required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent;args.output.mkdir(parents=True,exist_ok=True)
    if args.kind=="forward-sequences":
        cases=json.loads((root/"forward_subset_identifiers.json").read_text("utf8"));records=[]
        for accession in sorted({x["accession"]for x in cases}):
            request=urllib.request.Request("https://rest.uniprot.org/uniprotkb/"+accession+".fasta",headers={"User-Agent":"protein-reproduction"})
            text=urllib.request.urlopen(request,timeout=60).read().decode("utf8");sequence="".join(line.strip()for line in text.splitlines()if not line.startswith(">"));digest=hashlib.sha256(sequence.encode()).hexdigest()
            expected={x["sequence_sha256"]for x in cases if x["accession"]==accession}
            if digest not in expected:raise ValueError("Current public sequence differs from the frozen accession checksum: "+accession)
            records.append(dict(accession=accession,sequence=sequence,sequence_sha256=digest))
        (args.output/"forward_sequences.jsonl").write_text("\n".join(json.dumps(r)for r in records)+"\n","utf8")
    else:
        for source in json.loads((root/"public_label_sources.json").read_text("utf8")):
            request=urllib.request.Request(source["url"],headers={"User-Agent":"protein-reproduction"});data=urllib.request.urlopen(request,timeout=120).read()
            if hashlib.sha256(data).hexdigest()!=source["sha256"]:raise ValueError("Pinned public-table checksum differs.")
            (args.output/source["name"]).write_bytes(data)

if __name__=="__main__":main()
