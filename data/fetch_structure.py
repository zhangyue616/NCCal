"""Fetch and check one recorded PDB structure or SIFTS mapping."""
import argparse
import csv
from pathlib import Path
import tempfile
import urllib.request
import shutil
import os
from fetch import digest
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdb_id');p.add_argument('kind',choices=['sifts','mmcif']);p.add_argument('--dest',required=True,type=Path)
    a=p.parse_args()
    with Path(__file__).with_name('structure_hashes.tsv').open(encoding='utf-8') as s:
        row=next((r for r in csv.DictReader(s,delimiter='\t') if r['PDB']==a.pdb_id.upper() and r['kind']==a.kind),None)
    if row is None:p.error('identifier missing from this source index')
    target=a.dest/row['kind']/row['url'].rsplit('/',1)[-1]
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=row['sha256']:p.exit(1,'Existing file has a different digest\n')
        return
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as t:temp=Path(t.name)
    try:
        with urllib.request.urlopen(row['url'],timeout=90) as source,temp.open('wb') as out:
            shutil.copyfileobj(source,out)
        if digest(temp)!=row['sha256']:p.exit(1,'Snapshot differs from the recorded release\n')
        os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()
if __name__=='__main__':main()

