"""Recover the requested PubChem input strings by CID from a matching source snapshot."""
import argparse,csv,gzip
from pathlib import Path
from fetch import digest
import json
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot',type=Path)
    p.add_argument('--cohort',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=Path(__file__).resolve().parent
    pins=json.loads((data/'asset_index.json').read_text(encoding='utf-8'))
    pin=next(r for r in pins if r['id']=='molecules/CID-SMILES.gz')
    if digest(a.snapshot)!=pin['sha256']:p.exit(1,'The input does not match the historical snapshot.\n')
    with (data/'molecule_ids.tsv').open(encoding='utf-8') as s:
        rows=[r for r in csv.DictReader(s,delimiter='\t') if r['cohort']==a.cohort]
    if not rows:
        with gzip.open(data/'molecular_cohort_membership.tsv.gz','rt',encoding='utf-8',newline='') as s:
            rows=[r for r in csv.DictReader(s,delimiter='\t') if r['cohort']==a.cohort]
    if not rows:p.error('unknown cohort')
    remaining={r['cid'] for r in rows}; values={}
    with gzip.open(a.snapshot,'rt',encoding='utf-8') as s:
        for line in s:
            cid,smiles=line.rstrip('\r\n').split('\t',1)
            if cid in remaining:
                values[cid]=smiles; remaining.remove(cid)
                if not remaining: break
    if remaining:p.exit(1,f'{len(remaining)} CIDs were missing.\n')
    with a.out.open('x',encoding='utf-8',newline='') as s:
        writer=csv.writer(s,delimiter='\t');writer.writerow(['rank','cid','source_smiles'])
        for r in rows:writer.writerow([r['rank'],r['cid'],values[r['cid']]])
if __name__=='__main__':main()

