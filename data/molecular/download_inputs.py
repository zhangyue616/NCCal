"""Retrieve a requested CID subset from the official public property service.

This command performs network access only when explicitly run. Molecular
source strings remain in the user's external input/cache directory.
"""
from pathlib import Path
import argparse,csv,json,urllib.request

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ids',type=Path,default=Path(__file__).with_name('input_ids.tsv'));p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cohort',default='evaluation');p.add_argument('--limit',type=int,default=1);args=p.parse_args()
    with args.ids.open(encoding='utf8') as f:rows=[r for r in csv.DictReader(f,delimiter='\t') if r['cohort']==args.cohort][:args.limit]
    if not rows:raise ValueError('No requested identifiers.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for row in rows:
        cid=int(row['cid']);url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES/JSON'
        with urllib.request.urlopen(url,timeout=90) as response:data=response.read()
        json.loads(data);(args.output_dir/(str(cid)+'.json')).write_bytes(data)
if __name__=='__main__':main()
