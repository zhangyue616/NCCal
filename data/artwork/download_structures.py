"""Optional public-structure acquisition; never invoked by CPU figure rendering."""
import argparse,urllib.request
from pathlib import Path
STRUCTURES={'2CZL':'https://files.rcsb.org/download/2CZL.cif',
            '2XMX':'https://files.rcsb.org/download/2XMX.cif'}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download',action='store_true',help='Fetch coordinate files; requires network access.')
p.add_argument('--output',type=Path,default=Path('structure_inputs'))
a=p.parse_args()
if not a.download:
 for key,url in STRUCTURES.items():print(key,url)
else:
 a.output.mkdir(parents=True,exist_ok=True)
 for key,url in STRUCTURES.items():urllib.request.urlretrieve(url,str(a.output/(key+'.cif')))
