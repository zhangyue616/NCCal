import hashlib,json,urllib.request,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import pandas as pd
O=Path(__file__).resolve().parent/'author';O.mkdir(exist_ok=True)
def get(url,name):
 p=O/name
 if not p.exists(): urllib.request.urlretrieve(url,p)
 return {'url':url,'file':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
repo='https://raw.githubusercontent.com/Gleghorn-Lab/plm_interpretability/main/'
hf='https://huggingface.co/datasets/GleghornLab/esmc_native_units_results/resolve/c6db19f6/'
rr=[get(repo+'plm_interpretability/units/retest.py','retest.py'),get(hf+'retest/esm2_8m/e1_per_concept.parquet','e1_per_concept.parquet'),get(hf+'retest/esm2_8m/receipt.json','receipt.json'),get(hf+'retest/inputs_receipt.json','inputs_receipt.json')]
d=pd.read_parquet(O/'e1_per_concept.parquet');print(d.columns.tolist());s=d.astype(str).apply(lambda c:c.str.contains('disulf',case=False)).any(axis=1);print(d[s].to_string(index=False));d[s].to_csv(O/'disulfide_rows.tsv',sep='\t',index=False)
print((O/'receipt.json').read_text()[:12000]);print((O/'inputs_receipt.json').read_text()[:6000])
