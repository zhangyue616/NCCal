"""Fetch pinned author-published label/result tables only; no weights."""
from prepare import *
import requests
def main():
    hf=js(OUT/'sources/hallee_info.json')['response']['sha']
    gh=js(OUT/'sources/banerjee_tree.json')['response']['sha']
    paths=['retest/esm2_8m/e1_per_concept.parquet','retest/esm2_650m/e1_per_concept.parquet',
           'retest_esmc/esmc_600/e13_per_concept.parquet','retest_esmc/esmc_600_depth/e13_per_concept.parquet',
           'retest_esmc/esmc_600_mlp/e13_per_concept.parquet']
    urls=[(f'https://huggingface.co/datasets/GleghornLab/esmc_native_units_results/resolve/{hf}/{p}',p.replace('/','__')) for p in paths]
    for model in ['8M','35M']:
        p=f'analysis/esm{model}_500k_neuron_explanations.csv'
        urls.append((f'https://raw.githubusercontent.com/arjun-banerjee/PLMNeuron/{gh}/{p}',f'banerjee_{model}.csv'))
    manifest=[]
    for url,name in urls:
        p=OUT/'sources'/name
        if not p.exists():
            r=requests.get(url,timeout=60);r.raise_for_status();p.write_bytes(r.content)
        manifest.append({'url':url,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
        print(name,p.stat().st_size,flush=True)
    write('sources/TABLE_MANIFEST.json',{'utc':datetime.now(timezone.utc).isoformat(),'hallee_revision':hf,'banerjee_revision':gh,'files':manifest,'model_activation_read':False})
if __name__=='__main__':main()
