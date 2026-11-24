import sys,json,hashlib,urllib.request
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O/'vendor'))
import pandas as pd
import torch
d=pd.read_parquet(O/'author/e1_per_concept.parquet');s=d[(d.model=='esm2_8m')&(d.layer==2)&(d.concept=='Disulfide bond')]
rows=s[s.arm.isin(['mlp_poles','sae'])].to_dict('records')
root=O.parents[1];local=root/'assets/esm2_8m/model.safetensors'
api=json.load(urllib.request.urlopen('https://huggingface.co/api/models/facebook/esm2_t6_8M_UR50D/revision/c731040fcd8d73dceaa04b0a8e6329b345b0f5df?blobs=true'))
f=[x for x in api['siblings'] if x['rfilename']=='model.safetensors'][0];remote=f['lfs']['sha256'];actual=hashlib.sha256(local.read_bytes()).hexdigest();assert remote==actual
sae=O/'author/sae_layer2_normalized.pt';st=torch.load(sae,weights_only=True);assert st['encoder.weight'].shape==(10240,320)
receipt={'source':'https://huggingface.co/datasets/GleghornLab/esmc_native_units_results/tree/c6db19f6/retest/esm2_8m','published_rows':rows,'target':{'layer1indexed':2,'block_index':1,'hook':'intermediate.dense output pre-GELU','unit_index':361,'pole':'positive max(z,0)','unmasked':True},'model':{'repo':api['id'],'revision':api['sha'],'remote_weight_sha256':remote,'local_weight_sha256':actual,'exact_match':True},'sae':{'file':str(sae),'sha256':hashlib.sha256(sae.read_bytes()).hexdigest(),'repo':'Elana/InterPLM-esm2-8m','revision':'81d2429cd9dae7175f1dcd8b4c649a20cdc06c8c','layer':2,'unit':4181,'readout':'ReLU(W_4181 @ (h_layer2-bias)+encoder.bias_4181)','shape':list(st['encoder.weight'].shape),'normalization_to_sqrt_d':False,'weights_recovered':True},'original_sample_overlap':'UNKNOWN; accession hash is published but sample membership not present in downloaded archive receipt','original_F1_protocol':'Feature chosen on validation; for that feature test threshold maximized on test. Independent assay uses dev-fixed threshold, so no numeric equality expected.'}
(O/'AUTHOR_IDENTITY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
