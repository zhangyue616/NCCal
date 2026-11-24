"""Offline original ESM2/ESMC/E1 modules, explicit native readout families."""
import os,sys,gc,hashlib
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True
from prepare import ROOT
sys.path[:0]=[str(ROOT/'structural_readout_validation/scan/esmc_deps'),str(ROOT/'native_channels/e1/python_deps'),str(ROOT/'native_channels/e1')]
import torch,numpy as np
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
WEIGHTS={'esm2_8m':ROOT/'assets/esm2_8m','esm2_35m':ROOT/'exposure/assets/esm2_35m',
 'esm2_650m':__import__('pathlib').Path('external/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c'),
 'esmc_600':__import__('pathlib').Path('external/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/snapshots/e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c/data/weights/esmc_600m_2024_12_v0.pth')}
class Runtime:
    def __init__(self,name,seed=None):
        self.name=name;self.seed=seed;self.handles=[];self.collected={};self.identity_errors=[];torch.manual_seed(20261123021 if seed is None else seed)
        if name.startswith('esm2_'):
            from transformers import EsmConfig,EsmForMaskedLM,AutoTokenizer
            cfg=EsmConfig.from_pretrained(WEIGHTS[name],local_files_only=True);cfg._attn_implementation='sdpa';self.tok=AutoTokenizer.from_pretrained(WEIGHTS[name],local_files_only=True)
            self.model=(EsmForMaskedLM.from_pretrained(WEIGHTS[name],local_files_only=True,attn_implementation='sdpa') if seed is None else EsmForMaskedLM(cfg)).eval().float().cuda()
            self.mods=[b.intermediate.dense for b in self.model.esm.encoder.layer];self.prefix=1;self.kind='esm2';self.width=cfg.intermediate_size
        elif name=='esmc_600':
            from esm.models.esmc import ESMC
            from esm.tokenization import EsmSequenceTokenizer
            self.tok=EsmSequenceTokenizer();self.model=ESMC(d_model=1152,n_heads=18,n_layers=36,tokenizer=self.tok,use_flash_attn=False).eval().float()
            if seed is None:
                state=torch.load(WEIGHTS[name],map_location='cpu',weights_only=True);self.model.load_state_dict(state,strict=True);del state
            self.model.cuda();self.mods=[b.ffn[1] for b in self.model.transformer.blocks];self.prefix=1;self.kind='esmc';self.width=3072
        elif name=='e1_150m':
            import e1_runtime
            self.e1=e1_runtime;self.model=e1_runtime.get_model(trained=seed is None,seed=seed,device='cuda');self.mods=[b.ffn.mlp.w2 for b in self.model.model.layers];self.prefix=2;self.kind='e1';self.width=2304
        else:raise ValueError(name)
        sample=self.mods[0].weight.detach().float().flatten()[:1024].cpu().numpy()
    def observe(self,callback,blocks=None):
        self.clear_hooks();blocks=set(range(len(self.mods))) if blocks is None else set(blocks)
        for block,mod in enumerate(self.mods):
            if block not in blocks:continue
            if self.kind=='e1':h=mod.register_forward_pre_hook(lambda m,a,b=block:callback(b,a[0]))
            else:
                def checked(m,a,z,b=block):
                    if not self.identity_errors:
                        with torch.inference_mode():
                            expected=a[0][0,0].float()@m.weight[0].float()
                            if m.bias is not None:expected=expected+m.bias[0].float()
                            error=float((expected-z[0,0,0]).abs().cpu());assert error<=2e-5
                            self.identity_errors.append(dict(block0=b,explicit_linear_projection_max_abs_error=error,observation='module output before activation; same coordinates as weight rows'))
                    callback(b,z)
                h=mod.register_forward_hook(checked)
            self.handles.append(h)
    def clear_hooks(self):
        for h in self.handles:h.remove()
        self.handles=[]
    def forward(self,records,mask_positions=None):
        ss=[r['sequence'] for r in records]
        with torch.inference_mode():
            if self.kind=='esm2':
                tokens={k:v.cuda() for k,v in self.tok(ss,padding=True,return_tensors='pt').items()}
                if mask_positions is not None:
                    for i,p in enumerate(mask_positions):tokens['input_ids'][i,p+self.prefix]=self.tok.mask_token_id
                self.model.esm(**tokens)
            elif self.kind=='esmc':
                assert mask_positions is None;ids=[self.tok.encode(s) for s in ss];tokens=torch.full((len(ss),max(map(len,ids))),self.tok.pad_token_id,dtype=torch.long,device='cuda')
                for i,a in enumerate(ids):tokens[i,:len(a)]=torch.tensor(a,device='cuda')
                self.model(tokens)
            else:
                assert len(ss)==1 and mask_positions is None
                self.model.model(**self.e1.get_batch(ss[0],device='cuda'))
    def raw_product(self,z):
        if self.kind=='esmc':g,u=z.chunk(2,-1);return torch.nn.functional.silu(g)*u
        if self.kind=='esm2':return torch.nn.functional.gelu(z)
        return z
    def poles(self,z):
        if self.kind=='esmc':
            g,u=z.chunk(2,-1);prod=torch.nn.functional.silu(g)*u;return torch.cat([torch.where((g>0)&(u>0),prod,0),torch.where((g>0)&(u<0),-prod,0)],-1)
        return torch.cat([torch.relu(z),torch.relu(-z)],-1)
    def read_unit(self,z,unit):
        j=int(unit['unit0']);family=unit['family'];sign=float(unit.get('sign',1))
        if family=='pre_pole':return torch.relu(z[...,j] if unit['pole']=='positive' else -z[...,j])
        if family=='gate_tagged':
            g,u=z.chunk(2,-1);prod=torch.nn.functional.silu(g[...,j])*u[...,j]
            return torch.where((g[...,j]>0)&((u[...,j]>0) if unit['pole']=='positive' else (u[...,j]<0)),prod if unit['pole']=='positive' else -prod,0)
        if family=='native_signed':return self.raw_product(z)[...,j]*sign
        raise ValueError(family)
    def close(self):
        self.clear_hooks();self.model.cpu();del self.model;gc.collect();torch.cuda.empty_cache()
def batches(records,name):
    rr=sorted(records,key=lambda r:(len(r['sequence']),r['accession']));batch=[]
    for r in rr:
        if batch and (name=='e1_150m' or len(batch)>=8 or (len(batch)+1)*(len(r['sequence'])+2)>4096):yield batch;batch=[]
        batch.append(r)
    if batch:yield batch
