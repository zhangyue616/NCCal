from pathlib import Path
import sys, json, hashlib, time, os
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'native_channels'
sys.path[:0]=[str(ROOT/'esmc_deps'),str(OLD/'e1/python_deps'),str(OLD/'e1')]
# All actual weights and code are local. Optional E1 kernel discovery must not
# download a different backend after the validated PyTorch fallback was used.
os.environ['HF_HUB_OFFLINE']='1'
import torch
import numpy as np
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(name, seed=None, device='cuda'):
    if name=='E1':
        import e1_runtime
        model=e1_runtime.get_model(trained=seed is None,seed=seed,device=device)
        mods=[b.ffn.mlp.w2 for b in model.model.layers]
        def inputs(seq):return e1_runtime.get_batch(seq,device=next(model.parameters()).device)
        def forward(seq):return model.model(**inputs(seq)).last_hidden_state
        prefix=2; limit=8192
        meta=json.loads((OLD/'e1/trained/model_metadata.json').read_text())
        for old_key in ['chains_sha256','pairs_sha256']:
            meta.pop(old_key,None)
        meta.update(initializer='official E1ForMaskedLM constructor post_init; no weight load for random')
    elif name=='ESMC':
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer
        torch.manual_seed(88000 if seed is None else seed)
        tok=EsmSequenceTokenizer()
        model=ESMC(d_model=960,n_heads=15,n_layers=30,tokenizer=tok,use_flash_attn=False).eval()
        if seed is None:
            state=torch.load(ROOT/'esmc_weights/esmc_300m_2024_12_v0.pth',map_location='cpu',weights_only=True)
            result=model.load_state_dict(state,strict=True)
            assert not result.missing_keys and not result.unexpected_keys
            del state
        model.to(device=device,dtype=torch.float32)
        mods=[b.ffn[3] for b in model.transformer.blocks]
        def inputs(seq):
            ids=tok.encode(seq)
            assert len(ids)==len(seq)+2 and ids[0]==0 and ids[-1]==2
            assert tok.decode(ids[1:-1]).replace(' ','')==seq
            return torch.tensor([ids],device=next(model.parameters()).device)
        def forward(seq):return model(inputs(seq)).embeddings
        prefix=1;limit=2048
        meta=json.loads((ROOT/'esmc_acquisition.json').read_text())
        meta.update(model='ESM C300M official 2024-12',package='esm==3.2.1',attention='official non-flash SDPA',initializer='PyTorch constructor defaults used by official esm3.2.1 release; no special init method',strict_trained_load=seed is None)
    else:raise ValueError(name)
    width=mods[0].weight.shape[1]
    assert all(x.weight.shape[1]==width for x in mods)
    meta.update(layers=len(mods),units_per_layer=width,signed_candidates=2*width*len(mods),parameters=sum(p.numel() for p in model.parameters()),random_seed=seed,trained_weights=seed is None,dtype='float32',token_prefix_count=prefix,runtime_python=sys.executable,torch=torch.__version__,hook='post SwiGLU product; input of native output projection',max_tokens=limit,tf32=False)
    sample=mods[0].weight.detach().flatten()[:1024].float().cpu().numpy()
    meta.update(first_output_projection_sample_mean=float(sample.mean()),first_output_projection_sample_std=float(sample.std()),first_output_projection_sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest())
    return model,mods,forward,prefix,limit,meta

def preflight(name):
    rows=[json.loads(x) for x in (OLD/'cohort/chains.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    row=next(r for r in rows if r.get('uniprot')=='P69905') if any(r.get('uniprot')=='P69905' for r in rows) else rows[0]
    seq=row['sequence'];start=time.time()
    model,mods,forward,prefix,limit,meta=load(name,device='cpu')
    got=[]
    h=mods[len(mods)//2].register_forward_pre_hook(lambda m,a:got.append(a[0].detach().clone()))
    with torch.inference_mode():cpu=forward(seq).detach().cpu()
    model.cuda()
    with torch.inference_mode():gpu=forward(seq).detach().cpu()
    h.remove()
    diff=cpu-gpu; channel=got[0].cpu()-got[1].cpu()
    # A direct layer computation verifies the observed hook's semantic coordinate.
    if name=='ESMC':
        b=model.transformer.blocks[0]; x=model.embed(torch.tensor([[0,5,23,2]],device='cuda'))
        with torch.inference_mode():
            z=b.ffn[0](x);a,c=b.ffn[1](z).chunk(2,dim=-1);prod=torch.nn.functional.silu(a)*c
            explicit=prod@b.ffn[3].weight.T
            ffn_error=float((explicit-b.ffn(x)).abs().max())
    else:ffn_error=0.0 # Existing exact E1 backend semantic validation reused.
    report=dict(meta,sequence_record=row['record_id'],sequence_length=len(seq),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),observed_hook_shape=list(got[1].shape),cpu_gpu_rms=float(diff.square().mean().sqrt()),cpu_gpu_max=float(diff.abs().max()),channel_cpu_gpu_rms=float(channel.square().mean().sqrt()),ffn_explicit_max=ffn_error,ffn_check_origin='this preflight explicit official FFN algebra' if name=='ESMC' else 'inherited validated E1 preflight; not remeasured here',seconds=time.time()-start,labels_read=False)
    assert torch.isfinite(cpu).all() and torch.isfinite(gpu).all()
    assert report['cpu_gpu_rms']<1e-3 and report['channel_cpu_gpu_rms']<1e-3 and ffn_error<1e-4
    (ROOT/f'{name}_preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':preflight(sys.argv[1])
