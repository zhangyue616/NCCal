"""Read-only historical input; all outputs local to this calibration directory."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
threadpool_limits(4)
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODEL = ROOT/'assets/esm2_8m'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
P3 = module('readonly_p3',ROOT/'scripts/pilot3a.py')
KRR = module('readonly_krr',ROOT/'scripts/revision/protein_krr.py')

def source_inputs():
    d=read(ROOT/'results/pilot3a/D_CASES.json')['cases']
    m=read(ROOT/'results/revision/protein_D/D_INPUT_MANIFEST.json')
    t=read(ROOT/'data/revision/protein_fresh/SEALED_FRESH_CASES.json')
    assert len(d)==192 and len(m['cases'])==864 and len(t['cases'])==2608
    assert all(c['split']=='D_fit' for c in d[:128]) and all(c['split']=='D_select' for c in d[128:])
    assert not ({c['id'] for c in d} & {c['id'] for c in t['cases']})
    return d,m,t

def load_cache(path,cases):
    with np.load(path,allow_pickle=False) as a:
        ids=a['case_ids'].astype(str).tolist(); z=a['z'].copy()
    assert len(set(ids))==len(ids) and z.shape==(len(ids),1280) and np.isfinite(z).all()
    lookup={s:i for i,s in enumerate(ids)}
    return z[[lookup[c['case_id']] for c in cases]]

def trained_cache(d,m,t):
    zd=load_cache(ROOT/'results/pilot3a/D_NATIVE.npz',d)
    cache={}
    for p in KRR.native_paths():
        with np.load(p,allow_pickle=False) as a:
            for s,z in zip(a['case_ids'].astype(str),a['z']):
                assert s not in cache; cache[s]=z
    zm=np.stack([cache[c['case_id']] for c in m['cases']])
    zt=load_cache(ROOT/'results/revision/protein_T/T_NATIVE.npz',t['cases'])
    return zd,zm,zt

def orthogonal(r):
    path=OUT/'rotations'/f'haar_{r:02d}.npy'
    if path.exists(): return np.load(path,mmap_mode='r')
    rng=np.random.default_rng(2026111800+r)
    q,rr=np.linalg.qr(rng.standard_normal((1280,1280)))
    q*=np.where(np.diag(rr)>=0,1.,-1.)[None,:]
    err=float(np.max(np.abs(q.T@q-np.eye(1280))))
    assert err<1e-12
    path.parent.mkdir(exist_ok=True); np.save(path,q)
    write(path.with_suffix('.json'),{'seed':2026111800+r,'max_orthogonality_error':err,'sha256':sha(path),'distribution':'Haar O(1280), Gaussian QR diagonal-sign correction'})
    return q

def interval(v):
    v=np.asarray(v,dtype=np.float64); n=len(v)
    mu=float(v.mean()); se=float(v.std(ddof=1)/np.sqrt(n))
    half=float(student_t.ppf(.975,n-1)*se)
    return {'mean':mu,'CI95_t_conditional':[mu-half,mu+half],'SE':se,'groups':n}

def site_select(zd,d):
    xf0=P3.features(d[:128]); xs0=P3.features(d[128:]); sx=P3.fit_scaler(xf0)
    xf,xs=P3.transform(xf0,sx),P3.transform(xs0,sx)
    sz=P3.fit_scaler(zd[:128]); zf,zs=P3.transform(zd[:128],sz),P3.transform(zd[128:],sz)
    y=np.asarray([c['y'] for c in d]); base=P3.ridge_fit(xf,y[:128])
    base_loss=P3.brier(y[128:],P3.ridge_predict(base,xs))
    fits=P3.all_unit_ridges(xf,zf,y[:128],xs,zs)
    scores=P3.component_means(base_loss[:,None]-P3.brier(y[128:,None],fits['predictions']),d[128:]).mean(0)
    eligible=zd[:128].std(0)>1e-12
    unit=min(np.flatnonzero(eligible),key=lambda j:(-float(scores[j]),int(j)))
    # Polarity is model redundancy, not 2,560 candidates.
    neg=P3.all_unit_ridges(xf,-zf,y[:128],xs,-zs)
    pred_error=float(np.max(np.abs(fits['predictions']-neg['predictions'])))
    coef_error=float(np.max(np.abs(fits['coef_z']+neg['coef_z'])))
    assert pred_error<1e-11 and coef_error<1e-11
    aug=P3.ridge_fit(np.column_stack((xf,zf[:,unit])),y[:128])
    detail={'unit':int(unit),'D_select_gain':float(scores[unit]),'eligible_axes':int(eligible.sum()),
        'sign_flip_prediction_max_error':pred_error,'sign_flip_coefficient_max_error':coef_error}
    return detail,(sx,sz,base,aug)

def site_eval(zt,t,unit,fit):
    sx,sz,base,aug=fit
    x=P3.transform(P3.features(t['cases']),sx)
    zraw=zt if zt.ndim==1 else zt[:,unit]
    z=(zraw-sz['mean'][unit])/sz['scale'][unit]
    y=np.asarray([c['y'] for c in t['cases']])
    losses=np.column_stack((P3.brier(y,P3.ridge_predict(base,x)),P3.brier(y,P3.ridge_predict(aug,np.column_stack((x,z))))))
    g=np.asarray([losses[[p['a_index'],p['b_index']]].mean(0) for p in t['pairs']])
    assert len(set(p['group'] for p in t['pairs']))==1304
    return {'baseline_Brier':float(g[:,0].mean()),'augmented_Brier':float(g[:,1].mean()),'gain':interval(g[:,0]-g[:,1])},g

def paired_eval_direct(zm,zt,m,t,unit,run_dir):
    # Same manuscript KRR blocks, lambda grid, groups, folds and readout budget.
    dm=KRR.pair_data(m['cases'],m['pairs'],zm[:,unit]); dt=KRR.pair_data(t['cases'],t['pairs'],zt[:,unit])
    blocks=KRR.raw_blocks(dm); cross=KRR.raw_blocks(dt,dm)
    groups=np.asarray(m['groups']); folds=KRR.fixed_folds(groups)
    y=np.asarray([p['y_a'] for p in m['pairs']]); target=2*y-1
    ty=np.asarray([p['y_a'] for p in t['pairs']]); allidx=np.arange(len(y))
    scales=KRR.block_scales(blocks,allidx); development={}; losses={}; alphas={}; fits=0
    for model in KRR.MODELS:
        candidates=[]
        for lam in KRR.LAMBDAS:
            pred=np.empty(len(y))
            for fold in range(4):
                train=np.flatnonzero(folds!=fold); valid=np.flatnonzero(folds==fold)
                k=KRR.sum_kernel(blocks,KRR.block_scales(blocks,train),model)
                alpha,_=KRR.fit_krr(k[np.ix_(train,train)],target[train],lam); fits+=1
                pred[valid]=KRR.probability(k[np.ix_(valid,train)]@alpha)
            _,g=KRR.group_average((pred-y)**2,groups)
            candidates.append({'lambda':lam,'OOF_group_Brier':float(g.mean())})
        chosen=KRR.select_candidate(candidates); development[model]={'selected_lambda':chosen['lambda'],'candidates':candidates}
        kernel=KRR.sum_kernel(blocks,scales,model)
        alpha,_=KRR.fit_krr(kernel,target,chosen['lambda']); fits+=1; alphas[model]=alpha
        pred=KRR.probability(KRR.sum_kernel(cross,scales,model)@alpha)
        losses[model]=(pred-ty)**2
    assert fits==102
    contribution=KRR.paired_contributions(losses)
    np.savez_compressed(run_dir/'paired_arrays.npz',**{f'loss_{k}':v for k,v in losses.items()},contributions=contribution,
        groups=np.asarray([p['group'] for p in t['pairs']]),**{f'{k}_alpha':v for k,v in alphas.items()})
    # j -> -j leaves outer-product kernel unchanged, hence all KRR predictions unchanged.
    neg=KRR.pair_data(m['cases'],m['pairs'],-zm[:,unit])
    err=float(np.max(np.abs(blocks['j']-KRR.raw_blocks(neg)['j'])))
    assert err==0
    return {'fits':fits,'development_pairs':432,'development_groups':int(len(set(groups))),
        'confirmation_groups':1304,'models':development,'absolute_Brier':{k:float(v.mean()) for k,v in losses.items()},
        'endpoints':{k:interval(contribution[:,i]) for i,k in enumerate(KRR.ENDPOINTS)},'sign_flip_kernel_max_error':err}

_PAIR_CACHE=None
def paired_eval(zm,zt,m,t,unit,run_dir):
    """Exact rank-one update of the same KRR systems; reuse invariant X/P/Q."""
    global _PAIR_CACHE
    if _PAIR_CACHE is None:
        dm=KRR.pair_data(m['cases'],m['pairs'],np.zeros(len(m['cases'])))
        dt=KRR.pair_data(t['cases'],t['pairs'],np.zeros(len(t['cases'])))
        blocks=KRR.raw_blocks(dm); cross=KRR.raw_blocks(dt,dm)
        groups=np.asarray(m['groups']); folds=KRR.fixed_folds(groups)
        y=np.asarray([p['y_a'] for p in m['pairs']]); target=2*y-1
        ty=np.asarray([p['y_a'] for p in t['pairs']]); allidx=np.arange(len(y))
        systems={}; baseline={}
        for base in ['X','S','L']:
            candidates=[]
            for lam in KRR.LAMBDAS:
                pred=np.empty(len(y))
                for fold in [0,1,2,3,'final']:
                    train=allidx if fold=='final' else np.flatnonzero(folds!=fold)
                    valid=None if fold=='final' else np.flatnonzero(folds==fold)
                    scales=KRR.block_scales(blocks,train); kb=KRR.sum_kernel(blocks,scales,base)
                    a=kb[np.ix_(train,train)].copy(); a.flat[::len(train)+1]+=len(train)*lam
                    factor=cho_factor(a,lower=True,check_finite=False)
                    alpha=cho_solve(factor,target[train],check_finite=False)
                    test=KRR.sum_kernel(cross,scales,base) if fold=='final' else kb[np.ix_(valid,train)]
                    systems[base,lam,fold]=(train,valid,factor,alpha,test)
                    if fold!='final': pred[valid]=test@alpha
                _,g=KRR.group_average((KRR.probability(pred)-y)**2,groups)
                candidates.append({'lambda':lam,'OOF_group_Brier':float(g.mean())})
            chosen=KRR.select_candidate(candidates); lam=chosen['lambda']
            train,valid,factor,alpha,test=systems[base,lam,'final']
            baseline[base]={'development':{'selected_lambda':lam,'candidates':candidates},'alpha':alpha,
                'loss':(KRR.probability(test@alpha)-ty)**2}
        _PAIR_CACHE=(groups,folds,y,target,ty,allidx,systems,baseline)
    groups,folds,y,target,ty,allidx,systems,baseline=_PAIR_CACHE
    mz=zm if zm.ndim==1 else zm[:,unit]; tz=zt if zt.ndim==1 else zt[:,unit]
    dj=np.asarray([mz[p['a_index']]-mz[p['b_index']] for p in m['pairs']])
    tj=np.asarray([tz[p['a_index']]-tz[p['b_index']] for p in t['pairs']])
    development={k:v['development'] for k,v in baseline.items()}
    losses={k:v['loss'] for k,v in baseline.items()}; alphas={k:v['alpha'] for k,v in baseline.items()}
    def solve(base,lam,fold):
        train,valid,factor,alpha,test=systems[base,lam,fold]
        scale=float(np.mean(dj[train]**2))
        if scale<=0: return alpha,test@alpha
        u=dj[train]/np.sqrt(scale); v=(tj if fold=='final' else dj[valid])/np.sqrt(scale)
        invu=cho_solve(factor,u,check_finite=False)
        aug_alpha=alpha-invu*float(u@alpha)/(1+float(u@invu))
        return aug_alpha,test@aug_alpha+v*float(u@aug_alpha)
    for base in ['X','S','L']:
        model=base+'j'; candidates=[]
        for lam in KRR.LAMBDAS:
            pred=np.empty(len(y))
            for fold in range(4):
                _,scores=solve(base,lam,fold); pred[folds==fold]=KRR.probability(scores)
            _,g=KRR.group_average((pred-y)**2,groups)
            candidates.append({'lambda':lam,'OOF_group_Brier':float(g.mean())})
        chosen=KRR.select_candidate(candidates); lam=chosen['lambda']
        development[model]={'selected_lambda':lam,'candidates':candidates}
        alpha,scores=solve(base,lam,'final'); alphas[model]=alpha
        losses[model]=(KRR.probability(scores)-ty)**2
    contribution=KRR.paired_contributions(losses)
    np.savez_compressed(run_dir/'paired_arrays.npz',**{f'loss_{k}':v for k,v in losses.items()},contributions=contribution,
        groups=np.asarray([p['group'] for p in t['pairs']]),**{f'{k}_alpha':v for k,v in alphas.items()})
    return {'fits':102,'implementation':'same systems via cached base factorizations and Sherman-Morrison rank-one updates',
        'development_pairs':432,'development_groups':int(len(set(groups))),'confirmation_groups':1304,
        'models':development,'absolute_Brier':{k:float(v.mean()) for k,v in losses.items()},
        'endpoints':{k:interval(contribution[:,i]) for i,k in enumerate(KRR.ENDPOINTS)},'sign_flip_kernel_max_error':0.0}

def analyze(name,zd,zm,zt,paired=True):
    d,m,t=source_inputs(); nd=OUT/name; nd.mkdir(exist_ok=True)
    rows=[]
    for r in range(10):
        rd=nd/('native' if r==0 else f'rotated_{r:02d}'); rd.mkdir(exist_ok=True)
        result=rd/'RESULT.json'
        if result.exists(): rows.append(read(result)); continue
        tick=time.perf_counter()
        q=None if r==0 else orthogonal(r)
        dr=np.asarray(zd,dtype=np.float64) if q is None else zd@q
        detail,fit=site_select(dr,d); unit=detail['unit']
        tr=np.asarray(zt[:,unit],dtype=np.float64) if q is None else zt@q[:,unit]
        sr,sg=site_eval(tr,t,unit,fit)
        if name=='trained' and r==0:
            archived=read(ROOT/'results/pilot3a/D_SELECTED.json')
            assert unit==1246
            assert abs(detail['D_select_gain']-archived['D_select_selected_Brier_improvement'])<1e-12
        row={'network':name,'basis':r,'selected_axis':detail,'site_readout':sr,'paired_readout':None,'created_utc':utc()}
        np.savez_compressed(rd/'site_group_losses.npz',group_losses=sg)
        if paired:
            mr=np.asarray(zm[:,unit],dtype=np.float64) if q is None else zm@q[:,unit]
            row['paired_readout']=paired_eval(mr,tr,m,t,unit,rd)
        row['analysis_wall_seconds']=time.perf_counter()-tick
        write(result,row); rows.append(row)
        print(json.dumps({'network':name,'basis':r,'unit':unit,'gain':sr['gain']['mean'],'seconds':row['analysis_wall_seconds']}),flush=True)
    native=rows[0]['site_readout']['gain']['mean']; rot=[x['site_readout']['gain']['mean'] for x in rows[1:]]
    summary={'network':name,'site_G_native':native,'site_G_rotated':rot,'site_A':native-float(np.mean(rot)),
        'rotation_count':9,'readout':'historical single-center conditional ridge','rows':rows}
    if paired:
        summary['paired']={}
        for key in KRR.ENDPOINTS:
            n=rows[0]['paired_readout']['endpoints'][key]['mean']; r=[x['paired_readout']['endpoints'][key]['mean'] for x in rows[1:]]
            summary['paired'][key]={'G_native':n,'G_rotated':r,'A':n-float(np.mean(r))}
    write(nd/'SUMMARY.json',summary)
    return summary

def extract(index,device='cpu',limit=None,output_name=None):
    import torch, transformers
    from transformers import EsmConfig,EsmForMaskedLM,AutoTokenizer
    torch.set_num_threads(4)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    d,m,t=source_inputs(); cases=m['cases']+t['cases']
    name=output_name or ('trained_gpu' if index==0 else f'init_{index:03d}'); nd=OUT/name; nd.mkdir(exist_ok=True)
    torch.manual_seed(20261118000+index)
    config=EsmConfig.from_pretrained(MODEL,local_files_only=True)
    assert (config.hidden_size,config.intermediate_size,config.num_hidden_layers,config.hidden_act)==(320,1280,6,'gelu')
    model=(EsmForMaskedLM.from_pretrained(MODEL,local_files_only=True) if index==0 else EsmForMaskedLM(config)).float().eval()
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    ident={'created_utc':utc(),'seed':20261118000+index,'torch':torch.__version__,'transformers':transformers.__version__,
        'initializer_range':config.initializer_range,'model_type':config.model_type,'architecture':[320,1280,6,'gelu'],
        'state_sha256':h.hexdigest(),'parameter_count':sum(p.numel() for p in model.parameters()),
        'embedding_lm_head_tied':model.esm.embeddings.word_embeddings.weight.data_ptr()==model.lm_head.decoder.weight.data_ptr(),
        'config_sha256':sha(MODEL/'config.json'),'device':device,'threads':4}
    assert ident['embedding_lm_head_tied']
    prior=nd/'MODEL_IDENTITY.json'
    if prior.exists(): assert read(prior)['state_sha256']==ident['state_sha256']
    else: write(prior,ident)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model.esm.encoder.layer=torch.nn.ModuleList(list(model.esm.encoder.layer)[:3])
    model=model.to(device)
    captured={}
    def hook(_m,_i,out):
        captured['z']=out[torch.arange(len(captured['q']),device=out.device),captured['q']].detach().cpu().numpy().copy()
    handle=model.esm.encoder.layer[2].intermediate.register_forward_hook(hook)
    cd=nd/'activation_chunks'; cd.mkdir(exist_ok=True)
    n=len(cases) if limit is None else min(limit,len(cases)); tick=time.perf_counter(); completed=0
    for start in range(0,n,64):
        end=min(start+64,n); p=cd/f'{start:05d}_{end:05d}.npz'
        if p.exists():
            with np.load(p) as a: assert a['case_ids'].astype(str).tolist()==[c['case_id'] for c in cases[start:end]]
            continue
        values=[]; ct=time.perf_counter()
        batch_size=8 if device=='cuda' else 1
        for bs in range(start,end,batch_size):
            batch=cases[bs:min(bs+batch_size,end)]
            inputs=tokenizer([c['sequence'] for c in batch],return_tensors='pt',padding=True,truncation=False)
            qs=[]
            for b,c in enumerate(batch):
                q=int(c['q'])+1; assert int(inputs['attention_mask'][b].sum())==len(c['sequence'])+2
                assert inputs['input_ids'][b,q].item()==tokenizer.convert_tokens_to_ids(c['sequence'][c['q']])
                inputs['input_ids'][b,q]=tokenizer.mask_token_id; qs.append(q)
            captured['q']=torch.tensor(qs,device=device)
            inputs={k:v.to(device) for k,v in inputs.items()}
            with torch.inference_mode(): model.esm(**inputs)
            z=captured.pop('z'); assert z.shape==(len(batch),1280) and np.isfinite(z).all(); values.extend(z)
        z=np.stack(values); np.savez_compressed(p,z=z,case_ids=np.asarray([c['case_id'] for c in cases[start:end]]))
        completed+=len(values)
        write(p.with_suffix('.json'),{'start':start,'end':end,'wall_seconds':time.perf_counter()-ct,'device':device,'state_sha256':ident['state_sha256']})
        if end%512==0 or end==n:
            print(json.dumps({'network':name,'completed_through':end,'total':n,'seconds':time.perf_counter()-tick}),flush=True)
    handle.remove()
    write(nd/'EXTRACTION_RECEIPT.json',{'created_utc':utc(),'requested_cases':n,'full_cases':len(cases),
        'new_forwards_this_invocation':completed,'wall_seconds_this_invocation':time.perf_counter()-tick,'complete':n==len(cases),'device':device})
    if n<len(cases): return
    lookup={}
    for start in range(0,len(cases),64):
        end=min(start+64,len(cases)); p=cd/f'{start:05d}_{end:05d}.npz'
        with np.load(p) as a:
            for s,z in zip(a['case_ids'].astype(str),a['z']): lookup[s]=z
    get=lambda cc: np.stack([lookup[c['case_id']] for c in cc])
    zd,zm,zt=get(d),get(m['cases']),get(t['cases'])
    np.savez_compressed(nd/'ACTIVATIONS.npz',D=zd,M=zm,T=zt)

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['trained','extract','analyze-init','aggregate','batch'])
    p.add_argument('--index',type=int,default=1); p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--limit',type=int); args=p.parse_args()
    if args.command=='trained':
        d,m,t=source_inputs(); analyze('trained',*trained_cache(d,m,t))
    elif args.command=='extract': extract(args.index,args.device,args.limit)
    elif args.command=='analyze-init':
        name='trained_gpu' if args.index==0 else f'init_{args.index:03d}'
        with np.load(OUT/name/'ACTIVATIONS.npz') as a: analyze(name,a['D'],a['M'],a['T'])
    elif args.command=='batch':
        for index in range(args.index,200):
            nd=OUT/f'init_{index:03d}'
            if (nd/'SUMMARY.json').exists(): continue
            if not (nd/'ACTIVATIONS.npz').exists(): extract(index,args.device)
            with np.load(nd/'ACTIVATIONS.npz') as a: analyze(f'init_{index:03d}',a['D'],a['M'],a['T'])
            write(OUT/'BATCH_PROGRESS.json',{'created_utc':utc(),'last_completed':index,'planned_K':199,'complete':index==199})
    else:
        trained=read(OUT/'trained_gpu/SUMMARY.json'); initialized=[]
        for index in range(1,200):
            p=OUT/('runtime_check_init001' if index==1 else f'init_{index:03d}')/'SUMMARY.json'
            if p.exists(): initialized.append(read(p))
        k=len(initialized); result={'completed_K':k,'planned_K':199,'rank_resolution':1/(k+1),
            'trained_reference':'trained_gpu/SUMMARY.json','initialization_1_reference':'runtime_check_init001/SUMMARY.json',
            'interpretation':'empirical architectural reference ranks, not exact randomization p; retrospective old T',
            'single_center':{},'paired':{}}
        for key in ['site_G_native','site_A']:
            val=trained[key]; vals=[r[key] for r in initialized]
            result['single_center'][key]={'trained':val,'initialization_values':vals,'upper_rank_fraction':(1+sum(v>=val for v in vals))/(k+1)}
        for ep in KRR.ENDPOINTS:
            result['paired'][ep]={}
            for key in ['G_native','A']:
                val=trained['paired'][ep][key]; vals=[r['paired'][ep][key] for r in initialized]
                result['paired'][ep][key]={'trained':val,'initialization_values':vals,'upper_rank_fraction':(1+sum(v>=val for v in vals))/(k+1)}
        write(OUT/'AGGREGATE.json',result); print(json.dumps(result))
if __name__=='__main__': main()
