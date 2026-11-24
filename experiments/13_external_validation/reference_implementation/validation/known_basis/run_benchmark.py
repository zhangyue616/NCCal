"""Known-answer post-GELU coordinate benchmark; never edits source inputs."""
import os
for key in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
COHORT=PROJECT/'nccal_evidence_extension/cohort'
WEIGHTS=PROJECT/'exposure/assets/esm2_35m'
SUPPORT=np.array([814,17,101,233,377,509,641,773,905,1037,1169,1301,1433,1565,1697,1829])
NAMES=['identity','signed_swap','rotation15','rotation45','rotation75','hadamard4','hadamard16']

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def tsv(name,rows):
    rows=list(rows)
    if not rows:return
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,list(rows[0]),delimiter='\t');wr.writeheader();wr.writerows(rows)
def load():
    chains=[json.loads(s) for s in (COHORT/'chains.jsonl').read_text(encoding='utf-8').splitlines()]
    with (COHORT/'observed_residues.tsv').open(encoding='utf-8',newline='') as f:sites=list(csv.DictReader(f,delimiter='\t'))
    return chains,sites
def transform(name):
    q=np.eye(16)
    if name=='signed_swap':q[:2,:2]=[[0,-1],[1,0]]
    elif name.startswith('rotation'):
        angle=np.deg2rad(int(name[8:]));c,s=np.cos(angle),np.sin(angle)
        q[:2,:2]=[[c,-s],[s,c]]
    elif name.startswith('hadamard'):
        k=int(name[8:]);h=np.ones((1,1))
        while len(h)<k:h=np.block([[h,h],[h,-h]])
        q[:k,:k]=h/np.sqrt(k)
    return q
def freeze():
    if (ROOT/'FROZEN.json').exists():raise RuntimeError('Already frozen; use verify, never overwrite freeze')
    files=[ROOT/'PROTOCOL.md',ROOT/'run_benchmark.py',COHORT/'chains.jsonl',COHORT/'observed_residues.tsv',COHORT/'COHORT_FREEZE.json',WEIGHTS/'model.safetensors',WEIGHTS/'config.json',WEIGHTS/'vocab.txt']
    chains,sites=load()
    old=json.loads((COHORT/'COHORT_FREEZE.json').read_text(encoding='utf-8'))
    for f in files:
        if f.name in old['files'] and f.parent==COHORT:assert sha(f)==old['files'][f.name]['sha256']
    group_splits={}
    for c in chains:
        assert group_splits.setdefault(c['homology_group'],c['split'])==c['split']
        assert len(c['sequence'])<=1022 and sha_string(c['sequence'])==c['sequence_sha256']
        assert c['organism'] in ('Homo sapiens','Mus musculus','Saccharomyces cerevisiae','Arabidopsis thaliana','Drosophila melanogaster','Caenorhabditis elegans','Danio rerio','Rattus norvegicus','Bos taurus','Gallus gallus','Sus scrofa','Oryctolagus cuniculus','Equus caballus','Ovis aries') or c['taxid'] in ('9606','10090','4932','559292','3702','7227','6239','7955','10116','9913','9031','9823','9986','9796','9940')
    dump('FROZEN.json',{'utc':utc(),'before_new_forward':True,'files':{str(p):sha(p) for p in files},'chains':len(chains),'sites':len(sites),'groups':len(group_splits),'split_groups':{s:sum(v==s for v in group_splits.values()) for s in 'ABT'},'source_axis':814,'source_polarity':1,'coordinate_boundary':'block6 FFN post-GELU','support_order':SUPPORT.tolist(),'transformations':{n:transform(n).tolist() for n in NAMES}})
    print('FROZEN before any new result',flush=True)
def sha_string(x):return hashlib.sha256(x.encode()).hexdigest()
def guard():
    f=json.loads((ROOT/'FROZEN.json').read_text(encoding='utf-8'))
    for p,h in f['files'].items():
        if Path(p)==ROOT/'run_benchmark.py' and (ROOT/'IMPLEMENTATION_AMENDMENT.json').exists():
            amendment=json.loads((ROOT/'IMPLEMENTATION_AMENDMENT.json').read_text(encoding='utf-8'))
            assert amendment['original_frozen_code_sha256']==h
            h=amendment['corrected_code_sha256']
        if sha(p)!=h:raise RuntimeError('Frozen input/code drift: '+p)
    return f
def model():
    import torch
    from transformers import EsmForMaskedLM
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.20)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=EsmForMaskedLM.from_pretrained(str(WEIGHTS),local_files_only=True,dtype=torch.float32,attn_implementation='eager').to(dev).eval()
    m.requires_grad_(False)
    assert m.config.hidden_act=='gelu' and m.config.intermediate_size==1920 and len(m.esm.encoder.layer)==12
    vocab={s:i for i,s in enumerate((WEIGHTS/'vocab.txt').read_text().splitlines())}
    def tokens(seq):
        assert len(seq)<=1022 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY')
        return torch.tensor([[vocab['<cls>']]+[vocab[s] for s in seq]+[vocab['<eos>']]],device=dev)
    dump('RUNTIME.json',{'utc':utc(),'python':sys.executable,'torch':torch.__version__,'device':dev,'gpu':torch.cuda.get_device_name() if dev=='cuda' else None,'memory_fraction':.20,'dtype':'float32','tf32':False,'batch_size':1,'sleep_between_forwards_seconds':.08})
    return m,tokens,dev
def extract():
    guard();chains,sites=load();cache=ROOT/'activations';cache.mkdir(exist_ok=True)
    import torch
    m,tokens,dev=model();tap=m.esm.encoder.layer[5].intermediate
    seqs={c['sequence_sha256']:c['sequence'] for c in chains}
    class Reached(Exception):pass
    for n,(key,seq) in enumerate(sorted(seqs.items())):
        path=cache/(key+'.npy')
        if path.exists():continue
        captured=[]
        def hook(mod,args,out):
            assert out.shape[-1]==1920
            captured.append(out[0,1:len(seq)+1].detach().cpu().numpy().copy());raise Reached()
        handle=tap.register_forward_hook(hook)
        try:
            with torch.inference_mode():m(input_ids=tokens(seq))
        except Reached:pass
        finally:handle.remove()
        assert len(captured)==1 and captured[0].shape==(len(seq),1920) and np.isfinite(captured[0]).all()
        np.save(path,captured[0],allow_pickle=False)
        print(json.dumps({'phase':'extract','sequence':n+1,'of':len(seqs),'length':len(seq)}),flush=True);time.sleep(.08)
    cdict={c['chain_id']:c for c in chains}
    xx=np.empty((len(sites),1920),np.float32)
    for key in seqs:
        arr=np.load(cache/(key+'.npy'),allow_pickle=False)
        for i,r in enumerate(sites):
            c=cdict[r['chain_id']]
            if c['sequence_sha256']==key:
                pos=int(r['seq_index0']);assert r['aa']==c['sequence'][pos] and int(r['label_seq_id'])+int(c['native_offset0'])==pos+1
                xx[i]=arr[pos]
    np.save(ROOT/'observed_activations.npy',xx,allow_pickle=False)
    dump('EXTRACTION.json',{'utc':utc(),'chains':len(chains),'distinct_sequences':len(seqs),'observed_residue_records':len(xx),'width':1920,'finite':bool(np.isfinite(xx).all()),'max_length':max(map(len,seqs.values())),'old_files_written':False,'input_unmasked':True})
def weights(groups):
    u,c=np.unique(groups,return_counts=True);d=dict(zip(u,c));w=np.array([1/d[g] for g in groups],np.float64);return w/w.sum()
def metrics(y,p,w):
    mu=np.dot(w,y);var=np.dot(w,(y-mu)**2);mse=np.dot(w,(y-p)**2)
    cov=np.dot(w,(y-mu)*(p-np.dot(w,p)));vp=np.dot(w,(p-np.dot(w,p))**2)
    return {'r2':float(1-mse/var),'mse':float(mse),'target_variance':float(var),'pearson':float(cov/np.sqrt(var*vp)) if vp>0 else 0.,'amplitude_pred_on_truth':float(cov/var)}
def fit_axes(z,y,w):
    mz=w@z;my=w@y;zc=z-mz;yc=y-my
    beta=(w*yc)@zc/np.maximum(w@(zc*zc),1e-25);b=my-mz*beta
    mse=w@((y[:,None]-z*beta-b)**2)
    return beta,b,mse
def omp_fit(z,y,w,kmax=16):
    mu=w@z;sc=np.sqrt(w@((z-mu)**2));sc=np.maximum(sc,1e-10);a=(z-mu)/sc;ym=w@y;yy=y-ym
    chosen=[];models={};res=yy.copy();sw=np.sqrt(w)
    for k in range(1,kmax+1):
        scores=np.abs((w*res)@a);scores[chosen]=-np.inf;j=int(np.argmax(scores));chosen.append(j)
        coef=np.linalg.lstsq(a[:,chosen]*sw[:,None],yy*sw,rcond=None)[0];res=yy-a[:,chosen]@coef
        if k in (1,2,4,8,16):
            v=np.zeros(z.shape[1]);v[chosen]=coef/sc[chosen];models[k]=(chosen.copy(),v,float(ym-mu@v))
    return models
def analyze():
    guard();chains,sites=load();x=np.load(ROOT/'observed_activations.npy',allow_pickle=False).astype(np.float64)
    groups=np.array([r['homology_group'] for r in sites]);spl=np.array([r['split'] for r in sites]);idx={s:np.where(spl==s)[0] for s in ['A','B','T']};idx['AB']=np.where(spl!='T')[0]
    y=x[:,814];w={s:weights(groups[ix]) for s,ix in idx.items()};out=[];pergroup=[];predrows=[];fitout={};loading=[]
    ia,ib,it,itr=idx['A'],idx['B'],idx['T'],idx['AB'];mu=w['AB']@y[itr];sd=np.sqrt(w['AB']@((y[itr]-mu)**2))
    for name in NAMES:
        q=transform(name);z=x.copy();z[:,SUPPORT]=x[:,SUPPORT]@q;a=np.zeros(1920);a[SUPPORT]=q[0]
        assert np.max(np.abs(q@q.T-np.eye(16)))<1e-14
        beta,b,mse=fit_axes(z[itr],y[itr],w['AB']);j=int(np.argmin(mse))
        paths=omp_fit(z[ia],y[ia],w['A']);bm={k:float(w['B']@((y[ib]-(z[ib]@v+c))**2)) for k,(support,v,c) in paths.items()};tol=1e-10*float(w['A']@((y[ia]-w['A']@y[ia])**2));k=min(k for k,v in bm.items() if v<=min(bm.values())+tol)
        chosen=paths[k][0];zm=w['AB']@z[itr][:,chosen];ym=w['AB']@y[itr];zc=z[itr][:,chosen]-zm
        cc=np.linalg.lstsq(zc*np.sqrt(w['AB'])[:,None],(y[itr]-ym)*np.sqrt(w['AB']),rcond=None)[0];v=np.zeros(1920);v[chosen]=cc;c=float(ym-zm@cc)
        fitout[name]={'single_axis':j,'single_beta':float(beta[j]),'single_intercept':float(b[j]),'omp_selected_k':k,'omp_support':chosen,'omp_coef':v.tolist(),'omp_intercept':c,'B_selection_mse':bm,'direction':a.tolist(),'AB_target_mean':float(mu),'AB_target_sd':float(sd)}
        print(json.dumps({'phase':'AB_calibration_only','arm':name,'k':k,'support':chosen}),flush=True)
    dump('CALIBRATION_LOCK.json',{'utc':utc(),'before_T_diagnostic_scoring':True,'fit_uses':'A+B targets; OMP size chosen B using A fit; T unused for fitted selections','fits':fitout})
    for name in NAMES:
        q=transform(name);z=x.copy();z[:,SUPPORT]=x[:,SUPPORT]@q;a=np.zeros(1920);a[SUPPORT]=q[0]
        spec=fitout[name];j=spec['single_axis'];k=spec['omp_selected_k'];chosen=spec['omp_support'];v=np.array(spec['omp_coef']);c=spec['omp_intercept']
        p1=z[it,j]*spec['single_beta']+spec['single_intercept'];pred=z[it]@v+c;traced=z[it]@a
        bt,ct,mt=fit_axes(z[it],y[it],w['T']);jt=int(np.argmin(mt));oracle=z[it,jt]*bt[jt]+ct[jt]
        planted=set(np.where(np.abs(a)>1e-10)[0]);selected=set(chosen);signed=sum(np.sign(v[jj])==np.sign(a[jj]) for jj in planted&selected)
        row={'arm':name,'selected_single_axis':j,'oracle_T_axis':jt,'single_R2':metrics(y[it],p1,w['T'])['r2'],'oracle_T_single_R2':metrics(y[it],oracle,w['T'])['r2'],'OMP_R2':metrics(y[it],pred,w['T'])['r2'],'tracked_R2':metrics(y[it],traced,w['T'])['r2'],'OMP_k':k,'source_support_size':len(planted),'OMP_support_recall':len(planted&selected)/len(planted),'OMP_signed_support_recall':signed/len(planted),'OMP_direction_cosine':float(v@a/np.linalg.norm(v)),'OMP_direction_norm':float(np.linalg.norm(v)),'max_squared_loading':float(np.max(a*a)),'effective_axes':float(1/np.sum(a**4)),'inverse_max_abs':float(np.max(np.abs(traced-y[it]))),'single_MSE':metrics(y[it],p1,w['T'])['mse'],'OMP_MSE':metrics(y[it],pred,w['T'])['mse'],'single_amplitude':metrics(y[it],p1,w['T'])['amplitude_pred_on_truth'],'OMP_amplitude':metrics(y[it],pred,w['T'])['amplitude_pred_on_truth']}
        out.append(row)
        for jj in planted:loading.append({'arm':name,'transformed_axis':int(jj),'signed_loading':float(a[jj]),'squared_loading':float(a[jj]**2)})
        for g in np.unique(groups[it]):
            sel=np.where(groups[it]==g)[0];wg=np.full(len(sel),1/len(sel))
            for label,p in [('selected_single',p1),('oracle_T_single',oracle),('OMP',pred),('tracked',traced)]:pergroup.append({'arm':name,'group':g,'method':label,'n_sites':len(sel),**metrics(y[it][sel],p[sel],wg)})
        for jrow,ii in enumerate(it):predrows.append({'arm':name,'site_id':sites[ii]['site_id'],'group':groups[ii],'source_original814':float(y[ii]),'selected_single_prediction':float(p1[jrow]),'OMP_prediction':float(pred[jrow]),'tracked_prediction':float(traced[jrow])})
        # Exact signed partial inverse, ordered by absolute loading then index.
        order=sorted(planted,key=lambda jj:(-abs(a[jj]),jj));base=w['AB']@z[itr]
        for nk in sorted(set([1,min(2,len(planted)),min(4,len(planted)),len(planted)])):
            sub=order[:nk];partial=mu+(z[it][:,sub]-base[sub])@a[sub]
            pergroup.append({'arm':name,'group':'ALL_T_PARTIAL','method':'exact_partial_'+str(nk),'n_sites':len(it),**metrics(y[it],partial,w['T'])})
        print(json.dumps({'phase':'calibration_and_T','arm':name,'singleR2':row['single_R2'],'OMP_R2':row['OMP_R2'],'k':k}),flush=True)
    dump('SUMMARY.json',{'utc':utc(),'question':'known-answer coordinate sensitivity, not biological mechanism discovery','counts':{s:{'rows':len(ix),'groups':len(np.unique(groups[ix]))} for s,ix in idx.items()},'source_target_AB_mean':float(mu),'source_target_AB_sd':float(sd),'arms':out})
    tsv('diagnostics.tsv',out);tsv('true_loadings.tsv',loading);tsv('per_group_reconstruction.tsv',pergroup);tsv('T_predictions.tsv',predrows)
def output_checks():
    guard();chains,sites=load();fits=json.loads((ROOT/'CALIBRATION_LOCK.json').read_text(encoding='utf-8'))['fits'];plan={}
    for c in chains:
        if c['split']!='T':continue
        pos=sorted(int(s['seq_index0']) for s in sites if s['chain_id']==c['chain_id']);choose=np.unique(np.rint(np.linspace(0,len(pos)-1,min(8,len(pos)))).astype(int));plan[c['chain_id']]=[pos[i]+1 for i in choose]
    dump('OUTPUT_SITE_PLAN.json',{'utc':utc(),'before_output_comparisons':True,'positions_one_based':plan,'all_inputs':'complete unmasked natural sequences'})
    import torch
    m,tokens,dev=model();tap=m.esm.encoder.layer[5].intermediate;dense=m.esm.encoder.layer[5].output.dense;orig=dense.weight.detach().clone();sup=torch.tensor(SUPPORT,device=dev);rows=[];linear=[]
    def run(seq,positions,q=None,addition=None):
        handles=[];capture=[];qt=None if q is None else torch.as_tensor(q,device=dev,dtype=torch.float32)
        if q is not None:
            with torch.no_grad():dense.weight[:,sup]=orig[:,sup]@qt
        else:
            with torch.no_grad():dense.weight.copy_(orig)
        def hook(mod,args,out):
            h=out.clone();capture.append(out[0,positions].detach().cpu().double().numpy())
            if qt is not None:h[:,:,sup]=h[:,:,sup]@qt
            if addition is not None:h[0,positions,:]+=torch.as_tensor(addition,device=dev,dtype=torch.float32)
            return h
        handles.append(tap.register_forward_hook(hook))
        try:
            with torch.inference_mode():log=m(input_ids=tokens(seq)).logits[0,positions].detach().cpu().double().numpy()
        finally:
            for h in handles:h.remove()
            with torch.no_grad():dense.weight.copy_(orig)
        time.sleep(.08);return log,capture[0]
    def errors(ref,cur):
        delta=cur-ref;lp=ref-np.logaddexp.reduce(ref,axis=1)[:,None];lq=cur-np.logaddexp.reduce(cur,axis=1)[:,None];kl=np.maximum(0,(np.exp(lp)*(lp-lq)).sum(axis=1))
        return np.max(np.abs(delta),axis=1),np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(ref,axis=1),1e-20),kl
    for nc,c in enumerate([c for c in chains if c['chain_id'] in plan]):
        pos=plan[c['chain_id']];seq=c['sequence'];baseline,h=run(seq,pos);delta=.5*fits['identity']['AB_target_sd'];directadd=np.zeros_like(h);directadd[:,814]=delta;direct,_=run(seq,pos,addition=directadd)
        for name in NAMES:
            q=transform(name);a=np.zeros(1920);a[SUPPORT]=q[0];z=h.copy();z[:,SUPPORT]=h[:,SUPPORT]@q;spec=fits[name];j=spec['single_axis'];p1=z[:,j]*spec['single_beta']+spec['single_intercept'];pom=z@np.array(spec['omp_coef'])+spec['omp_intercept'];jj=min(np.where(abs(a)==max(abs(a)))[0]);single=np.zeros(1920);single[jj]=a[jj]
            cases=[('compensated_noop',None,baseline),('tracked_nonzero_writeback',np.tile(delta*a,(len(pos),1)),direct),('single_component_nonzero_writeback',np.tile(delta*single,(len(pos),1)),direct),('wrong_polarity_writeback',np.tile(-delta*a,(len(pos),1)),direct)]
            for label,p in [('single_decoder',p1),('OMP_decoder',pom)]:
                d=p-h[:,814];aa=np.zeros_like(h);aa[:,814]=d;directpred,_=run(seq,pos,addition=aa);cases.append((label+'_intact_fidelity',d[:,None]*a[None,:],baseline));cases.append((label+'_inverse_equivalence',d[:,None]*a[None,:],directpred))
            for label,add,ref in cases:
                cur,_=run(seq,pos,q=q,addition=add);ma,rl,kl=errors(ref,cur)
                for ir,p in enumerate(pos):rows.append({'chain_id':c['chain_id'],'group':c['homology_group'],'position_one_based':p,'arm':name,'comparison':label,'max_abs_logit_error':float(ma[ir]),'relative_L2_logit_error':float(rl[ir]),'KL_reference_to_current':float(kl[ir])})
            # Double-precision next-linear equivalence, real selected-site activations.
            ww=orig.detach().cpu().double().numpy();zz=h.copy();zz[:,SUPPORT]=h[:,SUPPORT]@q;wq=ww.copy();wq[:,SUPPORT]=ww[:,SUPPORT]@q
            for label,add,refh in [('noop',np.zeros_like(h),h),('tracked_writeback',np.tile(delta*a,(len(h),1)),h+directadd)]:
                true=refh@ww.T;test=(zz+add)@wq.T
                linear.append({'chain_id':c['chain_id'],'arm':name,'comparison':label,'max_abs':float(np.max(abs(test-true))),'relative_L2':float(np.linalg.norm(test-true)/np.linalg.norm(true))})
        print(json.dumps({'phase':'frozen_full_model','chain':nc+1,'of':len(plan),'chain_id':c['chain_id'],'sites':len(pos)}),flush=True)
        tsv('output_per_site.tsv',rows);tsv('linear_equivalence.tsv',linear)
    groupout=[]
    for name in NAMES:
        for label in sorted(set(r['comparison'] for r in rows)):
            for g in sorted(set(r['group'] for r in rows)):
                rr=[r for r in rows if r['arm']==name and r['comparison']==label and r['group']==g]
                groupout.append({'arm':name,'comparison':label,'group':g,'site_records':len(rr),'max_abs_logit_error':max(r['max_abs_logit_error'] for r in rr),'mean_relative_L2':float(np.mean([r['relative_L2_logit_error'] for r in rr])),'mean_KL':float(np.mean([r['KL_reference_to_current'] for r in rr]))})
    tsv('output_per_group.tsv',groupout)
    sums=[]
    for name in NAMES:
        for label in sorted(set(r['comparison'] for r in rows)):
            rr=[r for r in groupout if r['arm']==name and r['comparison']==label]
            sums.append({'arm':name,'comparison':label,'groups':len(rr),'chains':len(plan),'sites':sum(r['site_records'] for r in rr),'max_abs_logit_error':max(r['max_abs_logit_error'] for r in rr),'group_mean_relative_L2':float(np.mean([r['mean_relative_L2'] for r in rr])),'group_mean_KL':float(np.mean([r['mean_KL'] for r in rr]))})
    tsv('output_summary.tsv',sums);dump('OUTPUT_SUMMARY.json',{'utc':utc(),'arms':sums,'chains':len(plan),'groups':len(set(c['homology_group'] for c in chains if c['chain_id'] in plan)),'site_records':sum(map(len,plan.values())),'scope':'only fixed measured output sites; no protein functional claims'})
def acceptance():
    frozen=guard();s=json.loads((ROOT/'SUMMARY.json').read_text());o=json.loads((ROOT/'OUTPUT_SUMMARY.json').read_text());checks={}
    checks['inversion']=max(r['inverse_max_abs'] for r in s['arms'])<1e-10
    checks['function_and_inverse_writeback']=all(r['max_abs_logit_error']<2e-3 and r['group_mean_relative_L2']<1e-4 for r in o['arms'] if r['comparison'] in ('compensated_noop','tracked_nonzero_writeback','single_decoder_inverse_equivalence','OMP_decoder_inverse_equivalence'))
    with (ROOT/'linear_equivalence.tsv').open() as f:linear=list(csv.DictReader(f,delimiter='\t'))
    checks['double_linear_equivalence']=max(float(r['relative_L2']) for r in linear)<1e-10
    checks['orthogonal']=all(np.max(abs(transform(n)@transform(n).T-np.eye(16)))<1e-14 for n in NAMES)
    checks['all_counts']=frozen['sites']==sum(s['counts'][ss]['rows'] for ss in 'ABT') and o['chains']==17 and o['groups']==15 and o['site_records']==136
    dump('ACCEPTANCE.json',{'utc':utc(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'limits':['known scalar calibration target available A/B; not unsupervised mechanism discovery','R2 of exact inversion is guaranteed by construction','no exposure or biological endpoint scored here','T cohort shares earlier public host dataset; independent homology split, not new sampling','float32 full-model values measured only at136 declared sites','finite equivalence tolerance does not establish every possible input']})
    print(json.dumps(checks),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','extract','analyze','output','acceptance']);args=ap.parse_args()
    {'freeze':freeze,'extract':extract,'analyze':analyze,'output':output_checks,'acceptance':acceptance}[args.stage]()
