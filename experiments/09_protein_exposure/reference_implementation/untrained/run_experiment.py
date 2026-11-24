"""Offline, development-selected ESM-2 exposure initialization controls."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys
sys.dont_write_bytecode=True
import argparse,csv,hashlib,json,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch,transformers,scipy
from transformers import EsmConfig,EsmForMaskedLM
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
OLD=ROOT/'protein_extension'
RSA=OLD/'rsa'
sys.path.insert(0,str(RSA))
from model_inputs import read_rows,design_matrix,grouped_folds,weights,group_means,activations,STRUCTURE
from exposure import PENALTIES,sequence_fit,residual_slopes,development_signature
from paired_rsa import kernel_fit
threadpool_limits(2)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def tsv(p,head,rows):
    with Path(p).open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(head);w.writerows(rows)
def log(s):print(f'{utc()} {s}',flush=True)
def author_copy():
    AUTHOR.mkdir(parents=True,exist_ok=True)
    for name in ('PROTOCOL_A_UNTRAINED.md','FREEZE.json','TRAINED_REPLAY.json','TIMING.json','STATUS.json','DEVIATIONS.md','RESULT.md','NETWORK_RESULTS.tsv','INFERENCE.tsv','EXPOSURE_CONTROL_SKETCH.png'):
        p=OUT/name
        if p.exists():shutil.copy2(p,AUTHOR/name)
def status(state,**kw):
    write(OUT/'STATUS.json',dict(updated_utc=utc(),status=state,**kw));author_copy()

def freeze():
    if (OUT/'FREEZE.json').exists():return
    files=[OUT/'PROTOCOL_A_UNTRAINED.md',Path(__file__),
        *[RSA/n for n in ('METHOD.md','extract_axes.py','exposure.py','model_inputs.py','paired_rsa.py')],
        STRUCTURE/'residue_features.tsv',STRUCTURE/'matched_pairs.tsv',
        OLD/'calibration/run_calibration.py',OLD/'calibration/RUN_MANIFEST.json',
        ROOT/'assets/esm2_8m/config.json',RSA/'assets/esm2_35m/config.json']
    for model in ('8m','35m'):
        files += [RSA/'results'/model/n for n in ('selection.json','sequence_model.npz','results.json')]
    write(OUT/'FREEZE.json',dict(frozen_utc=utc(),before_untrained_confirmation=True,
        files={str(p):sha(p) for p in files},planned_K=199,fallback_K=99,seeds=[20261118000+i for i in range(1,200)],
        torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,scipy=scipy.__version__,
        python=sys.executable,device=torch.cuda.get_device_name(0),threads=2))
    (OUT/'DEVIATIONS.md').write_text('# Deviations\n\nNone at protocol freeze.\n',encoding='utf8')
    author_copy()

class Context:
    def __init__(self):
        self.d,self.cohort=read_rows('D');self.t,_=read_rows('T')
        self.x=design_matrix(self.d,self.cohort);self.xt=design_matrix(self.t,self.cohort)
        self.y=np.array([r[5] for r in self.d]);self.yt=np.array([r[5] for r in self.t])
        self.g=np.array([r[2] for r in self.d]);self.gt=np.array([r[2] for r in self.t])
        assert (len(self.d),len(np.unique(self.g)),len(self.t),len(np.unique(self.gt)))==(20974,325,173859,1099)
        f=grouped_folds(self.g);self.splits=[(f!=k,f==k) for k in range(4)]
        self.w=weights(self.g);losses={}
        for p in PENALTIES:
            pred=np.empty_like(self.y)
            for tr,va in self.splits:
                pred[va]=self.x[va]@sequence_fit(self.x[tr],self.y[tr],weights(self.g[tr]),p)
            losses[p]=float(np.mean(group_means(np.abs(self.y-pred),self.g)))
        self.penalty=min(PENALTIES,key=losses.get);self.basecv=losses[self.penalty]
        self.coef=sequence_fit(self.x,self.y,self.w,self.penalty)
        self.residuals=[]
        for tr,va in self.splits:
            b=sequence_fit(self.x[tr],self.y[tr],weights(self.g[tr]),self.penalty)
            self.residuals.append((self.y[tr]-self.x[tr]@b,self.y[va]-self.x[va]@b))
        self.base=self.xt@self.coef;self.bgroup=group_means(np.abs(self.yt-self.base),self.gt)
        self.dp=self.pairs(self.d,'D');self.tp=self.pairs(self.t,'T')
        de,dl,self.px,self.py,self.pg=self.dp;te,tl,pxt,self.pyt,self.pgt=self.tp
        self.pk=(self.px@self.px.T).toarray();self.pkt=(pxt@self.px.T).toarray()
        self.pfold=grouped_folds(self.pg)
        assert (len(self.pg),len(self.pgt))==(195,967)
        self.pbase=self.pair_predict(np.zeros(len(self.pg)),np.zeros(len(self.pgt)),False)
    def pairs(self,rows,split):
        lookup={(r[0],r[1]):i for i,r in enumerate(rows)};pairs=[];labs=[];groups=[]
        with (STRUCTURE/'matched_pairs.tsv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                if r['split']!=split:continue
                i=lookup[r['accession'],int(r['pos1'])];j=lookup[r['accession'],int(r['pos2'])]
                assert rows[i][2]==r['group'] and rows[i][3:5]==rows[j][3:5]
                assert rows[i][5]<=.10 and rows[j][5]>=.40
                pairs.append((i,j) if rows[i][1]<rows[j][1] else (j,i))
                labs.append(1. if rows[i][1]<rows[j][1] else -1.);groups.append(r['group'])
        assert len(groups)==len(set(groups))
        early,late=np.asarray(pairs).T;x=design_matrix(rows,self.cohort)
        return early,late,x[early]-x[late],np.asarray(labs),np.asarray(groups)
    def pair_predict(self,z,zt,augment):
        scores=[];target=(self.py+1)/2
        for p in PENALTIES:
            pred=np.empty(len(target))
            for k in range(4):
                tr=np.flatnonzero(self.pfold!=k);va=np.flatnonzero(self.pfold==k)
                pred[va]=kernel_fit(self.pk,self.pk,tr,va,z,z,self.py,p,augment)
            scores.append(float(np.mean((target-pred)**2)))
        p=PENALTIES[int(np.argmin(scores))]
        prob=kernel_fit(self.pk,self.pkt,np.arange(len(self.py)),np.arange(len(self.pyt)),z,zt,self.py,p,augment)
        return dict(penalty=p,cv=scores,prob=prob)
    def select(self,z,folder,model):
        loss=np.zeros(z.shape[1])
        for (tr,va),(rt,rv) in zip(self.splits,self.residuals):
            mu,beta=residual_slopes(z[tr],rt,weights(self.g[tr]));ids=np.flatnonzero(va)
            for start in range(0,len(ids),512):
                take=ids[start:start+512]
                e=rv[start:start+512,None]-(z[take]-mu)*beta
                loss+=self.w[take]@np.abs(e)
        axis=int(np.argmin(loss));selected=z[:,axis].copy()
        mu,beta=residual_slopes(z[:,[axis]],self.y-self.x@self.coef,self.w)
        rec=dict(model=model,axis_zero_based=axis,axis_candidates=z.shape[1],cv_augmented_mae=float(loss[axis]),
            cv_baseline_mae=self.basecv,penalty=self.penalty,mean=float(mu[0]),slope=float(beta[0]),
            development_data_signature=development_signature(self.d),selected_utc=utc(),before_confirmation_extraction=True)
        np.save(folder/'D_axis_losses.npy',loss,allow_pickle=False)
        np.save(folder/'D_selected.npy',selected,allow_pickle=False)
        np.savez(folder/'sequence_model.npz',coef=self.coef,mu=mu,beta=beta)
        write(folder/'SELECTION_LOCK.json',rec)
        return rec,selected
    def score(self,selection,zd,zt,folder):
        assert read(folder/'SELECTION_LOCK.json')==selection
        aug=self.base+(zt.astype(float)-selection['mean'])*selection['slope']
        agroup=group_means(np.abs(self.yt-aug),self.gt)
        de,dl,*_=self.dp;te,tl,*_=self.tp
        pair=self.pair_predict(zd[de].astype(float)-zd[dl],zt[te].astype(float)-zt[tl],True)
        target=(self.pyt+1)/2;bp=self.pbase['prob'];ap=pair['prob']
        be=(target-bp)**2;ae=(target-ap)**2
        tsv(folder/'group_errors.tsv',['group','sequence_mae','augmented_mae','gain'],zip(np.unique(self.gt),self.bgroup,agroup,self.bgroup-agroup))
        tsv(folder/'buried_exposed_group_errors.tsv',['group','target','sequence_probability','augmented_probability','sequence_brier','augmented_brier','gain'],zip(self.pgt,target,bp,ap,be,ae,be-ae))
        correct=int(np.sum(((ap>.5)&(target==1))|((ap<.5)&(target==0))))
        return dict(selection=selection,T_groups=1099,T_residues=173859,T_pairs=967,
            sequence_mae=float(self.bgroup.mean()),augmented_mae=float(agroup.mean()),mae_gain=float(np.mean(self.bgroup-agroup)),
            sequence_brier=float(be.mean()),augmented_brier=float(ae.mean()),brier_gain=float(np.mean(be-ae)),
            pair_correct=correct,pair_ties=int(np.sum(ap==.5)),pair_incorrect=int(967-correct-np.sum(ap==.5)),
            pair_penalties=[self.pbase['penalty'],pair['penalty']],pair_development_cv_brier=pair['cv'])

def replay(c):
    result={}
    for model,axis in (('8m',801),('35m',814)):
        folder=OUT/'trained_replay'/model;folder.mkdir(parents=True,exist_ok=True)
        zd=activations(c.d,model,'D',all_axes=True)
        s,selected=c.select(zd,folder,model)
        zt=activations(c.t,model,'T',axis=axis)
        r=c.score(s,selected,zt,folder);old=read(RSA/'results'/model/'results.json')
        assert s['axis_zero_based']==axis and s['penalty']==old['selection']['penalty']
        assert s['development_data_signature']==old['selection']['development_data_signature']
        errs={k:abs(a-b) for k,a,b in (
            ('cv_mae',s['cv_augmented_mae'],old['selection']['cv_augmented_mae']),
            ('mae_gain',r['mae_gain'],old['group_mae_difference']),
            ('brier_gain',r['brier_gain'],old['buried_exposed_pairs']['improvement']))}
        assert max(errs.values())<1e-12,errs
        mu,beta=residual_slopes((-selected)[:,None],c.y-c.x@c.coef,c.w)
        signerr=float(np.max(np.abs((selected.astype(float)-s['mean'])*s['slope']-(-selected.astype(float)-mu[0])*beta[0])))
        de,dl,*_=c.dp;dz=selected[de].astype(float)-selected[dl]
        kernerr=float(np.max(np.abs(np.outer(dz,dz)-np.outer(-dz,-dz))))
        assert signerr<1e-12 and kernerr==0
        result[model]=dict(status='PASS',differences=errs,sign_flip_prediction_max_error=signerr,
            sign_flip_pair_kernel_max_error=kernerr,results=r)
        write(folder/'RESULT.json',r);log(f'trained replay {model} PASS {errs}')
    write(OUT/'TRAINED_REPLAY.json',dict(completed_utc=utc(),status='PASS',models=result));author_copy()

def make_model(model,index,folder):
    path=ROOT/'assets/esm2_8m' if model=='8m' else RSA/'assets/esm2_35m'
    seed=20261118000+index;torch.manual_seed(seed)
    config=EsmConfig.from_pretrained(path,local_files_only=True)
    net=EsmForMaskedLM(config).float().eval()
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    identity=dict(created_utc=utc(),seed=seed,torch=torch.__version__,transformers=transformers.__version__,
        architecture=[config.hidden_size,config.intermediate_size,config.num_hidden_layers,config.hidden_act],
        state_sha256=h.hexdigest(),config_sha256=sha(path/'config.json'),parameter_count=sum(p.numel() for p in net.parameters()),
        embedding_lm_head_tied=net.esm.embeddings.word_embeddings.weight.data_ptr()==net.lm_head.decoder.weight.data_ptr(),device='cuda',threads=2)
    assert identity['embedding_lm_head_tied']
    if model=='8m' and index==1:
        identity['matches_historical_8m_init001']=h.hexdigest()==read(OLD/'calibration/init_001/MODEL_IDENTITY.json')['state_sha256']
        assert identity['matches_historical_8m_init001']
    write(folder/'MODEL_IDENTITY.json',identity)
    return net.to('cuda'),{token:i for i,token in enumerate((path/'vocab.txt').read_text().splitlines())}

class AtBlock(Exception):pass
def extract(net,vocab,rows,cohort,axis=None):
    width=net.config.intermediate_size
    result=np.empty((len(rows),width) if axis is None else (len(rows),),np.float32)
    by_acc={}
    for i,r in enumerate(rows):by_acc.setdefault(r[0],[]).append((i,r[1]-1))
    proteins=sorted(by_acc,key=lambda a:(len(cohort[a][2]),a));capture=[]
    def hook(_m,_in,out):
        capture.append((out if axis is None else out[:,:,axis]).detach());raise AtBlock()
    handle=net.esm.encoder.layer[net.config.num_hidden_layers//2-1].intermediate.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0,len(proteins),8):
                ids=proteins[start:start+8];seqs=[cohort[a][2] for a in ids]
                toks=[[vocab['<cls>'],*[vocab[c] for c in seq],vocab['<eos>']] for seq in seqs]
                inp=torch.full((len(ids),max(map(len,toks))),vocab['<pad>'],dtype=torch.long)
                for k,row in enumerate(toks):inp[k,:len(row)]=torch.tensor(row)
                inp=inp.cuda();capture.clear()
                try:net.esm(input_ids=inp,attention_mask=inp.ne(vocab['<pad>']).long())
                except AtBlock:pass
                assert len(capture)==1
                values=capture[0].cpu().numpy()
                for k,acc in enumerate(ids):
                    ix,pos=np.asarray(by_acc[acc]).T;result[ix]=values[k,pos+1]
    finally:handle.remove()
    assert np.isfinite(result).all()
    return result

def network(c,model,index):
    folder=OUT/model/f'init_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():return read(folder/'RESULT.json')
    tick=time.perf_counter();started=utc();times={}
    net,vocab=make_model(model,index,folder);t=time.perf_counter();times['initialization_seconds']=t-tick
    zd=extract(net,vocab,c.d,c.cohort);t2=time.perf_counter();times['development_extraction_seconds']=t2-t
    s,zsel=c.select(zd,folder,model);del zd;t3=time.perf_counter();times['development_selection_seconds']=t3-t2
    zt=extract(net,vocab,c.t,c.cohort,s['axis_zero_based']);t4=time.perf_counter();times['confirmation_extraction_seconds']=t4-t3
    np.save(folder/'T_selected.npy',zt,allow_pickle=False)
    r=c.score(s,zsel,zt,folder);del net;torch.cuda.empty_cache()
    times['scoring_and_output_seconds']=time.perf_counter()-t4
    times['total_seconds']=time.perf_counter()-tick
    r.update(index=index,seed=20261118000+index,model=model,started_utc=started,completed_utc=utc(),timing=times)
    write(folder/'RESULT.json',r);write(folder/'TIMING.json',times)
    log(f"{model} init {index} complete axis={s['axis_zero_based']} elapsed={times['total_seconds']:.2f}s")
    return r

def aggregate(K):
    rows=[];inf=[];trained=read(OUT/'TRAINED_REPLAY.json')['models']
    for model in ('8m','35m'):
        data=[read(OUT/model/f'init_{i:03d}/RESULT.json') for i in range(1,K+1)]
        for r in data:
            rows.append([model,r['index'],r['seed'],r['selection']['axis_zero_based'],r['selection']['cv_augmented_mae'],r['mae_gain'],r['brier_gain'],r['pair_correct'],r['pair_ties'],r['timing']['total_seconds']])
        for key in ('mae_gain','brier_gain'):
            v=np.array([r[key] for r in data]);value=trained[model]['results'][key];count=int(np.sum(v>=value))
            inf.append(dict(model=model,endpoint=key,K=K,trained_gain=value,untrained_ge=count,p_raw=(1+count)/(K+1),
                untrained_median=float(np.median(v)),untrained_min=float(v.min()),untrained_max=float(v.max()),p_holm=None))
    fam=[r for r in inf if r['model']=='35m'];running=0.
    for rank,r in enumerate(sorted(fam,key=lambda r:r['p_raw'])):
        running=max(running,min(1.,(2-rank)*r['p_raw']));r['p_holm']=running
    tsv(OUT/'NETWORK_RESULTS.tsv',['model','index','seed','axis_zero_based','development_cv_mae','mae_gain','brier_gain','pair_correct','pair_ties','wall_seconds'],rows)
    tsv(OUT/'INFERENCE.tsv',list(inf[0]),[list(r.values()) for r in inf])
    write(OUT/'AGGREGATE.json',dict(K=K,inference=inf,pass_both_35m=all(r['p_holm']<=.05 for r in fam)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,key,title in zip(axes,('mae_gain','brier_gain'),('Continuous MAE reduction','Matched-pair Brier reduction')):
        for x,model in enumerate(('8m','35m')):
            vals=np.array([r[5 if key=='mae_gain' else 6] for r in rows if r[0]==model])
            jitter=np.linspace(-.18,.18,K)
            ax.scatter(np.full(K,x)+jitter,vals,s=7,alpha=.5,color='#777777')
            ax.scatter([x],[trained[model]['results'][key]],s=60,marker='D',color='#005a8d',zorder=3)
        ax.axhline(0,color='#cccccc',linewidth=.7);ax.set_xticks([0,1],['8M','35M']);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Exposure initialization control: K={K} per size; blue = trained')
    fig.tight_layout();fig.savefig(OUT/'EXPOSURE_CONTROL_SKETCH.png',dpi=180);plt.close(fig)
    lines=['# Stage A completed results','',f'K = {K} per size. This is a post hoc control for the secondary 35M analysis.','',
        '|Model|Endpoint|Trained gain|Untrained median [min,max]|>= trained|p|35M Holm p|','|---|---|---:|---:|---:|---:|---:|']
    for r in inf:lines.append(f"|{r['model']}|{r['endpoint']}|{r['trained_gain']:.12g}|{r['untrained_median']:.12g} [{r['untrained_min']:.12g}, {r['untrained_max']:.12g}]|{r['untrained_ge']}|{r['p_raw']:.6g}|{r['p_holm'] if r['p_holm'] is not None else 'descriptive'}|")
    lines+=['',f"Both 35M Holm tests pass the frozen criterion: {all(r['p_holm']<=.05 for r in fam)}.",'','The original 8M continuous endpoint remains the prespecified primary analysis. The supplement does not change that status.']
    (OUT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');author_copy()

def main():
    cli=argparse.ArgumentParser();cli.add_argument('command',choices=['run','status','aggregate']);opt=cli.parse_args()
    if opt.command=='status':print((OUT/'STATUS.json').read_text());return
    if opt.command=='aggregate':aggregate(read(OUT/'TIMING.json')['chosen_K']);return
    freeze()
    freeze_data=read(OUT/'FREEZE.json')
    assert all(sha(p)==h for p,h in freeze_data['files'].items()),'Frozen source changed'
    start=time.perf_counter();started=utc();status('TRAINED_REPLAY',started_utc=started,pid=os.getpid(),cwd=str(OUT))
    try:
        c=Context();replay(c)
        status('TWO_NETWORK_TIMING',started_utc=started,pid=os.getpid())
        pilots=[network(c,m,1) for m in ('8m','35m')]
        elapsed=time.perf_counter()-start;pairsec=sum(r['timing']['total_seconds'] for r in pilots)
        estimates={str(k):elapsed+(k-1)*1.2*pairsec for k in (199,99)}
        K=199 if estimates['199']<=43200 else (99 if estimates['99']<=43200 else None)
        timing=dict(measured_utc=utc(),started_utc=started,preparation_replay_pilots_elapsed_seconds=elapsed,
            two_network_timings={r['model']:r['timing'] for r in pilots},margin=1.2,
            projected_total_seconds=estimates,chosen_K=K,selection_basis='timing only; gains not consulted')
        write(OUT/'TIMING.json',timing);author_copy()
        if K is None:
            status('STOP_K99_EXCEEDS_12H',timing=timing);return
        status('MAIN_QUEUE_RUNNING',chosen_K=K,complete_per_model=1,started_utc=started,pid=os.getpid(),timing=timing)
        for i in range(2,K+1):
            for model in ('8m','35m'):
                estimate=1.2*next(r for r in pilots if r['model']==model)['timing']['total_seconds']
                if time.perf_counter()-start+estimate>43200:
                    status('STOP_ACTUAL_12H_DEADLINE',chosen_K=K,next_model=model,next_index=i);return
                network(c,model,i)
                status('MAIN_QUEUE_RUNNING',chosen_K=K,last_complete_model=model,last_complete_index=i,started_utc=started,pid=os.getpid())
        aggregate(K);status('COMPLETE',chosen_K=K,total_elapsed_seconds=time.perf_counter()-start)
    except Exception as e:
        write(OUT/'ERROR.json',dict(utc=utc(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        status('STOP_ERROR',error_type=type(e).__name__,message=str(e));raise

if __name__=='__main__':main()
