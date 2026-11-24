"""Equal-group inference, independently selected orthonormal bases and fixed sequence judges."""
from pathlib import Path
import argparse,json,time,math
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from measure import ROOT,select,dump

def win(x):return (np.asarray(x)>0)+.5*(np.asarray(x)==0)
def grouped(v,rows):return pd.Series(np.asarray(v)).groupby(rows.group.to_numpy()).mean()

def uncertainty(x,null=.5,test=True):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return dict(mean=None,lo=None,hi=None,p=None,informative_groups=0)
    rng=np.random.default_rng(41624)
    boot=x[rng.integers(len(x),size=(9999,len(x)))].mean(axis=1)
    z=x-null;z=z[np.abs(z)>1e-12]
    p=None
    if test:
        if len(z)==0:p=1.
        elif len(z)<=18:
            ints=np.arange(2**len(z))[:,None];signs=1-2*((ints>>np.arange(len(z)))&1)
            p=float(np.mean(signs@z>=z.sum()-1e-12))
        else:
            rng=np.random.default_rng(41623);above=0
            for start in range(0,99999,2000):
                signs=2*rng.integers(2,size=(min(2000,99999-start),len(z)))-1
                above+=int((signs@z>=z.sum()-1e-12).sum())
            p=(above+1)/100000
    return dict(mean=float(x.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),p=p,informative_groups=len(z))

def rotate(X,seed,layer):
    rng=np.random.default_rng([77203,seed,layer]);r=np.asarray(X)
    for _ in range(2):
        perm=rng.permutation(r.shape[-1]);sgn=(2*rng.integers(0,2,r.shape[-1])-1).astype(np.float32)
        r=dct(np.ascontiguousarray(r[:,perm]*sgn),type=2,norm='ortho',axis=-1,workers=2)
    return r

def rotations(name):
    out=ROOT/name/'bases';out.mkdir(exist_ok=True)
    base=ROOT/name/'trained'
    tables={sp:pd.read_csv(base/f'{sp}_pairs.tsv',sep='\t',dtype={'group':str,'record_id':str}) for sp in ['D','T','E','H','F'] if (base/f'{sp}_pairs.tsv').exists()}
    values={sp:np.load(base/f'{sp}_delta.npy',mmap_mode='r') for sp in tables}
    for seed in range(1,10):
        lock=out/f'basis_{seed}_lock.json'
        if not lock.exists():
            data=np.empty(values['D'].shape,dtype=np.float32)
            errors=[]
            for l in range(data.shape[1]):
                data[:,l,:]=rotate(values['D'][:,l,:],seed,l)
                a=np.asarray(values['D'][:6,l,:],dtype=float);b=np.asarray(data[:6,l,:],dtype=float)
                relative=np.max(np.abs((a*a).sum(-1)-(b*b).sum(-1))/np.maximum((a*a).sum(-1),1e-30))
                errors.append(float(relative));assert relative<2e-5
            k=select(data,tables['D'])
            dump(lock,dict(model=name,seed=seed,selection=k,created_unix=time.time(),max_norm_relative_error=max(errors)),exclusive=True)
            del data
        k=json.loads(lock.read_text())['selection'];frames=[]
        for sp,rows in tables.items():
            result=rows.copy();r=rotate(values[sp][:,k['block_1']-1,:],seed,k['block_1']-1)
            result['delta']=k['sign']*r[:,k['channel_0']];result['correct']=win(result.delta);result['seed']=seed;result['model']=name
            frames.append(result)
        pd.concat(frames,ignore_index=True).to_csv(out/f'basis_{seed}_pairs.tsv',sep='\t',index=False)
        print('BASIS',name,seed,'done',flush=True)

ALPH=list('ACDEFGHIKLMNPQRSTVWY')+['?','.']
def window_features(rows,sequences):
    X=np.zeros((len(rows),8*len(ALPH)),dtype=float)
    for j,r in enumerate(rows.itertuples()):
        seq=sequences[r.record_id]
        for q,off in enumerate([-4,-3,-2,-1,1,2,3,4]):
            for pos,sgn in [(r.index_positive,1),(r.index_negative,-1)]:
                aa=seq[pos+off] if 0<=pos+off<len(seq) else '.'
                X[j,q*len(ALPH)+ALPH.index(aa if aa in ALPH else '?')]+=sgn
    return X

def native_predictions(name):
    base=ROOT/name/'trained'
    paths=[base/'signed_pairs.tsv']+([base/'F_signed_pairs.tsv'] if (base/'F_signed_pairs.tsv').exists() else [])
    rows=pd.concat([pd.read_csv(p,sep='\t',dtype={'group':str,'record_id':str}) for p in paths],ignore_index=True)
    sequences={}
    selection=json.loads((base/'selection_lock.json').read_text())
    paths={Path(selection['inputs']['chains_path']),ROOT.parent/'sheet/chains_F.jsonl'}
    if (base/'F_done.json').exists():paths.add(Path(json.loads((base/'F_done.json').read_text())['inputs']['chains_path']))
    for f in paths:
        if f.exists():sequences.update({r['record_id']:r['sequence'] for r in map(json.loads,f.read_text(encoding='utf-8-sig').splitlines())})
    X=window_features(rows,sequences);dev=rows.split=='D';d=rows.delta.to_numpy()
    groups=rows.loc[dev].group;weights=1/groups.map(groups.value_counts()).to_numpy();weights/=weights.mean()
    scale=float(np.sqrt(np.average(d[dev]**2,weights=weights)));assert scale>0
    coeff={};fitmeta={}
    for suffix,design in [('seq',X),('plus',np.column_stack([X,d/scale]))]:
        train=design[dev];clf=LogisticRegression(C=1,fit_intercept=False,solver='lbfgs',max_iter=1000)
        clf.fit(np.vstack([train,-train]),np.r_[np.ones(len(train)),np.zeros(len(train))],sample_weight=np.tile(weights,2))
        rows[f'prob_{suffix}']=clf.predict_proba(design)[:,1]
        rows[f'brier_{suffix}']=(1-rows[f'prob_{suffix}'])**2
        rows[f'correct_{suffix}']=win(rows[f'prob_{suffix}']-.5)
        coeff[suffix]=clf.coef_;fitmeta[suffix]=dict(iterations=int(clf.n_iter_[0]),feature_count=design.shape[1])
    np.savez(base/'sequence_coefficients.npz',**coeff,scale=np.array(scale))
    dump(base/'sequence_fit.json',dict(training_split='D only',radius=4,C=1,fit_intercept=False,mirrored_examples=True,group_equal_weights=True,channel_RMS_scale=scale,fits=fitmeta))
    rows.to_csv(base/'paired_predictions.tsv',sep='\t',index=False)
    return rows

def stats(rows):
    records=[]
    for sp,part in rows.groupby('split'):
        scopes=[('all',part)]
        for mask in ['sasa_matched_bool','position_length_matched_bool','confound_matched_bool','sheet_ge4_bool']:
            if mask in part:
                sub=part[part[mask].astype(str).str.lower().isin(['true','1','yes'])]
                scopes.append((mask,sub))
        for scope,sub in scopes:
            if not len(sub):continue
            measures={'single_channel':(win(sub.delta),.5),'accuracy_sequence':(sub.correct_seq,.5),'accuracy_plus':(sub.correct_plus,.5),'accuracy_gain':(sub.correct_plus-sub.correct_seq,0),'brier_sequence':(sub.brier_seq,0),'brier_plus':(sub.brier_plus,0),'brier_gain':(sub.brier_seq-sub.brier_plus,0)}
            for metric,(v,null) in measures.items():
                est=uncertainty(grouped(v,sub),null,test=scope=='all' and sp!='D' and metric in ['single_channel','accuracy_gain','brier_gain'])
                records.append(dict(model=sub.model.iloc[0],split=sp,scope=scope,metric=metric,groups=sub.group.nunique(),pairs=len(sub),pooled_mean=float(np.mean(v)),wins=int((sub.delta>0).sum()),losses=int((sub.delta<0).sum()),ties=int((sub.delta==0).sum()),descriptive=sp=='D' or est['informative_groups']<5 or scope!='all',**est))
    return records

def holm(ps):
    a=np.asarray(ps,dtype=float);order=np.argsort(a);res=np.empty_like(a)
    res[order]=np.maximum.accumulate(np.minimum(1,a[order]*np.arange(len(a),0,-1)));return res

def summarize(models):
    summaries=[];references=[];comparisons=[]
    for name in models:
        rows=native_predictions(name);summaries+=stats(rows)
        primary={sp:float(grouped(win(p.delta),p).mean()) for sp,p in rows.groupby('split')}
        families={'orthonormal_basis':[ROOT/name/'bases'/f'basis_{s}_pairs.tsv' for s in range(1,10)],'random_initialization':[ROOT/name/str(s)/'signed_pairs.tsv' for s in range(91001,91010)]}
        for fam,files in families.items():
            assert all(file.exists() for file in files),f'Need all nine {name} {fam} before final inference'
            accs={}
            for i,file in enumerate(files):
                if not file.exists():continue
                parts=[pd.read_csv(file,sep='\t',dtype={'group':str})]
                fresh=file.parent/'F_signed_pairs.tsv'
                if fam=='random_initialization' and fresh.exists():parts.append(pd.read_csv(fresh,sep='\t',dtype={'group':str}))
                frame=pd.concat(parts,ignore_index=True)
                for sp,p in frame.groupby('split'):
                    v=float(grouped(win(p.delta),p).mean());accs.setdefault(sp,[]).append(v)
                    references.append(dict(model=name,family=fam,reference=i+1,split=sp,accuracy=v,groups=p.group.nunique(),pairs=len(p)))
            for sp,vals in accs.items():
                assert len(vals)==9,(name,fam,sp,'Expected nine independently reselected controls')
                a=primary[sp];v=np.asarray(vals)
                comparisons.append(dict(model=name,split=sp,family=fam,n=len(v),trained_accuracy=a,reference_min=v.min(),reference_median=np.median(v),reference_max=v.max(),n_strictly_below=int((v<a-1e-12).sum()),n_tied=int((np.abs(v-a)<=1e-12).sum()),n_above=int((v>a+1e-12).sum()),rank_resolution=1/(len(v)+1)))
    table=pd.DataFrame(summaries);table['p_Holm']=np.nan
    for sp in ['T','E','H','F']:
        for metric in ['single_channel','accuracy_gain','brier_gain']:
            use=(table.split==sp)&(table.scope=='all')&(table.metric==metric)
            if use.sum()==2:table.loc[use,'p_Holm']=holm(table.loc[use,'p'])
    table.to_csv(ROOT/'inference_all.tsv',sep='\t',index=False)
    pd.DataFrame(references).to_csv(ROOT/'reference_scores.tsv',sep='\t',index=False)
    pd.DataFrame(comparisons).to_csv(ROOT/'reference_comparison.tsv',sep='\t',index=False)
    print(table[(table.scope=='all')&table.metric.isin(['single_channel','brier_gain'])].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['E1','ESMC']);ap.add_argument('--rotations-only',action='store_true');ap.add_argument('--summarize-only',action='store_true');a=ap.parse_args()
    if not a.summarize_only:
        for name in a.models:rotations(name)
    if not a.rotations_only:summarize(a.models)
