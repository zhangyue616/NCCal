"""Fixed published block channel: natural-protein observations only."""
import csv, json, os, sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WEIGHTS=ROOT/'exposure/assets/esm2_35m'
def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def forward(rows):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(str(WEIGHTS),local_files_only=True)
    model=EsmForMaskedLM.from_pretrained(str(WEIGHTS),local_files_only=True).eval()
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    holder={}
    def hook(mod,args,out):
        hidden=out[0] if isinstance(out,(list,tuple)) else out
        holder['selected']=hidden[0,:,204].detach().cpu()
    h=model.esm.encoder.layer[4].register_forward_hook(hook)
    assert model.config.hidden_size==480 and model.config.num_hidden_layers==12
    with torch.inference_mode():
        for row in rows:
            seq=row['sequence'];batch=tok(seq,return_tensors='pt').to(device)
            assert batch.input_ids.shape==(1,len(seq)+2)
            result=model(**batch,output_hidden_states=True)
            expected=result.hidden_states[5][0,:,204].detach().cpu()
            assert torch.allclose(expected,holder['selected'],rtol=1e-6,atol=1e-7)
            v=holder.pop('selected');row['act']=float(v.max());row['residue_max']=float(v[1:-1].max())
            row['special_strict_max']=int(row['act']>row['residue_max'])
    h.remove();return {'device':device,'n':len(rows),'utc':datetime.now(timezone.utc).isoformat(),'model_revision':'6fbf070e65b0b7291e7bbcd451118c216cff79d8'}
def partial_ranks(mat,which):
    z=rankdata(mat,axis=0); ys=z[:,0];x=z[:,which];c=np.delete(z,[0,which],axis=1)
    c=np.column_stack((np.ones(z.shape[0]),c))
    res=np.column_stack([ys,x])-c@np.linalg.lstsq(c,np.column_stack([ys,x]),rcond=None)[0]
    return float(np.corrcoef(res.T)[0,1])
def run():
    rows=json.loads((HERE/'cohort.json').read_text())
    if not rows:raise RuntimeError('no eligible cohort')
    meta=forward(rows)
    rowpublic=[{k:v for k,v in r.items() if k!='sequence'} for r in rows]
    with (HERE/'per_group.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rowpublic[0]),delimiter='\t');w.writeheader();w.writerows(rowpublic)
    keys=['act','charge74','sheet_fraction','length','propensity']; mat=np.array([[r[k] for k in keys] for r in rows])
    rrmax=np.array([r['residue_max'] for r in rows]); ch70=np.array([r['charge70'] for r in rows]);fold=np.array([r['fold'] for r in rows])
    n=len(rows); rng=np.random.default_rng(2204204)
    boot=rng.integers(0,n,(5000,n))
    prs=[]
    for i in [1,2]:
        values=[partial_ranks(mat[ix],i) for ix in boot]
        prs.append(dict(measure=keys[i],conditional_spearman=partial_ranks(mat,i),simultaneous_97_5=np.quantile(values,[.0125,.9875]).tolist(),residue_only=partial_ranks(np.column_stack([rrmax,mat[:,1:]]),i),raw_spearman=float(spearmanr(mat[:,0],mat[:,i]).statistic)))
    y=mat[:,2];base=np.column_stack([np.log(mat[:,3]),mat[:,1],mat[:,4]])
    preds=np.full((n,2),np.nan);high=np.zeros(n,dtype=bool);thresholds=[]
    for k in range(5):
        train=fold!=k;test=~train;assert test.any()
        cut=np.quantile(mat[train,1:3],.75,axis=0);thresholds.append(cut.tolist());high[test]=np.all(mat[test,1:3]>=cut,axis=1)
        for j,xx in enumerate([base,np.column_stack([base,mat[:,0]])]):
            pred=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(xx[train],y[train]);preds[test,j]=pred.predict(xx[test])
    diff=np.abs(y-preds[:,0])-np.abs(y-preds[:,1])
    xy=np.mean(np.abs(y[:,None]-preds),axis=0)
    desc=dict(zip(['length','sheet_propensity','absolute_charge_74','signed_charge_70'],[float(spearmanr(mat[:,0],x).statistic) for x in (mat[:,3],mat[:,4],np.abs(mat[:,1]),ch70)]))
    hh=dict(n=int(high.sum()),other=n-int(high.sum()),fold_thresholds_charge_and_sheet=thresholds)
    if high.any() and (~high).any():
        dif=mat[high,0,None]-mat[~high,0][None,:]
        hh['prob_larger_than_other']=float(((dif>0)+0.5*(dif==0)).mean())
    with (HERE/'application_predictions.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['group','fold','observed_sheet','sequence_prediction','plus_204','MAE_gain','high_high'])
        for i,r in enumerate(rows):w.writerow([r['group'],fold[i],y[i],*preds[i],diff[i],int(high[i])])
    out=dict(**meta,primary=prs,descriptive_spearman=desc,primary_special_token_max=sum(r['special_strict_max'] for r in rows),fold_counts=[int((fold==k).sum()) for k in range(5)],sheet_MAE_sequence=float(xy[0]),sheet_MAE_plus204=float(xy[1]),MAE_gain=float(np.mean(diff)),conditional_95_MAE_gain=np.quantile(np.mean(diff[boot],axis=1),[.025,.975]).tolist(),high_high=hh)
    save(HERE/'RESULT.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':run()
