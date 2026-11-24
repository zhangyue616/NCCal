"""Apply the two fixed original E1 channels to both helix questions on the same F proteins."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
F=HERE.parent/'fresh'


def signflip(v,rng):
    v=v[np.abs(v)>1e-12]
    cutoff=abs(v.sum())-1e-12
    if len(v)==0:
        return 1.
    if len(v)<=18:
        inds=np.arange(2**len(v))[:,None]
        signs=2*((inds>>np.arange(len(v)))&1)-1
        return float((np.abs(signs@v)>=cutoff).mean())
    hits=0
    for i in range(0,99999,2000):
        signs=rng.choice([-1,1],(min(2000,99999-i),len(v)))
        hits+=int((np.abs(signs@v)>=cutoff).sum())
    return (1+hits)/100000


def main():
    frozen=json.loads((HERE/'FREEZE.json').read_text(encoding='utf-8-sig'))
    assert not frozen['fresh_results_exist']
    assert hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()==frozen['protocol_sha256'].lower()
    ps=pd.read_csv(F/'pair_results.tsv',sep='\t',keep_default_na=False)
    strict=ps.same_helix_segment.astype(str).str.lower().eq('true')
    tests={'interior_vs_start':ps.loc[(ps.task=='helix_start_vs_internal')&strict].copy(),
           'helix_vs_strand':ps.loc[ps.task=='helix_vs_strand'].copy()}
    common=set.intersection(*(set(p.group) for p in tests.values()))
    if not common:
        (HERE/'F_status.json').write_text(json.dumps({'groups':0,'status':'no common eligible groups'}))
        return
    tables=[]
    for task,sub in tests.items():
        sub=sub.loc[sub.group.isin(common),['record_id','pair_id','group','sequence_sha256','index_positive','index_negative','win']].copy()
        sub['question']=task
        direction=-1 if task=='interior_vs_start' else 1
        for unit in (2192,1900):
            with np.load(F/f'unit{unit}_per_native_sequence.npz') as files:
                cache={key:files[key] for key in sub.sequence_sha256.unique()}
            values=[]
            for pair in sub.itertuples():
                x=cache[pair.sequence_sha256]
                values.append(direction*(float(x[pair.index_positive])-float(x[pair.index_negative])))
            values=np.array(values)
            sub[f'unit{unit}']=(values>0).astype(float)+.5*(values==0)
        assert (sub['unit2192' if direction==-1 else 'unit1900']==sub.win).all()
        tables.append(sub)
    all_scores=pd.concat(tables,ignore_index=True)
    all_scores.to_csv(HERE/'F_pairs.tsv',sep='\t',index=False)
    rates=all_scores.groupby(['group','question'])[['unit2192','unit1900']].mean().unstack('question')
    rate_counts=all_scores.groupby('question').size()
    n=len(rates)
    rng=np.random.default_rng(91220)
    draw=rng.integers(n,size=(9999,n))
    results=[]
    for key in rates.columns:
        v=rates[key].to_numpy()
        lo,hi=np.quantile(v[draw].mean(axis=1),[.025,.975])
        results.append(dict(metric='rate',question=key[1],unit=key[0],groups=n,pairs=int(rate_counts[key[1]]),
            mean=v.mean(),ci_low=lo,ci_high=hi,p=None,holm=None,informative_groups=None))
    for task,main_unit,other in [('interior_vs_start','unit2192','unit1900'),('helix_vs_strand','unit1900','unit2192')]:
        v=(rates[main_unit,task]-rates[other,task]).to_numpy()
        lo,hi=np.quantile(v[draw].mean(axis=1),[.025,.975])
        results.append(dict(metric='preassigned_advantage',question=task,unit=f'{main_unit} minus {other}',
            groups=n,pairs=int(rate_counts[task]),mean=v.mean(),ci_low=lo,ci_high=hi,p=signflip(v,rng),
            holm=None,informative_groups=int((np.abs(v)>1e-12).sum())))
    family=sorted(results[-2:],key=lambda row:row['p'])
    family[0]['holm']=min(1,2*family[0]['p'])
    family[1]['holm']=max(family[0]['holm'],family[1]['p'])
    rates.columns=['_'.join(col) for col in rates.columns]
    rates.to_csv(HERE/'F_groups.tsv',sep='\t')
    out=pd.DataFrame(results)
    out.to_csv(HERE/'F_summary.tsv',sep='\t',index=False)
    print(out.to_string(index=False))

if __name__=='__main__':
    main()
