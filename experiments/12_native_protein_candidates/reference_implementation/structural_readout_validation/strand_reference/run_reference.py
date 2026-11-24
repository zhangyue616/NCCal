"""Compare two development-fitted pair predictors on the same deposited strands."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
LEGACY = STUDY.parent/'native_channels/cohort'
KD = dict(I=4.5, V=4.2, L=3.8, F=2.8, C=2.5, M=1.9, A=1.8, G=-.4,
          T=-.7, S=-.8, W=-.9, Y=-1.3, P=-1.6, H=-3.2, E=-3.5, Q=-3.5,
          D=-3.5, N=-3.5, K=-3.9, R=-4.5)
FIELDS = ['log1p_length','charged_fraction','polar_fraction','hydropathy','hydropathy_alternation','polar_alternation']


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file, delimiter='\t'))


def write_table(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def describe(seq):
    hyd=np.array([KD[a] for a in seq])
    polar=np.array([float(a in 'STNQYCDEKRH') for a in seq])
    charged=np.array([a in 'DEKR' for a in seq])
    return np.array([np.log1p(len(seq)),charged.mean(),polar.mean(),hyd.mean(),
                     abs(hyd[::2].mean()-hyd[1::2].mean()),
                     abs(polar[::2].mean()-polar[1::2].mean())])


def fit_model(x, groups):
    group_counts=Counter(groups)
    weights=np.array([1/group_counts[g] for g in groups])
    weights *= len(weights)/weights.sum()
    scale=np.sqrt(np.average(x*x, axis=0, weights=weights))
    scale[scale==0]=1
    x=x/scale
    lx=np.vstack([x,-x])
    ly=np.r_[np.ones(len(x)),np.zeros(len(x))]
    mdl=LogisticRegression(C=1.,fit_intercept=False,solver='lbfgs',max_iter=2000,tol=1e-10)
    mdl.fit(lx,ly,sample_weight=np.r_[weights,weights])
    return dict(beta=mdl.coef_[0].tolist(),scale=scale.tolist(),n=len(groups),groups=len(group_counts))


def probability(model,x):
    return expit((x/np.array(model['scale']))@np.array(model['beta']))


def p_flip(v,rng):
    var=v[v!=0]
    threshold=abs(var.sum())-1e-13
    if len(var)<=16:
        return sum(abs(np.dot(s,var))>=threshold for s in itertools.product((-1.,1.),repeat=len(var)))/(2**len(var))
    hits=0
    for start in range(0,99999,2000):
        signs=rng.choice((-1.,1.),size=(min(2000,99999-start),len(var)))
        hits+=int((abs(signs@var)>=threshold).sum())
    return (hits+1)/100000


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('legacy','fresh'))
    parser.add_argument('--models',nargs='+',default=['E1','ESMC'])
    parser.add_argument('--development-chains',type=Path,default=LEGACY/'chains.jsonl')
    parser.add_argument('--fresh-chains',type=Path,default=STUDY/'fresh/chains.jsonl')
    args=parser.parse_args()
    freeze=json.loads((HERE/'FREEZE.json').read_text(encoding='utf-8-sig'))
    assert freeze['sha256'].lower()==hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()
    chain_path=args.development_chains if args.stage=='legacy' else args.fresh_chains
    chains={r['record_id']:r for r in (json.loads(s) for s in chain_path.read_text(encoding='utf-8-sig').splitlines() if s.strip())}
    panels, pairs_out, groups_out = [], [], []
    for name in args.models:
        location=STUDY/'scan'/name/'trained'
        path=location/('signed_pairs.tsv' if args.stage=='legacy' else 'F_signed_pairs.tsv')
        rows=read_table(path)
        if not rows:
            print(f'No eligible {args.stage} pairs for {name}',flush=True)
            continue
        record_key='record_id' if 'record_id' in rows[0] else 'chain_id'
        vectors=[]
        for r in rows:
            seq=chains[r[record_key]]['sequence']
            ends=[]
            for role in ('positive','negative'):
                a,b=int(r[role+'_strand_start_index0']),int(r[role+'_strand_end_index0'])
                assert 0<=a<b<len(seq) and len(seq[a:b+1])>=3
                ends.append(describe(seq[a:b+1]))
            vectors.append(ends[0]-ends[1])
        xs=np.array(vectors)
        xx=np.column_stack([xs,[float(r['delta']) for r in rows]])
        assert np.isfinite(xx).all() and len({r['pair_id'] for r in rows})==len(rows)
        groups=np.array([r.get('group') or r['homology_group'] for r in rows])
        splits=np.array([r['split'] if args.stage=='legacy' else 'F' for r in rows])
        dic_file=HERE/f'{name}_development_models.json'
        if args.stage=='legacy':
            train=splits=='D'
            selected_lock=location/'selection_lock.json'
            fitted=dict(strand=fit_model(xs[train],groups[train]),combined=fit_model(xx[train],groups[train]),
                        original_channel_lock=hashlib.sha256(selected_lock.read_bytes()).hexdigest(),
                        development_cohort=hashlib.sha256(chain_path.read_bytes()).hexdigest(),
                        fields=FIELDS+['signed_neuron'])
            dic_file.write_text(json.dumps(fitted,indent=2),encoding='utf-8')
        else:
            fitted=json.loads(dic_file.read_text())
        probabilities={key:probability(fitted[key],mat) for key,mat in [('strand',xs),('combined',xx)]}
        for i,r in enumerate(rows):
            pr=dict(model=name,split=splits[i],group=groups[i],pair_id=r['pair_id'])
            for k,v in probabilities.items():
                p=float(v[i])
                pr[k+'_p']=p
                pr[k+'_brier']=(1-p)**2
                pr[k+'_correct']=float(p>.5)+.5*float(p==.5)
            pr['brier_gain']=pr['strand_brier']-pr['combined_brier']
            pairs_out.append(pr)
        for split in sorted(set(splits)):
            split_rows=[r for r in pairs_out if r['split']==split and r['model']==name]
            for group in sorted({r['group'] for r in split_rows}):
                bucket=[r for r in split_rows if r['group']==group]
                grp=dict(model=name,split=split,group=group,pairs=len(bucket))
                for field in ('strand_correct','combined_correct','strand_brier','combined_brier','brier_gain'):
                    grp[field]=np.mean([r[field] for r in bucket])
                groups_out.append(grp)
            group_rows=[r for r in groups_out if r['model']==name and r['split']==split]
            rng=np.random.default_rng(77179+sum(map(ord,split)))
            g=len(group_rows)
            idx=rng.integers(g,size=(9999,g))
            for field in ('strand_correct','combined_correct','strand_brier','combined_brier','brier_gain'):
                v=np.array([r[field] for r in group_rows])
                l,h=np.quantile(v[idx].mean(axis=1),[.025,.975]) if g>=2 else (None,None)
                get_p=field=='brier_gain' and split!='D' and g>=2
                panels.append(dict(split=split,model=name,metric=field,groups=g,pairs=len(split_rows),
                                   mean=v.mean(),low=l,high=h,p=p_flip(v,rng) if get_p else None,holm=None))
    for split in sorted({r['split'] for r in panels}):
        family=sorted((r for r in panels if r['split']==split and r['p'] is not None),key=lambda r:r['p'])
        maximum=0.
        for i,r in enumerate(family):
            maximum=max(maximum,(len(family)-i)*r['p'])
            r['holm']=min(1,maximum)
    for title,data in [('summary',panels),('pairs',pairs_out),('groups',groups_out)]:
        write_table(HERE/f'{args.stage}_{title}.tsv',data)
    print([r for r in panels if r['metric']=='brier_gain'])


if __name__=='__main__':
    main()
