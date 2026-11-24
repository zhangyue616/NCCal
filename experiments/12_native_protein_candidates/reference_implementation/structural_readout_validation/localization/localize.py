"""Freeze five choices per window, calibrate on discovery proteins, test a fixed E1 channel."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

BASE = Path(__file__).resolve().parent
OLD = BASE.parent.parent / 'native_channels'
AA = 'ACDEFGHIKLMNPQRSTVWY'
SEED = 21921503


def digest(text):
    return hashlib.sha256(text.encode('ascii')).hexdigest()


def load_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def save_tsv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def yes(value):
    return str(value).lower() in ('true', '1')


def build(panel, prefix):
    chains = {r['record_id']: r for r in (
        json.loads(line) for line in (panel / 'chains.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()
    )}
    segments = load_tsv(panel / 'helix_segments.tsv')
    sites = defaultdict(set)
    for r in load_tsv(panel / 'residue_annotations.tsv'):
        if yes(r['backbone_quality']) and yes(r['local_backbone_continuous']):
            sites[r['record_id']].add(int(r['seq_index0']))
    starts = defaultdict(set)
    for seg in segments:
        starts[seg['record_id']].add(int(seg['start_index0']))
    by_window = defaultdict(list)
    funnel = Counter()
    for seg in segments:
        if not yes(seg['eligible_long_helix']):
            continue
        funnel['long_complete_helix_records'] += 1
        ch = chains[seg['record_id']]
        seq = ch['sequence']
        key = ch['sequence_sha256']
        assert digest(seq) == key
        true = int(seg['start_index0'])
        phase = int(digest('NCCal-five-site-v1|' + key)[:8], 16) % 5
        left = true - ((true - phase) % 5)
        if left < 4 or left + 9 >= len(seq) or set(seq[left-4:left+10]) - set(AA):
            funnel['sequence_bounds_or_noncanonical'] += 1
            continue
        if len(starts[ch['record_id']].intersection(range(left, left+5))) != 1:
            funnel['multiple_starts_in_window'] += 1
            continue
        if not set(range(left, left+10)) <= sites[ch['record_id']]:
            funnel['incomplete_local_backbone'] += 1
            continue
        split = ch.get('split', 'F') if prefix == 'legacy' else 'F'
        by_window[(key, left)].append(dict(window=digest(f'{key}|{left}'), split=split,
            group=ch.get('group') or ch['homology_group'], record=ch['record_id'],
            seqhash=key, left=left, true=true, offset=true-left, segment=seg['segment_id']))
    selected = []
    for duplicates in by_window.values():
        if len({r['true'] for r in duplicates}) > 1:
            funnel['conflicting_native_window'] += 1
            continue
        assert len({(r['split'],r['group']) for r in duplicates}) == 1
        selected.append(min(duplicates, key=lambda r:r['record']))
        funnel['duplicate_depositions'] += len(duplicates)-1
    counts = Counter()
    final = []
    for r in sorted(selected, key=lambda r:r['window']):
        if counts[r['group']] == 12:
            funnel['group_cap'] += 1
            continue
        final.append(r)
        counts[r['group']] += 1
    if not final:
        raise RuntimeError(f'No eligible windows in {panel}; exclusions {dict(funnel)}')
    groups_by_split = {s: {r['group'] for r in final if r['split'] == s} for s in {r['split'] for r in final}}
    assert all(a.isdisjoint(b) for a,b in itertools.combinations(groups_by_split.values(),2))
    windows_path = BASE / f'{prefix}_windows.tsv'
    save_tsv(windows_path, final)
    save_json(BASE/f'{prefix}_window_freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        inputs=str(panel),selection_sha256=hashlib.sha256(windows_path.read_bytes()).hexdigest(),funnel=dict(funnel),
        samples={s: {'groups':len(g),'windows':sum(r['split']==s for r in final)} for s,g in groups_by_split.items()}))
    return chains, final


def activations(chains, rows, prefix, external_cache):
    wanted = {r['seqhash'] for r in rows}
    values = {}
    paths = [OLD/'terminal_specificity/unit2192_per_native_sequence.npz']
    if external_cache:
        paths.append(external_cache)
    own = BASE/f'{prefix}_additional_2192.npz'
    paths.append(own)
    for path in paths:
        if path.exists():
            with np.load(path) as cache:
                for key in wanted.intersection(cache.files):
                    values[key] = cache[key]
    missing = wanted - values.keys()
    by_hash = {c['sequence_sha256']:c['sequence'] for c in chains.values()}
    if missing:
        import torch
        sys.path.insert(0, str(OLD/'e1'))
        import e1_runtime as rt
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3*1024**3 else 'cpu'
        model = rt.get_model(trained=True, device=device)
        capture = {}
        def keep(_, args):
            capture['values'] = args[0][0,2:-2,2192].detach().cpu().numpy().copy()
        hook = model.model.layers[14].ffn.mlp.w2.register_forward_pre_hook(keep)
        for key in sorted(missing):
            with torch.inference_mode():
                model.model(**rt.get_batch(by_hash[key],device=device))
            values[key] = capture.pop('values')
        hook.remove()
        if own.exists():
            with np.load(own) as previous:
                retained = {k:previous[k] for k in previous.files}
        else:
            retained = {}
        retained.update({key:values[key] for key in missing})
        np.savez_compressed(own, **retained)
    for key,v in values.items():
        assert v.ndim == 1 and len(v) == len(by_hash[key]) and np.isfinite(v).all()
    return values, by_hash


def features(rows, values, seqs):
    seq = np.zeros((len(rows), 5, len(AA)*9), dtype=np.float64)
    neuron = np.empty((len(rows), 5), dtype=np.float64)
    for i,r in enumerate(rows):
        native, a = seqs[r['seqhash']], values[r['seqhash']].astype(np.float64)
        for j in range(5):
            k = r['left'] + j
            neuron[i,j] = a[k+3:k+6].mean() - a[k:k+2].mean()
            for pos,letter in enumerate(native[k-4:k+5]):
                seq[i,j, pos*len(AA)+AA.index(letter)] = 1.
    return seq, neuron


def conditional_fit(x, target, weights):
    def value_grad(beta):
        logits = x @ beta
        loss = np.dot(weights,logsumexp(logits,axis=1)-logits[np.arange(len(target)),target]) + .5*np.dot(beta,beta)
        residual = softmax(logits,axis=1)
        residual[np.arange(len(target)),target] -= 1.
        grad = np.einsum('ij,ijk,i->k',residual,x,weights,optimize=True) + beta
        return loss,grad
    opt = minimize(value_grad, np.zeros(x.shape[2]),jac=True,method='L-BFGS-B',
                   options={'maxiter':1500,'ftol':1e-13,'gtol':1e-8})
    if not opt.success or np.max(np.abs(opt.jac)) > 2e-4:
        raise ValueError(str(opt.message)+f' residual {np.max(np.abs(opt.jac))}')
    return opt.x


def fit(x, n, rows):
    ids = [i for i,r in enumerate(rows) if r['split'] == 'D']
    groups = Counter(rows[i]['group'] for i in ids)
    weights = np.array([1/groups[rows[i]['group']] for i in ids])
    target = np.array([rows[i]['offset'] for i in ids])
    mean, sd = n[ids].mean(), n[ids].std()
    assert sd > 0 and abs(sum(weights)-len(groups)) < 1e-8
    xx = np.concatenate((x[ids], ((n[ids]-mean)/sd)[:,:,None]),axis=2)
    coefficients = dict(seq=conditional_fit(x[ids],target,weights), plus=conditional_fit(xx,target,weights),
                        mean=mean,sd=sd)
    np.savez(BASE/'development_model.npz', **coefficients)
    save_json(BASE/'model_lock.json',dict(utc=datetime.now(timezone.utc).isoformat(),groups=len(groups),
        windows=len(ids),raw_rule='mean(a[k+3:k+6])-mean(a[k:k+2])',ridge=1,
        arrays_sha256=hashlib.sha256((BASE/'development_model.npz').read_bytes()).hexdigest()))
    return coefficients


def summarize(rows, x, n, coef, prefix):
    plus = np.concatenate((x, ((n-coef['mean'])/coef['sd'])[:,:,None]),axis=2)
    seq_log = x @ coef['seq']
    comb_log = plus @ coef['plus']
    per_window, per_group, summaries = [],[],[]
    for i,row in enumerate(rows):
        target = row['offset']
        out = dict(row)
        for name,z in [('uniform',np.zeros(5)),('center',-np.abs(np.arange(5)-2)),('neuron',n[i]),
                       ('sequence',seq_log[i]),('combined',comb_log[i])]:
            best = np.flatnonzero(z==z.max())
            error = np.abs(best-target)
            out[name+'_correct'] = np.mean(error==0)
            out[name+'_within1'] = np.mean(error<=1)
            out[name+'_abs_error'] = error.mean()
            if name in ('sequence','combined'):
                out[name+'_nll'] = logsumexp(z) - z[target]
                out[name+'_probabilities'] = ','.join(format(v,'.12g') for v in softmax(z))
        out['raw_scores'] = ','.join(format(v,'.12g') for v in n[i])
        out['nll_gain'] = out['sequence_nll'] - out['combined_nll']
        per_window.append(out)
    keys=[k for k in per_window[0] if k.endswith(('_correct','_within1','_abs_error','_nll'))]+['nll_gain']
    for group in sorted({r['group'] for r in per_window}):
        bucket=[r for r in per_window if r['group']==group]
        per_group.append(dict(group=group,split=bucket[0]['split'],windows=len(bucket),
                              **{k:np.mean([r[k] for r in bucket]) for k in keys}))
    for split in sorted({r['split'] for r in per_group}):
        groups=[r for r in per_group if r['split']==split]
        w=[r for r in per_window if r['split']==split]
        count=len(groups)
        rng=np.random.default_rng(SEED+ord(split[0]))
        m=np.array([[r[k] for k in keys] for r in groups])
        idx=rng.integers(count,size=(9999,count))
        for j,k in enumerate(keys):
            v=m[:,j]
            ci=np.quantile(v[idx].mean(axis=1),[.025,.975]) if count>=2 else [None,None]
            data=dict(split=split,measure=k,groups=count,windows=len(w),mean=v.mean(),ci_low=ci[0],ci_high=ci[1],
                      window_mean=np.mean([r[k] for r in w]),two_sided_primary_p=None,
                      true_offsets=','.join(str(sum(r['offset']==a for r in w)) for a in range(5)))
            if k=='nll_gain' and count>=2:
                nonzero=v[v!=0]
                threshold=abs(nonzero.sum())-1e-12
                if len(nonzero)<=16:
                    tails=sum(abs(np.dot(s,nonzero))>=threshold for s in itertools.product((-1.,1.),repeat=len(nonzero)))
                    data['two_sided_primary_p']=tails/(2**len(nonzero))
                else:
                    hits=0
                    for start in range(0,99999,4000):
                        signs=rng.choice((-1.,1.),size=(min(4000,99999-start),len(nonzero)))
                        hits+=int((np.abs(signs@nonzero)>=threshold).sum())
                    data['two_sided_primary_p']=(1+hits)/100000
            summaries.append(data)
    save_tsv(BASE/f'{prefix}_by_window.tsv',per_window)
    save_tsv(BASE/f'{prefix}_by_group.tsv',per_group)
    save_tsv(BASE/f'{prefix}_summary.tsv',summaries)
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prefix',choices=['legacy','fresh'])
    parser.add_argument('panel',type=Path)
    parser.add_argument('--fresh-values',type=Path)
    args=parser.parse_args()
    lock=json.loads((BASE.parent/'DESIGN_FREEZE.json').read_text(encoding='utf-8-sig'))
    assert lock['localization'].lower()==hashlib.sha256((BASE/'PROTOCOL.md').read_bytes()).hexdigest()
    chains,rows=build(args.panel,args.prefix)
    values,seqs=activations(chains,rows,args.prefix,args.fresh_values)
    x,n=features(rows,values,seqs)
    if args.prefix=='legacy':
        coeff=fit(x,n,rows)
    else:
        with np.load(BASE/'development_model.npz') as f:
            coeff=dict(f)
    summary=summarize(rows,x,n,coeff,args.prefix)
    for r in summary:
        if r['measure'] in ('neuron_correct','sequence_correct','combined_correct','nll_gain'):
            print(r,flush=True)


if __name__=='__main__':
    main()
