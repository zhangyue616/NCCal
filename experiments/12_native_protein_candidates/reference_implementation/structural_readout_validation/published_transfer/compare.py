"""Compare previously locked individual readouts on exactly the same natural H/E pairs."""
import hashlib
import json
import os
import sys
from pathlib import Path

for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[var]="1"
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,EsmForMaskedLM

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
PUBLISHED=ROOT/"native_channels/published"
FRESH=BASE.parent/"fresh"
sys.path.insert(0,str(PUBLISHED))
import run_unirep341 as uni


@torch.inference_mode()
def hallee(records):
    model_dir=ROOT/"assets/esm2_8m"
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    network=EsmForMaskedLM.from_pretrained(model_dir,local_files_only=True).eval()
    captured={}
    def positive(module,inputs,output):
        raw=output[:,:,786]
        if 'direct_error' not in captured:
            direct=inputs[0]@module.weight[786]+module.bias[786]
            captured['direct_error']=float((direct-raw).abs().max())
            assert captured['direct_error']<2e-5
        captured['values']=raw.clamp(min=0).numpy()
    hook=network.esm.encoder.layer[4].intermediate.dense.register_forward_hook(positive)
    values={}
    try:
        for start in range(0,len(records),6):
            items=records[start:start+6]
            batch=tokenizer([r['sequence'] for r in items],return_tensors='pt',padding=True)
            assert not torch.any(batch.input_ids==tokenizer.mask_token_id)
            network(**batch)
            for j,r in enumerate(items):
                n=len(r['sequence'])
                assert int(batch.attention_mask[j].sum())==n+2
                values[r['sequence_sha256']]=captured['values'][j,1:n+1].copy()
    finally:
        hook.remove()
    return values,captured['direct_error']


def main():
    torch.set_num_threads(3)
    reg=json.loads((BASE/'FREEZE.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((BASE/'PROTOCOL.md').read_bytes()).hexdigest()==reg['protocol_sha256'].lower()
    f_lock=json.loads((FRESH/'COHORT_FREEZE.json').read_text())
    for key in ('chains.jsonl','pairs.tsv'):
        assert uni.sha(FRESH/key)==f_lock['files'][key]
    pdb=pd.read_csv(FRESH/'pairs.tsv',sep='\t')
    ps=pdb.loc[pdb.task=='helix_vs_strand'].copy().sort_values('pair_id').reset_index(drop=True)
    assert len(ps)>0 and ps.pair_id.is_unique
    all_records=[json.loads(s) for s in (FRESH/'chains.jsonl').read_text(encoding='utf8').splitlines() if s]
    known={r['record_id']:r for r in all_records}
    for r in ps.itertuples():
        c=known[r.record_id]
        assert c['group']==r.group and c['sequence'][r.index_positive]==r.aa==c['sequence'][r.index_negative]
    lookup={known[r]['sequence_sha256']:known[r] for r in ps.record_id}
    records=sorted(lookup.values(),key=lambda r:(len(r['sequence']),r['sequence_sha256']))
    old_lock=json.loads((PUBLISHED/'HALLEE_HELIX_LOCK.json').read_text())['selected']
    assert (old_lock['layer'],old_lock['best_feature'],old_lock['arm'])==(5,786,'mlp_positive')
    assert uni.UNIT==341
    trained={'Alley_UniRep341':{},'Hallee_ESM2_786':{}}
    weights=uni.load_weights()
    for start in range(0,len(records),12):
        items=records[start:start+12]
        batch=uni.forward([r['sequence'] for r in items],weights)
        for r,v in zip(items,batch):
            assert v.shape==(len(r['sequence']),)
            trained['Alley_UniRep341'][r['sequence_sha256']]=v
    trained['Hallee_ESM2_786'],direct_error=hallee(records)
    with np.load(FRESH/'unit1900_per_native_sequence.npz') as prior:
        trained['E1_1900']={k:prior[k] for k in lookup}
    score_cols=[]
    for name,v in trained.items():
        assert all(np.isfinite(s).all() for s in v.values())
        d=np.asarray([float(v[r.sequence_sha256][r.index_positive])-float(v[r.sequence_sha256][r.index_negative]) for r in ps.itertuples()])
        ps[name+'_delta']=d
        ps[name]=(d>0).astype(float)+.5*(d==0)
        score_cols.append(name)
    groups=ps.groupby('group',sort=True)[score_cols].mean()
    groups['pairs']=ps.groupby('group',sort=True).size()
    summary=[]
    rng=np.random.default_rng(41624)
    idx=rng.integers(len(groups),size=(9999,len(groups)))
    for name in score_cols:
        stats=groups[name].to_numpy()
        c_low,c_high=np.quantile(stats[idx].mean(1),[.025,.975])
        summary.append(dict(readout=name,pairs=len(ps),groups=len(groups),wins=int((ps[name]==1).sum()),
            losses=int((ps[name]==0).sum()),ties=int((ps[name]==.5).sum()),accuracy=stats.mean(),low=c_low,high=c_high))
    contrasts=[]
    for name in score_cols[:2]:
        diff=(groups['E1_1900']-groups[name]).to_numpy()
        lo,hi=np.quantile(diff[idx].mean(1),[.0125,.9875])
        contrasts.append(dict(contrast='E1_1900 minus '+name,within_group_difference=diff.mean(),low_97p5=lo,high_97p5=hi))
    by_id=pd.read_csv(FRESH/'pair_results.tsv',sep='\t').set_index('pair_id').loc[ps.pair_id]
    assert np.array_equal(ps['E1_1900'].to_numpy(),by_id['win'].to_numpy())
    ps[['pair_id','group','record_id','sequence_sha256','aa','index_positive','index_negative']+
       [x for n in score_cols for x in (n,n+'_delta')]].to_csv(BASE/'F_pairs.tsv',sep='\t',index=False)
    groups.to_csv(BASE/'F_groups.tsv',sep='\t')
    for label,rows in [('F_readouts',summary),('F_paired_contrasts',contrasts)]:
        pd.DataFrame(rows).to_csv(BASE/(label+'.tsv'),sep='\t',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    receipt=dict(n_native_sequences=len(records),same_pair_ids=True,hallee_positive_pole_direct_max_abs_error=direct_error,
                 repeated_E1_pair_outcomes_identical=True,published_lock=reg,cohort_hashes=f_lock['files'])
    (BASE/'CHECK.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
