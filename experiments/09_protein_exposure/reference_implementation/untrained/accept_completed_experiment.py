"""One bounded final acceptance of the integrated frozen experiment."""
import sys
sys.dont_write_bytecode=True
import csv
import run_experiment as r
import numpy as np

assert r.read(r.OUT/'STATUS.json')['status']=='COMPLETE'
K=r.read(r.OUT/'TIMING.json')['chosen_K'];checks=[]
freeze=r.read(r.OUT/'FREEZE.json')
assert all(r.sha(p)==h for p,h in freeze['files'].items())
checks.append('Frozen protocol, runner and registered scientific inputs unchanged')
aggregate=r.read(r.OUT/'AGGREGATE.json')
for model,width in (('8m',1280),('35m',1920)):
    states=set();data=[]
    for i in range(1,K+1):
        folder=r.OUT/model/f'init_{i:03d}';d=r.read(folder/'RESULT.json');ident=r.read(folder/'MODEL_IDENTITY.json')
        assert d['index']==i and d['seed']==20261118000+i==ident['seed']
        assert 0<=d['selection']['axis_zero_based']<width
        assert d['selection']['axis_candidates']==width
        assert (d['T_groups'],d['T_residues'],d['T_pairs'])==(1099,173859,967)
        assert d['pair_correct']+d['pair_incorrect']+d['pair_ties']==967
        assert d['selection']['development_data_signature']=='9df331898ba9726f1da7b8b3304fe490f6af541d23619ac3af7f30289716c25c'
        states.add(ident['state_sha256']);data.append(d)
    assert len(states)==K
    for endpoint in ('mae_gain','brier_gain'):
        row=next(x for x in aggregate['inference'] if x['model']==model and x['endpoint']==endpoint)
        n=sum(d[endpoint]>=row['trained_gain'] for d in data)
        assert row['untrained_ge']==n and row['p_raw']==(1+n)/(K+1)
    folder=r.OUT/model/f'init_{K:03d}';d=data[-1]
    with (folder/'group_errors.tsv').open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==1099 and abs(np.mean([float(x['sequence_mae'])-float(x['augmented_mae']) for x in rows])-d['mae_gain'])<1e-12
    with (folder/'buried_exposed_group_errors.tsv').open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==967 and abs(np.mean([float(x['sequence_brier'])-float(x['augmented_brier']) for x in rows])-d['brier_gain'])<1e-12
    correct=sum((float(x['augmented_probability'])>.5 and float(x['target'])==1) or (float(x['augmented_probability'])<.5 and float(x['target'])==0) for x in rows)
    assert correct==d['pair_correct']
    checks.append(f'{model}: all {K} seeds, architecture domains, denominators and independent state identities; endpoint tail counts checked; final network group-loss and correct-count recomputation')
fam=sorted((x for x in aggregate['inference'] if x['model']=='35m'),key=lambda x:x['p_raw'])
assert fam[0]['p_holm']==min(1.,2*fam[0]['p_raw'])
assert fam[1]['p_holm']==max(fam[0]['p_holm'],fam[1]['p_raw'])
checks.append('Holm family contains exactly the two 35M endpoints; 8M remains descriptive')
r.write(r.OUT/'ACCEPTANCE.json',dict(status='PASS',completed_utc=r.utc(),K=K,network_count=2*K,checks=checks,
    limitations=['No manuscript or scientific promotion decision is made here','Forward extraction cross-check was bounded to 8 eligible proteins per model/split; full endpoint replay used original activation caches']))
r.shutil.copy2(r.OUT/'ACCEPTANCE.json',r.AUTHOR/'ACCEPTANCE.json')
print('Final integrated acceptance PASS',flush=True)
