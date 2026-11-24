"""Frozen alanine readout transfer; all new files stay beside this script."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys
sys.dont_write_bytecode=True
import csv, hashlib, importlib.util, json, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EXT=ROOT/'cross_architecture'
MODEL=ROOT/'assets/esm2_8m'
FIT=ROOT/'results/revision/protein_D/FINAL_MODELS.npz'
SEED=20261122
ARMS=('X','Xj','L','Lj')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write(name,obj):
    p=OUT/name
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
KRR=module('frozen_krr',ROOT/'scripts/revision/protein_krr.py')

def prepare():
    if (OUT/'BINDINGS.json').exists(): raise RuntimeError('Bindings already exist; do not silently regenerate')
    paths=[OUT/'PROTOCOL.md',Path(__file__),FIT,MODEL/'config.json',MODEL/'model.safetensors',
        MODEL/'vocab.txt',MODEL/'tokenizer_config.json',MODEL/'special_tokens_map.json',
        ROOT/'scripts/revision/protein_krr.py',
        ROOT/'protein_extension/calibration/run_calibration.py',
        EXT/'external/sites.tsv',EXT/'external/sequences.jsonl',EXT/'validation/HOMOLOGY.json',
        ROOT/'data/revision/protein_fresh/SEALED_FRESH_CASES.json',
        ROOT/'results/revision/protein_T/T_NATIVE.npz',
        ROOT/'results/revision/protein_T/T_PREDICTIONS.npz']
    identities=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
    assert sha(FIT)=='f8dc035e77a0535832127f1946bc202bb0e7d42a0e4443f34144c025a3980d0b'
    write('BINDINGS.json',{'created_utc':utc(),'status':'BOUND_BEFORE_NEW_EXTERNAL_PREDICTIONS',
        'identities':identities,'new_external_predictions_at_binding':0,'refits_authorized':False,
        'device':'cpu','threads':2,'bootstrap_draws':10000,'seed':SEED,
        'scoring_direction':'earlier minus later; y_a=1 exactly when earlier is H; p_a=clip((1+score)/2,0,1)',
        'length_rule':'canonical full sequences 2..1024, satisfying saved config token capacity; no crops'})
    seqs=[json.loads(s) for s in (EXT/'external/sequences.jsonl').read_text().splitlines() if s.strip()]
    rows=list(csv.DictReader((EXT/'external/sites.tsv').open(newline='',encoding='utf-8-sig'),delimiter='\t'))
    strict=set(read(EXT/'validation/HOMOLOGY.json')['retained_groups'])
    assert len({s['group'] for s in seqs})==len(seqs)
    assert strict <= {s['group'] for s in seqs}
    bygroup=defaultdict(list)
    for row in rows:
        assert row['main_clean']=='True' and row['interface']=='False'
        assert row['secondary']==row['P_SEA'] and row['aa']==row['AA'] and row['pos']==row['position']
        bygroup[row['group']].append(row)
    selected=[]; audit=[]; cases=[]
    config=read(MODEL/'config.json')
    for s in sorted(seqs,key=lambda x:x['group']):
        group,seq=s['group'],s['sequence']; rr=bygroup[group]
        assert all(r['accession']==s['accession'] and r['chain']==s['chain'] and r['pdb_id']==s['pdb_id'] for r in rr)
        assert len({int(r['pos']) for r in rr})==len(rr)
        for r in rr: assert seq[int(r['pos'])-1]==r['aa']
        reason=None
        if set(seq)-set(KRR.AA): reason='noncanonical_sequence'
        elif not 2<=len(seq)<=config['max_position_embeddings']-2: reason='full_sequence_exceeds_saved_token_capacity'
        for aa in 'ALV':
            hs=[r for r in rr if r['aa']==aa and r['secondary']=='H']
            es=[r for r in rr if r['aa']==aa and r['secondary']=='E']
            candidates=[]
            if reason is None:
                for h in hs:
                    for e in es:
                        early,late=sorted((int(h['pos']),int(e['pos'])))
                        if late-early>30:
                            digest=hashlib.sha256(f'NCCal-HS-transfer-v1|{group}|{aa}|{early}|{late}'.encode()).hexdigest()
                            candidates.append((digest,early,late))
            audit.append({'group':group,'accession':s['accession'],'identity':aa,'strict_retained':group in strict,
                'native_length':len(seq),'H_sites':len(hs),'E_sites':len(es),'eligible_pairs':len(candidates),
                'exclusion':reason or (None if candidates else 'no_same_identity_H_E_pair_separation_gt30')})
            if not candidates: continue
            digest,early,late=min(candidates)
            lookup={int(r['pos']):r for r in rr}; er,lr=lookup[early],lookup[late]
            assert er['chain']==lr['chain'] and er['aa']==lr['aa']==aa and {er['secondary'],lr['secondary']}=={'H','E'}
            start=len(cases)
            pair={'group':group,'id':s['accession'],'identity':aa,'strict_retained':group in strict,
                'pdb_id':s['pdb_id'],'chain':s['chain'],'early':early,'late':late,'native_length':len(seq),
                'early_secondary':er['secondary'],'late_secondary':lr['secondary'],
                'y_a':int(er['secondary']=='H'),'selection_sha256':digest,'eligible_pairs':len(candidates),
                'a_index':start,'b_index':start+1,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest()}
            for pos,row in ((early,er),(late,lr)):
                cases.append({'id':s['accession'],'group':group,'sequence':seq,'q':pos-1,'aa':aa,
                    'case_id':f"{s['accession']}:{aa}:{pos}",'y':int(row['secondary']=='H')})
            selected.append(pair)
    assert all(p['late']-p['early']>30 for p in selected)
    counts={frame:{aa:sum(p['identity']==aa and (frame=='structure_cluster' or p['strict_retained']) for p in selected) for aa in 'ALV'}
            for frame in ('direct_sequence_screen','structure_cluster')}
    write('PAIR_SELECTION.json',{'created_utc':utc(),'pairs':selected,'cohort_groups':len(seqs),
        'strict_retained_groups':len(strict),'counts':counts,'audit':audit,'site_rows':len(rows),
        'excluded_sequence_count':sum(bool(set(s['sequence'])-set(KRR.AA)) or not 2<=len(s['sequence'])<=config['max_position_embeddings']-2 for s in seqs),
        'unique_external_forward_cases':len(cases),'new_predictions':0})
    # No sequences are copied into outputs: reconstruct them only in memory.
    print(json.dumps({'phase':'selected_before_scoring','counts':counts,'forward_cases':len(cases)}),flush=True)
    return selected,cases

def reconstruct():
    selected=read(OUT/'PAIR_SELECTION.json')['pairs']
    ss={s['accession']:s['sequence'] for s in [json.loads(x) for x in (EXT/'external/sequences.jsonl').read_text().splitlines() if x.strip()]}
    cases=[]
    for p in selected:
        seq=ss[p['id']];assert hashlib.sha256(seq.encode()).hexdigest()==p['sequence_sha256']
        for pos,y in ((p['early'],p['y_a']),(p['late'],1-p['y_a'])):
            assert seq[pos-1]==p['identity']
            cases.append({'id':p['id'],'group':p['group'],'sequence':seq,'q':pos-1,'aa':p['identity'],
                'case_id':f"{p['id']}:{p['identity']}:{pos}",'y':y})
    return selected,cases

def load_forward():
    import torch,transformers
    from transformers import AutoTokenizer,EsmForMaskedLM
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    model=EsmForMaskedLM.from_pretrained(MODEL,local_files_only=True).float().eval()
    assert (model.config.hidden_size,model.config.intermediate_size,model.config.num_hidden_layers,model.config.hidden_act)==(320,1280,6,'gelu')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model.esm.encoder.layer=torch.nn.ModuleList(list(model.esm.encoder.layer)[:3])
    captured={}
    def hook(_m,_i,out): captured['j']=float(out[0,captured['q'],1246].detach().cpu())
    handle=model.esm.encoder.layer[2].intermediate.register_forward_hook(hook)
    def forward(c):
        seq,q=c['sequence'],int(c['q']);assert seq[q]==c['aa']
        inputs=tokenizer(seq,return_tensors='pt',padding=False,truncation=False)
        assert inputs['input_ids'].shape==(1,len(seq)+2)
        assert int(inputs['attention_mask'].sum())==len(seq)+2
        assert inputs['input_ids'][0,q+1].item()==tokenizer.convert_tokens_to_ids(seq[q])
        original=inputs['input_ids'].clone();inputs['input_ids'][0,q+1]=tokenizer.mask_token_id
        assert int((inputs['input_ids']!=original).sum())==1
        assert int((inputs['input_ids']==tokenizer.mask_token_id).sum())==1
        captured['q']=q+1
        with torch.inference_mode(): model.esm(**inputs)
        value=captured.pop('j');assert np.isfinite(value)
        return value
    write('MODEL_IDENTITY.json',{'created_utc':utc(),'torch':torch.__version__,'transformers':transformers.__version__,
        'device':'cpu','threads':2,'precision':'float32','model_weight_sha256':sha(MODEL/'model.safetensors'),
        'layer':'block3 (0-based2) intermediate post-GELU','unit':1246,'forward':'one masked centre per independent full-length sequence; first three blocks evaluated; no crops'})
    return forward,handle

def replay(forward):
    t=read(ROOT/'data/revision/protein_fresh/SEALED_FRESH_CASES.json')
    indices=sorted(range(len(t['pairs'])),key=lambda i:hashlib.sha256(('transfer-replay-v1|'+t['pairs'][i]['group']).encode()).hexdigest())[:3]
    pp=[];cc=[];oldi=[]
    for idx in indices:
        p=t['pairs'][idx];assert t['cases'][p['a_index']]['q']<t['cases'][p['b_index']]['q']
        pp.append(dict(p,a_index=len(cc),b_index=len(cc)+1));oldi.append(idx)
        for field in ('a_index','b_index'):
            c=dict(t['cases'][p[field]],aa='A');assert c['sequence'][c['q']]=='A';cc.append(c)
    with np.load(ROOT/'results/revision/protein_T/T_NATIVE.npz') as a:
        lookup={s:float(v) for s,v in zip(a['case_ids'].astype(str),a['z'][:,1246])}
    oldj=np.array([lookup[c['case_id']] for c in cc]);newj=np.array([forward(c) for c in cc])
    diff=float(np.max(np.abs(oldj-newj)));assert diff<1e-5,(oldj,newj,diff)
    archive=np.load(ROOT/'results/revision/protein_T/T_PREDICTIONS.npz')
    dataold=KRR.pair_data(cc,pp,oldj);datanew=KRR.pair_data(cc,pp,newj)
    results={}
    for m in ARMS:
        so,po=KRR.predict_saved(dataold,m,FIT);sn,pn=KRR.predict_saved(datanew,m,FIT)
        err=float(np.max(np.abs(so-archive[m+'_score'][oldi])))
        assert err<1e-10
        assert np.max(np.abs(sn-so))<1e-4
        results[m]={'archived_score_max_error_using_archived_j':err,'fresh_vs_archived_probability_max_error':float(np.max(np.abs(pn-po))),
            'archived_score':so.tolist(),'fresh_score':sn.tolist(),'archived_p_a':po.tolist(),'fresh_p_a':pn.tolist()}
    write('REPLAY.json',{'status':'PASS','created_utc':utc(),'historical_pairs':3,'historical_forward_cases':6,
        'old_pair_indices':indices,'case_ids':[c['case_id'] for c in cc],'old_j':oldj.tolist(),'fresh_j':newj.tolist(),
        'j_max_abs_error':diff,'arms':results,'new_fits':0})
    print(json.dumps({'phase':'replay_PASS','j_max_abs_error':diff,'historical_pairs':3}),flush=True)

def summarize(pairs,j):
    _,cases=reconstruct();data=KRR.pair_data(cases,pairs,j);y=np.array([p['y_a'] for p in pairs])
    predictions={}
    rows=[]
    for m in ARMS:
        s,p=KRR.predict_saved(data,m,FIT);loss=(p-y)**2
        acc=np.where(p==.5,.5,(p>.5)==y).astype(float)
        assert np.array_equal(p,KRR.probability(s))
        predictions[m]={'score':s,'p_a':p,'loss':loss,'accuracy':acc}
    # Frozen directional algebra: swapped paired features reverse scores and complement probabilities.
    swapped={'dx':-data['dx'],'pa':data['pb'],'pb':data['pa'],'dj':-data['dj']}
    direction_errors={m:float(np.max(np.abs(KRR.predict_saved(swapped,m,FIT)[0]+predictions[m]['score']))) for m in ARMS}
    assert max(direction_errors.values())<1e-10
    for i,p in enumerate(pairs):
        row={k:v for k,v in p.items() if k not in ('a_index','b_index')}
        row.update(j_early=float(j[2*i]),j_late=float(j[2*i+1]),dj_early_minus_late=float(j[2*i]-j[2*i+1]))
        for m in ARMS:
            for k,v in predictions[m].items():row[m+'_'+k]=float(v[i])
        row['gain_L_minus_Lj']=row['L_loss']-row['Lj_loss'];row['gain_X_minus_Xj']=row['X_loss']-row['Xj_loss']
        rows.append(row)
    with (OUT/'PAIRED_SCORES.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    summary=[]
    for frame in ('direct_sequence_screen','structure_cluster'):
        for aa in 'ALV':
            idx=np.array([i for i,p in enumerate(pairs) if p['identity']==aa and (frame=='structure_cluster' or p['strict_retained'])],dtype=int)
            n=len(idx);groups=[pairs[i]['group'] for i in idx];assert len(set(groups))==n
            r={'frame':frame,'identity':aa,'n_pairs':n,'n_groups':n,'sites':2*n,'arms':{},'comparisons':{}}
            if not n:r['status']='UNDEFINED_NO_ELIGIBLE_PAIR';summary.append(r);continue
            rng=np.random.default_rng(SEED);boot=rng.integers(0,n,size=(10000,n))
            def intervals(v):
                means=np.asarray(v)[boot].mean(axis=1)
                return {'mean':float(np.mean(v)),'CI95_percentile':np.quantile(means,[.025,.975]).tolist(),
                    **({'CI97_5_simultaneous_percentile':np.quantile(means,[.0125,.9875]).tolist()} if frame=='direct_sequence_screen' and aa in 'LV' else {})}
            for m in ARMS:
                pred=predictions[m];p=pred['p_a'][idx];s=pred['score'][idx]
                r['arms'][m]={'Brier':intervals(pred['loss'][idx]),'conditional_pair_accuracy':intervals(pred['accuracy'][idx]),
                    'correct_non_tied':int(np.sum(((p>.5)==y[idx]) & (p!=.5))),'incorrect_non_tied':int(np.sum(((p>.5)!=y[idx]) & (p!=.5))),
                    'probability_ties':int(np.sum(p==.5)),'score_ties':int(np.sum(s==0))}
            for b,a in (('L','Lj'),('X','Xj')):
                gain=predictions[b]['loss'][idx]-predictions[a]['loss'][idx]
                z=intervals(gain);z['positive_pairs']=int(np.sum(gain>0));z['negative_pairs']=int(np.sum(gain<0));z['exact_loss_ties']=int(np.sum(gain==0))
                z['uncertainty']='interval_spans_zero' if z.get('CI97_5_simultaneous_percentile',z['CI95_percentile'])[0]<=0<=z.get('CI97_5_simultaneous_percentile',z['CI95_percentile'])[1] else 'interval_excludes_zero'
                r['comparisons'][b+'_minus_'+a]=z
            r['status']='COMPLETE';summary.append(r)
    write('SUMMARY.json',{'created_utc':utc(),'status':'COMPLETE_FROZEN_READOUT_NO_REFIT','results':summary,
        'unit':1246,'layer':'block3.post-GELU','model':'ESM2 8M','bootstrap':{'draws':10000,'seed':SEED,'unit':'independent group, paired shared resampling indices within each identity/frame',
        'primary_family':'L vs Lj for L and V, no pooling','simultaneous_level':.975},
        'direction_errors':direction_errors,'scope':'Observational structure-conditioned paired discrimination, conditional on saved fits and archived cohort; no sequence or function optimization',
        'annotation_shift':'historical development UniProt annotation; external experimental-coordinate P-SEA H/E',
        'new_fits':0,'new_sequence_or_structure_acquisition':0})
    write('ACCEPTANCE.json',{'status':'PASS','created_utc':utc(),'pair_rows':len(pairs),'external_sites':len(j),
        'denominator_check':'one pair per group and identity; exactly two sites per pair; strict frame subset of original frame',
        'direction_check':'early minus late; H label encoded only in y; swapped input score reversal tested for four arms',
        'same_sequence_chain_residue_check':'asserted for every selected pair before forwarding',
        'masking':'one original centre token replaced, input tokens len(sequence)+2, no truncation; asserted every forward',
        'exclusions':'complete group-by-identity H/E/site/candidate/exclusion ledger in PAIR_SELECTION.json',
        'replay':'REPLAY.json PASS before external predictions',
        'source_bindings':'BINDINGS.json','source_binding_recheck':'PASS',
        'outputs':{n:sha(OUT/n) for n in ('PROTOCOL.md','BINDINGS.json','PAIR_SELECTION.json','MODEL_IDENTITY.json','REPLAY.json','ACTIVATIONS.npz','PAIRED_SCORES.tsv','SUMMARY.json')},
        'confidence':'high for computation and fixed sample; external sample uncertainty reported explicitly'})
    print(json.dumps({'phase':'complete','results':[{k:r[k] for k in ('frame','identity','n_pairs','comparisons')} for r in summary]}),flush=True)

def main():
    command=sys.argv[1] if len(sys.argv)>1 else 'run'
    if command=='prepare':prepare();return
    bindings=read(OUT/'BINDINGS.json')
    for x in bindings['identities']:assert sha(x['path'])==x['sha256'],x['path']
    pairs,cases=reconstruct()
    if command=='summarize':
        with np.load(OUT/'ACTIVATIONS.npz') as a:j=a['j1246'];assert a['case_ids'].astype(str).tolist()==[c['case_id'] for c in cases]
        summarize(pairs,j);return
    if (OUT/'STARTED.json').exists():raise RuntimeError('Already started; no unrecorded automatic rerun')
    write('STARTED.json',{'created_utc':utc(),'bindings_sha256':sha(OUT/'BINDINGS.json'),'selected_sha256':sha(OUT/'PAIR_SELECTION.json'),'external_cases':len(cases)})
    forward,handle=load_forward();replay(forward)
    tick=time.perf_counter();j=[]
    for i,c in enumerate(cases):
        j.append(forward(c))
        if (i+1)%32==0 or i+1==len(cases):
            write('PROGRESS.json',{'created_utc':utc(),'completed':i+1,'planned':len(cases),'wall_seconds':time.perf_counter()-tick})
            print(json.dumps({'phase':'external','completed':i+1,'planned':len(cases),'seconds':time.perf_counter()-tick}),flush=True)
    handle.remove();j=np.asarray(j,dtype=np.float64)
    np.savez_compressed(OUT/'ACTIVATIONS.npz',j1246=j,case_ids=np.asarray([c['case_id'] for c in cases]))
    for x in bindings['identities']:assert sha(x['path'])==x['sha256'],x['path']
    summarize(pairs,j)

if __name__=='__main__':main()
