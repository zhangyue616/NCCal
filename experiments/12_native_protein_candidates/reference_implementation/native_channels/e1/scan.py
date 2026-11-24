"""All-block native-unit census, development lock then held-out scoring."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, time, gc, math
import numpy as np
import e1_runtime as rt
import torch

ROOT = rt.ROOT
COHORT = ROOT.parent/'cohort'
TASKS = ('helix_vs_strand', 'helix_start_vs_internal')
LAYERS, UNITS = 20, 2304


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_tsv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def load_data():
    assert (COHORT/'READY.json').exists(), 'Need final cohort READY receipt'
    chains = [json.loads(x) for x in (COHORT/'chains.jsonl').read_text().splitlines() if x.strip()]
    pairs = list(csv.DictReader(open(COHORT/'pairs.tsv', newline='', encoding='utf-8-sig'), delimiter='\t'))
    by_id = {x['record_id']:x for x in chains}
    assert len(by_id) == len(chains)
    seen = set()
    group_splits = defaultdict(set)
    for row in pairs:
        assert row['pair_id'] not in seen
        seen.add(row['pair_id'])
        assert row['split'] in ('D', 'T', 'E') and row['task'] in TASKS
        row['index_positive'] = int(row['index_positive'])
        row['index_negative'] = int(row['index_negative'])
        chain = by_id[row['record_id']]
        assert chain['group'] == row['group'] and chain['split'] == row['split']
        seq = chain['sequence']
        assert seq[row['index_positive']] == seq[row['index_negative']] == row['aa']
        assert len(seq) + 4 <= 8192, 'No truncation allowed'
        group_splits[row['group']].add(row['split'])
    assert all(len(x)==1 for x in group_splits.values()), 'Group leakage across splits'
    return chains, pairs


def extract(model, chains, pairs, out, keep_matrix):
    matrix_dir = out/'pair_deltas'
    if keep_matrix:
        matrix_dir.mkdir(exist_ok=True)
        matrices = [np.lib.format.open_memmap(matrix_dir/f'layer_{l:02d}.npy', mode='w+', dtype=np.float32,
                                            shape=(len(pairs),UNITS)) for l in range(LAYERS)]
    else:
        matrices = [np.empty((len(pairs),UNITS), dtype=np.float32) for _ in range(LAYERS)]
    by_chain = defaultdict(list)
    for i,row in enumerate(pairs):
        by_chain[row['record_id']].append(i)
    context = {}
    def capture(layer, activation):
        idx = context['rows']
        pos, neg = context['positive'], context['negative']
        delta = (activation[0,pos,:].float() - activation[0,neg,:].float()).detach().cpu().numpy()
        assert delta.shape == (len(idx),UNITS) and np.isfinite(delta).all()
        matrices[layer][idx] = delta
        context['layers'].add(layer)
    handles = rt.attach_unit_capture(model, capture)
    forward_rows = []
    start = time.time()
    for count,chain in enumerate(chains):
        indices = by_chain.get(chain['record_id'],[])
        if not indices:
            continue
        sequence = chain['sequence']
        context.update(rows=np.array(indices),
                       positive=torch.tensor([pairs[i]['index_positive']+2 for i in indices], device='cuda'),
                       negative=torch.tensor([pairs[i]['index_negative']+2 for i in indices], device='cuda'), layers=set())
        t0 = time.time()
        with torch.inference_mode():
            outputs = model.model(**rt.get_batch(sequence))
        assert context['layers'] == set(range(LAYERS))
        assert torch.isfinite(outputs.last_hidden_state).all()
        forward_rows.append({'record_id':chain['record_id'], 'group':chain['group'], 'split':chain['split'],
                             'sequence_length':len(sequence), 'pairs':len(indices), 'seconds':time.time()-t0,
                             'formal_forwards':1, 'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest()})
        del outputs
        if len(forward_rows)%25 == 0:
            print('FORWARD',out.name,len(forward_rows),round(time.time()-start,2),flush=True)
    for h in handles:
        h.remove()
    if keep_matrix:
        for m in matrices:
            m.flush()
    write_tsv(out/'forward_coverage.tsv',forward_rows)
    write_tsv(out/'pair_index.tsv', [dict(matrix_row=i, **row) for i,row in enumerate(pairs)])
    return matrices, forward_rows


def select_and_evaluate(matrices, pairs, out, full_ranking):
    all_locks = []
    # Selection uses only D. Confirmation evaluation is below the closed lock file.
    for task in TASKS:
        groups = sorted({r['group'] for r in pairs if r['task']==task and r['split']=='D'})
        assert groups, task
        group_rows = [np.array([i for i,r in enumerate(pairs) if r['task']==task and r['split']=='D' and r['group']==g]) for g in groups]
        # Exact integer macro weighting preserves mathematical ties. Every
        # group has <=16 pairs by frozen cohort construction.
        common = math.lcm(*(len(ix) for ix in group_rows))
        denom = 2 * common * len(groups)
        assert denom < np.iinfo(np.int64).max
        numerators = np.zeros((LAYERS,UNITS),dtype=np.int64)
        for layer, matrix in enumerate(matrices):
            for ix in group_rows:
                d = matrix[ix]
                doubled_correct = 2*(d>0).sum(axis=0) + (d==0).sum(axis=0)
                numerators[layer] += doubled_correct * (common//len(ix))
        # ravel axes [layer, unit, sign(+ then -)] implement specified tie rule.
        candidate_numerators = np.stack([numerators,denom-numerators],axis=-1).ravel()
        candidate_scores = candidate_numerators / denom
        order = np.argsort(-candidate_numerators, kind='stable')
        top = int(order[0]); layer,unit,sign_idx = np.unravel_index(top,(LAYERS,UNITS,2))
        all_locks.append({'task':task,'layer0':int(layer),'layer1':int(layer)+1,'unit0':int(unit),'unit1':int(unit)+1,
                          'sign':1 if sign_idx==0 else -1,'development_group_equal_accuracy':float(candidate_scores[top]),
                          'development_groups':len(groups),'development_pairs':sum(map(len,group_rows)),
                          'physical_units':LAYERS*UNITS,'signed_candidates':LAYERS*UNITS*2})
        if full_ranking:
            with open(out/f'development_ranking_{task}.tsv','w',newline='',encoding='utf-8') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['rank','layer0','layer1','unit0','unit1','sign','group_equal_accuracy'])
                for rank,candidate in enumerate(order,1):
                    l,u,s = np.unravel_index(candidate,(LAYERS,UNITS,2))
                    w.writerow([rank,l,l+1,u,u+1,1 if s==0 else -1,repr(float(candidate_scores[candidate]))])
    lock = {'created_unix':time.time(), 'rule':'D group-equal; tie layer0,unit0,+ first; no T/E selection',
            'pairs_sha256':sha(COHORT/'pairs.tsv'),'chains_sha256':sha(COHORT/'chains.jsonl'), 'selections':all_locks}
    lock_path = out/'selection_lock.json'
    with open(lock_path,'x',encoding='utf-8') as f:
        json.dump(lock,f,indent=2)
        f.flush()
    # Closed, immutable-on-this-run selection lock precedes confirmation scoring.
    lock_sha = sha(lock_path)
    group_rows, pair_rows, summaries = [], [], []
    for selected in all_locks:
        task=selected['task']; layer=selected['layer0']; unit=selected['unit0']; sign=selected['sign']
        task_indices = [i for i,r in enumerate(pairs) if r['task']==task]
        for i in task_indices:
            row=pairs[i]; d=float(matrices[layer][i,unit]); sd=sign*d
            pair_rows.append(dict(row,layer0=layer,layer1=layer+1,unit0=unit,unit1=unit+1,sign=sign,
                                  raw_delta=d,signed_delta=sd,correct=1.0 if sd>0 else .0 if sd<0 else .5))
        for split in ('D','T','E'):
            split_rows=[r for r in pair_rows if r['task']==task and r['split']==split]
            groups=sorted({r['group'] for r in split_rows})
            vals=[]
            for g in groups:
                gr=[r for r in split_rows if r['group']==g]
                acc=float(np.mean([r['correct'] for r in gr]));vals.append(acc)
                group_rows.append({'task':task,'split':split,'group':g,'n_pairs':len(gr),'accuracy':acc,
                                   'ties':sum(r['correct']==.5 for r in gr),'layer0':layer,'unit0':unit,'sign':sign})
            if groups:
                summaries.append({'task':task,'split':split,'n_groups':len(groups),'n_pairs':len(split_rows),
                                  'group_equal_accuracy':float(np.mean(vals)),
                                  'pair_pooled_accuracy':float(np.mean([r['correct'] for r in split_rows])),
                                  'ties':sum(r['correct']==.5 for r in split_rows),
                                  'layer0':layer,'layer1':layer+1,'unit0':unit,'unit1':unit+1,'sign':sign})
    write_tsv(out/'selected_pair_scores.tsv',pair_rows)
    write_tsv(out/'selected_group_scores.tsv',group_rows)
    write_tsv(out/'split_summary.tsv',summaries)
    result={'status':'COMPLETED','selection_lock_sha256':lock_sha,'selections':all_locks,'summary':summaries}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--random-seed',type=int);args=parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    chains,pairs=load_data()
    random=args.random_seed is not None
    out=ROOT/f'random_{args.random_seed}' if random else ROOT/'trained'
    out.mkdir(exist_ok=True)
    assert not (out/'selection_lock.json').exists(), 'Do not overwrite existing selection lock'
    metadata={'model':'Profluent-Bio/E1-150m','weight_revision':'c4dbfe827e4aa6ed7f95eaef50dc1e084f4d77dc',
              'source_revision':'bfd2620a602248499f3d2583d85a7ecddf0b6e02','trained_weights':not random,'random_seed':args.random_seed,
              'dtype':'float32','torch':torch.__version__,'attention':'single-sequence SDPA; original E1 everything else',
              'hook':'w2 input = silu(w1(x))*w3(x)', 'layers':LAYERS,'units_per_layer':UNITS,
              'signed_candidates_per_task':LAYERS*UNITS*2,'token_prefix_count':2,
              'parameter_count':154423330,'checkpoint_storage_dtype':'bfloat16',
              'weight_sha256':json.loads((ROOT/'preflight.json').read_text())['weight_sha256'],
              'transformers':__import__('transformers').__version__,
              'runtime_python':__import__('sys').executable,
              'chains_sha256':sha(COHORT/'chains.jsonl'),'pairs_sha256':sha(COHORT/'pairs.tsv')}
    (out/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    model=rt.get_model(trained=not random,seed=args.random_seed)
    matrix,coverage=extract(model,chains,pairs,out,keep_matrix=not random)
    del model;gc.collect();torch.cuda.empty_cache()
    result=select_and_evaluate(matrix,pairs,out,full_ranking=not random)
    result['formal_forwards']=len(coverage)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
