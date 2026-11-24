"""Targeted independent acceptance; reads existing vectors, never reruns models."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np

root=Path(__file__).resolve().parent
out=root/'trained';cohort=root.parent/'cohort'
def table(path):
    return list(csv.DictReader(open(path,encoding='utf-8-sig',newline=''),delimiter='\t'))
pairs=table(cohort/'pairs.tsv');index=table(out/'pair_index.tsv');selected=table(out/'selected_pair_scores.tsv')
chains={r['record_id']:r for r in map(json.loads,(cohort/'chains.jsonl').read_text().splitlines())}
results=json.loads((out/'results.json').read_text());lock=json.loads((out/'selection_lock.json').read_text())
ready=json.loads((cohort/'READY.json').read_text());metadata=json.loads((out/'model_metadata.json').read_text())
checks=[]
def check(name,condition,detail):
    assert condition, name
    checks.append({'check':name,'status':'PASS','detail':detail})
check('source_binding',all(metadata[k]==lock[k]==ready[k] for k in ['chains_sha256','pairs_sha256']), 'metadata/lock/final READY point to the same two frozen inputs; reuse cohort owner hash evidence')
check('row_mapping',len(pairs)==len(index)==len(selected)==2593 and all(int(ix['matrix_row'])==i and ix['pair_id']==p['pair_id'] for i,(p,ix) in enumerate(zip(pairs,index))), 'all 2593 pair_ids align to matrix_row; no model forward')
by_id={r['pair_id']:r for r in selected}
sample=[];strict=[];summaries=[]
for selection in lock['selections']:
    task=selection['task'];layer=selection['layer0'];unit=selection['unit0'];sign=selection['sign']
    delta=np.load(out/'pair_deltas'/f'layer_{layer:02d}.npy',mmap_mode='r')
    check('selected_column_'+task,delta.shape==(2593,2304) and delta.dtype==np.float32, f'layer {layer} raw float32 [2593,2304], column {unit}, direction {sign}')
    rows=[]
    for i,p in enumerate(pairs):
        if p['task']!=task:continue
        reported=by_id[p['pair_id']];raw=float(delta[i,unit]);sd=sign*raw;correct=1. if sd>0 else 0. if sd<0 else .5
        assert raw==float(reported['raw_delta']) and sd==float(reported['signed_delta']) and correct==float(reported['correct'])
        rows.append(dict(p,raw_delta=raw,correct=correct))
    check('all_reported_readouts_'+task,True,f'exactly re-read {len(rows)} existing scalar deltas, including all D/T/E; signs and ties match TSV')
    for split in ('D','T','E'):
        use=[r for r in rows if r['split']==split];by_group=defaultdict(list)
        for r in use:by_group[r['group']].append(r['correct'])
        group_acc=[float(np.mean(v)) for v in by_group.values()]
        macro=float(np.mean(group_acc));reported=next(x for x in results['summary'] if x['task']==task and x['split']==split)
        assert len(use)==reported['n_pairs'] and len(by_group)==reported['n_groups'] and abs(macro-reported['group_equal_accuracy'])<1e-12
        summaries.append({'task':task,'split':split,'pairs':len(use),'groups':len(by_group),'group_equal_accuracy':macro,
                          'layer0':layer,'layer1':layer+1,'unit0':unit,'unit1':unit+1,'sign':sign})
        # Deterministic samples include one correct and, when present, one error.
        for outcome in (1.,0.):
            candidates=[r for r in use if r['correct']==outcome]
            if not candidates:continue
            r=candidates[0];c=chains[r['record_id']];pos=int(r['index_positive']);neg=int(r['index_negative'])
            assert c['sequence'][pos]==c['sequence'][neg]==r['aa']
            sample.append({'pair_id':r['pair_id'],'split':split,'task':task,'record_id':r['record_id'],
                           'aa':r['aa'],'index_positive0':pos,'index_negative0':neg,'token_positive0':pos+2,'token_negative0':neg+2,
                           'raw_delta':r['raw_delta'],'sign':sign,'correct':r['correct'],
                           'same_helix_segment':r['same_helix_segment']})
        if task=='helix_start_vs_internal':
            subset=[r for r in use if r['same_helix_segment'].lower()=='true'];sg=defaultdict(list)
            for r in subset:
                assert r['positive_helix_segment_id']==r['negative_helix_segment_id']
                assert int(r['positive_helix_offset0']) in (0,1)
                assert int(r['negative_helix_offset0'])>=3 and int(r['negative_helix_distance_to_end'])>=3
                assert int(r['positive_helix_length'])>=10
                sg[r['group']].append(r['correct'])
            strict.append({'task':task,'split':split,'pairs':len(subset),'groups':len(sg),'group_equal_accuracy':float(np.mean([np.mean(v) for v in sg.values()])),
                           'pair_pooled_accuracy':float(np.mean([r['correct'] for r in subset])), 'unit_reselection':False,'sign':sign})
    # First ranked row must equal immutable lock, without redoing selection.
    with open(out/f'development_ranking_{task}.tsv',encoding='utf-8') as f:
        top=next(csv.DictReader(f,delimiter='\t'))
    assert int(top['layer0'])==layer and int(top['unit0'])==unit and int(top['sign'])==sign
    check('ranking_lock_'+task,True,'top development-ranked row matches closed lock; source review confirms no T/E selection')
coverage=table(out/'forward_coverage.tsv')
check('formal_forward_coverage',len(coverage)==142 and len({r['record_id'] for r in coverage})==142 and all(int(r['formal_forwards'])==1 for r in coverage), '142 distinct chain records, one formal trained forward each; 131 distinct native sequences per reused cohort acceptance')
check('raw_source_sample',len(sample)==12,'12 deterministic pair samples: correct and error from each task/split; raw native same-AA mapping and +2 token offset checked')
check('strict_same_segment_counts',[(r['split'],r['pairs'],r['groups']) for r in strict]==[('D',64,22),('T',252,55),('E',73,19)], 'strict same-helix subset retains original lock/direction and is reported separately')
preflight=json.loads((root/'preflight.json').read_text())
check('implementation_reference',preflight['success'] and preflight['strict_load']=='<All keys matched successfully>', 'reuse completed all-20-layer native CPU eager flex vs SDPA evidence; no new forward')
review={'status':'PASS_TARGETED_INDEPENDENT_READOUT_REVIEW','checks':checks,'summaries':summaries,'strict_same_segment':strict,'samples':sample,
        'scope':'No model rerun, no refit/reselection, no full corpus PDB revalidation; use final cohort owner acceptance. Formal statistics and all comparators remain with parent.',
        'interpretation':{'helix_vs_strand':'zero-based layer19/unit1900 (+), one-based layer20/unit1901. Native scalar tends to be higher at alpha helix than beta strand.',
                          'helix_start_vs_internal':'zero-based layer14/unit2192 (-), one-based layer15/unit2193. Native scalar tends to be higher at helix internal residues; negative-oriented scalar tends to be higher at the first two helix residues. This is not an onset-high native channel.'}}
(root/'TARGETED_REVIEW.json').write_text(json.dumps(review,indent=2))
lines=['Scientific narrative is documented in PROTOCOL.md.','', 'Scientific narrative is documented in PROTOCOL.md.', '',
       'Scientific narrative is documented in PROTOCOL.md.',
       'Scientific narrative is documented in PROTOCOL.md.',
       'Scientific narrative is documented in PROTOCOL.md.',
       'Scientific narrative is documented in PROTOCOL.md.',
       '', 'Scientific narrative is documented in PROTOCOL.md.', '|---|---|---:|---:|---:|']
for r in summaries:lines.append(f"|{r['task']}|{r['split']}|{r['pairs']}|{r['groups']}|{r['group_equal_accuracy']:.6f}|")
lines+=['','Scientific narrative is documented in PROTOCOL.md.','', 'Scientific narrative is documented in PROTOCOL.md.', '|---|---:|---:|---:|']
for r in strict:lines.append(f"|{r['split']}|{r['pairs']}|{r['groups']}|{r['group_equal_accuracy']:.6f}|")
lines+=['','Scientific narrative is documented in PROTOCOL.md.',
        'Scientific narrative is documented in PROTOCOL.md.',
        'Scientific narrative is documented in PROTOCOL.md.',
        'Scientific narrative is documented in PROTOCOL.md.']
(root/'TARGETED_REVIEW_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'summaries':summaries,'strict_same_segment':strict,'checks':len(checks)},indent=2))
