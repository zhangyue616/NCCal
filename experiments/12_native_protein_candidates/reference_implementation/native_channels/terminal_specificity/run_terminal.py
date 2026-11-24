"""Closed post-hoc two-terminal sensitivity; one previously selected native E1 unit."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np

P = Path(__file__).resolve().parent
BASE = P.parent
C = BASE / "cohort"
E = BASE / "e1"
TASKS = ("N_vs_C", "C_vs_interior4")
LAYER0, UNIT0, SEED = 14, 2192, 20261123


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def build():
    chains = [json.loads(s) for s in (C / "chains.jsonl").read_text().splitlines() if s.strip()]
    by_id = {r["record_id"]: r for r in chains}
    annotations = {(r["record_id"], int(r["seq_index0"])): r for r in read_tsv(C / "residue_annotations.tsv")}
    segments = read_tsv(C / "helix_segments.tsv")
    candidates, funnel = [], Counter()
    for seg in segments:
        funnel["segments_all"] += 1
        if seg["eligible_long_helix"] != "True":
            continue
        assert seg["boundary_and_full_backbone_continuous"] == "True"
        r = by_id[seg["record_id"]]
        start, end, length = int(seg["start_index0"]), int(seg["end_index0"]), int(seg["length"])
        assert end - start + 1 == length and length >= 10
        funnel["segments_eligible"] += 1
        seqhash = hashlib.sha256(r["sequence"].encode()).hexdigest()
        assert seqhash == r["sequence_sha256"]
        sets = {
            "N_vs_C": (range(start, start + 2), range(end - 1, end + 1)),
            "C_vs_interior4": (range(end - 1, end + 1), range(start + 4, end - 3)),
        }
        for task, (low, high) in sets.items():
            for a, b in itertools.product(low, high):
                if r["sequence"][a] != r["sequence"][b]:
                    continue
                for pos in (a, b):
                    ann = annotations[(r["record_id"], pos)]
                    assert ann["secondary"] == "H" and ann["backbone_quality"] == "True"
                    assert ann["long_helix_complete"] == "True" and ann["helix_segment_id"] == seg["segment_id"]
                    assert ann["aa"] == r["sequence"][pos]
                key = f"{task}|{seqhash}|{min(a,b)}|{max(a,b)}"
                candidates.append(dict(pair_key=key, selection_hash=hashlib.sha256(key.encode()).hexdigest(),
                    task=task, record_id=r["record_id"], group=r["group"], split=r["split"],
                    sequence_sha256=seqhash, segment_id=seg["segment_id"], helix_start_index0=start,
                    helix_end_index0=end, helix_length=length, aa=r["sequence"][a],
                    index_low_expected=a, index_high_expected=b, low_offset0=a-start,
                    low_distance_to_C=end-a, high_offset0=b-start, high_distance_to_C=end-b,
                    source_cif=r["source_cif"],
                ))
    write_tsv(P / "candidate_pairs.tsv", sorted(candidates, key=lambda r: (r["selection_hash"],r["record_id"],r["segment_id"])))
    unique, exclusions = {}, []
    for r in sorted(candidates, key=lambda r: (r["record_id"],r["segment_id"])):
        if r["pair_key"] in unique:
            old = unique[r["pair_key"]]
            assert (old["split"],old["group"]) == (r["split"],r["group"])
            exclusions.append(dict(r, exclusion_reason="duplicate_native_sequence_positions"))
        else:
            unique[r["pair_key"]] = r
    selected, counts, used = [], Counter(), defaultdict(set)
    for r in sorted(unique.values(), key=lambda r: (r["selection_hash"],r["record_id"],r["segment_id"])):
        key = (r["task"],r["group"])
        sites = {(r["sequence_sha256"],r["index_low_expected"]),(r["sequence_sha256"],r["index_high_expected"])}
        reason = "group_max16" if counts[key] >= 16 else "native_site_already_used" if used[key].intersection(sites) else None
        if reason:
            exclusions.append(dict(r, exclusion_reason=reason))
            continue
        counts[key] += 1
        used[key].update(sites)
        selected.append(dict(pair_id="terminal-"+r["selection_hash"][:24], **r))
    write_tsv(P / "pairs.tsv", selected)
    write_tsv(P / "pair_selection_exclusions.tsv", exclusions)
    old_pairs = read_tsv(E / "trained/pair_index.tsv")
    # Selection below does not access old raw_delta, correctness, or scores.
    calibration = []
    for split in ("D","T","E"):
        eligible = sorted((r for r in old_pairs if r["split"] == split and r["task"] == "helix_start_vs_internal"),key=lambda r:r["pair_id"])
        calibration.extend(eligible[:3])
    write_tsv(P / "old_pair_calibration_selection.tsv", calibration)
    paths = [P / "LOCKED_PROTOCOL.md", Path(__file__), C / "chains.jsonl",C / "helix_segments.tsv",C / "residue_annotations.tsv",C / "READY.json",P / "pairs.tsv",E / "e1_runtime.py",E / "scan.py",E / "trained/selection_lock.json",E / "trained/model_metadata.json"]
    meta = json.loads((E / "trained/model_metadata.json").read_text())
    lock = dict(created_utc=datetime.now(timezone.utc).isoformat(),status="LOCKED_BEFORE_NEW_TERMINAL_FORWARD",
        posthoc=True, layer0=LAYER0, layer1=LAYER0+1, unit0=UNIT0, unit1=UNIT0+1,original_sign=-1,
        raw_unit_read="w2 input post-SwiGLU, untransformed signed value",direction={"N_vs_C":"N lower than C","C_vs_interior4":"C lower than interior"},
        funnel=dict(funnel),candidate_pairs=len(candidates),unique_native_position_pairs=len(unique),selected_pairs=len(selected),
        selection_counts=[dict(task=t,split=s,n_groups=len({r['group'] for r in selected if r['task']==t and r['split']==s}),n_pairs=sum(r['task']==t and r['split']==s for r in selected)) for t in TASKS for s in ('D','T','E')],
        model_weight_sha256_reused_from_accepted_E1=meta['weight_sha256'],source_revision=meta['source_revision'],weight_revision=meta['weight_revision'],
        inputs=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in paths])
    dump(P / "PRE_FORWARD_FREEZE.json", lock)
    print("PAIRS_FROZEN",json.dumps(lock['selection_counts']),flush=True)
    return chains, selected, calibration, lock


def extract(chains, pairs, calibration):
    sys.path.insert(0,str(E))
    import e1_runtime as rt
    import torch
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3 * 1024**3 else "cpu"
    wanted = {r['record_id'] for r in pairs} | {r['record_id'] for r in calibration}
    needed = [r for r in chains if r['record_id'] in wanted]
    by_seq = defaultdict(list)
    for r in needed:
        by_seq[r['sequence_sha256']].append(r)
    model = rt.get_model(trained=True,device=device)
    context = {}
    def capture(_module, args):
        a = args[0]
        assert a.shape[0] == 1 and a.shape[-1] == 2304
        context['value'] = a[0,2:-2,UNIT0].detach().float().cpu().numpy().copy()
    handle = model.model.layers[LAYER0].ffn.mlp.w2.register_forward_pre_hook(capture)
    values, coverage = {}, []
    start = time.time()
    for n,(seqhash,rs) in enumerate(sorted(by_seq.items())):
        r = sorted(rs,key=lambda r:r['record_id'])[0]
        seq = r['sequence']
        assert all(x['sequence'] == seq for x in rs)
        with torch.inference_mode():
            out = model.model(**rt.get_batch(seq,device=device))
        a = context.pop('value')
        assert len(a) == len(seq) and np.isfinite(a).all() and torch.isfinite(out.last_hidden_state).all()
        values[seqhash] = a
        coverage.append(dict(sequence_sha256=seqhash,record_ids=';'.join(sorted(x['record_id'] for x in rs)),sequence_length=len(seq),formal_forwards=1))
        if (n+1)%20 == 0:
            print('FORWARD',n+1,'/',len(by_seq),round(time.time()-start,1),device,flush=True)
    handle.remove()
    np.savez_compressed(P / "unit2192_per_native_sequence.npz",**values)
    write_tsv(P / "forward_coverage.tsv",coverage)
    raw_old = np.load(E / "trained/pair_deltas/layer_14.npy",mmap_mode='r')
    by_id = {r['record_id']:r for r in chains}
    check_rows=[]
    for r in calibration:
        a=values[by_id[r['record_id']]['sequence_sha256']]
        pos,neg=int(r['index_positive']),int(r['index_negative'])
        now=float(a[pos]-a[neg]);old=float(raw_old[int(r['matrix_row']),UNIT0])
        passed=bool(np.isclose(now,old,atol=1e-4,rtol=1e-4))
        check_rows.append(dict(pair_id=r['pair_id'],split=r['split'],matrix_row=r['matrix_row'],old_delta=old,new_delta=now,abs_error=abs(now-old),PASS=passed))
    write_tsv(P / "old_pair_calibration.tsv",check_rows)
    assert all(r['PASS'] for r in check_rows), check_rows
    for r in pairs:
        a=values[r['sequence_sha256']]
        r['value_low_expected']=float(a[r['index_low_expected']])
        r['value_high_expected']=float(a[r['index_high_expected']])
        r['signed_delta']=r['value_high_expected']-r['value_low_expected']
        r['correct']=1. if r['signed_delta']>0 else 0. if r['signed_delta']<0 else .5
    write_tsv(P / "pair_results.tsv",pairs)
    return pairs,dict(device=device,torch=torch.__version__,dtype='float32',single_native_sequence=True,unique_sequence_forwards=len(values),record_ids_covered=len(needed),seconds=time.time()-start,calibration_pairs=len(check_rows),calibration_max_abs_error=max((r['abs_error'] for r in check_rows),default=None))


def signflip_p(group_accuracies, rng):
    vals=np.asarray(group_accuracies,dtype=float)-.5
    nz=vals[vals!=0]
    if len(vals)<2:
        return None,'too_few_groups'
    if len(nz)==0:
        return 1.,'all_groups_no_direction'
    observed=vals.sum()
    if len(nz)<=18:
        flips=np.asarray(list(itertools.product((-1.,1.),repeat=len(nz))),dtype=np.float64)
        scores=flips@nz
        return float(np.mean(scores>=observed-1e-12)),'exact_group_signflip'
    count=0
    for j in range(0,100000,10000):
        flips=rng.integers(0,2,(10000,len(nz)),dtype=np.int8)*2-1
        count+=int(np.sum(flips@nz>=observed-1e-12))
    return (count+1)/100001,'Monte_Carlo_group_signflip_100000_plus1'


def summarize(pairs):
    rng=np.random.default_rng(SEED)
    group_rows,summary=[],[]
    for task in TASKS:
        for split in ('D','T','E'):
            ps=[r for r in pairs if r['task']==task and r['split']==split]
            groups=sorted({r['group'] for r in ps})
            gs=[]
            for g in groups:
                rr=[r for r in ps if r['group']==g]
                gr=dict(task=task,split=split,group=g,n_pairs=len(rr),n_correct=sum(r['correct']==1 for r in rr),n_wrong=sum(r['correct']==0 for r in rr),n_ties=sum(r['correct']==.5 for r in rr),accuracy=float(np.mean([r['correct'] for r in rr])),mean_signed_delta=float(np.mean([r['signed_delta'] for r in rr])))
                group_rows.append(gr);gs.append(gr)
            vals=np.array([r['accuracy'] for r in gs])
            if len(vals)>=2:
                b=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1)
                ci=np.quantile(b,[.025,.975]).tolist()
            else:ci=None
            p,method=signflip_p(vals,rng)
            summary.append(dict(task=task,split=split,n_groups=len(groups),n_pairs=len(ps),n_correct=sum(r['correct']==1 for r in ps),n_wrong=sum(r['correct']==0 for r in ps),n_ties=sum(r['correct']==.5 for r in ps),group_equal_accuracy=float(vals.mean()) if len(vals) else None,group_bootstrap_95ci=ci,pair_pooled_accuracy=float(np.mean([r['correct'] for r in ps])) if ps else None,group_equal_mean_signed_delta=float(np.mean([r['mean_signed_delta'] for r in gs])) if gs else None,p_one_sided=p,p_method=method))
    for split in ('D','T','E'):
        rows=sorted([r for r in summary if r['split']==split],key=lambda r:r['p_one_sided'] if r['p_one_sided'] is not None else 1.)
        last=0.
        for rank,r in enumerate(rows):
            p=r['p_one_sided'] if r['p_one_sided'] is not None else 1.
            last=max(last,min(1.,(2-rank)*p))
            r['p_holm_two_tasks']=last if r['p_one_sided'] is not None else None
    write_tsv(P / "group_results.tsv",group_rows)
    write_tsv(P / "split_summary.tsv",summary)
    dump(P / "STATISTICS.json",dict(posthoc=True,unit0=UNIT0,layer1=LAYER0+1,seed=SEED,bootstrap_draws=10000,holm_family='two new contrasts within each split',summary=summary))
    return group_rows,summary


def acceptance(chains,pairs,groups,summary,lock,runtime):
    by_id={r['record_id']:r for r in chains}
    site_keys=set();pair_keys=set()
    for r in pairs:
        assert r['pair_key'] not in pair_keys
        pair_keys.add(r['pair_key'])
        c=by_id[r['record_id']]
        assert c['split']==r['split'] and c['group']==r['group']
        assert c['sequence'][r['index_low_expected']]==c['sequence'][r['index_high_expected']]==r['aa']
        assert r['helix_start_index0']<=r['index_low_expected']<=r['helix_end_index0']
        assert r['helix_start_index0']<=r['index_high_expected']<=r['helix_end_index0']
        if r['task']=='N_vs_C':
            assert r['low_offset0'] in (0,1) and r['high_distance_to_C'] in (0,1)
        else:
            assert r['low_distance_to_C'] in (0,1) and r['high_offset0']>=4 and r['high_distance_to_C']>=4
        for pos in (r['index_low_expected'],r['index_high_expected']):
            key=(r['task'],r['group'],r['sequence_sha256'],pos)
            assert key not in site_keys
            site_keys.add(key)
    assert all(g['n_pairs']<=16 for g in groups)
    assert all(r['n_correct']+r['n_wrong']+r['n_ties']==r['n_pairs'] for r in summary)
    for f in lock['inputs']:
        assert sha(f['path'])==f['sha256'],f['path']
    receipt=dict(status='PASS_CONSOLIDATED_TERMINAL_SENSITIVITY',created_utc=datetime.now(timezone.utc).isoformat(),posthoc=True,
        no_core_results_recomputed=True,no_other_unit_search=True,full_native_sequences_unchanged=True,
        same_helix_same_AA_and_endpoint_distances=True,native_position_pairs_deduplicated=True,
        no_station_reuse_within_group_task=True,max16_per_group_task=True,pre_forward_bindings_match=True,
        runtime=runtime,scope='terminal_specificity only',scientific_limits='posthoc; failure to reject is not equality; no equivalence test')
    dump(P / 'ACCEPTANCE.json',receipt)
    return receipt


def report(summary,runtime,receipt):
    """Write the English rendering; analysis and numerical results are unchanged."""
    dump(P / 'REPORT.json', dict(summary=summary, runtime=runtime, receipt=receipt))


def main():
    assert not (P/'PRE_FORWARD_FREEZE.json').exists(),'Do not overwrite the new assay freeze'
    chains,pairs,calibration,lock=build()
    pairs,runtime=extract(chains,pairs,calibration)
    groups,summary=summarize(pairs)
    receipt=acceptance(chains,pairs,groups,summary,lock,runtime)
    report(summary,runtime,receipt)
    print(json.dumps(dict(summary=summary,runtime=runtime,status=receipt['status']),indent=2),flush=True)


if __name__=='__main__':
    main()
