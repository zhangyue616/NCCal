"""Consolidated acceptance and human-readable results; no new scientific fits."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import run_calibration as c

manifest=c.read(c.OUT/'RUN_MANIFEST.json')
assert all(c.sha(p)==s for p,s in manifest['source_hashes'].items()),'Bound source drift'
assert c.read(c.OUT/'runtime_checks/CHECKS.json')['status']=='PASS'
for r in range(1,10):
    rotation=c.OUT/'rotations'/f'haar_{r:02d}.npy'
    identity=c.read(rotation.with_suffix('.json'))
    assert identity['seed']==2026111800+r and c.sha(rotation)==identity['sha256']
    assert identity['max_orthogonality_error']<1e-12
trained=c.read(c.OUT/'trained_gpu/SUMMARY.json')
networks=[]
for i in range(1,200):
    folder='runtime_check_init001' if i==1 else f'init_{i:03d}'
    summary=c.read(c.OUT/folder/'SUMMARY.json')
    identity=c.read(c.OUT/folder/'MODEL_IDENTITY.json')
    assert identity['seed']==20261118000+i
    assert identity['architecture']==[320,1280,6,'gelu'] and identity['embedding_lm_head_tied']
    assert identity['torch']=='2.11.0+cu128' and identity['transformers']=='5.20.0'
    assert identity['device']=='cuda'
    assert len(summary['rows'])==10 and summary['rotation_count']==9
    assert [r['basis'] for r in summary['rows']]==list(range(10))
    assert all(r['selected_axis']['sign_flip_prediction_max_error']<1e-11 for r in summary['rows'])
    assert all(r['paired_readout']['development_pairs']==432 and r['paired_readout']['development_groups']==421 and r['paired_readout']['confirmation_groups']==1304 for r in summary['rows'])
    assert c.read(c.OUT/folder/'EXTRACTION_RECEIPT.json')['complete']
    networks.append((i,summary,identity))
assert len({n[2]['state_sha256'] for n in networks})==199
assert trained['rows'][0]['selected_axis']['unit']==1246

def metrics(s):
    result={'single_center_native':s['site_G_native'],'single_center_native_minus_rotated':s['site_A']}
    for ep in c.KRR.ENDPOINTS:
        result[f'paired_{ep}_native']=s['paired'][ep]['G_native']
        result[f'paired_{ep}_native_minus_rotated']=s['paired'][ep]['A']
    return result

def plot_values(s):
    result=metrics(s)
    result['single_center_mean_rotated']=float(np.mean(s['site_G_rotated']))
    for ep in c.KRR.ENDPOINTS:
        result[f'paired_{ep}_mean_rotated']=float(np.mean(s['paired'][ep]['G_rotated']))
    return result

c.write(c.OUT/'METRIC_DICTIONARY.json',{'scope':'All metrics are retrospective evaluations of old published 1304 groups',
    'axis_selection':'Original Pilot3A D_fit64/D_select32 component rule; separate native and each rotated basis winner; paired fitting does not select a new axis',
    'single_center_native':'Primary new transport endpoint: original single-center conditional ridge X64 versus X64+selected axis; two site losses averaged within group',
    'paired_gX_native':'Diagnostic: paired composition/position-summary X kernel versus X+j',
    'paired_gS_native':'Ordered-window diagnostic: paired S=X+P, where P retains each visible residue position, versus S+j',
    'paired_gL_native':'Window-interaction diagnostic: paired L=X+P+Q, where Q adds second-order position interactions, versus L+j',
    'paired_psi_native':'Diagnostic interaction contrast gX-gS; not gL and not the incremental effect of Q',
    'suffix_native_minus_rotated':'Within-network native G minus arithmetic mean of nine G values; each Haar basis separately chooses its own best axis on original development data',
    'suffix_mean_rotated':'Arithmetic mean of nine separately selected rotated-basis G values in this network; not a pooled 1791-rotation independent reference',
    'rotation_unit':'Nine matched Haar bases within each network; reused across networks; they are not independent outer-network samples',
    'outer_reference_unit':'One independently initialized network; K=199 (199 initialization seeds)',
    'reference_rank':'(1 + count(initialization metric >= trained metric))/(199+1); descriptive empirical rank, not exact exchangeability p',
    'column_mapping':'NETWORK_REFERENCE.csv has one row per network, BASIS_RESULTS.csv one row per basis; network_index 0 denotes pretrained',
    'coordinate_index':'Zero-based block3 post-GELU axis; historical pretrained native axis is 1246',
    'main_secondary_order':'Single-center conditional ridge preserves primary historical selection/readout; paired gS and gL are manuscript-linked diagnostics with original 432-pair fitting'})

reference=[]
for key,value in metrics(trained).items():
    ref=np.asarray([metrics(s)[key] for _,s,_ in networks])
    count=int(np.sum(ref>=value))
    reference.append({'metric':key,'trained':value,'initialization_median':float(np.median(ref)),
        'initialization_q025':float(np.quantile(ref,.025)),'initialization_q975':float(np.quantile(ref,.975)),
        'initialization_min':float(ref.min()),'initialization_max':float(ref.max()),'n_at_least_trained':count,
        'K':199,'upper_reference_rank_fraction':(1+count)/200,'rank_resolution':.005})

trained_group_values={}
for basis in range(10):
    folder='native' if basis==0 else f'rotated_{basis:02d}'
    with np.load(c.OUT/'trained_gpu'/folder/'site_group_losses.npz') as a:
        loss=a['group_losses']; site=loss[:,0]-loss[:,1]
    with np.load(c.OUT/'trained_gpu'/folder/'paired_arrays.npz') as a:
        contributions=a['contributions'].copy()
    trained_group_values[basis]=np.column_stack((site,contributions))
native_groups=trained_group_values[0]
rotation_mean=np.mean([trained_group_values[r] for r in range(1,10)],axis=0)
group_summary={name:{'native':c.interval(native_groups[:,i]),'mean_rotated':c.interval(rotation_mean[:,i]),
    'native_minus_mean_rotated':c.interval(native_groups[:,i]-rotation_mean[:,i])}
    for i,name in enumerate(['single_center']+list(c.KRR.ENDPOINTS))}
c.write(c.OUT/'TRAINED_GROUP_SUMMARY.json',{'created_utc':c.utc(),'conditional_intervals_only':True,
    'groups':1304,'group_summaries':group_summary})

with (c.OUT/'NETWORK_REFERENCE.csv').open('w',encoding='utf-8',newline='') as f:
    rows=[dict(index=0,network='trained_gpu',**plot_values(trained))]+[dict(index=i,network=s['network'],**plot_values(s)) for i,s,_ in networks]
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
with (c.OUT/'BASIS_RESULTS.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['network_index','network','basis','selected_axis','D_select_gain','single_center_G','paired_gX','paired_gS','paired_gL','paired_psi'])
    writer.writeheader()
    for i,s in [(0,trained)]+[(i,s) for i,s,_ in networks]:
        for row in s['rows']:
            writer.writerow(dict(network_index=i,network=s['network'],basis=row['basis'],selected_axis=row['selected_axis']['unit'],
                D_select_gain=row['selected_axis']['D_select_gain'],single_center_G=row['site_readout']['gain']['mean'],
                **{f'paired_{ep}':row['paired_readout']['endpoints'][ep]['mean'] for ep in c.KRR.ENDPOINTS}))
c.write(c.OUT/'REFERENCE_SUMMARY.json',{'created_utc':c.utc(),'status':'COMPLETE_199_INITIALIZATIONS',
    'completed_K':199,'rotation_bases_per_network':9,'evaluation_groups':1304,'development_groups_for_pair_readout':421,
    'primary_selection_components':{'D_fit':64,'D_select':32},'metric_reference':reference,
    'interpretation':'retrospective empirical architectural reference ranks, no exchangeability-based exact p value',
    'initialization_seeds':[n[2]['seed'] for n in networks]})

lines=['# Dual-reference calibration: completed results','',f'Completed {c.utc()}. All 199 predeclared initializations and all nine rotations per network completed. Trained plus initialization: 200 networks × 10 bases = 2,000 selected axes. Each basis searched 1,280 candidates using the historical D_fit64/D_select32 rule. The original native trained winner remains zero-based 1246.','',
    'The primary single-center conditional ridge endpoint and manuscript-linked paired KRR diagnostic remain separate. The latter fits each already-selected axis using the same 432 development pairs/421 groups and evaluates 1,304 old published groups. Neither is an independent new confirmation cohort.','',
    'The calibrated object is the ESM-2 8M block3 post-GELU masked-alanine HELIX-versus-STRAND task. These controls do not calibrate RSA-selected axes, 35M, other residue types, or structural mechanisms.','',
    'In the paired diagnostic, gS is improvement beyond the ordered single-position window reference S=X+P; gL is improvement beyond the richer window-interaction reference L=X+P+Q. They are distinct from the primary new single-center endpoint and from psi=gX-gS. METRIC_DICTIONARY.json specifies every plotting column.','',
    '| Metric | Trained | Init median | Init 2.5–97.5% | Init >= trained | Empirical upper rank |','|---|---:|---:|---:|---:|---:|']
for r in reference:
    lines.append(f"| {r['metric']} | {r['trained']:.8f} | {r['initialization_median']:.8f} | {r['initialization_q025']:.8f}, {r['initialization_q975']:.8f} | {r['n_at_least_trained']}/199 | {r['upper_reference_rank_fraction']:.3f} |")
lines+=['','A = native G minus the mean G of the nine independently development-selected rotated bases. Native G comparisons are J1-style architectural references; A comparisons are J2-style calibration. The empirical rank denominator is 200. A value 0.005 is the finite reference ranking resolution. Init 2.5–97.5% are empirical reference quantiles, not confidence intervals. Pretrained weights are not exchangeable with fresh initializations under a demonstrated null, so these are not exact randomization p values. The four paired endpoint summaries are descriptive; no confirmatory multiplicity-adjusted family is asserted.','',
    '## Absolute losses for trained native 1246','', '| Readout | Brier |','|---|---:|']
native=trained['rows'][0]
lines += [f"| Site X | {native['site_readout']['baseline_Brier']:.8f} |",f"| Site X+axis | {native['site_readout']['augmented_Brier']:.8f} |"]
lines += [f'| Paired {k} | {v:.8f} |' for k,v in native['paired_readout']['absolute_Brier'].items()]
lines += ['','The conditional group-level 95% intervals for native, mean rotated and their paired difference are in TRAINED_GROUP_SUMMARY.json. They condition on the fitted selected axes and do not express uncertainty over random network generation or the nine-basis Monte Carlo reference.','', '## Verification and scope','',
    '- Historical D_select winner and score replayed before calibration. Native paired KRR replayed the old published estimates. All 2,000 basis analyses passed unrestricted positive/negative-axis equivalence; signed axes are not double counted.',
    '- Full trained historical-cache/GPU activations differed by at most 2.82e-5; all ten selected axes remained the same. Full initialized CPU/GPU activations differed by at most 2.27e-6; all ten selected axes remained the same. Rank-one KRR and direct fits differed by at most 1.86e-14. See runtime_checks/CHECKS.json.',
    '- Final references use the matching GPU float32 batch8 execution for all 200 networks; CPU init001 is retained as verification only. Initialization tensors were generated on CPU before GPU transfer, with exact installed implementation versions and 199 distinct tensor hashes recorded.',
    '- Equivalent 102-fit KRR procedures reuse invariant baseline factorizations and exact single-axis rank-one updates. The per-row fits field describes the 102-fit protocol, not 102 new independent numerical factorizations.',
    '- No refitting or coordinate selection used old T. T is nevertheless retrospective and cannot become fresh independent evidence. Unknown homology and pretraining overlap remain. Random initialization is an architectural reference, not a biological mechanism or training-causality experiment.',
    '- Only this calibration directory was written. No Git, upload, manuscript change, new model family, or structural/RSA model fitting.','',
    '## Artifacts','', 'REFERENCE_SUMMARY.json contains aggregate ranks and completed denominators; NETWORK_REFERENCE.csv has one row per network; BASIS_RESULTS.csv has one row per basis. Each network directory retains activation chunks, model identity, selected-axis development scores, absolute/group losses and paired readout results. DESIGN.md fixes statistical scope; RUN_MANIFEST.json binds original inputs. run_calibration.py batch --index 2 --device cuda resumes completed-network/chunk checkpoints.']
(c.OUT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
c.write(c.OUT/'ACCEPTANCE.json',{'created_utc':c.utc(),'status':'PASS_COMPLETE_199_RETROSPECTIVE_CALIBRATION',
    'completed_initializations':199,'unique_initialization_tensor_hashes':199,'networks_with_trained':200,'bases_per_network':10,
    'selection_rule_replayed':True,'all_2000_basis_polarity_checks':True,'same_pair_diagnostic_protocol':True,
    'source_bindings_unchanged':True,'runtime_equivalence':'runtime_checks/CHECKS.json',
    'result':'RESULT.md','open_limits':['old T retrospective','empirical reference ranks not exact p','unknown homology/pretraining overlap']})
print(json.dumps({'status':'PASS','K':199,'reference':reference}))
