"""Final concentrated acceptance: denominators plus three independent score samples."""
import sys
sys.dont_write_bytecode=True
import csv, hashlib, json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=read(OUT/'SUMMARY.json');selection=read(OUT/'PAIR_SELECTION.json')
table=list(csv.DictReader((OUT/'PAIRED_SCORES.tsv').open(encoding='utf-8'),delimiter='\t'))
assert len(table)==len(selection['pairs'])==153
assert all('sequence' not in row for row in table)
source=ROOT/'cross_architecture/external'
seqs={s['accession']:s for s in [json.loads(t) for t in (source/'sequences.jsonl').read_text().splitlines() if t.strip()]}
sites={(s['group'],int(s['pos'])):s for s in csv.DictReader((source/'sites.tsv').open(encoding='utf-8'),delimiter='\t')}
strict=set(read(source.parent/'validation/HOMOLOGY.json')['retained_groups'])
aa_order='ACDEFGHIKLMNPQRSTVWY'
offsets=[*range(-15,0),*range(1,16)]
def features(seq,q):
    out=[]
    for rad in (None,7,15):
        lo=0 if rad is None else max(0,q-rad)
        hi=len(seq) if rad is None else min(len(seq),q+rad+1)
        visible=seq[lo:q]+seq[q+1:hi]
        out.extend(visible.count(a)/len(visible) for a in aa_order)
    relative=q/(len(seq)-1)
    return np.array(out+[relative,relative**2,len(seq)/256,(min(q,15)+min(len(seq)-q-1,15))/30])
def positions(seq,q):return np.array([aa_order.index(seq[q+d]) if 0<=q+d<len(seq) else 20 for d in offsets])
sample=[]
with np.load(ROOT/'results/revision/protein_D/FINAL_MODELS.npz') as fit:
    for aa in 'ALV':
        row=min((r for r in table if r['identity']==aa),key=lambda r:r['selection_sha256'])
        seq=seqs[row['id']]['sequence'];early,late=int(row['early']),int(row['late']);qa,qb=early-1,late-1
        a,b=sites[row['group'],early],sites[row['group'],late]
        assert early<late and late-early>30 and a['chain']==b['chain']==row['chain']
        assert seq[qa]==seq[qb]==a['aa']==b['aa']==aa
        assert a['main_clean']==b['main_clean']=='True' and a['interface']==b['interface']=='False'
        assert {a['P_SEA'],b['P_SEA']}=={'H','E'} and int(row['y_a'])==int(a['P_SEA']=='H')
        dx=features(seq,qa)-features(seq,qb);pa,pb=positions(seq,qa),positions(seq,qb)
        matches=[(p==fit[key]).sum(axis=1) for p,key in ((pa,'train_pa'),(pa,'train_pb'),(pb,'train_pa'),(pb,'train_pb'))]
        q=[m*(m-1)/(30*29) for m in matches]
        raw={'X':dx@fit['train_dx'].T,'P':matches[0]-matches[1]-matches[2]+matches[3],
            'Q':q[0]-q[1]-q[2]+q[3],'j':(float(row['j_early'])-float(row['j_late']))*fit['train_dj']}
        errors={}
        for arm,blocks in (('X',('X',)),('Xj',('X','j')),('L',('X','P','Q')),('Lj',('X','P','Q','j'))):
            kernel=sum(raw[k]/float(fit['scale_'+k]) for k in blocks)
            score=float(kernel@fit[arm+'_alpha']);p=float(np.clip((1+score)/2,0,1))
            error=abs(score-float(row[arm+'_score']));assert error<1e-10
            assert abs(p-float(row[arm+'_p_a']))<1e-12
            errors[arm]=error
        sample.append({'identity':aa,'id':row['id'],'group':row['group'],'early':early,'late':late,'y_a':int(row['y_a']),
            'same_chain_identity_filters':'PASS','manual_kernel_score_max_errors':errors})
denominators=[]
for result in summary['results']:
    filtered=[r for r in table if r['identity']==result['identity'] and (result['frame']=='structure_cluster' or r['group'] in strict)]
    assert len(filtered)==result['n_pairs']==result['n_groups']
    assert len({r['group'] for r in filtered})==len(filtered)
    for arm in ('X','Xj','L','Lj'):
        y=np.array([int(r['y_a']) for r in filtered]);p=np.array([float(r[arm+'_p_a']) for r in filtered])
        loss=(p-y)**2
        assert abs(loss.mean()-result['arms'][arm]['Brier']['mean'])<1e-14
        assert all(abs(v-float(r[arm+'_loss']))<1e-14 for v,r in zip(loss,filtered))
        assert result['arms'][arm]['probability_ties']==result['arms'][arm]['score_ties']==0
        assert sum(result['arms'][arm][k] for k in ('correct_non_tied','incorrect_non_tied','probability_ties'))==len(filtered)
    denominators.append({'frame':result['frame'],'identity':result['identity'],'denominator':len(filtered),
        'no_eligible_pair':(55 if result['frame']=='direct_sequence_screen' else 118)-len(filtered)})
audit={'status':'PASS','new_forwards':0,'new_fits':0,'three_manual_kernel_samples':sample,
    'all_six_denominators_and_Brier_means':denominators,'no_prediction_ties':True,
    'normalization':'raw X/P/Q/j blocks each divided by archived D mean diagonal; no coordinate normalization or external fitting',
    'orientation':'earlier minus later; p_earlier_helix=clip((1+score)/2,0,1); label used only for loss',
    'plan_sha256':sha(OUT/'PROTOCOL.md'),'model_sha256':sha(ROOT/'results/revision/protein_D/FINAL_MODELS.npz')}
(OUT/'FINAL_SPOTCHECK.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
def f(v):return f'{v:.6f}'
def ci(v):return '['+', '.join(f(x) for x in v)+']'
lines=['# Frozen alanine readout transfer to leucine and valine','',
'Status: complete; concentrated acceptance and independent manual-score spot checks PASS. Computational confidence: high. Statistical conclusions remain conditional on the archived cohort and frozen readout.','',
'## Primary direct-sequence-screen frame','',
'Positive gain means lower paired Brier loss after adding neuron 1246. L is the local interaction comparator; Lj adds the fixed neuron. Each identity contributes at most one helix/strand pair per independent group.','',
'| Covered residue | Pairs/groups | L Brier | Lj Brier | L − Lj | 97.5% simultaneous percentile interval | Pair accuracy L → Lj |',
'|---|---:|---:|---:|---:|---|---|']
for r in summary['results']:
    if r['frame']!='direct_sequence_screen' or r['identity']=='A':continue
    g=r['comparisons']['L_minus_Lj'];arms=r['arms']
    lines.append(f"| {r['identity']} | {r['n_pairs']} | {f(arms['L']['Brier']['mean'])} | {f(arms['Lj']['Brier']['mean'])} | {f(g['mean'])} | {ci(g['CI97_5_simultaneous_percentile'])} | {100*arms['L']['conditional_pair_accuracy']['mean']:.2f}% → {100*arms['Lj']['conditional_pair_accuracy']['mean']:.2f}% |")
lines += ['', 'Both primary intervals span zero. Point estimates favor adding the fixed neuron for both residues, while the current pair counts do not resolve the sign of the gain. No pooling or comparator selection was performed.','',
'Suggested manuscript wording: “With the alanine-fitted readout held fixed, adding neuron 1246 lowered paired Brier loss for leucine and valine in experimental structures. The simultaneous intervals spanned zero, quantifying the uncertainty of this cross-residue transfer estimate.”','',
'## All specified frames and references','',
'| Frame | Residue | Pairs/groups | X Brier | Xj Brier | L Brier | Lj Brier | L − Lj (95% interval) | X − Xj (95% interval) |',
'|---|---|---:|---:|---:|---:|---:|---|---|']
for r in summary['results']:
    a=r['arms'];g=r['comparisons']['L_minus_Lj'];x=r['comparisons']['X_minus_Xj']
    lines.append('| '+' | '.join([r['frame'],r['identity'],str(r['n_pairs']),*[f(a[m]['Brier']['mean']) for m in ('X','Xj','L','Lj')],f(g['mean'])+' '+ci(g['CI95_percentile']),f(x['mean'])+' '+ci(x['CI95_percentile'])])+' |')
lines += ['', 'The alanine reference and original structure-cluster frame are descriptive. The original cluster-frame V comparison against X has a positive 95% interval; it does not replace the locked local-interaction primary comparison or the direct-sequence-screen frame.','',
'| Frame | Residue | Pair accuracy X / Xj / L / Lj | Exact prediction ties, all arms |','|---|---|---|---:|']
for r in summary['results']:
    lines.append('| '+' | '.join([r['frame'],r['identity'],' / '.join(f"{100*r['arms'][m]['conditional_pair_accuracy']['mean']:.2f}%" for m in ('X','Xj','L','Lj')),'0'])+' |')
lines += ['', 'Accuracy uses half credit for probability or score ties; none occurred. Raw unbounded scores, clipped probabilities, both losses, paired gains and centre activations are retained for every pair in `PAIRED_SCORES.tsv`, with no sequences. Full 95% intervals for absolute losses and accuracies are in `SUMMARY.json`.','',
'## Coverage and implementation','',
'The archived tranche has 118 natural full sequences (97–511 aa); 55 groups remain after the existing direct-sequence screen and 63 are outside that primary frame. No sequences were excluded for noncanonical residues or length. Strict eligible-pair counts are A19/L28/V21; original structure-cluster counts are A43/L58/V52. Missing pairs arise from the locked H/E identity and separation requirements: strict A36/L27/V34 groups; original A75/L60/V66 groups. A complete group-by-identity eligibility ledger, including H/E site counts and candidate denominators, is retained in `PAIR_SELECTION.json`.','',
'Pairs use the same natural sequence and mapped chain, the same covered identity, main-clean non-interface sites, P-SEA H versus E, and separation >30 residues. The minimum prescribed SHA-256 selects the pair independently of activation or helix ordering. Inputs always retain earlier/later order. Every independent full-sequence pass masks exactly the target centre and preserves all other residues. No cropping, new acquisition, fitting, sequence design or biological-function optimization occurred.','',
'Three deterministic archived alanine pairs were replayed before external scoring: six centre activations agreed to a maximum absolute error of 3.595e−7, and all three pair scores in each of four archived paired-readout arms were reproduced to <1e−10 using archived activations. The external extraction used 306 forward passes (plus six replay passes), float32 CPU, two threads, in about 11.49 s for the external passes. The hook is ESM-2 8M block 3 (zero-based 2), intermediate post-GELU unit 1246. The input sequence is full length; only the first three blocks need evaluation to obtain that hook.','',
'All 153 paired rows, all six aggregate denominators and Brier means passed concentrated checks. Three separately calculated score samples (one per identity) reproduced the archived-scale kernel algebra, central identity, same-chain filters and orientation; scores also reverse when the paired inputs are swapped. Bootstrap inference uses 10,000 paired group draws, seed 20261122, with 95% descriptive intervals and 97.5% simultaneous intervals for the two primary identities.','',
'The follow-up was specified before scoring this endpoint in the external cohort; that cohort had previously been examined for solvent exposure and remote contacts. Training labels were UniProt annotations; external labels are experimental-coordinate P-SEA assignments. These are observations of paired structure discrimination conditional on the saved fit and cohort.','',
'## Bound artifacts','',
f"- Plan SHA-256: `{sha(OUT/'PROTOCOL.md')}`.",
f"- Frozen paired model SHA-256: `{audit['model_sha256']}`.",
'- `BINDINGS.json`: plan, script, model weights, readout, source tables and archived replay identities, bound before new predictions.',
'- `REPLAY.json`, `MODEL_IDENTITY.json`, `ACTIVATIONS.npz`: actual replay and extraction evidence.',
'- `SUMMARY.json`, `PAIRED_SCORES.tsv`: complete six-frame results and pair-level attachment.',
'- `ACCEPTANCE.json`, `FINAL_SPOTCHECK.json`: computational acceptance and manual-score spot checks.',
'- `run_transfer.py`: executable evaluation; `finish_report.py`: report and final checks. Historical inputs were read only.','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'status':'PASS','pair_rows':len(table),'sample_count':len(sample),'report':str(OUT/'REPORT.md')}))
