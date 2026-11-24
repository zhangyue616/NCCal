"""Required final descriptive summaries; never summarize incomplete cohorts."""
import sys
sys.dont_write_bytecode=True
import run_experiment as r
import numpy as np

K=r.read(r.OUT/'TIMING.json')['chosen_K']
assert r.read(r.OUT/'STATUS.json')['status']=='COMPLETE'
records=[]
for model in ('8m','35m'):
    data=[r.read(r.OUT/model/f'init_{i:03d}/RESULT.json') for i in range(1,K+1)]
    metrics={
        'pair_correct':[d['pair_correct'] for d in data],
        'selected_axis_zero_based':[d['selection']['axis_zero_based'] for d in data],
        'development_cv_mae':[d['selection']['cv_augmented_mae'] for d in data],
        'mae_gain':[d['mae_gain'] for d in data],
        'brier_gain':[d['brier_gain'] for d in data]}
    for metric,values in metrics.items():
        x=np.asarray(values)
        records.append([model,K,metric,float(np.median(x)),float(x.min()),float(x.max())])
r.tsv(r.OUT/'DESCRIPTIVE_SUMMARY.tsv',['model','K','metric','median','minimum','maximum'],records)
r.shutil.copy2(r.OUT/'DESCRIPTIVE_SUMMARY.tsv',r.AUTHOR/'DESCRIPTIVE_SUMMARY.tsv')
print('Final descriptive summaries complete',flush=True)
