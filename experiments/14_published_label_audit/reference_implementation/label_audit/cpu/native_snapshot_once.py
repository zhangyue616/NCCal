"""One native-only result snapshot after all six readouts and 215 labels close."""
import sys
sys.dont_write_bytecode=True
import csv,json,time,collections,shutil
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent.parent;CPU=HERE/'cpu';sys.path.insert(0,str(HERE))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def yes(x):return str(x).lower()in('true','1')
def table(p):
 with Path(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f,delimiter='\t'))
def number(v):return float(v)if v not in(None,'','None','nan')else None
if(CPU/'NATIVE_STAGE_SUMMARY.json').exists():print('NATIVE_SNAPSHOT_ALREADY_COMPLETE');raise SystemExit(0)
catalog=read(HERE/'UNIT_MANIFEST.json')['units'];expected={(u['unit_id'],u['concept'])for u in catalog};readouts=[HERE/'readouts'/m/'trained'for m in('esm2_8m','esm2_35m','esm2_650m','esmc_600','e1_150m')]+[HERE/'readouts/esm2_8m/trained_masked']
while True:
 if any(p.exists()for p in[HERE/'STOP_SECTION6.json',HERE.parent/'STOP_SECTION6.json']):print('NATIVE_SNAPSHOT_STOPPED_AT_AUTHOR_GATE');raise SystemExit(0)
 complete={}
 for p in(CPU/'unit_results').glob('*/COMPLETE.json'):
  try:r=read(p)
  except(json.JSONDecodeError,OSError):continue
  if r.get('reference')=='trained'and{f['frame']for f in r.get('frames',[])}=={'T','NEW'}:complete[r['unit_id'],r['concept']]=r
 if all((p/'COMPLETE.json').exists()for p in readouts)and expected.issubset(complete):break
 time.sleep(20)
import aggregate_cpu
aggregate_cpu.aggregate(199)
rows=table(HERE/'results/per_unit_results.tsv');summary=read(HERE/'results/SUMMARY.json');assert summary['pending_native_label_frames']==0
public=[r for r in rows if r['frame']=='T'and r['family']=='PUBLIC'and r['status']=='EVALUABLE'and int(r.get('auc_groups')or 0)>=20]
exclusive=dict(collections.Counter(r['classification']for r in public));assert sum(exclusive.values())==len(public)
flags={k:sum(yes(r.get(k))for r in public)for k in('AA_compatible_flag','same_AA_flag','strong_increment_flag')}
named=[]
for r in rows:
 if r['family']=='OWN'and(r['model'],r['unit0'])in[('esm2_35m','814'),('esm2_650m','4566')]:named.append({k:r.get(k)for k in('unit_id','model','unit0','frame','D_groups','groups','auc_groups','pair_accuracy','pair_q','linear_mean','linear_CI_lo','linear_CI_hi','linear_p','linear_q','strong_mean','strong_CI_lo','strong_CI_hi','strong_p','strong_q','classification')})
alpha_fix=[]
for r in rows:
 if r['frame']=='NEW'and r['family']=='OWN'and r['concept']=='OWN_EXPOSURE':
  assert int(r['auc_groups'])==154 and int(r['groups'])==136
  alpha_fix.append(dict(model=r['model'],alpha_groups=154,beta_gamma_groups=136))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');snapshot=CPU/('native_snapshot_'+stamp);snapshot.mkdir()
for name in('per_unit_results.tsv','summary_by_source_category.tsv','SUMMARY.json','REPORT_ZH.md'):shutil.copy2(HERE/'results'/name,snapshot/name)
out=dict(status='NATIVE_TABLE_COMPLETE_REFERENCE_RANKS_STILL_RUNNING',utc=datetime.now(timezone.utc).isoformat(),native_readout_directories=[str(p)for p in readouts],frozen_label_rows=len(expected),completed_native_label_rows=len(expected&set(complete)),public_T_original_selected_rows=sum(u['source']=='Hallee'for u in catalog),public_T_evaluable_rows=len(public),public_T_mutually_exclusive_classification_counts=exclusive,public_T_independent_flag_counts=flags,flags_note='Independently passed columns may overlap; do not sum them as label totals. Exclusive prose counts use classification priority strong->sameAA->AAcompatible->unresolved.',public_T_AA_compatible_flag_fraction=summary['public_T_AA_compatible_fraction'],NEW_public_distinct_physical_units=summary['NEW_public_distinct_units'],NEW_own_distinct_physical_units=summary['NEW_own_distinct_units'],NEW_cross_source_condition_met=summary['NEW_cross_source_condition_met'],native_only_guide_outcome=summary['outcome'],named_814_4566=named,alpha_correction_002_merge_acceptance=alpha_fix,snapshot_directory=str(snapshot),reference_validation='NOT_YET_COMPLETE; final K199/9 expected-universe audit and independent numerical acceptance remain pending')
(CPU/'NATIVE_STAGE_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
