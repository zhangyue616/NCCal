"""Audit actual finished reference families; no statistical definitions changed."""
import sys
sys.dont_write_bytecode=True
import collections,csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;CPU=HERE/'cpu';RESULTS=HERE/'results'
catalog=json.loads((HERE/'UNIT_MANIFEST.json').read_text(encoding='utf-8'))['units'];byunit={u['unit_id']:u for u in catalog}
with(HERE/'inputs/UNIT_ELIGIBILITY.tsv').open(encoding='utf-8-sig',newline='')as f:eligibility=list(csv.DictReader(f,delimiter='\t'))
expected=set()
for r in eligibility:
 if int(r['D_pairs'])>=20 and int(r['T_pairs'])>=20:
  u=byunit[r['unit_id']];assert (r['model'],r['concept'],r['family'],r['masked'].lower()=='true')==(u['model'],u['concept'],u['family'],bool(u.get('masked')))
  expected.add((u['model'],u['concept'],u['family'],bool(u.get('masked'))))
jobs=[json.loads(p.read_text(encoding='utf-8'))for p in(CPU/'unit_results').glob('*/COMPLETE.json')];families=collections.defaultdict(lambda:dict(random=set(),rotation=set(),directories=set()))
for job in jobs:
 ref=str(job.get('reference',''))
 if ref=='trained':continue
 for r in job.get('frames',[]):
  key=(r['model'],r['concept'],r.get('original_family'),bool(r.get('masked')))
  if ref.startswith('random_'):families[key]['random'].add(ref)
  elif ref.startswith('rotation_'):families[key]['rotation'].add(ref)
  families[key]['directories'].add(job.get('directory',''))
rows=[]
observed=set(families)
for key in sorted(expected|observed,key=str):
 model,concept,original,masked=key;v=families[key]
 required_random={('random_masked_'if masked else'random_')+f'{i:03d}'for i in range(1,200)};required_rotation={('rotation_masked_'if masked else'rotation_')+f'{i:02d}'for i in range(1,10)}
 random_missing=required_random-v['random'];rotation_missing=required_rotation-v['rotation'];unexpected=(v['random']-required_random)|(v['rotation']-required_rotation)
 rows.append(dict(model=model,exactconcept=concept,original_family=original,masked=masked,expected_from_frozen_manifest_and_D_T_gate=key in expected,random_networks_completed=len(v['random']),random_K199_complete=not random_missing and not unexpected,rotation_axes_completed=len(v['rotation']),rotation_9_complete=not rotation_missing and not unexpected,all_199_plus_9_complete=not random_missing and not rotation_missing and not unexpected,missing_random_ids='|'.join(sorted(random_missing)),missing_rotation_ids='|'.join(sorted(rotation_missing)),unexpected_reference_ids='|'.join(sorted(unexpected)),entire_expected_family_missing=key in expected and not v['random']and not v['rotation'],readout_directories='|'.join(sorted(v['directories']))))
RESULTS.mkdir(exist_ok=True)
with(RESULTS/'reference_family_completion.tsv').open('w',encoding='utf-8',newline='')as f:
 fields=list(rows[0])if rows else['model','exactconcept','original_family','masked','random_networks_completed','random_K199_complete','rotation_axes_completed','rotation_9_complete','all_199_plus_9_complete','readout_directories'];w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
expected_rows=[r for r in rows if r['expected_from_frozen_manifest_and_D_T_gate']]
summary=dict(expected_families=len(expected),observed_families=len(observed&expected),missing_expected_family_count=sum(r['entire_expected_family_missing']for r in expected_rows),incomplete_expected_family_count=sum(not r['all_199_plus_9_complete']for r in expected_rows),unexpected_observed_family_count=len(observed-expected),K199_complete_families=sum(r['random_K199_complete']for r in expected_rows),nine_rotation_complete_families=sum(r['rotation_9_complete']for r in expected_rows),both_complete_families=sum(r['all_199_plus_9_complete']for r in expected_rows),status='EXPECTED_FROZEN_REFERENCE_UNIVERSE_COMPARED_WITH_ACTUAL_RECEIPTS',source='Frozen UNIT_MANIFEST plus D/T pair groups>=20; actual CPU jobs accepted only after GPU COMPLETE. Wholly absent expected keys remain in denominator.')
(RESULTS/'REFERENCE_COMPLETION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary))
