"""Use one freed CPU slot for native AUC/pair descriptions needing no D fit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import cpu_runner as run
HERE=Path(__file__).resolve().parent
run.input_gate()
if not(HERE/'FREEZE.json').exists():raise RuntimeError('joint_FREEZE_required')
catalog=run.read(HERE/'UNIT_MANIFEST.json')['units'];done=0
for marker in sorted((HERE/'readouts').glob('*/trained*/COMPLETE.json')):
 directory=marker.parent
 for u in catalog:
  if u['model']!=directory.parent.name or bool(u.get('masked'))!=('masked'in directory.name):continue
  entry=run.feature_entry(u['concept'])
  if entry['baseline_eligible']:continue
  run.score_one(dict(directory=str(directory),unit=u['unit_id'],concept=u['concept']));done+=1
  if done%10==0:print('NATIVE_DESCRIPTIVE_COMPLETE',done,flush=True)
run.dump(HERE/'cpu/EARLY_DESCRIPTIVE_RECEIPT.json',dict(status='COMPLETE_AVAILABLE_NATIVE_D_INSUFFICIENT_DESCRIPTIONS',unit_jobs=done,additional_fits=0,workers=1,source='only GPU COMPLETE native folders, frozen worker unchanged'))
print('EARLY_DESCRIPTIVE_DONE',done,flush=True)
