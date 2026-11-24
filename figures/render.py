"""Render supported manuscript figures from repository-local intermediates."""
import argparse, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
M=ROOT/'results/artwork/main';D=ROOT/'data/artwork/molecules'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('figures',nargs='*',help='Main numbers 1-6 or supplementary identifiers such as S-C2.')
p.add_argument('--all',action='store_true',help='Render all 6 main and 23 supplementary figures.')
p.add_argument('--output',type=Path,default=HERE/'rendered')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
if a.all:a.figures=['1','2','3','4','5','6','S-B1','S-C1','S-C2','S-C3','S-C4','S-C5','S-C6','S-E1','S-G1','S-G2','S-G3','S-G4','S-I1','S-J1','S-J2','S-M1','S-N1','S-O1','S-P1','S-G5','S-Q1','S-R1','S-T1']
if not a.figures:p.error('Supply figure identifiers or --all.')
arguments={
 '1':[D/'fig1_mol2d.json',D/'ketone2_heat.json',D/'fig4_mol2d.json',M/'activation_output.json',M/'calibration.json',M/'concept_protein.json',M/'atom_activations.npz',D/'selected_atom_alignment.json',ROOT/'data/artwork/generated/prot_gray_two_sites.png',ROOT/'data/artwork/generated/protein_masked_sites_manifest.json',a.output,M/'exposure_example.json'],
 '2':[M/'calibration.json',M/'pilot.json',M/'concept_protein.json'],
 '3':[M/'concept_protein.json',M/'activation_output.json',M/'reconstruction_panel.json',D/'fig4_mol2d.json',M/'atom_activations.npz'],
 '4':[M/'activation_output.json',M/'task_output_panel.json',M/'reconstruction_panel.json',M/'concept_protein.json',M/'output_deviations.npz',M/'matched_dose.npz',M/'mechanism.json',M/'paired_dose.json'],
}
env=dict(os.environ);env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
done=set()
supplement_groups=[(['S-B1','S-I1'],'supplement_audits.py'),(['S-C1'],'supplement_tokenizer.py'),(['S-C2','S-C3','S-C4','S-C5'],'supplement_examples.py'),(['S-C6'],'supplement_pilot.py'),(['S-E1'],'supplement_mechanism.py'),(['S-G4'],'supplement_protein_validation.py'),(['S-J1'],'supplement_paired_dose.py'),(['S-J2'],'supplement_notation.py'),(['S-M1','S-N1'],'supplement_matched_controls.py'),(['S-O1','S-P1'],'supplement_layer_dictionary.py')]
for name in a.figures:
 if name in done:continue
 if name in ['5','6']:
  examples=ROOT/'data/artwork/examples';numbers=ROOT/'results/artwork/examples'
  if name=='5':
   cmd=[sys.executable,'-B',str(HERE/'native/main_5.py'),str(ROOT),str(examples/'hydroxyl_pair_inputs.jsonl.gz'),str(a.output),str(numbers/'Fig5d_CID66277342_three_SMILES.json')]
  else:
   subprocess.run([sys.executable,'-B',str(HERE/'validate_exposure_example.py'),str(ROOT)],check=True,env=env)
   cmd=[sys.executable,'-B',str(HERE/'native/main_6.py'),str(ROOT),str(numbers/'Fig6d_Q9I398_E1_1900.json'),str(a.output)]
  figure_env=dict(env);figure_env['FIGSW']='146'
  subprocess.run(cmd,check=True,env=figure_env)
  done.add(name);continue
 group=next(((names,script)for names,script in supplement_groups if name in names),None)
 if group:
  names,script=group
  subprocess.run([sys.executable,'-B',str(HERE/'native'/script),str(a.output)],check=True,env=env)
  done.update(names);continue
 if name=='S-G2':
  S=ROOT/'results/artwork/supplement'
  subprocess.run([sys.executable,'-B',str(HERE/'native/supplement_exposure_endpoints.py')]+[str(S/n)for n in ['exposure_networks.tsv','exposure_inference.tsv','exposure_rotations.tsv']]+[str(a.output)],check=True,env=env)
  done.add(name);continue
 if name in ['S-G1','S-G3']:
  asset=ROOT/'data/artwork/generated';asset.mkdir(parents=True,exist_ok=True)
  if not (asset/'prot_gray_two_sites.png').exists():subprocess.run([sys.executable,'-B',str(HERE/'native/protein_masked.py'),str(ROOT/'data/artwork/protein_masked_geometry.json'),str(asset)],check=True,env=env)
  component=a.output/'components';component.mkdir(exist_ok=True)
  inputs=[M/n for n in ['concept_protein.json','protein_group_values.npz','protein_site_predictions.npz']]+[asset/n for n in ['prot_gray_two_sites.png','protein_masked_sites_manifest.json']]+[component]
  subprocess.run([sys.executable,'-B',str(HERE/'native/supplement_masked_components.py')]+list(map(str,inputs)),check=True,env=env)
  subprocess.run([sys.executable,'-B',str(HERE/'compose_masked_supplement.py'),str(a.output)],check=True,env=env)
  done.update(['S-G1','S-G3']);continue
 if name=='S-G5':
  asset=ROOT/'data/artwork/generated';asset.mkdir(parents=True,exist_ok=True)
  for script,geometry,image in [('protein_masked.py','protein_masked_geometry.json','prot_gray_two_sites.png'),('protein_exposure.py','protein_exposure_geometry.json','prot_exposure_pair_gray.png')]:
   if not (asset/image).exists():subprocess.run([sys.executable,'-B',str(HERE/'native'/script),str(ROOT/'data/artwork'/geometry),str(asset)],check=True,env=env)
  component=a.output/'components';component.mkdir(exist_ok=True)
  U=ROOT/'results/artwork/use';S=ROOT/'results/artwork/supplement'
  inputs=[M/'concept_protein.json',asset/'prot_gray_two_sites.png',asset/'protein_masked_sites_manifest.json',U/'s_g5c_network_reference.csv',U/'s_g5c_basis_results.csv',U/'6d_and_s_g5_pairs.tsv',U/'6d_classification_summary.tsv',U/'s_g5a_paired_activations_814.tsv',S/'exposure_networks.tsv',asset/'prot_exposure_pair_gray.png',asset/'protein_exposure_pair_manifest.json',U/'6d_network_results.tsv',component]
  subprocess.run([sys.executable,'-B',str(HERE/'native/protein_panels.py')]+list(map(str,inputs)),check=True,env=env)
  subprocess.run([sys.executable,'-B',str(HERE/'plot_protein_supplement.py'),str(a.output)],check=True,env=env)
  continue
 if name in arguments:
  if name=='1':
   asset=ROOT/'data/artwork/generated';asset.mkdir(parents=True,exist_ok=True)
   if not (asset/'prot_gray_two_sites.png').exists():
    subprocess.run([sys.executable,'-B',str(HERE/'native/protein_masked.py'),str(ROOT/'data/artwork/protein_masked_geometry.json'),str(asset)],check=True,env=env)
  cmd=[sys.executable,'-B',str(HERE/'native'/('main_'+name+'.py'))]+list(map(str,arguments[name]))+([]if name=='1'else[str(a.output)])
 elif name in ['S-Q1','S-R1','S-T1']:
  label='Figure_'+name.replace('-','_')
  cmd=[sys.executable,'-B',str(HERE/'plot_use_results.py'),label,str(a.output)]
 else:raise SystemExit('Figure '+name+' is recorded in the inventory but its portable renderer is not ready.')
 subprocess.run(cmd,check=True,env=env)
