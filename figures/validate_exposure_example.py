"""Validate the structural-only selection and required Figure 6a drawing inputs."""
import csv,json,subprocess,sys
from pathlib import Path
from PIL import Image

root=Path(sys.argv[1]).resolve()
examples=root/'data/artwork/examples'
spec=json.loads((examples/'exposure_pair_selection.json').read_text(encoding='utf-8'))
with (examples/'exposure_pair_selection.tsv').open(encoding='utf-8',newline='')as stream:
    pairs=list(csv.DictReader(stream,delimiter='\t'))
assert len(pairs)==967
eligible=[r for r in pairs if r['aa']==spec['filter']['amino_acid']
          and r['secondary']==spec['filter']['secondary_class_for_both_sites']
          and float(r['mapped_chain_fraction'])>=spec['filter']['minimum_mapped_chain_fraction']]
eligible.sort(key=lambda r:(abs(int(r['native_length'])-270),r['group']))
assert len(eligible)==spec['eligible_pairs']==52 and eligible[0]==spec['selected']
selected=eligible[0]
geometry=json.loads((root/spec['structural_component']).read_text(encoding='utf-8'))
assert geometry['pdb_id']==selected['pdb_id'].upper()=='2XMX'and geometry['chain']==selected['chain']=='A'
assert geometry['metadata']['group']==selected['group']
assert len(geometry['residues'])==270
assert all(all(k in r for k in ['N','CA','C','O','author_residue','sequence_position','secondary'])for r in geometry['residues'])
image=root/spec['png'];manifest=root/spec['site_manifest']
if not image.exists()or not manifest.exists():
    image.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B',str(root/spec['component_renderer']),str(root/spec['structural_component']),str(image.parent)],check=True)
info=json.loads(manifest.read_text(encoding='utf-8'));info=info[0]if isinstance(info,list)else info
assert info['pdb_id']=='2XMX'and info['chain']=='A'and info['accession']=='P05820'
assert info['group']==selected['group']and Image.open(image).size==tuple(info['size_px'])
for label,position_key,rsa_key in [('buried','pos1','rsa1'),('exposed','pos2','rsa2')]:
    site=info['sites'][label]
    assert site['position']==int(selected[position_key])and site['class']=='H'and site['residue'].startswith('ALA')
    assert site==geometry['sites'][label]
    assert abs(site['rsa']-float(selected[rsa_key]))<1e-12
    assert len(site['window_sequence'])==7 and site['window_sequence'][3]=='A'
    x,y=info['anchor_xy_px'][label]
    assert 0<=x<info['size_px'][0]and 0<=y<info['size_px'][1]
print('Figure 6a: 967 structural pairs; 52 eligible; selected 2XMX A/P05820, A86 and A248; backbone, site anchors and image complete.')
